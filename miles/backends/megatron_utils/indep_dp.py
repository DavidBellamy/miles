import logging
import time
from collections.abc import Sequence
from datetime import timedelta
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from miles.utils.indep_dp import IndepDPInfo
from miles.utils.process_group_utils import GeneralPGUtil, GroupInfo, collective_bool_and

from ..training_utils.parallel import ParallelState

if TYPE_CHECKING:
    from megatron.core.distributed import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

_INDEP_DP_ALLREDUCE_POLL_INTERVAL: float = 0.05  # 50ms
_INDEP_DP_ALLREDUCE_TIMEOUT: float = 60.0  # seconds


def create_indep_dp_group(
    store_addr: str | None,
    indep_dp_info: IndepDPInfo,
    megatron_rank: int,
    megatron_world_size: int,
) -> GroupInfo:
    if indep_dp_info.alive_size <= 1:
        return GroupInfo(rank=0, size=1, group=None)

    try:
        from torchft.process_group import ProcessGroupGloo, ProcessGroupNCCL
    except ImportError as e:
        raise ImportError("torchft is required for indep_dp. Install with: pip install torchft") from e

    def _create(pg_cls: type, backend_name: str) -> dist.ProcessGroup:
        # Must be large enough to tolerate cross-cell step-time skew (~30s observed),
        # but not so large that a truly dead cell takes minutes to detect.
        # TODO: tune this value based on production workload profiling.
        pg = pg_cls(timeout=timedelta(seconds=120))
        pg.configure(
            store_addr=f"{store_addr}/indep_dp/{backend_name}/{indep_dp_info.quorum_id}/{megatron_rank}",
            replica_id=str(indep_dp_info.cell_index),
            rank=indep_dp_info.alive_rank,
            world_size=indep_dp_info.alive_size,
            quorum_id=indep_dp_info.quorum_id,
            group_rank=megatron_rank,
            group_world_size=megatron_world_size,
        )
        return pg

    nccl_pg = _create(ProcessGroupNCCL, "nccl")
    gloo_pg = _create(ProcessGroupGloo, "gloo")
    logger.info(
        f"Configured independent DP PG: {indep_dp_info}, "
        f"megatron_rank={megatron_rank}, megatron_world_size={megatron_world_size}"
    )
    return GroupInfo(rank=indep_dp_info.alive_rank, size=indep_dp_info.alive_size, group=nccl_pg, gloo_group=gloo_pg)


def reconfigure_indep_dp_group(
    parallel_state: ParallelState,
    store_addr: str | None,
    indep_dp_info: IndepDPInfo,
    megatron_rank: int,
    megatron_world_size: int,
) -> None:
    """Shutdown old indep_dp PGs and create new ones with a fresh quorum_id."""
    old = parallel_state.indep_dp
    for g in [old.group, old.gloo_group]:
        if g is not None:
            g.shutdown()

    parallel_state.indep_dp = create_indep_dp_group(
        store_addr=store_addr,
        indep_dp_info=indep_dp_info,
        megatron_rank=megatron_rank,
        megatron_world_size=megatron_world_size,
    )
    logger.info(f"Reconfigured indep_dp PG with quorum_id={indep_dp_info.quorum_id}")


def _poll_work_until_complete(work: dist._Work, pg: dist.ProcessGroup, timeout: float) -> None:
    """Poll a non-blocking NCCL work until completion or timeout.

    On timeout, calls ``pg.shutdown()`` (non-blocking, avoids ``ncclCommAbort``
    which can hang on NVLink) and raises ``TimeoutError``.
    """
    deadline = time.monotonic() + timeout
    while not work.is_completed():
        if time.monotonic() > deadline:
            logger.error("indep_dp allreduce timed out after %.0fs, shutting down PG (no abort)", timeout)
            pg.shutdown()
            raise TimeoutError(f"indep_dp allreduce timed out after {timeout}s")
        time.sleep(_INDEP_DP_ALLREDUCE_POLL_INTERVAL)
    success = work.wait()
    if not success:
        raise RuntimeError("indep_dp allreduce failed (wait returned False)")


def _allreduce_grads_across_replicas(args, model: Sequence["DDP"], parallel_state: ParallelState) -> bool:
    assert not args.calculate_per_token_loss, "calculate_per_token_loss is not supported with indep_dp yet"
    assert parallel_state.intra_dp.size == 1, (
        f"indep_dp requires intra_dp.size == 1, got {parallel_state.intra_dp.size}. "
        "Simultaneous intra and indep DP is not supported."
    )

    pg = parallel_state.indep_dp.group

    allreduce_success = True
    try:
        for model_chunk in model:
            # mimic: DistributedDataParallel.start_grad_sync
            for bucket_group in model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups:
                for bucket in bucket_group.buckets:
                    opts = dist.AllreduceOptions()
                    opts.reduceOp = dist.ReduceOp.SUM
                    work = pg.allreduce([bucket.grad_data], opts)
                    _poll_work_until_complete(work, pg, timeout=_INDEP_DP_ALLREDUCE_TIMEOUT)
    except Exception:
        allreduce_success = False
        logger.exception("Gradient allreduce across replicas failed")

    try:
        cross_cell_ok = collective_bool_and(value=allreduce_success, group=parallel_state.indep_dp.gloo_group)
    except Exception:
        logger.exception("collective_bool_and for gradient allreduce failed (peer cell likely dead)")
        cross_cell_ok = False

    # Synchronize within the cell: if ANY rank in this cell detected failure,
    # ALL ranks must return False so the entire cell discards this step.
    ok_tensor = torch.tensor([1 if cross_cell_ok else 0], dtype=torch.int32, device="cuda")
    dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
    return bool(ok_tensor.item())
