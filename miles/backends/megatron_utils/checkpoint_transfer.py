import logging
from collections.abc import Sequence
from datetime import timedelta

import torch

try:
    from torchft.checkpointing.pg_transport import PGTransport
except ImportError:
    PGTransport = None

from miles.backends.megatron_utils.in_memory_checkpoint import InMemoryCheckpointManager, save_to_memory
from miles.utils.process_group_utils import GroupInfo

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = timedelta(seconds=120)


def send_ckpt(
    *,
    indep_dp: GroupInfo,
    model: Sequence,
    optimizer: object,
    opt_param_scheduler: object,
    iteration: int,
    dst_rank: int,
    timeout: timedelta = _DEFAULT_TIMEOUT,
) -> None:
    """Send in-memory checkpoint to a destination cell via torchft PGTransport."""
    import time as _time

    logger.info("send_ckpt: starting save_to_memory")
    t0 = _time.monotonic()
    state_dict = save_to_memory(
        iteration=iteration,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
    )
    logger.info("send_ckpt: save_to_memory done (%.1fs)", _time.monotonic() - t0)

    payload = {"iteration": iteration, "state_dict": state_dict}
    transport = _create_transport(indep_dp, timeout)
    # Warm up the PG connection before P2P transfer.
    # Gloo/NCCL P2P requires the communicator to be established first.
    import torch.distributed as dist

    logger.info("send_ckpt: warming up PG connection via allreduce")
    warmup_t = torch.ones(1, device="cpu")
    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    indep_dp.gloo_group.allreduce([warmup_t], opts).wait()
    logger.info("send_ckpt: warmup allreduce done (value=%s)", warmup_t.item())

    logger.info("send_ckpt: starting send_checkpoint to dst_rank=%d (pg=%s)", dst_rank, type(indep_dp.gloo_group).__name__)
    t1 = _time.monotonic()
    transport.send_checkpoint(
        dst_ranks=[dst_rank],
        step=0,
        state_dict=payload,
        timeout=timeout,
    )
    logger.info("send_ckpt: send_checkpoint done (%.1fs)", _time.monotonic() - t1)
    transport.disallow_checkpoint()
    logger.info(f"Sent checkpoint (iteration={iteration}) to alive_rank={dst_rank}")


def recv_ckpt(
    *,
    indep_dp: GroupInfo,
    src_rank: int,
    timeout: timedelta = _DEFAULT_TIMEOUT,
) -> InMemoryCheckpointManager:
    """Receive checkpoint from a healthy cell via torchft PGTransport.

    Returns an InMemoryCheckpointManager containing the received state_dict,
    ready to be passed to initialize_model_and_optimizer.

    Args:
        indep_dp: Independent DP group info (provides the torchft PG).
        src_rank: Source alive_rank in the indep_dp process group.
        timeout: Timeout for the NCCL recv operation.

    Returns:
        InMemoryCheckpointManager with state_dict loaded, ready for
        initialize_model_and_optimizer to consume.
    """
    # Warm up the PG connection before P2P transfer.
    import torch.distributed as dist

    logger.info("recv_ckpt: warming up PG connection via allreduce")
    warmup_t = torch.ones(1, device="cpu")
    opts = dist.AllreduceOptions()
    opts.reduceOp = dist.ReduceOp.SUM
    indep_dp.gloo_group.allreduce([warmup_t], opts).wait()
    logger.info("recv_ckpt: warmup allreduce done (value=%s)", warmup_t.item())

    transport = _create_transport(indep_dp, timeout)
    payload = transport.recv_checkpoint(
        src_rank=src_rank,
        metadata=transport.metadata(),
        step=0,
        timeout=timeout,
    )
    iteration = payload["iteration"]
    state_dict = payload["state_dict"]
    logger.info(f"Received checkpoint (iteration={iteration}) from alive_rank={src_rank}")

    manager = InMemoryCheckpointManager()
    manager.save(state_dict, iteration=iteration)
    return manager


def _create_transport(indep_dp: GroupInfo, timeout: timedelta) -> PGTransport:
    return PGTransport(
        pg=indep_dp.gloo_group,
        timeout=timeout,
        device=torch.device("cpu"),
    )
