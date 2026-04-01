import os
from typing import Any

import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.megatron_utils.indep_dp import _intra_cell_consensus


def _init_gloo(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _run(fn: Any, world_size: int = 2) -> None:
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)


def _worker_consensus(rank: int, world_size: int, *, success_by_rank: dict[int, bool], expected: bool) -> None:
    _init_gloo(rank, world_size)
    try:
        group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        result = _intra_cell_consensus(success=success_by_rank[rank], gloo_group=group)
        assert result is expected, f"rank {rank}: expected {expected}, got {result}"
    finally:
        dist.destroy_process_group()


class TestIntraCellConsensus:
    def test_all_true(self) -> None:
        def _worker(rank: int, world_size: int) -> None:
            _worker_consensus(rank, world_size, success_by_rank={0: True, 1: True}, expected=True)

        _run(_worker)

    def test_all_false(self) -> None:
        def _worker(rank: int, world_size: int) -> None:
            _worker_consensus(rank, world_size, success_by_rank={0: False, 1: False}, expected=False)

        _run(_worker)

    def test_mixed_returns_false(self) -> None:
        def _worker(rank: int, world_size: int) -> None:
            _worker_consensus(rank, world_size, success_by_rank={0: True, 1: False}, expected=False)

        _run(_worker)
