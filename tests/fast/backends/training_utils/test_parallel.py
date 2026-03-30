"""Tests for GroupsInfo and all_reduce_multi using real Gloo process groups."""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.training_utils.parallel import GroupInfo, GroupsInfo, all_reduce_multi


def _init_gloo(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _cleanup() -> None:
    dist.destroy_process_group()


class TestGroupsInfo:
    def test_from_single(self) -> None:
        info = GroupInfo(rank=2, size=4, group=None)
        groups_info = GroupsInfo.from_single(info)

        assert groups_info.rank == 2
        assert groups_info.size == 4
        assert groups_info.groups_inner_to_outer == [None]

    def test_from_pair(self) -> None:
        inner = GroupInfo(rank=1, size=3, group=None)
        outer = GroupInfo(rank=2, size=4, group=None)
        groups_info = GroupsInfo.from_pair(inner=inner, outer=outer)

        assert groups_info.rank == 2 * 3 + 1  # outer.rank * inner.size + inner.rank = 7
        assert groups_info.size == 4 * 3  # outer.size * inner.size = 12
        assert groups_info.groups_inner_to_outer == [None, None]

    def test_from_pair_rank_zero_only_when_both_zero(self) -> None:
        groups_info = GroupsInfo.from_pair(
            inner=GroupInfo(rank=0, size=2, group=None),
            outer=GroupInfo(rank=0, size=3, group=None),
        )
        assert groups_info.rank == 0
        assert groups_info.size == 6

    def test_from_pair_rank_nonzero_when_inner_nonzero(self) -> None:
        groups_info = GroupsInfo.from_pair(
            inner=GroupInfo(rank=1, size=2, group=None),
            outer=GroupInfo(rank=0, size=3, group=None),
        )
        assert groups_info.rank == 1  # 0 * 2 + 1

    def test_from_pair_rank_nonzero_when_outer_nonzero(self) -> None:
        groups_info = GroupsInfo.from_pair(
            inner=GroupInfo(rank=0, size=2, group=None),
            outer=GroupInfo(rank=1, size=3, group=None),
        )
        assert groups_info.rank == 2  # 1 * 2 + 0


class TestAllReduceMulti:
    @staticmethod
    def _run(fn, world_size: int) -> None:
        mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)

    @staticmethod
    def _single_group_worker(rank: int, world_size: int) -> None:
        _init_gloo(rank, world_size)
        try:
            group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
            tensor = torch.tensor([float(rank + 1)])

            all_reduce_multi(tensor, [group], op=dist.ReduceOp.SUM)

            expected = sum(range(1, world_size + 1))
            assert tensor.item() == expected, f"rank {rank}: expected {expected}, got {tensor.item()}"
        finally:
            _cleanup()

    def test_single_group(self) -> None:
        self._run(self._single_group_worker, world_size=4)

    @staticmethod
    def _two_groups_worker(rank: int, world_size: int) -> None:
        """4 ranks split into 2 inner groups of 2, then outer group connects inner rank 0s.

        After SUM on inner then SUM on outer, rank 0 should have the global sum.
        """
        _init_gloo(rank, world_size)
        try:
            inner_ranks = [0, 1] if rank < 2 else [2, 3]
            inner_group = dist.new_group(ranks=inner_ranks, backend="gloo")

            outer_ranks = [0, 2]
            outer_group = dist.new_group(ranks=outer_ranks, backend="gloo")

            tensor = torch.tensor([float(rank + 1)])

            # Step 1: inner allreduce
            all_reduce_multi(tensor, [inner_group], op=dist.ReduceOp.SUM)

            inner_sum = sum(r + 1 for r in inner_ranks)
            assert tensor.item() == inner_sum, f"rank {rank}: after inner, expected {inner_sum}, got {tensor.item()}"

            # Step 2: outer allreduce (only ranks 0 and 2 participate)
            if rank in outer_ranks:
                all_reduce_multi(tensor, [outer_group], op=dist.ReduceOp.SUM)
                global_sum = sum(range(1, world_size + 1))
                assert tensor.item() == global_sum, (
                    f"rank {rank}: after outer, expected {global_sum}, got {tensor.item()}"
                )
        finally:
            _cleanup()

    def test_two_groups(self) -> None:
        self._run(self._two_groups_worker, world_size=4)

    @staticmethod
    def _sequential_worker(rank: int, world_size: int) -> None:
        """Two sequential allreduces on the same group: SUM twice = SUM * world_size."""
        _init_gloo(rank, world_size)
        try:
            group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
            tensor = torch.tensor([float(rank)])

            all_reduce_multi(tensor, [group, group], op=dist.ReduceOp.SUM)

            # First SUM: 0+1+2+3 = 6. Second SUM: 6*4 = 24
            first_sum = sum(range(world_size))
            expected = first_sum * world_size
            assert tensor.item() == expected, f"rank {rank}: expected {expected}, got {tensor.item()}"
        finally:
            _cleanup()

    def test_sequential(self) -> None:
        self._run(self._sequential_worker, world_size=4)

    @staticmethod
    def _empty_groups_worker(rank: int, world_size: int) -> None:
        """Empty groups list should be a no-op."""
        _init_gloo(rank, world_size)
        try:
            tensor = torch.tensor([42.0])
            all_reduce_multi(tensor, [], op=dist.ReduceOp.SUM)
            assert tensor.item() == 42.0
        finally:
            _cleanup()

    def test_empty_groups(self) -> None:
        self._run(self._empty_groups_worker, world_size=2)
