"""Tests for GroupsInfo and all_reduce_multi using real Gloo process groups."""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

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
        result = GroupsInfo.from_single(info)

        assert result.rank == 2
        assert result.size == 4
        assert result.groups_inner_to_outer == [None]

    def test_from_pair(self) -> None:
        inner = GroupInfo(rank=1, size=3, group=None)
        outer = GroupInfo(rank=2, size=4, group=None)
        result = GroupsInfo.from_pair(inner=inner, outer=outer)

        assert result.rank == 2 * 3 + 1  # 7
        assert result.size == 4 * 3  # 12
        assert result.groups_inner_to_outer == [None, None]

    def test_from_pair_rank_zero_only_when_both_zero(self) -> None:
        result = GroupsInfo.from_pair(
            inner=GroupInfo(rank=0, size=2, group=None),
            outer=GroupInfo(rank=0, size=3, group=None),
        )
        assert result.rank == 0
        assert result.size == 6

    def test_from_pair_rank_nonzero_when_inner_nonzero(self) -> None:
        result = GroupsInfo.from_pair(
            inner=GroupInfo(rank=1, size=2, group=None),
            outer=GroupInfo(rank=0, size=3, group=None),
        )
        assert result.rank == 1

    def test_from_pair_rank_nonzero_when_outer_nonzero(self) -> None:
        result = GroupsInfo.from_pair(
            inner=GroupInfo(rank=0, size=2, group=None),
            outer=GroupInfo(rank=1, size=3, group=None),
        )
        assert result.rank == 2


class TestAllReduceMulti:
    """Test all_reduce_multi with real Gloo groups via 2D DeviceMesh.

    Uses a (2, 2) mesh with dimensions ("inner", "outer") on 4 ranks.
    Mesh layout:
        rank 0: inner=0, outer=0
        rank 1: inner=1, outer=0
        rank 2: inner=0, outer=1
        rank 3: inner=1, outer=1

    inner groups: [0, 1] and [2, 3]
    outer groups: [0, 2] and [1, 3]
    """

    @staticmethod
    def _run(fn, world_size: int = 4) -> None:
        mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)

    @staticmethod
    def _worker_single_group(rank: int, world_size: int) -> None:
        """all_reduce_multi with one group == standard allreduce."""
        _init_gloo(rank, world_size)
        try:
            mesh = init_device_mesh("cpu", mesh_shape=(2, 2), mesh_dim_names=("inner", "outer"))
            inner_group = mesh.get_group("inner")

            tensor = torch.tensor([float(rank + 1)])
            all_reduce_multi(tensor, [inner_group], op=dist.ReduceOp.SUM)

            # inner groups: [0,1] sum=1+2=3, [2,3] sum=3+4=7
            expected = {0: 3.0, 1: 3.0, 2: 7.0, 3: 7.0}[rank]
            assert tensor.item() == expected, f"rank {rank}: expected {expected}, got {tensor.item()}"
        finally:
            _cleanup()

    def test_single_group(self) -> None:
        self._run(self._worker_single_group)

    @staticmethod
    def _worker_two_groups(rank: int, world_size: int) -> None:
        """all_reduce_multi([inner, outer]) should produce the global sum on all ranks."""
        _init_gloo(rank, world_size)
        try:
            mesh = init_device_mesh("cpu", mesh_shape=(2, 2), mesh_dim_names=("inner", "outer"))
            inner_group = mesh.get_group("inner")
            outer_group = mesh.get_group("outer")

            tensor = torch.tensor([float(rank + 1)])
            all_reduce_multi(tensor, [inner_group, outer_group], op=dist.ReduceOp.SUM)

            # Step 1 inner SUM: [0,1]->3, [2,3]->7. Step 2 outer SUM: [0,2]->3+7=10, [1,3]->3+7=10
            global_sum = float(sum(range(1, world_size + 1)))  # 10
            assert tensor.item() == global_sum, f"rank {rank}: expected {global_sum}, got {tensor.item()}"
        finally:
            _cleanup()

    def test_two_groups(self) -> None:
        self._run(self._worker_two_groups)

    @staticmethod
    def _worker_empty_groups(rank: int, world_size: int) -> None:
        """Empty groups list should be a no-op."""
        _init_gloo(rank, world_size)
        try:
            tensor = torch.tensor([42.0])
            all_reduce_multi(tensor, [], op=dist.ReduceOp.SUM)
            assert tensor.item() == 42.0
        finally:
            _cleanup()

    def test_empty_groups(self) -> None:
        self._run(self._worker_empty_groups)
