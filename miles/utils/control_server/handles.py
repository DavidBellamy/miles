from __future__ import annotations

import abc

from miles.ray.train.group import RayTrainGroup


class _CellHandle(abc.ABC):
    @property
    def cell_id(self) -> str:
        return f"{self.cell_type}-{self.cell_index}"

    @property
    @abc.abstractmethod
    def cell_type(self) -> str: ...

    @property
    @abc.abstractmethod
    def cell_index(self) -> int: ...

    @abc.abstractmethod
    async def stop(self, timeout_seconds: int) -> None: ...

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def get_status(self) -> str: ...

    @abc.abstractmethod
    async def get_node_ids(self) -> list[str]: ...


class _ActorCellHandle(_CellHandle):
    def __init__(self, *, group: RayTrainGroup, cell_index: int) -> None:
        self._group = group
        self._cell_index = cell_index

    @property
    def cell_type(self) -> str:
        return "actor"

    @property
    def cell_index(self) -> int:
        return self._cell_index

    async def stop(self, timeout_seconds: int) -> None:
        self._group.stop_cell(self._cell_index)

    async def start(self) -> None:
        self._group.start_cell(self._cell_index)

    async def get_status(self) -> str:
        return self._group._cells[self._cell_index].status

    async def get_node_ids(self) -> list[str]:
        return []


# TODO the code will NOT work before implementing rollout ft
class _RolloutCellHandle(_CellHandle):
    def __init__(self, *, rollout_manager: object, cell_index: int) -> None:
        self._rollout_manager = rollout_manager
        self._cell_index = cell_index

    @property
    def cell_type(self) -> str:
        return "rollout"

    @property
    def cell_index(self) -> int:
        return self._cell_index

    async def stop(self, timeout_seconds: int) -> None:
        await self._rollout_manager.stop_cell.remote(self._cell_index, timeout_seconds)

    async def start(self) -> None:
        await self._rollout_manager.start_cell.remote(self._cell_index)

    async def get_status(self) -> str:
        return await self._rollout_manager.get_cell_status.remote(self._cell_index)

    async def get_node_ids(self) -> list[str]:
        return await self._rollout_manager.get_cell_node_ids.remote(self._cell_index)
