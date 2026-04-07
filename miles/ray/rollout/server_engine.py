from dataclasses import dataclass

import ray
from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class ServerCellConfig:
    TODO


class ServerCell:
    def __init__(self, config: ServerCellConfig):
        self.config = config
        TODO


# ------------------------- states -----------------------------


class _StateBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class _StateStopped(_StateBase):
    pass


class _StateAllocatedBase(_StateBase):
    actor_handles: list[ray.actor.ActorHandle]


class _StateAllocatedUninitialized(_StateAllocatedBase):
    pass


class _StateAllocatedAlive(_StateAllocatedBase):
    pass


CellState = _StateStopped | _StateAllocatedUninitialized | _StateAllocatedAlive
