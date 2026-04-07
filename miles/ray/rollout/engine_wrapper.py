import ray
from pydantic import BaseModel, ConfigDict


class EngineWrapper:
    def __init__(self):
        self._state = _StateStopped()
        TODO


# ------------------------- states -----------------------------


class _StateBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class _StateStopped(_StateBase):
    pass


class _StateAllocatedBase(_StateBase):
    actor_handle: ray.actor.ActorHandle


class _StateAllocatedUninitialized(_StateAllocatedBase):
    pass


class _StateAllocatedAlive(_StateAllocatedBase):
    pass


CellState = _StateStopped | _StateAllocatedUninitialized | _StateAllocatedAlive
