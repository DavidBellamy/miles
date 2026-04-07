from __future__ import annotations

import ray
from pydantic import BaseModel, ConfigDict


# NOTE: better name may be `ServerEngine`, but we explicitly name it `wrapper` here to
#       avoid confusion w/ the old `engine`s which are indeed ray `ActorHandle`s
class EngineWrapper:
    def __init__(self):
        self._state = _StateStopped()

    # TODO: unify w/ trainer `change_state`
    def _change_state(
        self,
        debug_name: str,
        old_state_cls: type[_State] | tuple[type[_State], ...],
        new_state: _State,
    ) -> None:
        logger.info(f"{debug_name} start old={self._state}")
        assert isinstance(self._state, old_state_cls), f"{self.cell_index=} {self._state=}"
        self._state = new_state
        logger.info(f"{debug_name} end new={self._state}")


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


_State = _StateStopped | _StateAllocatedUninitialized | _StateAllocatedAlive
