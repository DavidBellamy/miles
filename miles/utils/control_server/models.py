from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class _CellInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_id: str
    cell_type: Literal["actor", "rollout"]
    status: Literal["running", "stopped", "pending", "errored"]
    node_ids: list[str]


class _StopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: int = 30


class _OkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
