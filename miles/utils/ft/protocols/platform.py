from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable


class JobStatus(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    PENDING = "pending"


STOP_TRAINING_TIMEOUT_SECONDS: int = 300


@runtime_checkable
class NodeManagerProtocol(Protocol):
    async def mark_node_bad(self, node_id: str, reason: str) -> None: ...

    async def unmark_node_bad(self, node_id: str) -> None: ...

    async def get_bad_nodes(self) -> list[str]: ...


@runtime_checkable
class TrainingJobProtocol(Protocol):
    async def stop_training(self, timeout_seconds: int = STOP_TRAINING_TIMEOUT_SECONDS) -> None: ...

    async def submit_training(
        self,
        excluded_node_ids: list[str] | None = None,
    ) -> str: ...

    async def get_training_status(self) -> JobStatus: ...


@runtime_checkable
class NotifierProtocol(Protocol):
    async def send(self, title: str, content: str, severity: str) -> None: ...

    async def aclose(self) -> None: ...
