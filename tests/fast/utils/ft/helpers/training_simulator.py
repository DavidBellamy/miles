"""Remote-controlled training job for local_ray integration tests.

Provides a TrainingJobProtocol implementation whose state lives in a separate
Ray actor, so the test driver can mutate it (crash, hang, recover) while the
FtController actor reads it during its tick loop.
"""
from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

import ray

from miles.utils.ft.protocols.platform import JobStatus

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0, num_gpus=0)
class TrainingStateActor:
    """Holds mutable state shared between the test driver and RemoteControlledTrainingJob.

    Methods are synchronous (not async) since state updates are trivial.
    """

    def __init__(self) -> None:
        self._status: str = JobStatus.RUNNING.value
        self._run_id: str = uuid4().hex[:8]
        self._submit_count: int = 0
        self._stop_called: bool = False
        self._excluded_node_ids: list[str] = []

    def get_status(self) -> str:
        return self._status

    def set_status(self, status: str) -> None:
        self._status = status

    def get_run_id(self) -> str:
        return self._run_id

    def submit(self, excluded_node_ids: list[str] | None = None) -> str:
        self._submit_count += 1
        self._run_id = uuid4().hex[:8]
        self._status = JobStatus.RUNNING.value
        self._stop_called = False
        self._excluded_node_ids = excluded_node_ids or []
        return self._run_id

    def stop(self) -> None:
        self._stop_called = True
        self._status = JobStatus.STOPPED.value

    def get_submit_count(self) -> int:
        return self._submit_count

    def get_stop_called(self) -> bool:
        return self._stop_called

    def get_excluded_node_ids(self) -> list[str]:
        return self._excluded_node_ids


class RemoteControlledTrainingJob:
    """TrainingJobProtocol that delegates to a TrainingStateActor.

    Instances are serialized into the FtControllerActor via cloudpickle.
    All state queries go through Ray RPCs to the shared TrainingStateActor,
    so the test driver can control the training job externally.
    """

    def __init__(self, state_actor: ray.actor.ActorHandle) -> None:
        self._state = state_actor

    async def get_training_status(self) -> JobStatus:
        status_str: str = await self._state.get_status.remote()
        return JobStatus(status_str)

    async def stop_training(self, timeout_seconds: int = 300) -> None:
        await self._state.stop.remote()

    async def submit_training(
        self, excluded_node_ids: list[str] | None = None,
    ) -> str:
        run_id: str = await self._state.submit.remote(excluded_node_ids)
        return run_id
