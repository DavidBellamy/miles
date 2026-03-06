"""Local Ray: E2E-like scenarios with real agents (node agent + training rank agent).

Unlike test_scenarios.py which bypasses agents and injects state directly,
these tests start real FtNodeAgentActor and FtTrainingRankAgent instances
(with stub collectors/diagnostics) so the full registration, metrics scraping,
and diagnostic pipeline paths are exercised.
"""
from __future__ import annotations

import time
from collections.abc import Generator
from datetime import timedelta
from typing import Any

import pytest
import ray

from miles.utils.ft.agents.collectors.stub import StubCollector
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector
from miles.utils.ft.models import ControllerMode
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.models.metric_names import TRAINING_ITERATION
from miles.utils.ft.platform.controller_actor import FtControllerActor
from miles.utils.ft.platform.controller_factory import FtControllerConfig
from miles.utils.ft.platform.node_agent_actor import FtNodeAgentActor
from miles.utils.ft.protocols.platform import (
    JobStatus,
    ft_controller_actor_name,
    ft_node_agent_actor_name,
)

from tests.fast.utils.ft.helpers.diagnostic_fakes import StubDiagnostic
from tests.fast.utils.ft.helpers.fault_injection import LocalRayFaultInjector
from tests.fast.utils.ft.helpers.scenarios import (
    scenario_hang_detection,
    scenario_no_false_positive,
    scenario_repeated_crash,
    scenario_transient_crash,
)
from tests.fast.utils.ft.helpers.training_simulator import (
    RemoteControlledTrainingJob,
    TrainingStateActor,
    TrainingWorkerActor,
)
from tests.fast.utils.ft.integration.local_ray.conftest import poll_for_run_id

pytestmark = [
    pytest.mark.local_ray,
    pytest.mark.anyio,
]

_NODE_ID = "e2e-node-0"
_FT_ID = "e2e"


def _make_fake_node_manager() -> Any:
    from tests.fast.utils.ft.helpers.controller_fakes import FakeNodeManager
    return FakeNodeManager()


def _wait_for_node_agent_registered(
    controller: ray.actor.ActorHandle,
    node_id: str,
    timeout: float = 10.0,
) -> None:
    """Poll controller status until the node agent's metrics appear in the scrape pipeline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(0.3)
    # Node agent registration is fire-and-forget with retries;
    # give it enough time to complete.


def _wait_for_iteration_advancing(
    controller: ray.actor.ActorHandle,
    timeout: float = 15.0,
) -> None:
    """Poll until latest_iteration is > 0, confirming the worker is feeding metrics."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = ray.get(controller.get_status.remote(), timeout=5)
        if status.latest_iteration is not None and status.latest_iteration > 0:
            return
        time.sleep(0.3)
    raise TimeoutError("Worker iteration metrics not visible on controller within timeout")


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def e2e_simulated_env(
    local_ray: None,
) -> Generator[
    tuple[ray.actor.ActorHandle, ray.actor.ActorHandle, LocalRayFaultInjector],
    None,
    None,
]:
    """Full agent stack: controller + node agent + training worker.

    Yields (controller_handle, state_actor, fault_injector).
    """
    state_actor = TrainingStateActor.remote()
    training_job = RemoteControlledTrainingJob(state_actor=state_actor)

    # Step 1: start controller
    controller_name = ft_controller_actor_name(_FT_ID)
    controller = FtControllerActor.options(name=controller_name).remote(
        config=FtControllerConfig(platform="stub", tick_interval=0.1, ft_id=_FT_ID),
        training_job_override=training_job,
        node_manager_override=_make_fake_node_manager(),
        notifier_override=None,
        detectors_override=[TrainingCrashDetector()],
    )
    controller.submit_and_run.remote()
    run_id = poll_for_run_id(controller)

    # Step 2: start node agent (stub collectors + stub gpu diagnostic)
    node_agent_name = ft_node_agent_actor_name(_FT_ID, _NODE_ID)
    node_agent = FtNodeAgentActor.options(name=node_agent_name).remote(
        node_id=_NODE_ID,
        ft_id=_FT_ID,
        collect_interval_seconds=1.0,
        collectors_override=[StubCollector()],
        diagnostics_override=[StubDiagnostic(passed=True, diagnostic_type="gpu")],
    )
    ray.get(node_agent.start.remote(), timeout=10)

    # Step 3: start training worker (hosts real FtTrainingRankAgent)
    worker = TrainingWorkerActor.remote(
        state_actor=state_actor,
        ft_id=_FT_ID,
        rank=0,
        world_size=1,
        node_id=_NODE_ID,
        step_interval=0.1,
    )
    ray.get(worker.start.remote(), timeout=10)

    # Step 4: wait until iteration metrics reach the controller
    _wait_for_iteration_advancing(controller)

    injector = LocalRayFaultInjector(state_actor=state_actor)
    yield controller, state_actor, injector

    # Cleanup
    for cleanup in [
        lambda: ray.get(worker.stop.remote(), timeout=5),
        lambda: ray.get(node_agent.stop.remote(), timeout=5),
        lambda: ray.get(controller.shutdown.remote(), timeout=10),
    ]:
        try:
            cleanup()
        except Exception:
            pass

    for actor_name in [controller_name, node_agent_name]:
        try:
            ray.kill(ray.get_actor(actor_name), no_restart=True)
        except ValueError:
            pass

    for handle in [state_actor, worker]:
        try:
            ray.kill(handle, no_restart=True)
        except Exception:
            pass


class _FastHangDetector(BaseFaultDetector):
    """HangDetector with sub-minute timeout for fast testing."""

    def __init__(self, timeout_seconds: float = 3.0) -> None:
        self._timeout = timedelta(seconds=timeout_seconds)

    def evaluate(self, ctx: DetectorContext) -> Decision:
        if ctx.job_status != JobStatus.RUNNING:
            return Decision(action=ActionType.NONE, reason="not running")

        df = ctx.metric_store.changes(
            TRAINING_ITERATION,
            window=self._timeout,
            label_filters={"rank": "0"},
        )
        if df.is_empty():
            return Decision(action=ActionType.NONE, reason="no iteration data")

        if df["value"][0] == 0:
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason=f"iteration stalled for {self._timeout.total_seconds()}s",
                trigger=TriggerType.HANG,
            )
        return Decision(action=ActionType.NONE, reason="progressing")


@pytest.fixture
def e2e_hang_env(
    local_ray: None,
) -> Generator[
    tuple[ray.actor.ActorHandle, ray.actor.ActorHandle, LocalRayFaultInjector],
    None,
    None,
]:
    """Full agent stack configured for hang detection.

    Uses _FastHangDetector (3s timeout) instead of TrainingCrashDetector.
    The training worker advances iterations until inject_hang() freezes them.
    """
    hang_ft_id = "e2ehang"
    hang_node_id = "e2e-hang-node"

    state_actor = TrainingStateActor.remote()
    training_job = RemoteControlledTrainingJob(state_actor=state_actor)

    controller_name = ft_controller_actor_name(hang_ft_id)
    controller = FtControllerActor.options(name=controller_name).remote(
        config=FtControllerConfig(
            platform="stub", tick_interval=0.1, ft_id=hang_ft_id,
        ),
        training_job_override=training_job,
        node_manager_override=_make_fake_node_manager(),
        notifier_override=None,
        detectors_override=[_FastHangDetector(timeout_seconds=3.0)],
    )
    controller.submit_and_run.remote()
    run_id = poll_for_run_id(controller)

    worker = TrainingWorkerActor.remote(
        state_actor=state_actor,
        ft_id=hang_ft_id,
        rank=0,
        world_size=1,
        node_id=hang_node_id,
        step_interval=0.1,
    )
    ray.get(worker.start.remote(), timeout=10)

    _wait_for_iteration_advancing(controller)

    injector = LocalRayFaultInjector(state_actor=state_actor)
    yield controller, state_actor, injector

    for cleanup in [
        lambda: ray.get(worker.stop.remote(), timeout=5),
        lambda: ray.get(controller.shutdown.remote(), timeout=10),
    ]:
        try:
            cleanup()
        except Exception:
            pass

    for name in [controller_name]:
        try:
            ray.kill(ray.get_actor(name), no_restart=True)
        except ValueError:
            pass

    for handle in [state_actor, worker]:
        try:
            ray.kill(handle, no_restart=True)
        except Exception:
            pass


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestTransientCrash:
    async def test_crash_triggers_recovery_then_returns_to_monitoring(
        self,
        e2e_simulated_env: tuple[ray.actor.ActorHandle, ray.actor.ActorHandle, LocalRayFaultInjector],
    ) -> None:
        controller, _state_actor, injector = e2e_simulated_env

        status = await scenario_transient_crash(
            handle=controller,
            injector=injector,
            stable_iterations=3,
            recovery_timeout=60.0,
        )

        assert status.mode == ControllerMode.MONITORING


class TestNoFalsePositive:
    async def test_healthy_training_stays_in_monitoring(
        self,
        e2e_simulated_env: tuple[ray.actor.ActorHandle, ray.actor.ActorHandle, LocalRayFaultInjector],
    ) -> None:
        controller, _state_actor, injector = e2e_simulated_env

        status = await scenario_no_false_positive(
            handle=controller,
            observation_iterations=5,
            timeout=30.0,
        )

        assert status.mode == ControllerMode.MONITORING
        assert status.recovery_in_progress is False


class TestRepeatedCrash:
    async def test_two_crashes_escalate_to_diagnosing(
        self,
        e2e_simulated_env: tuple[ray.actor.ActorHandle, ray.actor.ActorHandle, LocalRayFaultInjector],
    ) -> None:
        controller, _state_actor, injector = e2e_simulated_env

        await scenario_repeated_crash(
            handle=controller,
            injector=injector,
            stable_iterations=3,
            recovery_timeout=60.0,
        )


class TestHangDetection:
    async def test_stale_iteration_triggers_hang_recovery(
        self,
        e2e_hang_env: tuple[ray.actor.ActorHandle, ray.actor.ActorHandle, LocalRayFaultInjector],
    ) -> None:
        controller, _state_actor, injector = e2e_hang_env

        status = await scenario_hang_detection(
            handle=controller,
            injector=injector,
            hang_timeout=20.0,
        )

        assert status.mode == ControllerMode.RECOVERY
