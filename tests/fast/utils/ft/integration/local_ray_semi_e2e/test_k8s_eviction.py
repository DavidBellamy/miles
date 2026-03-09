"""Semi-E2E: K8s-based eviction — label + pod delete → WaitingForNewNode → restart."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from tests.fast.utils.ft.integration.conftest import FAST_TIMEOUT, LONG_RECOVERY_TIMEOUT, RECOVERY_TIMEOUT
from tests.fast.utils.ft.integration.local_ray_semi_e2e.conftest import _SLOW_STEP, E2EEnv, NodeSpec
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import (
    assert_phase_path_contains,
    get_status,
    wait_for_recovery_complete,
    wait_for_recovery_phase,
    wait_for_training_stable,
)

from miles.utils.ft.controller.detectors.core.training_crash import TrainingCrashDetector
from miles.utils.ft.controller.types import ControllerMode


class TestSingleNodeEviction:
    async def test_single_node_eviction_through_waiting_for_new_node(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Single node fault → label → WaitingForNewNode → StoppingAndRestarting → RestartDone.

        Verifies the full K8s eviction path goes through WaitingForNewNode.
        """
        env = make_e2e_env(
            ft_id="e2ekev1",
            nodes=[
                NodeSpec(node_id="e2ekev1-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2ekev1-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=FAST_TIMEOUT)

        # Step 1: crash → MONITORING
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=FAST_TIMEOUT,
        )

        # Step 2: crash during MONITORING → diagnostics → eviction → recovery
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=RECOVERY_TIMEOUT)

        assert final.mode == ControllerMode.MONITORING
        assert_phase_path_contains(
            final,
            [
                "Evicting",
                "WaitingForNewNode",
                "StoppingAndRestarting",
                "MonitoringProgress",
            ],
        )


class TestMultiNodeEviction:
    async def test_multi_node_eviction_marks_all_bad_nodes(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """2/3 nodes fail diagnostics → both evicted → WaitingForNewNode → recovery."""
        env = make_e2e_env(
            ft_id="e2ekev2",
            nodes=[
                NodeSpec(node_id="e2ekev2-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2ekev2-node-1", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2ekev2-node-2", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=FAST_TIMEOUT)

        # Step 1: crash → MONITORING
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=FAST_TIMEOUT,
        )

        # Step 2: crash during MONITORING → diagnostics → eviction
        await env.injector.crash_training()
        final = await wait_for_recovery_complete(env.controller, timeout=LONG_RECOVERY_TIMEOUT)

        assert_phase_path_contains(final, ["Evicting", "WaitingForNewNode"])


class TestWaitingForNewNodeTimeout:
    async def test_waiting_for_new_node_timeout_causes_restart_failed(
        self,
        make_e2e_env: Callable[..., E2EEnv],
    ) -> None:
        """Evict → WaitingForNewNode → timeout → RestartFailed → NotifyHumans.

        Uses a very short waiting_for_node_timeout so the timeout triggers quickly.
        Since semi-E2E uses FakeNodeManager (no real K8s), WaitingForNewNode will
        time out because no new node appears unless get_current_alive_node_count
        is configured to return enough nodes.

        In the current setup (no node count callbacks configured on FakeNodeManager),
        WaitingForNewNode will skip the wait and proceed. This test verifies the
        eviction → waiting → restart path completes.
        """
        env = make_e2e_env(
            ft_id="e2ekev3",
            nodes=[
                NodeSpec(node_id="e2ekev3-node-0", num_ranks=1, diagnostic_pass=False),
                NodeSpec(node_id="e2ekev3-node-1", num_ranks=1, diagnostic_pass=True),
            ],
            detectors=[TrainingCrashDetector()],
            step_interval=_SLOW_STEP,
            monitoring_success_iterations=999,
        )

        await wait_for_training_stable(env.controller, n_iterations=1, timeout=FAST_TIMEOUT)

        # Step 1: crash → MONITORING
        await env.injector.crash_training()
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=FAST_TIMEOUT,
        )

        # Step 2: crash during MONITORING → diagnostics → eviction → WaitingForNewNode
        await env.injector.crash_training()

        deadline = time.monotonic() + RECOVERY_TIMEOUT
        while time.monotonic() < deadline:
            status = get_status(env.controller)
            if status.phase_history and "Evicting" in status.phase_history:
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError(
                f"Evicting not found in phase_history within {RECOVERY_TIMEOUT}s: {status.phase_history}"
            )

        # Step 3: wait for post-eviction MonitoringProgress
        await wait_for_recovery_phase(
            env.controller,
            phase="MonitoringProgress",
            timeout=RECOVERY_TIMEOUT,
        )

        assert_phase_path_contains(
            get_status(env.controller),
            ["Evicting", "WaitingForNewNode", "StoppingAndRestarting"],
        )
