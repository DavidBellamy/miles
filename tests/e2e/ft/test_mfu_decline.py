"""E2E: MFU decline detection via GPU stress.

MfuDeclineDetector returns MARK_BAD_AND_RESTART (when high GPU temperature
correlates with MFU drop) or NOTIFY_HUMAN (when the decline persists without
identifiable cause).  Neither action enters ControllerMode.RECOVERY — that
only happens via ENTER_RECOVERY (TrainingCrashDetector / HangDetector).

Possible outcomes under GPU stress:
  A) MARK_BAD_AND_RESTART — target node evicted, training restarts.
  B) NOTIFY_HUMAN — notification sent, no eviction, training continues.
  C) GPU stress causes a crash → TrainingCrashDetector fires ENTER_RECOVERY.
     This is a valid but unintended path (tests crash recovery, not MFU decline).

The test accepts all three outcomes but logs which path was taken.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest
import ray
from miles.utils.ft.models import ControllerMode
from miles.utils.ft.platform.k8s_node_manager import K8sNodeManager
from tests.e2e.ft.conftest import (
    FaultInjectorFactory,
    get_status,
    wait_for_recovery_complete,
    wait_for_training_stable,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(900),
]

_DETECTION_TIMEOUT = 600.0
_POLL_INTERVAL = 10.0


async def test_mfu_decline_detection(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    target_node: str,
    _cleanup_node_manager: K8sNodeManager,
) -> None:
    await wait_for_training_stable(
        handle=ft_controller_handle,
        n_iterations=20,
        timeout=600.0,
    )

    injector = fault_injector.deploy_to(node_id=target_node)
    stress_pid = ray.get(injector.start_gpu_stress.remote())
    logger.info("gpu_stress_started pid=%d node=%s", stress_pid, target_node)

    try:
        evicted = False
        crash_recovery = False
        deadline = time.monotonic() + _DETECTION_TIMEOUT

        while time.monotonic() < deadline:
            bad_nodes = set(await _cleanup_node_manager.get_bad_nodes())
            if target_node in bad_nodes:
                evicted = True
                break

            status = get_status(ft_controller_handle)
            if status.mode == ControllerMode.RECOVERY:
                crash_recovery = True
                break

            await asyncio.sleep(_POLL_INTERVAL)

        if evicted:
            logger.info(
                "mfu_decline_path=MARK_BAD_AND_RESTART node=%s "
                "(temperature correlated eviction)",
                target_node,
            )
            await wait_for_training_stable(
                handle=ft_controller_handle,
                n_iterations=10,
                timeout=300.0,
            )

        elif crash_recovery:
            logger.warning(
                "mfu_decline_path=CRASH_RECOVERY node=%s "
                "(GPU stress caused training crash, not MFU decline detection)",
                target_node,
            )
            status = await wait_for_recovery_complete(
                handle=ft_controller_handle,
                timeout=300.0,
            )
            assert status.mode == ControllerMode.MONITORING

        else:
            logger.info(
                "mfu_decline_path=NOTIFY_OR_UNDETECTED node=%s "
                "(NOTIFY_HUMAN sent, or decline below detection threshold)",
                target_node,
            )
            status = get_status(ft_controller_handle)
            assert status.mode == ControllerMode.MONITORING, (
                f"Expected MONITORING after timeout, got {status.mode}"
            )

    finally:
        try:
            ray.get(injector.stop_gpu_stress.remote(pid=stress_pid))
        except Exception:
            logger.warning("stop_gpu_stress_failed pid=%d", stress_pid, exc_info=True)
