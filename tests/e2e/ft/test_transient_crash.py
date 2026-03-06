"""E2E: Transient crash — single kill → auto-recovery.

Uses the shared scenario_transient_crash from helpers/scenarios.py,
with E2eFaultInjector providing the real process-kill implementation.
"""

from __future__ import annotations

import time

import ray
from miles.utils.ft.models import RecoveryPhase
from tests.e2e.ft.conftest import (
    E2eFaultInjector,
    FaultInjectorFactory,
    assert_phase_path_contains,
    get_status,
    wait_for_training_stable,
)
from tests.fast.utils.ft.helpers.scenarios import scenario_transient_crash


async def test_transient_crash_auto_recovery(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    fault = E2eFaultInjector(
        injector_handle=fault_injector.deploy_to(node_id=target_node),
        target_node=target_node,
    )

    t_inject = time.monotonic()
    await scenario_transient_crash(
        handle=ft_controller_handle,
        injector=fault,
        stable_iterations=3,
        stable_timeout=180.0,
        recovery_timeout=300.0,
    )
    t_recover = time.monotonic() - t_inject

    await wait_for_training_stable(
        handle=ft_controller_handle,
        n_iterations=5,
        timeout=300.0,
    )

    final_status = get_status(ft_controller_handle)
    assert_phase_path_contains(final_status, [
        RecoveryPhase.CHECK_ALERTS,
        RecoveryPhase.REATTEMPTING,
        RecoveryPhase.MONITORING,
        RecoveryPhase.DONE,
    ])

    assert t_recover < 300.0, f"Recovery took too long: {t_recover:.1f}s"
