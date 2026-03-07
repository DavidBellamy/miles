"""E2E: Transient crash — single kill → auto-recovery.

Uses the shared scenario_transient_crash from helpers/scenarios.py,
with E2eFaultInjector providing the real process-kill implementation.
"""

from __future__ import annotations

import time

import ray
from tests.e2e.ft.conftest import E2eFaultInjector, FaultInjectorFactory
from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import scenario_transient_crash


async def test_transient_crash_auto_recovery(
    ft_controller_handle: ray.actor.ActorHandle,
    fault_injector: FaultInjectorFactory,
    target_node: str,
) -> None:
    fault = E2eFaultInjector(
        injector_handle=fault_injector.deploy_to(node_id=target_node),
        target_node=target_node,
    )

    t0 = time.monotonic()
    await scenario_transient_crash(
        handle=ft_controller_handle,
        injector=fault,
        stable_iterations=3,
        stable_timeout=180.0,
        recovery_timeout=300.0,
        post_recovery_iterations=5,
        post_recovery_timeout=300.0,
    )
    assert time.monotonic() - t0 < 300.0
