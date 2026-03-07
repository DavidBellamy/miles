"""E2E negative test: no false positives during normal training."""

from __future__ import annotations

import ray

from tests.fast.utils.ft.integration.local_ray_semi_e2e.scenarios import scenario_no_false_positive


async def test_no_false_positive_during_normal_training(
    ft_controller_handle: ray.actor.ActorHandle,
) -> None:
    """Controller should not trigger recovery when training runs normally."""
    await scenario_no_false_positive(
        handle=ft_controller_handle,
        observation_iterations=10,
        timeout=120.0,
        poll_interval=5.0,
    )
