"""E2E negative test: no false positives during normal training.

Validates that the controller does NOT trigger recovery when training
runs normally without any fault injection. This catches overly sensitive
detectors or noisy metric thresholds.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from miles.utils.ft.models import ControllerMode
from tests.e2e.ft.conftest import FtSystem, get_iteration_count

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(600),
]

_TARGET_ITERATIONS = 50
_POLL_INTERVAL = 5.0


async def test_no_false_positive_during_normal_training(
    ft_system: FtSystem,
) -> None:
    """Controller should not trigger recovery when training runs normally."""
    controller = ft_system.controller
    mini_wandb = ft_system.mini_wandb

    baseline = get_iteration_count(mini_wandb=mini_wandb)
    recovery_triggered = False

    while True:
        status = controller.get_status()

        if status.mode == ControllerMode.RECOVERY:
            recovery_triggered = True
            logger.error(
                "false_positive_detected status=%s iteration=%d",
                status,
                get_iteration_count(mini_wandb=mini_wandb),
            )
            break

        current = get_iteration_count(mini_wandb=mini_wandb)
        progress = current - baseline
        if progress >= _TARGET_ITERATIONS:
            logger.info(
                "no_false_positive iterations=%d/%d",
                progress, _TARGET_ITERATIONS,
            )
            break

        await asyncio.sleep(_POLL_INTERVAL)

    assert not recovery_triggered, (
        f"Controller entered recovery during normal training at iteration "
        f"{get_iteration_count(mini_wandb=mini_wandb)}: {controller.get_status()}"
    )
