import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from miles.utils.mini_ft_controller.controller import CellSnapshot, MiniFTController


def _make_cell(
    *,
    name: str = "cell-0",
    healthy_status: str = "True",
    healthy_reason: str | None = None,
) -> CellSnapshot:
    return CellSnapshot(name=name, healthy_status=healthy_status, healthy_reason=healthy_reason)


def _make_controller(
    *,
    get_cells: AsyncMock | None = None,
    suspend_cell: AsyncMock | None = None,
    resume_cell: AsyncMock | None = None,
    poll_interval: float = 0.01,
    resume_delay: float = 0.0,
    max_consecutive_failures: int = 5,
) -> MiniFTController:
    return MiniFTController(
        get_cells=get_cells or AsyncMock(return_value=[]),
        suspend_cell=suspend_cell or AsyncMock(),
        resume_cell=resume_cell or AsyncMock(),
        poll_interval=poll_interval,
        resume_delay=resume_delay,
        max_consecutive_failures=max_consecutive_failures,
    )


@pytest.mark.asyncio
async def test_heal_on_fatal_error() -> None:
    """Fatal cell triggers suspend → sleep → resume in order."""
    fatal_cell = _make_cell(name="cell-0", healthy_status="False", healthy_reason="Fatal")
    get_cells = AsyncMock(return_value=[fatal_cell])
    suspend_cell = AsyncMock()
    resume_cell = AsyncMock()

    controller = _make_controller(
        get_cells=get_cells,
        suspend_cell=suspend_cell,
        resume_cell=resume_cell,
    )

    # Run one poll cycle then stop
    get_cells.side_effect = [
        [fatal_cell],
        asyncio.CancelledError(),
    ]

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(controller.run(), timeout=5.0)

    suspend_cell.assert_called_once_with("cell-0")
    resume_cell.assert_called_once_with("cell-0")


@pytest.mark.asyncio
async def test_skip_degraded_cell() -> None:
    """Degraded cell does not trigger heal."""
    degraded_cell = _make_cell(name="cell-0", healthy_status="False", healthy_reason="Degraded")
    suspend_cell = AsyncMock()
    resume_cell = AsyncMock()

    controller = _make_controller(
        get_cells=AsyncMock(return_value=[degraded_cell]),
        suspend_cell=suspend_cell,
        resume_cell=resume_cell,
    )
    await controller._poll_and_heal()

    suspend_cell.assert_not_called()
    resume_cell.assert_not_called()


@pytest.mark.asyncio
async def test_skip_healthy_cell() -> None:
    """Healthy=True cell does not trigger heal."""
    healthy_cell = _make_cell(name="cell-0", healthy_status="True")
    suspend_cell = AsyncMock()
    resume_cell = AsyncMock()

    controller = _make_controller(
        get_cells=AsyncMock(return_value=[healthy_cell]),
        suspend_cell=suspend_cell,
        resume_cell=resume_cell,
    )
    await controller._poll_and_heal()

    suspend_cell.assert_not_called()
    resume_cell.assert_not_called()


@pytest.mark.asyncio
async def test_heal_multiple_fatal_cells() -> None:
    """Multiple Fatal cells are each healed."""
    cells = [
        _make_cell(name="cell-0", healthy_status="False", healthy_reason="Fatal"),
        _make_cell(name="cell-1", healthy_status="False", healthy_reason="Fatal"),
    ]
    suspend_cell = AsyncMock()
    resume_cell = AsyncMock()

    controller = _make_controller(
        get_cells=AsyncMock(return_value=cells),
        suspend_cell=suspend_cell,
        resume_cell=resume_cell,
    )
    await controller._poll_and_heal()

    assert suspend_cell.call_count == 2
    assert resume_cell.call_count == 2
    suspend_cell.assert_any_call("cell-0")
    suspend_cell.assert_any_call("cell-1")
    resume_cell.assert_any_call("cell-0")
    resume_cell.assert_any_call("cell-1")


@pytest.mark.asyncio
async def test_backoff_on_heal_failure() -> None:
    """Suspend raises → consecutive_failures increments, next_attempt_at increases."""
    fatal_cell = _make_cell(name="cell-0", healthy_status="False", healthy_reason="Fatal")
    suspend_cell = AsyncMock(side_effect=RuntimeError("connection failed"))

    controller = _make_controller(
        get_cells=AsyncMock(return_value=[fatal_cell]),
        suspend_cell=suspend_cell,
        max_consecutive_failures=10,
    )

    await controller._poll_and_heal()

    backoff = controller._cell_backoffs["cell-0"]
    assert backoff.consecutive_failures == 1
    assert backoff.next_attempt_at > 0


@pytest.mark.asyncio
async def test_exponential_backoff_timing() -> None:
    """Verify backoff delays: 5*2^1=10, 5*2^2=20, 5*2^3=40, ..., capped at 300."""
    fatal_cell = _make_cell(name="cell-0", healthy_status="False", healthy_reason="Fatal")
    suspend_cell = AsyncMock(side_effect=RuntimeError("fail"))

    controller = _make_controller(
        get_cells=AsyncMock(return_value=[fatal_cell]),
        suspend_cell=suspend_cell,
        max_consecutive_failures=20,
    )

    expected_delays = [10, 20, 40, 80, 160, 300, 300]
    for expected_delay in expected_delays:
        # Reset next_attempt_at so the poll actually attempts heal
        backoff = controller._cell_backoffs.get("cell-0")
        if backoff:
            backoff.next_attempt_at = 0.0

        before = time.monotonic()
        await controller._poll_and_heal()
        after = time.monotonic()

        backoff = controller._cell_backoffs["cell-0"]
        actual_delay = backoff.next_attempt_at - after
        assert abs(actual_delay - expected_delay) < 2.0, (
            f"Expected delay ~{expected_delay}, got {actual_delay:.1f}"
        )


@pytest.mark.asyncio
async def test_given_up_after_max_failures() -> None:
    """After N consecutive failures, given_up=True, no more attempts."""
    fatal_cell = _make_cell(name="cell-0", healthy_status="False", healthy_reason="Fatal")
    suspend_cell = AsyncMock(side_effect=RuntimeError("fail"))

    max_failures = 3
    controller = _make_controller(
        get_cells=AsyncMock(return_value=[fatal_cell]),
        suspend_cell=suspend_cell,
        max_consecutive_failures=max_failures,
    )

    for _ in range(max_failures):
        backoff = controller._cell_backoffs.get("cell-0")
        if backoff:
            backoff.next_attempt_at = 0.0
        await controller._poll_and_heal()

    backoff = controller._cell_backoffs["cell-0"]
    assert backoff.given_up is True
    assert backoff.consecutive_failures == max_failures

    # One more poll should not attempt heal
    suspend_cell.reset_mock()
    await controller._poll_and_heal()
    suspend_cell.assert_not_called()


@pytest.mark.asyncio
async def test_successful_heal_resets_backoff() -> None:
    """Successful heal resets consecutive_failures to 0."""
    fatal_cell = _make_cell(name="cell-0", healthy_status="False", healthy_reason="Fatal")

    call_count = 0

    async def failing_then_succeeding_suspend(name: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("fail")

    controller = _make_controller(
        get_cells=AsyncMock(return_value=[fatal_cell]),
        suspend_cell=AsyncMock(side_effect=failing_then_succeeding_suspend),
        resume_cell=AsyncMock(),
    )

    # Step 1: First attempt fails
    await controller._poll_and_heal()
    backoff = controller._cell_backoffs["cell-0"]
    assert backoff.consecutive_failures == 1

    # Step 2: Reset next_attempt_at, second attempt succeeds
    backoff.next_attempt_at = 0.0
    await controller._poll_and_heal()
    assert backoff.consecutive_failures == 0
    assert backoff.next_attempt_at == 0.0


@pytest.mark.asyncio
async def test_poll_continues_after_get_cells_failure() -> None:
    """get_cells raises → controller does not exit."""
    call_count = 0

    async def failing_get_cells() -> list[CellSnapshot]:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise RuntimeError("network error")
        raise asyncio.CancelledError()

    controller = _make_controller(
        get_cells=AsyncMock(side_effect=failing_get_cells),
    )

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(controller.run(), timeout=5.0)

    assert call_count == 3


@pytest.mark.asyncio
async def test_request_stop_exits_loop() -> None:
    """call request_stop → run() returns."""
    controller = _make_controller()

    async def stop_after_first_poll() -> list[CellSnapshot]:
        controller.request_stop()
        return []

    controller._get_cells = AsyncMock(side_effect=stop_after_first_poll)

    await asyncio.wait_for(controller.run(), timeout=5.0)


@pytest.mark.asyncio
async def test_no_action_when_all_healthy() -> None:
    """All Healthy=True → no suspend/resume calls."""
    cells = [
        _make_cell(name="cell-0", healthy_status="True"),
        _make_cell(name="cell-1", healthy_status="True"),
    ]
    suspend_cell = AsyncMock()
    resume_cell = AsyncMock()

    controller = _make_controller(
        get_cells=AsyncMock(return_value=cells),
        suspend_cell=suspend_cell,
        resume_cell=resume_cell,
    )
    await controller._poll_and_heal()

    suspend_cell.assert_not_called()
    resume_cell.assert_not_called()
