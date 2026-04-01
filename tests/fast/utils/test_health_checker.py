import asyncio

import pytest

from miles.utils.clock import FakeClock
from miles.utils.health_checker import SimpleHealthChecker


def _make_checker(
    *,
    check_fn=None,
    on_result=None,
    interval: float = 1.0,
    first_wait: float = 0.0,
    name: str = "test",
    clock: FakeClock | None = None,
) -> SimpleHealthChecker:
    if check_fn is None:

        async def check_fn() -> None:
            pass

    if on_result is None:
        on_result = lambda success: None

    return SimpleHealthChecker(
        name=name,
        check_fn=check_fn,
        on_result=on_result,
        interval=interval,
        first_wait=first_wait,
        clock=clock or FakeClock(),
    )


async def _tick() -> None:
    await asyncio.sleep(0)


class TestStartStop:
    async def test_start_creates_task(self):
        checker = _make_checker()
        assert checker._task is None

        await checker.start()
        assert checker._task is not None

        checker.stop()
        assert checker._task is None

    async def test_start_is_idempotent(self):
        checker = _make_checker()
        await checker.start()
        task = checker._task

        await checker.start()
        assert checker._task is task

        checker.stop()

    async def test_stop_without_start_is_noop(self):
        checker = _make_checker()
        checker.stop()


class TestCheckFnCalled:
    async def test_check_fn_called_on_each_tick(self):
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        checker = _make_checker(check_fn=check_fn)
        await checker.start()

        await _tick()
        assert call_count == 1

        await _tick()
        assert call_count == 2

        checker.stop()

    async def test_first_wait_still_allows_sleep_to_complete(self):
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        checker = _make_checker(check_fn=check_fn, first_wait=300.0)
        await checker.start()

        # FakeClock.sleep returns immediately, so check runs after a tick
        await _tick()
        assert call_count == 1

        checker.stop()


class TestOnResult:
    async def test_on_result_true_on_success(self):
        results: list[bool] = []

        checker = _make_checker(on_result=lambda s: results.append(s))
        await checker.start()
        await _tick()
        checker.stop()

        assert results == [True]

    async def test_on_result_false_on_failure(self):
        results: list[bool] = []

        async def check_fn() -> None:
            raise RuntimeError("boom")

        checker = _make_checker(check_fn=check_fn, on_result=lambda s: results.append(s))
        await checker.start()
        await _tick()
        checker.stop()

        assert results == [False]

    async def test_loop_continues_after_failure(self):
        results: list[bool] = []

        async def check_fn() -> None:
            raise RuntimeError("boom")

        checker = _make_checker(check_fn=check_fn, on_result=lambda s: results.append(s))
        await checker.start()

        await _tick()
        await _tick()
        checker.stop()

        assert len(results) >= 2
        assert all(r is False for r in results)

    async def test_intermittent_failure(self):
        call_count = 0
        results: list[bool] = []

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("intermittent")

        checker = _make_checker(check_fn=check_fn, on_result=lambda s: results.append(s))
        await checker.start()

        for _ in range(5):
            await _tick()
        checker.stop()

        assert len(results) >= 4
        assert True in results
        assert False in results


class TestPauseResume:
    async def test_paused_checker_does_not_call_check_fn(self):
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        checker = _make_checker(check_fn=check_fn)
        checker.pause()

        await checker.start()
        await _tick()
        await _tick()
        checker.stop()

        assert call_count == 0

    async def test_resume_after_pause_resumes_checking(self):
        call_count = 0

        async def check_fn() -> None:
            nonlocal call_count
            call_count += 1

        checker = _make_checker(check_fn=check_fn)
        checker.pause()

        await checker.start()
        await _tick()
        assert call_count == 0

        checker.resume()
        await _tick()
        checker.stop()

        assert call_count >= 1

    async def test_pause_resume_flags(self):
        checker = _make_checker()
        assert not checker._paused

        checker.pause()
        assert checker._paused

        checker.resume()
        assert not checker._paused
