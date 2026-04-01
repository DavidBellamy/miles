import asyncio
import time

import pytest

from miles.utils.retry_utils import retry

pytestmark = pytest.mark.asyncio


class TestRetryBasic:
    async def test_succeeds_immediately(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1

        await retry(fn, initial_delay=0.0)
        assert call_count == 1

    async def test_retries_then_succeeds(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise ValueError("not yet")

        await retry(fn, initial_delay=0.0)
        assert call_count == 4

    async def test_single_retry(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail once")

        await retry(fn, initial_delay=0.0)
        assert call_count == 2


class TestRetryLogging:
    async def test_logs_on_retry(self, caplog):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("boom")

        with caplog.at_level("WARNING"):
            await retry(fn, initial_delay=0.0)

        retry_messages = [r for r in caplog.records if "retrying" in r.message]
        assert len(retry_messages) == 2

    async def test_no_log_on_first_success(self, caplog):
        async def fn():
            pass

        with caplog.at_level("WARNING"):
            await retry(fn, initial_delay=0.0)

        assert not any("retrying" in r.message for r in caplog.records)

    async def test_logs_include_exc_info(self, caplog):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("detail")

        with caplog.at_level("WARNING"):
            await retry(fn, initial_delay=0.0)

        retry_records = [r for r in caplog.records if "retrying" in r.message]
        assert len(retry_records) == 1
        assert retry_records[0].exc_info is not None

    async def test_log_message_includes_delay(self, caplog):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail")

        with caplog.at_level("WARNING"):
            await retry(fn, initial_delay=2.5)

        retry_records = [r for r in caplog.records if "retrying" in r.message]
        assert len(retry_records) == 1
        assert "2.5s" in retry_records[0].message


class TestRetryBackoff:
    async def test_sleeps_between_retries(self):
        """Verify actual wall-clock delay between retries."""
        timestamps: list[float] = []

        async def fn():
            timestamps.append(time.monotonic())
            if len(timestamps) < 3:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=0.1, max_delay=1.0, backoff_factor=2.0)

        assert len(timestamps) == 3
        gap1 = timestamps[1] - timestamps[0]
        gap2 = timestamps[2] - timestamps[1]
        assert gap1 >= 0.08  # ~0.1s (allow small timing slack)
        assert gap2 >= 0.16  # ~0.2s (doubled)

    async def test_delay_doubles_each_retry(self):
        """Track delays via log messages to verify exponential growth."""
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count <= 4:
                raise RuntimeError("fail")

        delays_seen: list[str] = []
        import logging

        class _DelayCapture(logging.Handler):
            def emit(self, record):
                if "retrying in" in record.message:
                    delays_seen.append(record.message)

        handler = _DelayCapture()
        logger = logging.getLogger("miles.utils.retry_utils")
        logger.addHandler(handler)
        try:
            await retry(fn, initial_delay=1.0, max_delay=100.0, backoff_factor=2.0)
        finally:
            logger.removeHandler(handler)

        assert len(delays_seen) == 4
        assert "1.0s" in delays_seen[0]
        assert "2.0s" in delays_seen[1]
        assert "4.0s" in delays_seen[2]
        assert "8.0s" in delays_seen[3]

    async def test_delay_capped_at_max(self):
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                raise RuntimeError("fail")

        delays_seen: list[str] = []
        import logging

        class _DelayCapture(logging.Handler):
            def emit(self, record):
                if "retrying in" in record.message:
                    delays_seen.append(record.message)

        handler = _DelayCapture()
        logger = logging.getLogger("miles.utils.retry_utils")
        logger.addHandler(handler)
        try:
            await retry(fn, initial_delay=1.0, max_delay=3.0, backoff_factor=2.0)
        finally:
            logger.removeHandler(handler)

        # 1.0, 2.0, 3.0 (capped), 3.0 (capped), 3.0 (capped)
        assert len(delays_seen) == 5
        assert "1.0s" in delays_seen[0]
        assert "2.0s" in delays_seen[1]
        assert "3.0s" in delays_seen[2]
        assert "3.0s" in delays_seen[3]
        assert "3.0s" in delays_seen[4]

    async def test_default_params_are_reasonable(self):
        """Default initial_delay=1.0, max_delay=60.0, backoff_factor=2.0."""
        from miles.utils.retry_utils import _DEFAULT_BACKOFF_FACTOR, _DEFAULT_INITIAL_DELAY, _DEFAULT_MAX_DELAY

        assert _DEFAULT_INITIAL_DELAY == 1.0
        assert _DEFAULT_MAX_DELAY == 60.0
        assert _DEFAULT_BACKOFF_FACTOR == 2.0

    async def test_zero_initial_delay_no_sleep(self):
        """With initial_delay=0, retries happen without sleeping."""
        timestamps: list[float] = []

        async def fn():
            timestamps.append(time.monotonic())
            if len(timestamps) < 3:
                raise RuntimeError("fail")

        await retry(fn, initial_delay=0.0)

        assert len(timestamps) == 3
        total_elapsed = timestamps[-1] - timestamps[0]
        assert total_elapsed < 0.1  # should be near-instant
