import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


class SimpleHealthChecker:
    """Async periodic health checker with pause/resume support.

    Calls ``check_fn`` every ``interval`` seconds. If ``check_fn`` raises,
    calls ``on_failure`` and stops the loop.
    """

    def __init__(
        self,
        *,
        name: str,
        check_fn: Callable[[], Coroutine[Any, Any, None]],
        on_failure: Callable[[], None],
        interval: float,
        first_wait: float = 0.0,
    ) -> None:
        self._name = name
        self._check_fn = check_fn
        self._on_failure = on_failure
        self._interval = interval
        self._first_wait = first_wait

        self._paused: bool = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    async def _loop(self) -> None:
        await asyncio.sleep(self._first_wait)

        while True:
            if not self._paused:
                try:
                    await self._check_fn()
                except Exception:
                    logger.error(f"Health check failed for {self._name}", exc_info=True)
                    self._on_failure()
                    return

            await asyncio.sleep(self._interval)
