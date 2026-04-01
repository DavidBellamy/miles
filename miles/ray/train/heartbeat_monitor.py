import asyncio
import logging
import time

from miles.ray.train.cell import RayTrainCell
from miles.utils.simple_health_checker import SimpleHealthChecker

logger = logging.getLogger(__name__)


class TrainerHeartbeatMonitor:
    """Per-cell heartbeat monitors for trainer actors.

    Creates one ``SimpleHealthChecker`` per cell. Each checker periodically
    calls ``heartbeat()`` on every actor in the cell and verifies the returned
    timestamp is not stale.
    """

    def __init__(
        self,
        *,
        cells: list[RayTrainCell],
        first_wait: float,
        interval: float,
        timeout: float,
        staleness: float,
    ) -> None:
        self._checkers: list[SimpleHealthChecker] = []
        for cell in cells:
            checker = SimpleHealthChecker(
                name=f"trainer-cell-{cell.cell_index}",
                check_fn=lambda c=cell: _check_cell(
                    cell=c, timeout=timeout, staleness=staleness,
                ),
                on_failure=cell._mark_as_errored,
                interval=interval,
                first_wait=first_wait,
            )
            self._checkers.append(checker)

    async def start(self) -> None:
        for checker in self._checkers:
            await checker.start()

    def stop(self) -> None:
        for checker in self._checkers:
            checker.stop()

    def pause(self) -> None:
        for checker in self._checkers:
            checker.pause()

    def resume(self) -> None:
        for checker in self._checkers:
            checker.resume()


async def _check_cell(
    *,
    cell: RayTrainCell,
    timeout: float,
    staleness: float,
) -> None:
    if not cell.is_alive:
        return

    now = time.time()
    futures = [actor.heartbeat.remote() for actor in cell._get_actor_handles()]

    for future in futures:
        status = await asyncio.wait_for(future, timeout=timeout)
        delta = now - status.last_active_timestamp
        if delta > staleness:
            raise RuntimeError(
                f"Heartbeat stale: last_active={status.last_active_timestamp:.1f}, "
                f"now={now:.1f}, delta={delta:.1f}s, bump_count={status.bump_count}"
            )
