from __future__ import annotations

import asyncio
import logging
import time

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.agents.utils.prometheus_exporter import PrometheusExporter
from miles.utils.ft.models.metrics import GaugeSample

logger = logging.getLogger(__name__)

_STALENESS_METRIC = "ft_collector_last_success_timestamp"
_CONSECUTIVE_FAILURES_METRIC = "ft_collector_consecutive_failures"


class MetricCollectionLoop:
    """Runs collectors as background tasks and feeds metrics to a PrometheusExporter."""

    def __init__(
        self,
        node_id: str,
        collectors: list[BaseCollector],
        exporter: PrometheusExporter,
    ) -> None:
        self._node_id = node_id
        self._collectors = collectors
        self._exporter = exporter
        self._tasks: list[asyncio.Task[None]] = []
        self._stopped = False

    @property
    def tasks(self) -> list[asyncio.Task[None]]:
        return self._tasks

    async def start(self) -> None:
        if self._stopped or self._tasks:
            return

        loop = asyncio.get_running_loop()
        for collector in self._collectors:
            task = loop.create_task(self._run_single_collector(collector))
            self._tasks.append(task)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True

        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for collector in self._collectors:
            try:
                await collector.close()
            except Exception:
                logger.warning(
                    "Collector %s.close() failed on node %s",
                    type(collector).__name__,
                    self._node_id,
                    exc_info=True,
                )

    async def _run_single_collector(self, collector: BaseCollector) -> None:
        collector_name = type(collector).__name__
        consecutive_failures = 0

        while True:
            try:
                result = await collector.collect()
                self._exporter.update_metrics(result.metrics)
                consecutive_failures = 0
                self._emit_staleness_metrics(
                    collector_name=collector_name,
                    last_success_timestamp=time.time(),
                    consecutive_failures=0,
                )
            except Exception:
                consecutive_failures += 1
                logger.warning(
                    "Collector %s failed on node %s (consecutive_failures=%d)",
                    collector_name,
                    self._node_id,
                    consecutive_failures,
                    exc_info=True,
                )
                self._emit_staleness_metrics(
                    collector_name=collector_name,
                    last_success_timestamp=0.0,
                    consecutive_failures=consecutive_failures,
                )

            await asyncio.sleep(collector.collect_interval)

    def _emit_staleness_metrics(
        self,
        *,
        collector_name: str,
        last_success_timestamp: float,
        consecutive_failures: int,
    ) -> None:
        labels = {"collector": collector_name, "node_id": self._node_id}
        staleness_metrics: list[GaugeSample] = []

        if last_success_timestamp > 0:
            staleness_metrics.append(
                GaugeSample(name=_STALENESS_METRIC, labels=labels, value=last_success_timestamp)
            )

        staleness_metrics.append(
            GaugeSample(
                name=_CONSECUTIVE_FAILURES_METRIC,
                labels=labels,
                value=float(consecutive_failures),
            )
        )
        self._exporter.update_metrics(staleness_metrics)
