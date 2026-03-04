from __future__ import annotations

import asyncio
import logging
from typing import Any

from prometheus_client import CollectorRegistry, Gauge, start_http_server

from miles.utils.ft.agents.collectors.base import BaseCollector
from miles.utils.ft.models import DiagnosticResult, MetricSample

logger = logging.getLogger(__name__)

_DEFAULT_COLLECT_INTERVAL_SECONDS = 10.0

_GaugeKey = tuple[str, frozenset[str]]


class FtNodeAgent:
    def __init__(
        self,
        node_id: str,
        collectors: list[BaseCollector] | None = None,
        collect_interval_seconds: float = _DEFAULT_COLLECT_INTERVAL_SECONDS,
        start_loop: bool = True,
    ) -> None:
        self._node_id = node_id
        self._collectors = collectors or []
        self._collect_interval = collect_interval_seconds

        self._registry = CollectorRegistry()
        self._gauges: dict[_GaugeKey, Gauge] = {}

        httpd, port = start_http_server(port=0, registry=self._registry)
        self._httpd = httpd
        self._port = port

        self._loop_task: asyncio.Task[None] | None = None
        if start_loop and self._collectors:
            self._loop_task = asyncio.get_event_loop().create_task(
                self._collection_loop()
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_exporter_address(self) -> str:
        return f"http://localhost:{self._port}"

    async def stop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

        self._httpd.shutdown()

    # ------------------------------------------------------------------
    # Stub methods (future milestones)
    # ------------------------------------------------------------------

    async def collect_logs(self) -> dict[str, str]:
        raise NotImplementedError(
            "collect_logs will be implemented in diag-framework milestone"
        )

    async def run_diagnostic(self, diagnostic_type: str) -> DiagnosticResult:
        raise NotImplementedError(
            "run_diagnostic will be implemented in diag-framework milestone"
        )

    async def cleanup_training_processes(self, training_job_id: str) -> None:
        raise NotImplementedError(
            "cleanup_training_processes will be implemented in platform-adapters milestone"
        )

    # ------------------------------------------------------------------
    # Background collection loop
    # ------------------------------------------------------------------

    async def _collection_loop(self) -> None:
        while True:
            all_metrics: list[MetricSample] = []

            for collector in self._collectors:
                try:
                    output = await collector.collect()
                    all_metrics.extend(output.metrics)
                except Exception:
                    logger.warning(
                        "Collector %s failed on node %s",
                        type(collector).__name__,
                        self._node_id,
                        exc_info=True,
                    )

            self._update_exporter(all_metrics)
            await asyncio.sleep(self._collect_interval)

    # ------------------------------------------------------------------
    # Exporter update
    # ------------------------------------------------------------------

    def _update_exporter(self, metrics: list[MetricSample]) -> None:
        for sample in metrics:
            gauge_name = f"ft_node_{sample.name}"
            label_keys = frozenset(sample.labels.keys())
            key: _GaugeKey = (gauge_name, label_keys)

            gauge = self._gauges.get(key)
            if gauge is None:
                sorted_keys = sorted(label_keys)
                gauge = Gauge(
                    gauge_name,
                    f"FT node metric: {sample.name}",
                    labelnames=sorted_keys,
                    registry=self._registry,
                )
                self._gauges[key] = gauge

            if sample.labels:
                gauge.labels(**sample.labels).set(sample.value)
            else:
                gauge.set(sample.value)
