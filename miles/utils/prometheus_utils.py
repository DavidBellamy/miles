import logging
from typing import Any

import ray

logger = logging.getLogger(__name__)

_collector_handle: Any = None


class _PrometheusCollector:
    """Ray actor that owns all Prometheus gauges and exposes an HTTP endpoint."""

    def __init__(self, *, prometheus_port: int, run_name: str) -> None:
        from prometheus_client import Gauge, start_http_server

        self._Gauge = Gauge
        self._prometheus_port = prometheus_port
        self._run_name = run_name
        self._custom_gauges: dict[str, Any] = {}

        start_http_server(prometheus_port)
        logger.info("Prometheus HTTP server started on port %d", prometheus_port)

    def set_gauge_with_labels(
        self, name: str, label_keys: list[str], label_values: list[str], value: float
    ) -> None:
        """Set a gauge with custom labels (different from the default run_name labels)."""
        if name not in self._custom_gauges:
            self._custom_gauges[name] = self._Gauge(name, name, label_keys)
        self._custom_gauges[name].labels(*label_values).set(value)


def init_prometheus(*, prometheus_port: int, run_name: str) -> None:
    """Create the singleton _PrometheusCollector Ray actor."""
    global _collector_handle
    _collector_handle = ray.remote(_PrometheusCollector).remote(
        prometheus_port=prometheus_port, run_name=run_name
    )


def get_prometheus() -> Any:
    """Return the collector actor handle, or None if not initialised."""
    return _collector_handle


def set_prometheus_gauge(
    name: str, label_keys: list[str], label_values: list[str], value: float
) -> None:
    """Set a gauge with custom labels via the collector actor. No-op if Prometheus is not enabled."""
    handle = get_prometheus()
    if handle is None:
        return

    try:
        ray.get(handle.set_gauge_with_labels.remote(name, label_keys, label_values, value))
    except Exception:
        logger.warning("Failed to set prometheus gauge %s", name, exc_info=True)
