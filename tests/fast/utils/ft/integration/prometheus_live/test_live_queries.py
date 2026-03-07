"""Integration tests against a real Prometheus instance.

These tests download the Prometheus binary, start a local server,
push metrics via a local exporter, and query via PrometheusClient.
"""

from __future__ import annotations

import time
from datetime import timedelta

import pytest

from miles.utils.ft.controller.metrics.prometheus_api.store import PrometheusClient
from tests.fast.utils.ft.integration.prometheus_live.conftest import ExporterInfo

pytestmark = pytest.mark.integration

_POLL_INTERVAL = 1.0
_POLL_TIMEOUT = 15.0


def _poll_until_nonempty(
    client: PrometheusClient,
    metric_name: str,
    label_filters: dict[str, str] | None = None,
    timeout: float = _POLL_TIMEOUT,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        df = client.query_latest(metric_name, label_filters=label_filters)
        if not df.is_empty():
            return
        time.sleep(_POLL_INTERVAL)
    raise TimeoutError(
        f"Metric {metric_name!r} (filters={label_filters}) not available after {timeout}s"
    )


class TestLiveQueryLatest:
    def test_query_latest_returns_correct_value(
        self, prometheus_server: str, local_exporter: ExporterInfo,
    ) -> None:
        client = PrometheusClient(url=prometheus_server)
        _poll_until_nonempty(client, "test_gauge")

        df = client.query_latest("test_gauge")

        assert df.shape[0] >= 1
        assert "__name__" in df.columns
        assert "value" in df.columns
        assert df["value"][0] == 42.0


class TestLiveQueryRange:
    def test_query_range_returns_time_series(
        self, prometheus_server: str, local_exporter: ExporterInfo,
    ) -> None:
        client = PrometheusClient(url=prometheus_server, range_query_step_seconds=1)
        _poll_until_nonempty(client, "test_gauge")

        time.sleep(3)

        df = client.query_range("test_gauge", window=timedelta(seconds=30))

        assert df.shape[0] >= 2, f"Expected multiple data points, got {df.shape[0]}"
        assert "timestamp" in df.columns
        assert "value" in df.columns


class TestLiveChanges:
    def test_changes_detects_value_change(
        self, prometheus_server: str, local_exporter: ExporterInfo,
    ) -> None:
        client = PrometheusClient(url=prometheus_server)
        _poll_until_nonempty(client, "test_gauge")

        local_exporter.test_gauge.set(99.0)
        time.sleep(3)

        df = client.changes("test_gauge", window=timedelta(seconds=30))

        assert not df.is_empty()
        assert df["value"][0] >= 1.0

        local_exporter.test_gauge.set(42.0)


class TestLiveLabelFilters:
    def test_label_filters_return_matching_series(
        self, prometheus_server: str, local_exporter: ExporterInfo,
    ) -> None:
        client = PrometheusClient(url=prometheus_server)
        _poll_until_nonempty(client, "test_labeled_gauge")

        df = client.query_latest(
            "test_labeled_gauge",
            label_filters={"device": "ib0"},
        )

        assert df.shape[0] == 1
        assert df["value"][0] == 1.0


class TestLiveEmptyResult:
    def test_nonexistent_metric_returns_empty(
        self, prometheus_server: str,
    ) -> None:
        client = PrometheusClient(url=prometheus_server)

        df = client.query_latest("completely_nonexistent_metric_xyz")

        assert df.is_empty()
        assert "__name__" in df.columns
        assert "value" in df.columns
