"""Schema conformance: MiniPrometheus and PrometheusClient must produce
DataFrames with identical column names and compatible dtypes for every
MetricQueryProtocol method.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import httpx
import polars as pl
import pytest

from miles.utils.ft.controller.metrics.mini_prometheus.storage import MiniPrometheus
from miles.utils.ft.controller.metrics.prometheus_api.store import PrometheusClient
from miles.utils.ft.models.metrics import GaugeSample


METRIC_NAME = "test_metric"
LABELS = {"node_id": "node-0", "device": "ib0"}
VALUE = 42.5


def _make_response(json_data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json=json_data,
        request=httpx.Request("GET", "http://fake:9090/api/v1/query"),
    )


def _mini_store_with_data() -> MiniPrometheus:
    store = MiniPrometheus()
    store.ingest_samples(
        target_id=LABELS["node_id"],
        samples=[GaugeSample(name=METRIC_NAME, labels={"device": LABELS["device"]}, value=VALUE)],
    )
    return store


def _vector_json(
    metric_name: str = METRIC_NAME,
    labels: dict[str, str] | None = None,
    value: float = VALUE,
) -> dict[str, Any]:
    labels = labels or LABELS
    metric = {"__name__": metric_name, **labels}
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [{"metric": metric, "value": [time.time(), str(value)]}],
        },
    }


def _matrix_json(
    metric_name: str = METRIC_NAME,
    labels: dict[str, str] | None = None,
    values: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    labels = labels or LABELS
    metric = {"__name__": metric_name, **labels}
    if values is None:
        now = time.time()
        values = [(now - 10, VALUE), (now - 5, VALUE + 1), (now, VALUE + 2)]
    return {
        "status": "success",
        "data": {
            "resultType": "matrix",
            "result": [
                {"metric": metric, "values": [[ts, str(v)] for ts, v in values]},
            ],
        },
    }


@contextmanager
def _prom_client(response_json: dict[str, Any]) -> Iterator[PrometheusClient]:
    from unittest.mock import patch

    with patch.object(httpx.Client, "get", return_value=_make_response(response_json)):
        yield PrometheusClient(url="http://fake:9090")


def _assert_columns_match(df_mini: pl.DataFrame, df_prom: pl.DataFrame) -> None:
    assert sorted(df_mini.columns) == sorted(df_prom.columns), (
        f"Column mismatch: mini={sorted(df_mini.columns)} prom={sorted(df_prom.columns)}"
    )


def _assert_core_dtypes_match(
    df_mini: pl.DataFrame,
    df_prom: pl.DataFrame,
    columns: list[str],
) -> None:
    for col in columns:
        if col in df_mini.columns and col in df_prom.columns:
            assert df_mini[col].dtype == df_prom[col].dtype, (
                f"dtype mismatch for '{col}': mini={df_mini[col].dtype} prom={df_prom[col].dtype}"
            )


class TestQueryLatestConformance:
    def test_columns_match(self) -> None:
        store = _mini_store_with_data()
        df_mini = store.query_latest(METRIC_NAME)

        with _prom_client(_vector_json()) as client:
            df_prom = client.query_latest(METRIC_NAME)

        _assert_columns_match(df_mini, df_prom)

    def test_core_dtypes_match(self) -> None:
        store = _mini_store_with_data()
        df_mini = store.query_latest(METRIC_NAME)

        with _prom_client(_vector_json()) as client:
            df_prom = client.query_latest(METRIC_NAME)

        _assert_core_dtypes_match(df_mini, df_prom, ["__name__", "value"])

    def test_value_semantics_match(self) -> None:
        store = _mini_store_with_data()
        df_mini = store.query_latest(METRIC_NAME)

        with _prom_client(_vector_json()) as client:
            df_prom = client.query_latest(METRIC_NAME)

        assert df_mini.shape[0] == df_prom.shape[0]
        assert df_mini["value"][0] == df_prom["value"][0] == VALUE

    def test_empty_result_schema_match(self) -> None:
        store = MiniPrometheus()
        df_mini = store.query_latest("nonexistent")

        empty_json = {
            "status": "success",
            "data": {"resultType": "vector", "result": []},
        }
        with _prom_client(empty_json) as client:
            df_prom = client.query_latest("nonexistent")

        _assert_columns_match(df_mini, df_prom)
        _assert_core_dtypes_match(df_mini, df_prom, ["__name__", "value"])


class TestQueryRangeConformance:
    def test_columns_match(self) -> None:
        store = _mini_store_with_data()
        df_mini = store.query_range(METRIC_NAME, window=timedelta(hours=1))

        with _prom_client(_matrix_json()) as client:
            df_prom = client.query_range(METRIC_NAME, window=timedelta(hours=1))

        _assert_columns_match(df_mini, df_prom)

    @pytest.mark.xfail(
        reason="Known issue: MiniPrometheus returns datetime, PrometheusClient returns float",
        strict=True,
    )
    def test_timestamp_dtype_match(self) -> None:
        store = _mini_store_with_data()
        df_mini = store.query_range(METRIC_NAME, window=timedelta(hours=1))

        with _prom_client(_matrix_json()) as client:
            df_prom = client.query_range(METRIC_NAME, window=timedelta(hours=1))

        _assert_core_dtypes_match(df_mini, df_prom, ["timestamp"])

    def test_value_dtype_match(self) -> None:
        store = _mini_store_with_data()
        df_mini = store.query_range(METRIC_NAME, window=timedelta(hours=1))

        with _prom_client(_matrix_json()) as client:
            df_prom = client.query_range(METRIC_NAME, window=timedelta(hours=1))

        _assert_core_dtypes_match(df_mini, df_prom, ["__name__", "value"])

    def test_empty_result_schema_match(self) -> None:
        store = MiniPrometheus()
        df_mini = store.query_range("nonexistent", window=timedelta(hours=1))

        empty_json = {
            "status": "success",
            "data": {"resultType": "matrix", "result": []},
        }
        with _prom_client(empty_json) as client:
            df_prom = client.query_range("nonexistent", window=timedelta(hours=1))

        _assert_columns_match(df_mini, df_prom)
        _assert_core_dtypes_match(df_mini, df_prom, ["__name__", "timestamp", "value"])


class TestRangeAggregationConformance:
    """changes / count_over_time / avg_over_time all return instant-style DataFrames."""

    @pytest.fixture()
    def mini_store(self) -> MiniPrometheus:
        store = MiniPrometheus()
        store.ingest_samples(
            target_id="node-0",
            samples=[GaugeSample(name=METRIC_NAME, labels={"device": "ib0"}, value=1.0)],
        )
        store.ingest_samples(
            target_id="node-0",
            samples=[GaugeSample(name=METRIC_NAME, labels={"device": "ib0"}, value=2.0)],
        )
        return store

    def test_changes_columns_and_dtypes_match(self, mini_store: MiniPrometheus) -> None:
        df_mini = mini_store.changes(METRIC_NAME, window=timedelta(hours=1))

        with _prom_client(_vector_json(value=1.0)) as client:
            df_prom = client.changes(METRIC_NAME, window=timedelta(hours=1))

        _assert_columns_match(df_mini, df_prom)
        _assert_core_dtypes_match(df_mini, df_prom, ["__name__", "value"])

    def test_count_over_time_columns_and_dtypes_match(self, mini_store: MiniPrometheus) -> None:
        df_mini = mini_store.count_over_time(METRIC_NAME, window=timedelta(hours=1))

        with _prom_client(_vector_json(value=2.0)) as client:
            df_prom = client.count_over_time(METRIC_NAME, window=timedelta(hours=1))

        _assert_columns_match(df_mini, df_prom)
        _assert_core_dtypes_match(df_mini, df_prom, ["__name__", "value"])

    def test_avg_over_time_columns_and_dtypes_match(self, mini_store: MiniPrometheus) -> None:
        df_mini = mini_store.avg_over_time(METRIC_NAME, window=timedelta(hours=1))

        with _prom_client(_vector_json(value=1.5)) as client:
            df_prom = client.avg_over_time(METRIC_NAME, window=timedelta(hours=1))

        _assert_columns_match(df_mini, df_prom)
        _assert_core_dtypes_match(df_mini, df_prom, ["__name__", "value"])
