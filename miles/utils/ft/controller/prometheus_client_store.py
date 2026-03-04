from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import httpx
import polars as pl

logger = logging.getLogger(__name__)


class PrometheusClient:
    """MetricStoreProtocol implementation backed by a real Prometheus HTTP API.

    Queries are forwarded to Prometheus and results parsed into Polars DataFrames
    with the same schema as MiniPrometheus, so detectors work without modification.
    """

    def __init__(self, url: str, timeout: float = 10.0) -> None:
        self._url = url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def instant_query(self, query: str) -> pl.DataFrame:
        data = self._fetch_json(
            endpoint="/api/v1/query",
            params={"query": query},
            error_label="prometheus_instant_query_failed",
            query=query,
        )
        if data is None:
            return _empty_instant_dataframe()

        return _parse_instant_response(data)

    def range_query(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> pl.DataFrame:
        data = self._fetch_json(
            endpoint="/api/v1/query_range",
            params={
                "query": query,
                "start": start.timestamp(),
                "end": end.timestamp(),
                "step": step.total_seconds(),
            },
            error_label="prometheus_range_query_failed",
            query=query,
        )
        if data is None:
            return _empty_range_dataframe()

        return _parse_range_response(data)

    def _fetch_json(
        self,
        endpoint: str,
        params: dict[str, Any],
        error_label: str,
        query: str,
    ) -> dict[str, Any] | None:
        try:
            response = self._client.get(f"{self._url}{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except Exception:
            logger.warning("%s query=%s", error_label, query, exc_info=True)
            return None

    def close(self) -> None:
        self._client.close()


def _empty_instant_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {"__name__": pl.Series([], dtype=pl.Utf8), "value": pl.Series([], dtype=pl.Float64)}
    )


def _empty_range_dataframe() -> pl.DataFrame:
    return pl.DataFrame({
        "__name__": pl.Series([], dtype=pl.Utf8),
        "timestamp": pl.Series([], dtype=pl.Float64),
        "value": pl.Series([], dtype=pl.Float64),
    })


def _extract_results(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]] | None:
    if data.get("status") != "success":
        logger.warning("prometheus_query_error response=%s", data)
        return None

    data_section: dict[str, Any] = data.get("data") or {}
    result: list[dict[str, Any]] = data_section.get("result") or []
    if not result:
        return None

    return data_section.get("resultType", ""), result


def _parse_instant_response(data: dict[str, Any]) -> pl.DataFrame:
    extracted = _extract_results(data)
    if extracted is None:
        return _empty_instant_dataframe()

    result_type, result = extracted
    if result_type == "vector":
        return _parse_vector(result)
    if result_type == "scalar":
        return _parse_scalar(result)

    logger.warning("prometheus_unsupported_result_type type=%s", result_type)
    return _empty_instant_dataframe()


def _parse_vector(result: list[dict[str, Any]]) -> pl.DataFrame:
    rows: list[tuple[dict[str, str], float]] = []
    all_label_keys: set[str] = set()

    for item in result:
        metric: dict[str, str] = item.get("metric") or {}
        value_pair = item.get("value") or [0, "0"]
        try:
            parsed_value = float(value_pair[1])
        except (IndexError, TypeError, ValueError):
            continue
        all_label_keys.update(metric.keys())
        rows.append((metric, parsed_value))

    sorted_label_keys = _collect_sorted_label_keys(all_label_keys)
    records: list[dict[str, object]] = []
    for metric, value in rows:
        record = _build_label_record(metric=metric, sorted_label_keys=sorted_label_keys)
        record["value"] = value
        records.append(record)

    if not records:
        return _empty_instant_dataframe()

    return pl.DataFrame(records)


def _parse_scalar(result: list[Any]) -> pl.DataFrame:
    if len(result) != 2:
        return _empty_instant_dataframe()

    try:
        return pl.DataFrame({"__name__": [""], "value": [float(result[1])]})
    except (TypeError, ValueError):
        return _empty_instant_dataframe()


def _parse_range_response(data: dict[str, Any]) -> pl.DataFrame:
    extracted = _extract_results(data)
    if extracted is None:
        return _empty_range_dataframe()

    result_type, result = extracted
    if result_type != "matrix":
        logger.warning("prometheus_unsupported_range_result_type type=%s", result_type)
        return _empty_range_dataframe()

    return _parse_matrix(result)


def _parse_matrix(result: list[dict[str, Any]]) -> pl.DataFrame:
    all_label_keys: set[str] = set()
    for item in result:
        metric = item.get("metric") or {}
        all_label_keys.update(metric.keys())

    sorted_label_keys = _collect_sorted_label_keys(all_label_keys)
    records: list[dict[str, object]] = []
    for item in result:
        metric: dict[str, str] = item.get("metric") or {}
        for ts, value_str in (item.get("values") or []):
            try:
                parsed_value = float(value_str)
                parsed_ts = float(ts)
            except (TypeError, ValueError):
                continue

            record = _build_label_record(metric=metric, sorted_label_keys=sorted_label_keys)
            record["timestamp"] = parsed_ts
            record["value"] = parsed_value
            records.append(record)

    if not records:
        return _empty_range_dataframe()

    return pl.DataFrame(records)


def _collect_sorted_label_keys(all_label_keys: set[str]) -> list[str]:
    return sorted(k for k in all_label_keys if k != "__name__")


def _build_label_record(
    metric: dict[str, str],
    sorted_label_keys: list[str],
) -> dict[str, object]:
    record: dict[str, object] = {"__name__": metric.get("__name__", "")}
    for key in sorted_label_keys:
        record[key] = metric.get(key, "")
    return record
