"""Tests for miles.utils.ft.controller.metrics.aggregation_mixin."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from miles.utils.ft.controller.metrics.aggregation_mixin import RangeAggregationMixin


class _ConcreteAggregation(RangeAggregationMixin):
    """Concrete implementation that records calls to _dispatch_range_function."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, timedelta, dict[str, str] | None]] = []

    def _dispatch_range_function(
        self,
        func_name: str,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None,
    ) -> pl.DataFrame:
        self.calls.append((func_name, metric_name, window, label_filters))
        return pl.DataFrame()


class TestRangeAggregationMixin:
    def test_changes_dispatches_correctly(self) -> None:
        agg = _ConcreteAggregation()
        window = timedelta(minutes=5)

        agg.changes("my_metric", window=window, label_filters={"a": "b"})

        assert len(agg.calls) == 1
        assert agg.calls[0] == ("changes", "my_metric", window, {"a": "b"})

    def test_count_over_time_dispatches_correctly(self) -> None:
        agg = _ConcreteAggregation()
        window = timedelta(hours=1)

        agg.count_over_time("counter_metric", window=window)

        assert len(agg.calls) == 1
        assert agg.calls[0] == ("count_over_time", "counter_metric", window, None)

    def test_avg_over_time_dispatches_correctly(self) -> None:
        agg = _ConcreteAggregation()
        window = timedelta(seconds=30)

        agg.avg_over_time("gauge_metric", window=window, label_filters={"node": "n1"})

        assert len(agg.calls) == 1
        assert agg.calls[0] == ("avg_over_time", "gauge_metric", window, {"node": "n1"})
