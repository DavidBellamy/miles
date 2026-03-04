from datetime import datetime, timedelta

import polars as pl
import pytest

from miles.utils.ft.controller.metric_store import (
    CompareExpr,
    CompareOp,
    LabelMatchOp,
    LabelMatcher,
    MetricSelector,
    MiniPrometheus,
    MiniPrometheusConfig,
    RangeFunction,
    RangeFunctionCompare,
    _parse_prometheus_text,
    parse_promql,
)
from miles.utils.ft.models import MetricSample


# ---------------------------------------------------------------------------
# PromQL parser tests
# ---------------------------------------------------------------------------


class TestParsePromQL:
    def test_simple_metric_name(self) -> None:
        expr = parse_promql("ft_node_gpu_available")
        assert isinstance(expr, MetricSelector)
        assert expr.name == "ft_node_gpu_available"
        assert expr.matchers == []

    def test_metric_with_label_filter(self) -> None:
        expr = parse_promql('ft_node_xid_code_recent{xid="48"}')
        assert isinstance(expr, MetricSelector)
        assert expr.name == "ft_node_xid_code_recent"
        assert len(expr.matchers) == 1
        assert expr.matchers[0].label == "xid"
        assert expr.matchers[0].op == LabelMatchOp.EQ
        assert expr.matchers[0].value == "48"

    def test_metric_with_neq_label(self) -> None:
        expr = parse_promql('gpu_available{node_id!="node-0"}')
        assert isinstance(expr, MetricSelector)
        assert expr.matchers[0].op == LabelMatchOp.NEQ

    def test_metric_with_regex_label(self) -> None:
        expr = parse_promql('gpu_available{node_id=~"node-.*"}')
        assert isinstance(expr, MetricSelector)
        assert expr.matchers[0].op == LabelMatchOp.RE
        assert expr.matchers[0].value == "node-.*"

    def test_compare_eq(self) -> None:
        expr = parse_promql("ft_node_gpu_available == 0")
        assert isinstance(expr, CompareExpr)
        assert expr.selector.name == "ft_node_gpu_available"
        assert expr.op == CompareOp.EQ
        assert expr.threshold == 0.0

    def test_compare_gt(self) -> None:
        expr = parse_promql("gpu_temperature_celsius > 90")
        assert isinstance(expr, CompareExpr)
        assert expr.op == CompareOp.GT
        assert expr.threshold == 90.0

    def test_compare_lte(self) -> None:
        expr = parse_promql("disk_available_bytes <= 1000000")
        assert isinstance(expr, CompareExpr)
        assert expr.op == CompareOp.LTE

    def test_range_function_count_over_time(self) -> None:
        expr = parse_promql("count_over_time(nic_alert[5m])")
        assert isinstance(expr, RangeFunction)
        assert expr.func_name == "count_over_time"
        assert expr.selector.name == "nic_alert"
        assert expr.duration == timedelta(minutes=5)

    def test_range_function_changes(self) -> None:
        expr = parse_promql("changes(training_iteration[10m])")
        assert isinstance(expr, RangeFunction)
        assert expr.func_name == "changes"
        assert expr.duration == timedelta(minutes=10)

    def test_range_function_with_compare(self) -> None:
        expr = parse_promql("count_over_time(nic_alert[5m]) >= 2")
        assert isinstance(expr, RangeFunctionCompare)
        assert expr.func.func_name == "count_over_time"
        assert expr.op == CompareOp.GTE
        assert expr.threshold == 2.0

    def test_changes_with_compare(self) -> None:
        expr = parse_promql("changes(training_iteration[10m]) == 0")
        assert isinstance(expr, RangeFunctionCompare)
        assert expr.func.func_name == "changes"
        assert expr.op == CompareOp.EQ
        assert expr.threshold == 0.0

    def test_range_function_with_labels(self) -> None:
        expr = parse_promql('count_over_time(xid_code_recent{xid="48"}[5m])')
        assert isinstance(expr, RangeFunction)
        assert expr.selector.name == "xid_code_recent"
        assert len(expr.selector.matchers) == 1
        assert expr.selector.matchers[0].value == "48"

    def test_duration_seconds(self) -> None:
        expr = parse_promql("count_over_time(metric[30s])")
        assert isinstance(expr, RangeFunction)
        assert expr.duration == timedelta(seconds=30)

    def test_duration_hours(self) -> None:
        expr = parse_promql("count_over_time(metric[1h])")
        assert isinstance(expr, RangeFunction)
        assert expr.duration == timedelta(hours=1)


# ---------------------------------------------------------------------------
# Prometheus text format parser tests
# ---------------------------------------------------------------------------


class TestParsePrometheusText:
    def test_simple_metric(self) -> None:
        text = "gpu_temperature_celsius 75.0\n"
        samples = _parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].name == "gpu_temperature_celsius"
        assert samples[0].value == 75.0
        assert samples[0].labels == {}

    def test_metric_with_labels(self) -> None:
        text = 'gpu_temperature_celsius{gpu="0",node="n1"} 82.5\n'
        samples = _parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].labels == {"gpu": "0", "node": "n1"}
        assert samples[0].value == 82.5

    def test_skips_comments_and_help(self) -> None:
        text = (
            "# HELP gpu_temp GPU temperature\n"
            "# TYPE gpu_temp gauge\n"
            "gpu_temp 42.0\n"
        )
        samples = _parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].name == "gpu_temp"

    def test_multiple_metrics(self) -> None:
        text = (
            "metric_a 1.0\n"
            "metric_b 2.0\n"
            "metric_c{label=\"x\"} 3.0\n"
        )
        samples = _parse_prometheus_text(text)
        assert len(samples) == 3

    def test_metric_with_timestamp(self) -> None:
        text = "http_requests_total 1000 1700000000000\n"
        samples = _parse_prometheus_text(text)
        assert len(samples) == 1
        assert samples[0].value == 1000.0


# ---------------------------------------------------------------------------
# MiniPrometheus instant query tests
# ---------------------------------------------------------------------------


class TestMiniPrometheusInstantQuery:
    def _make_store(self) -> MiniPrometheus:
        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
        return store

    def test_simple_metric_query(self) -> None:
        store = self._make_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=75.0)],
        )

        df = store.instant_query("gpu_temp")
        assert len(df) == 1
        assert df["value"][0] == 75.0

    def test_latest_value_returned(self) -> None:
        store = self._make_store()
        t1 = datetime(2026, 1, 1, 0, 0, 0)
        t2 = datetime(2026, 1, 1, 0, 0, 10)

        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=70.0)],
            timestamp=t1,
        )
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=80.0)],
            timestamp=t2,
        )

        df = store.instant_query("gpu_temp")
        assert len(df) == 1
        assert df["value"][0] == 80.0

    def test_compare_eq_zero(self) -> None:
        store = self._make_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                MetricSample(name="gpu_available", labels={"gpu": "0"}, value=1.0),
                MetricSample(name="gpu_available", labels={"gpu": "1"}, value=0.0),
            ],
        )

        df = store.instant_query("gpu_available == 0")
        assert len(df) == 1
        assert df["gpu"][0] == "1"

    def test_label_filter(self) -> None:
        store = self._make_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                MetricSample(name="xid_code_recent", labels={"xid": "48"}, value=1.0),
                MetricSample(name="xid_code_recent", labels={"xid": "31"}, value=1.0),
            ],
        )

        df = store.instant_query('xid_code_recent{xid="48"}')
        assert len(df) == 1
        assert df["xid"][0] == "48"

    def test_label_neq_filter(self) -> None:
        store = self._make_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[
                MetricSample(name="gpu_available", labels={"gpu": "0"}, value=1.0),
            ],
        )
        store.ingest_samples(
            target_id="node-1",
            samples=[
                MetricSample(name="gpu_available", labels={"gpu": "0"}, value=0.0),
            ],
        )

        df = store.instant_query('gpu_available{node_id!="node-0"}')
        assert len(df) == 1
        assert df["node_id"][0] == "node-1"

    def test_multiple_targets(self) -> None:
        store = self._make_store()
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=70.0)],
        )
        store.ingest_samples(
            target_id="node-1",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=80.0)],
        )

        df = store.instant_query("gpu_temp")
        assert len(df) == 2

    def test_empty_result(self) -> None:
        store = self._make_store()
        df = store.instant_query("nonexistent_metric")
        assert df.is_empty()


class TestMiniPrometheusRangeFunctions:
    def _make_store_with_samples(self) -> MiniPrometheus:
        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
        now = datetime.utcnow()

        for i in range(5):
            store.ingest_samples(
                target_id="node-0",
                samples=[MetricSample(
                    name="nic_alert",
                    labels={},
                    value=1.0,
                )],
                timestamp=now - timedelta(minutes=4 - i),
            )

        return store

    def test_count_over_time(self) -> None:
        store = self._make_store_with_samples()
        df = store.instant_query("count_over_time(nic_alert[5m])")
        assert len(df) == 1
        assert df["value"][0] == 5.0

    def test_count_over_time_shorter_window(self) -> None:
        store = self._make_store_with_samples()
        df = store.instant_query("count_over_time(nic_alert[2m])")
        assert len(df) == 1
        assert df["value"][0] <= 3.0  # only recent samples in 2m window

    def test_changes(self) -> None:
        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
        now = datetime.utcnow()

        # iteration stays at 100 (no changes)
        for i in range(3):
            store.ingest_samples(
                target_id="node-0",
                samples=[MetricSample(
                    name="training_iteration",
                    labels={"rank": "0"},
                    value=100.0,
                )],
                timestamp=now - timedelta(minutes=2 - i),
            )

        df = store.instant_query("changes(training_iteration[5m])")
        assert len(df) == 1
        assert df["value"][0] == 0.0  # no changes

    def test_changes_with_actual_changes(self) -> None:
        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
        now = datetime.utcnow()

        values = [100.0, 101.0, 102.0, 102.0, 103.0]
        for i, val in enumerate(values):
            store.ingest_samples(
                target_id="node-0",
                samples=[MetricSample(
                    name="training_iteration",
                    labels={"rank": "0"},
                    value=val,
                )],
                timestamp=now - timedelta(minutes=4 - i),
            )

        df = store.instant_query("changes(training_iteration[5m])")
        assert df["value"][0] == 3.0  # 100->101, 101->102, 102->103

    def test_count_over_time_with_compare(self) -> None:
        store = self._make_store_with_samples()
        df = store.instant_query("count_over_time(nic_alert[5m]) >= 2")
        assert len(df) == 1
        assert df["value"][0] >= 2.0

    def test_min_over_time(self) -> None:
        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
        now = datetime.utcnow()

        for i, val in enumerate([75.0, 80.0, 70.0, 85.0]):
            store.ingest_samples(
                target_id="node-0",
                samples=[MetricSample(
                    name="gpu_temp",
                    labels={"gpu": "0"},
                    value=val,
                )],
                timestamp=now - timedelta(minutes=3 - i),
            )

        df = store.instant_query("min_over_time(gpu_temp[5m])")
        assert df["value"][0] == 70.0

    def test_avg_over_time(self) -> None:
        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
        now = datetime.utcnow()

        for i, val in enumerate([10.0, 20.0, 30.0]):
            store.ingest_samples(
                target_id="node-0",
                samples=[MetricSample(
                    name="metric_a",
                    labels={},
                    value=val,
                )],
                timestamp=now - timedelta(minutes=2 - i),
            )

        df = store.instant_query("avg_over_time(metric_a[5m])")
        assert df["value"][0] == pytest.approx(20.0)


class TestMiniPrometheusRangeQuery:
    def test_range_query_returns_time_series(self) -> None:
        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
        now = datetime.utcnow()
        t1 = now - timedelta(minutes=10)
        t2 = now - timedelta(minutes=5)
        t3 = now

        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=70.0)],
            timestamp=t1,
        )
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=75.0)],
            timestamp=t2,
        )
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=80.0)],
            timestamp=t3,
        )

        df = store.range_query(
            query="gpu_temp",
            start=now - timedelta(minutes=15),
            end=now + timedelta(minutes=1),
            step=timedelta(minutes=5),
        )
        assert len(df) == 3
        values = sorted(df["value"].to_list())
        assert values == [70.0, 75.0, 80.0]


class TestMiniPrometheusRetention:
    def test_expired_data_evicted(self) -> None:
        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=5),
        ))
        now = datetime.utcnow()

        # Old sample outside retention
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=70.0)],
            timestamp=now - timedelta(minutes=10),
        )

        # New sample within retention
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=80.0)],
            timestamp=now,
        )

        df = store.instant_query("gpu_temp")
        assert len(df) == 1
        assert df["value"][0] == 80.0


class TestMiniPrometheusIngestSamples:
    def test_ingest_adds_node_id_label(self) -> None:
        store = MiniPrometheus()
        store.ingest_samples(
            target_id="node-42",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=75.0)],
        )

        df = store.instant_query("gpu_temp")
        assert "node_id" in df.columns
        assert df["node_id"][0] == "node-42"

    def test_ingest_multiple_targets_separate(self) -> None:
        store = MiniPrometheus()
        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_available", labels={"gpu": "0"}, value=1.0)],
        )
        store.ingest_samples(
            target_id="node-1",
            samples=[MetricSample(name="gpu_available", labels={"gpu": "0"}, value=0.0)],
        )

        df = store.instant_query("gpu_available == 0")
        assert len(df) == 1
        assert df["node_id"][0] == "node-1"
