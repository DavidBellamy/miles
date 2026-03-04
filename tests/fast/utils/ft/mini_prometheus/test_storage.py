from datetime import datetime, timedelta

import pytest

from miles.utils.ft.controller.mini_prometheus.storage import (
    MiniPrometheus,
    MiniPrometheusConfig,
)
from miles.utils.ft.models import MetricSample


class TestMiniPrometheusInstantQuery:
    def _make_store(self) -> MiniPrometheus:
        return MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))

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
                samples=[MetricSample(name="nic_alert", labels={}, value=1.0)],
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
        assert df["value"][0] <= 3.0

    def test_changes_no_change(self) -> None:
        store = MiniPrometheus(config=MiniPrometheusConfig(
            retention=timedelta(minutes=60),
        ))
        now = datetime.utcnow()

        for i in range(3):
            store.ingest_samples(
                target_id="node-0",
                samples=[MetricSample(
                    name="training_iteration", labels={"rank": "0"}, value=100.0,
                )],
                timestamp=now - timedelta(minutes=2 - i),
            )

        df = store.instant_query("changes(training_iteration[5m])")
        assert len(df) == 1
        assert df["value"][0] == 0.0

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
                    name="training_iteration", labels={"rank": "0"}, value=val,
                )],
                timestamp=now - timedelta(minutes=4 - i),
            )

        df = store.instant_query("changes(training_iteration[5m])")
        assert df["value"][0] == 3.0

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
                samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=val)],
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
                samples=[MetricSample(name="metric_a", labels={}, value=val)],
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

        store.ingest_samples(
            target_id="node-0",
            samples=[MetricSample(name="gpu_temp", labels={"gpu": "0"}, value=70.0)],
            timestamp=now - timedelta(minutes=10),
        )
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
