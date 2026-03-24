from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, patch

import prometheus_client
import pytest

import miles.utils.prometheus_utils as prometheus_mod
from miles.utils.prometheus_utils import _PrometheusCollector, set_prometheus_gauge

_registry = prometheus_client.REGISTRY


def _make_args(
    prometheus_port: int = 9090,
    prometheus_run_name: str | None = None,
    wandb_group: str | None = None,
) -> Namespace:
    return Namespace(
        prometheus_port=prometheus_port,
        prometheus_run_name=prometheus_run_name,
        wandb_group=wandb_group,
    )


@pytest.fixture()
def collector() -> _PrometheusCollector:
    with patch("prometheus_client.start_http_server"):
        c = _PrometheusCollector(_make_args(prometheus_run_name="test-run"))

    yield c

    for gauge in list(c._gauges.values()):
        _registry.unregister(gauge)
    for gauge in list(c._custom_gauges.values()):
        _registry.unregister(gauge)


def _sample(name: str, labels: dict[str, str]) -> float | None:
    return _registry.get_sample_value(name, labels)


class TestPrometheusCollectorInit:
    def test_run_name_from_prometheus_run_name(self) -> None:
        with patch("prometheus_client.start_http_server"):
            c = _PrometheusCollector(_make_args(prometheus_run_name="my-run"))
        assert c._run_name == "my-run"

    def test_run_name_falls_back_to_wandb_group(self) -> None:
        with patch("prometheus_client.start_http_server"):
            c = _PrometheusCollector(_make_args(wandb_group="wandb-group"))
        assert c._run_name == "wandb-group"

    def test_run_name_defaults_to_miles_training(self) -> None:
        with patch("prometheus_client.start_http_server"):
            c = _PrometheusCollector(_make_args())
        assert c._run_name == "miles_training"

    def test_prometheus_run_name_takes_priority_over_wandb_group(self) -> None:
        with patch("prometheus_client.start_http_server"):
            c = _PrometheusCollector(
                _make_args(prometheus_run_name="prom", wandb_group="wandb")
            )
        assert c._run_name == "prom"

    def test_starts_http_server_on_given_port(self) -> None:
        with patch("prometheus_client.start_http_server") as mock_start:
            _PrometheusCollector(_make_args(prometheus_port=8888))
        mock_start.assert_called_once_with(8888)

    def test_ping_returns_true(self, collector: _PrometheusCollector) -> None:
        assert collector.ping() is True


class TestSetGauge:
    def test_value_readable_from_registry(self, collector: _PrometheusCollector) -> None:
        collector.set_gauge("test_sg_val", 42.0)
        assert _sample("test_sg_val", {"run_name": "test-run"}) == 42.0

    def test_update_overwrites_previous_value(self, collector: _PrometheusCollector) -> None:
        collector.set_gauge("test_sg_overwrite", 1.0)
        collector.set_gauge("test_sg_overwrite", 99.0)
        assert _sample("test_sg_overwrite", {"run_name": "test-run"}) == 99.0

    def test_uses_run_name_label(self, collector: _PrometheusCollector) -> None:
        collector.set_gauge("test_sg_label", 1.0)
        assert _sample("test_sg_label", {"run_name": "test-run"}) == 1.0
        assert _sample("test_sg_label", {"run_name": "wrong"}) is None


class TestUpdate:
    def test_adds_prefix_and_sets_values(self, collector: _PrometheusCollector) -> None:
        collector.update({"loss": 0.5, "mfu": 0.3})
        assert _sample("miles_metric_loss", {"run_name": "test-run"}) == 0.5
        assert _sample("miles_metric_mfu", {"run_name": "test-run"}) == 0.3

    def test_skips_non_numeric_values(self, collector: _PrometheusCollector) -> None:
        collector.update({"loss": 0.5, "name": "hello"})
        assert _sample("miles_metric_loss", {"run_name": "test-run"}) == 0.5
        assert _sample("miles_metric_name", {"run_name": "test-run"}) is None

    def test_sanitizes_slash(self, collector: _PrometheusCollector) -> None:
        collector.update({"train/loss": 1.0})
        assert _sample("miles_metric_train_loss", {"run_name": "test-run"}) == 1.0

    def test_sanitizes_dash(self, collector: _PrometheusCollector) -> None:
        collector.update({"grad-norm": 2.0})
        assert _sample("miles_metric_grad_norm", {"run_name": "test-run"}) == 2.0

    def test_sanitizes_at_sign(self, collector: _PrometheusCollector) -> None:
        collector.update({"lr@step": 3.0})
        assert _sample("miles_metric_lr_at_step", {"run_name": "test-run"}) == 3.0

    def test_int_values_accepted(self, collector: _PrometheusCollector) -> None:
        collector.update({"step": 100})
        assert _sample("miles_metric_step", {"run_name": "test-run"}) == 100.0

    def test_empty_dict_is_noop(self, collector: _PrometheusCollector) -> None:
        collector.update({})
        assert len(collector._gauges) == 0


class TestSetGaugeWithLabels:
    def test_value_readable_from_registry(self, collector: _PrometheusCollector) -> None:
        collector.set_gauge_with_labels(
            name="test_sgwl_val",
            label_keys=["session_id", "cell_id"],
            label_values=["sess-1", "cell-0"],
            value=1.0,
        )
        assert _sample("test_sgwl_val", {"session_id": "sess-1", "cell_id": "cell-0"}) == 1.0

    def test_update_overwrites_previous_value(self, collector: _PrometheusCollector) -> None:
        collector.set_gauge_with_labels(
            name="test_sgwl_overwrite",
            label_keys=["cell_id"],
            label_values=["cell-0"],
            value=1.0,
        )
        collector.set_gauge_with_labels(
            name="test_sgwl_overwrite",
            label_keys=["cell_id"],
            label_values=["cell-0"],
            value=0.0,
        )
        assert _sample("test_sgwl_overwrite", {"cell_id": "cell-0"}) == 0.0

    def test_different_label_values_independent(self, collector: _PrometheusCollector) -> None:
        collector.set_gauge_with_labels(
            name="test_sgwl_multi",
            label_keys=["cell_id"],
            label_values=["cell-0"],
            value=1.0,
        )
        collector.set_gauge_with_labels(
            name="test_sgwl_multi",
            label_keys=["cell_id"],
            label_values=["cell-1"],
            value=0.0,
        )
        assert _sample("test_sgwl_multi", {"cell_id": "cell-0"}) == 1.0
        assert _sample("test_sgwl_multi", {"cell_id": "cell-1"}) == 0.0

    def test_does_not_interfere_with_regular_gauges(self, collector: _PrometheusCollector) -> None:
        collector.set_gauge("test_sgwl_regular", 1.0)
        collector.set_gauge_with_labels(
            name="test_sgwl_custom",
            label_keys=["k"],
            label_values=["v"],
            value=2.0,
        )

        assert _sample("test_sgwl_regular", {"run_name": "test-run"}) == 1.0
        assert _sample("test_sgwl_custom", {"k": "v"}) == 2.0


class TestSetPrometheusGauge:
    @patch.object(prometheus_mod, "_collector_handle", None)
    def test_noop_when_collector_is_none(self) -> None:
        set_prometheus_gauge(
            name="miles_rollout_cell_alive",
            label_keys=["cell_id"],
            label_values=["cell-0"],
            value=1.0,
        )

    def test_happy_path_calls_remote(self) -> None:
        mock_handle = MagicMock()
        mock_remote_ref = MagicMock()
        mock_handle.set_gauge_with_labels.remote.return_value = mock_remote_ref

        with (
            patch.object(prometheus_mod, "_collector_handle", mock_handle),
            patch.object(prometheus_mod.ray, "get") as mock_ray_get,
        ):
            set_prometheus_gauge(
                name="miles_rollout_cell_alive",
                label_keys=["session_id", "cell_id"],
                label_values=["sess-1", "cell-0"],
                value=1.0,
            )

        mock_handle.set_gauge_with_labels.remote.assert_called_once_with(
            "miles_rollout_cell_alive",
            ["session_id", "cell_id"],
            ["sess-1", "cell-0"],
            1.0,
        )
        mock_ray_get.assert_called_once_with(mock_remote_ref)

    def test_swallows_exception_and_logs_warning(self) -> None:
        mock_handle = MagicMock()
        mock_handle.set_gauge_with_labels.remote.side_effect = RuntimeError("ray down")

        with (
            patch.object(prometheus_mod, "_collector_handle", mock_handle),
            patch.object(prometheus_mod, "logger") as mock_logger,
        ):
            set_prometheus_gauge(
                name="my_gauge",
                label_keys=["k"],
                label_values=["v"],
                value=1.0,
            )

        mock_logger.warning.assert_called_once()
        assert "my_gauge" in mock_logger.warning.call_args[0][1]
