"""Tests for conftest.py builder helpers."""

from tests.fast.utils.ft.conftest import (
    make_fake_metric_store,
    make_fake_mini_wandb,
    make_metric,
)


class TestMakeMetric:
    def test_basic(self) -> None:
        m = make_metric("gpu_temp", 75.0)
        assert m.name == "gpu_temp"
        assert m.value == 75.0
        assert m.labels == {}

    def test_with_labels(self) -> None:
        m = make_metric("gpu_temp", 75.0, labels={"gpu": "0"})
        assert m.labels == {"gpu": "0"}


class TestMakeFakeMetricStore:
    def test_empty_store(self) -> None:
        store = make_fake_metric_store()
        df = store.instant_query("anything")
        assert df.is_empty()

    def test_with_metrics(self) -> None:
        metrics = [
            make_metric("gpu_temp", 75.0, labels={"gpu": "0"}),
            make_metric("gpu_temp", 82.0, labels={"gpu": "1"}),
        ]
        store = make_fake_metric_store(metrics=metrics)
        df = store.instant_query("gpu_temp")
        assert len(df) == 2


class TestMakeFakeMiniWandb:
    def test_empty_wandb(self) -> None:
        wandb = make_fake_mini_wandb()
        assert wandb.latest(metric_name="loss", rank=0) is None

    def test_with_steps(self) -> None:
        wandb = make_fake_mini_wandb(steps={
            1: {"loss": 3.0, "grad_norm": 1.0},
            2: {"loss": 2.5, "grad_norm": 0.8},
        })
        assert wandb.latest(metric_name="loss", rank=0) == 2.5
        result = wandb.query_last_n_steps(metric_name="loss", rank=0, last_n=10)
        assert len(result) == 2
        assert result[0] == (1, 3.0)
        assert result[1] == (2, 2.5)
