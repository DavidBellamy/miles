from datetime import datetime, timedelta

from miles.utils.ft.controller.metric_store import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import MetricSample


def make_metric(
    name: str,
    value: float,
    labels: dict[str, str] | None = None,
) -> MetricSample:
    return MetricSample(name=name, labels=labels or {}, value=value)


def make_fake_metric_store(
    metrics: list[MetricSample] | None = None,
    target_id: str = "node-0",
) -> MiniPrometheus:
    store = MiniPrometheus(config=MiniPrometheusConfig(
        retention=timedelta(minutes=60),
    ))
    if metrics:
        store.ingest_samples(target_id=target_id, samples=metrics)
    return store


def make_fake_mini_wandb(
    steps: dict[int, dict[str, float]] | None = None,
    run_id: str = "test-run",
    rank: int = 0,
) -> MiniWandb:
    wandb = MiniWandb(active_run_id=run_id)
    if steps:
        for step_num, metrics in sorted(steps.items()):
            wandb.log_step(run_id=run_id, rank=rank, step=step_num, metrics=metrics)
    return wandb
