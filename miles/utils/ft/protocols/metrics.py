from __future__ import annotations

from datetime import datetime, timedelta
from typing import NamedTuple, Protocol

import polars as pl


class StepValue(NamedTuple):
    step: int
    value: float


class TimedStepValue(NamedTuple):
    step: int
    timestamp: datetime
    value: float


class MetricStoreProtocol(Protocol):
    def query_latest(
        self,
        metric_name: str,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def query_range(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def changes(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def count_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def avg_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def min_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    def max_over_time(
        self,
        metric_name: str,
        window: timedelta,
        label_filters: dict[str, str] | None = None,
    ) -> pl.DataFrame: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class ScrapeTargetManagerProtocol(Protocol):
    def add_scrape_target(self, target_id: str, address: str) -> None: ...

    def remove_scrape_target(self, target_id: str) -> None: ...


class TrainingMetricStoreProtocol(Protocol):
    def latest(self, metric_name: str, rank: int | None = None) -> float | None: ...

    def query_last_n_steps(
        self, metric_name: str, last_n: int, rank: int | None = None,
    ) -> list[StepValue]: ...

    def query_time_window(
        self, metric_name: str, window: timedelta, rank: int | None = None,
    ) -> list[TimedStepValue]: ...
