from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Protocol

import polars as pl

from miles.utils.ft.models import MetricSample

logger = logging.getLogger(__name__)


class MetricStoreProtocol(Protocol):
    def instant_query(self, query: str) -> pl.DataFrame: ...

    def range_query(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> pl.DataFrame: ...


# ---------------------------------------------------------------------------
# PromQL subset AST
# ---------------------------------------------------------------------------


class LabelMatchOp(Enum):
    EQ = "="
    NEQ = "!="
    RE = "=~"


@dataclass
class LabelMatcher:
    label: str
    op: LabelMatchOp
    value: str


@dataclass
class MetricSelector:
    name: str
    matchers: list[LabelMatcher] = field(default_factory=list)


class CompareOp(Enum):
    EQ = "=="
    NEQ = "!="
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="


@dataclass
class CompareExpr:
    selector: MetricSelector
    op: CompareOp
    threshold: float


@dataclass
class RangeFunction:
    func_name: str  # count_over_time, changes, min_over_time, avg_over_time
    selector: MetricSelector
    duration: timedelta


@dataclass
class RangeFunctionCompare:
    func: RangeFunction
    op: CompareOp
    threshold: float


PromQLExpr = MetricSelector | CompareExpr | RangeFunction | RangeFunctionCompare


# ---------------------------------------------------------------------------
# PromQL subset parser
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"(\d+)([smhd])")
_COMPARE_OPS = ["==", "!=", ">=", "<=", ">", "<"]
_RANGE_FUNCTIONS = {
    "count_over_time",
    "changes",
    "min_over_time",
    "avg_over_time",
    "max_over_time",
}


def _parse_duration(text: str) -> timedelta:
    match = _DURATION_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Invalid duration: {text!r}")
    value = int(match.group(1))
    unit = match.group(2)
    mapping = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}
    return timedelta(**{mapping[unit]: value})


def _parse_label_matchers(text: str) -> list[LabelMatcher]:
    text = text.strip()
    if not text:
        return []

    matchers: list[LabelMatcher] = []
    for part in text.split(","):
        part = part.strip()
        for op_str in ["!=", "=~", "="]:
            if op_str in part:
                label, value = part.split(op_str, 1)
                value = value.strip().strip('"').strip("'")
                matchers.append(LabelMatcher(
                    label=label.strip(),
                    op=LabelMatchOp(op_str),
                    value=value,
                ))
                break

    return matchers


def _parse_metric_selector(text: str) -> MetricSelector:
    text = text.strip()
    if "{" in text:
        name_part, rest = text.split("{", 1)
        labels_part = rest.rstrip("}")
        return MetricSelector(
            name=name_part.strip(),
            matchers=_parse_label_matchers(labels_part),
        )
    return MetricSelector(name=text)


def _find_compare_op(text: str) -> tuple[str, CompareOp, str] | None:
    for op_str in _COMPARE_OPS:
        parts = text.split(op_str)
        if len(parts) == 2:
            return parts[0].strip(), CompareOp(op_str), parts[1].strip()
    return None


def parse_promql(query: str) -> PromQLExpr:
    query = query.strip()

    for func_name in _RANGE_FUNCTIONS:
        prefix = f"{func_name}("
        if query.startswith(prefix):
            inner_and_rest = query[len(prefix):]

            paren_depth = 1
            func_end = -1
            for i, ch in enumerate(inner_and_rest):
                if ch == "(":
                    paren_depth += 1
                elif ch == ")":
                    paren_depth -= 1
                    if paren_depth == 0:
                        func_end = i
                        break

            if func_end < 0:
                raise ValueError(f"Unmatched parenthesis in: {query!r}")

            inner = inner_and_rest[:func_end]
            after_func = inner_and_rest[func_end + 1:].strip()

            bracket_match = re.search(r"\[([^\]]+)\]", inner)
            if not bracket_match:
                raise ValueError(f"Missing range selector [duration] in: {query!r}")

            duration = _parse_duration(bracket_match.group(1))
            selector_text = inner[:bracket_match.start()]
            selector = _parse_metric_selector(selector_text)

            range_func = RangeFunction(
                func_name=func_name,
                selector=selector,
                duration=duration,
            )

            if after_func:
                compare = _find_compare_op(after_func)
                if compare:
                    _, op, threshold_str = compare
                    return RangeFunctionCompare(
                        func=range_func,
                        op=op,
                        threshold=float(threshold_str),
                    )

            return range_func

    compare = _find_compare_op(query)
    if compare:
        left, op, right = compare
        try:
            threshold = float(right)
            return CompareExpr(
                selector=_parse_metric_selector(left),
                op=op,
                threshold=threshold,
            )
        except ValueError:
            pass

    return _parse_metric_selector(query)


# ---------------------------------------------------------------------------
# Time series storage
# ---------------------------------------------------------------------------

_SeriesKey = tuple[str, frozenset[tuple[str, str]]]


@dataclass
class _TimeSeriesSample:
    timestamp: datetime
    value: float


@dataclass
class MiniPrometheusConfig:
    scrape_interval: timedelta = field(default_factory=lambda: timedelta(seconds=10))
    retention: timedelta = field(default_factory=lambda: timedelta(minutes=60))


class MiniPrometheus:
    def __init__(self, config: MiniPrometheusConfig | None = None) -> None:
        self._config = config or MiniPrometheusConfig()
        self._series: dict[_SeriesKey, deque[_TimeSeriesSample]] = {}
        self._label_maps: dict[_SeriesKey, dict[str, str]] = {}
        self._scrape_targets: dict[str, str] = {}
        self._running = False

    def add_scrape_target(self, target_id: str, address: str) -> None:
        self._scrape_targets[target_id] = address

    def remove_scrape_target(self, target_id: str) -> None:
        self._scrape_targets.pop(target_id, None)

    def ingest_samples(
        self,
        target_id: str,
        samples: list[MetricSample],
        timestamp: datetime | None = None,
    ) -> None:
        ts = timestamp or datetime.utcnow()
        for sample in samples:
            labels = dict(sample.labels)
            labels["node_id"] = target_id
            key: _SeriesKey = (sample.name, frozenset(labels.items()))
            if key not in self._series:
                self._series[key] = deque()
                self._label_maps[key] = labels
            self._series[key].append(_TimeSeriesSample(timestamp=ts, value=sample.value))

        self._evict_expired()

    async def scrape_once(self) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            for target_id, address in list(self._scrape_targets.items()):
                try:
                    response = await client.get(f"{address}/metrics")
                    response.raise_for_status()
                    samples = _parse_prometheus_text(response.text)
                    self.ingest_samples(target_id=target_id, samples=samples)
                except Exception:
                    logger.warning(
                        "Failed to scrape target %s at %s",
                        target_id,
                        address,
                        exc_info=True,
                    )

    async def start(self) -> None:
        import asyncio

        self._running = True
        while self._running:
            await self.scrape_once()
            await asyncio.sleep(self._config.scrape_interval.total_seconds())

    async def stop(self) -> None:
        self._running = False

    def instant_query(self, query: str) -> pl.DataFrame:
        expr = parse_promql(query)
        return self._evaluate_instant(expr)

    def range_query(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> pl.DataFrame:
        expr = parse_promql(query)
        return self._evaluate_range(expr, start=start, end=end, step=step)

    # -------------------------------------------------------------------
    # Internal: instant evaluation
    # -------------------------------------------------------------------

    def _evaluate_instant(self, expr: PromQLExpr) -> pl.DataFrame:
        if isinstance(expr, MetricSelector):
            return self._instant_selector(expr)

        if isinstance(expr, CompareExpr):
            df = self._instant_selector(expr.selector)
            if df.is_empty():
                return df
            return df.filter(_compare_col(pl.col("value"), expr.op, expr.threshold))

        if isinstance(expr, RangeFunction):
            return self._instant_range_function(expr)

        if isinstance(expr, RangeFunctionCompare):
            df = self._instant_range_function(expr.func)
            if df.is_empty():
                return df
            return df.filter(_compare_col(pl.col("value"), expr.op, expr.threshold))

        raise ValueError(f"Unsupported expression type: {type(expr)}")

    def _instant_selector(self, selector: MetricSelector) -> pl.DataFrame:
        rows: list[dict] = []
        for key, samples in self._series.items():
            metric_name, _ = key
            if metric_name != selector.name:
                continue
            labels = self._label_maps[key]
            if not _match_labels(labels, selector.matchers):
                continue
            if not samples:
                continue

            latest = samples[-1]
            row: dict = {"__name__": metric_name, "value": latest.value}
            row.update(labels)
            rows.append(row)

        if not rows:
            return pl.DataFrame({"__name__": [], "value": []})
        return pl.DataFrame(rows)

    def _instant_range_function(self, func: RangeFunction) -> pl.DataFrame:
        now = datetime.utcnow()
        window_start = now - func.duration
        rows: list[dict] = []

        for key, samples in self._series.items():
            metric_name, _ = key
            if metric_name != func.selector.name:
                continue
            labels = self._label_maps[key]
            if not _match_labels(labels, func.selector.matchers):
                continue

            window_samples = [
                s for s in samples if s.timestamp >= window_start
            ]
            if not window_samples:
                continue

            value = _apply_range_function(func.func_name, window_samples)
            row: dict = {"__name__": metric_name, "value": value}
            row.update(labels)
            rows.append(row)

        if not rows:
            return pl.DataFrame({"__name__": [], "value": []})
        return pl.DataFrame(rows)

    # -------------------------------------------------------------------
    # Internal: range evaluation
    # -------------------------------------------------------------------

    def _evaluate_range(
        self,
        expr: PromQLExpr,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> pl.DataFrame:
        if isinstance(expr, MetricSelector):
            return self._range_selector(expr, start=start, end=end, step=step)

        if isinstance(expr, CompareExpr):
            df = self._range_selector(expr.selector, start=start, end=end, step=step)
            if df.is_empty():
                return df
            return df.filter(_compare_col(pl.col("value"), expr.op, expr.threshold))

        raise ValueError(
            f"range_query not yet supported for expression type: {type(expr)}"
        )

    def _range_selector(
        self,
        selector: MetricSelector,
        start: datetime,
        end: datetime,
        step: timedelta,
    ) -> pl.DataFrame:
        rows: list[dict] = []
        for key, samples in self._series.items():
            metric_name, _ = key
            if metric_name != selector.name:
                continue
            labels = self._label_maps[key]
            if not _match_labels(labels, selector.matchers):
                continue

            for sample in samples:
                if start <= sample.timestamp <= end:
                    row: dict = {
                        "__name__": metric_name,
                        "timestamp": sample.timestamp,
                        "value": sample.value,
                    }
                    row.update(labels)
                    rows.append(row)

        if not rows:
            return pl.DataFrame({"__name__": [], "timestamp": [], "value": []})
        return pl.DataFrame(rows)

    # -------------------------------------------------------------------
    # Internal: eviction
    # -------------------------------------------------------------------

    def _evict_expired(self) -> None:
        cutoff = datetime.utcnow() - self._config.retention
        empty_keys: list[_SeriesKey] = []

        for key, samples in self._series.items():
            while samples and samples[0].timestamp < cutoff:
                samples.popleft()
            if not samples:
                empty_keys.append(key)

        for key in empty_keys:
            del self._series[key]
            self._label_maps.pop(key, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compare_col(col: pl.Expr, op: CompareOp, threshold: float) -> pl.Expr:
    if op == CompareOp.EQ:
        return col == threshold
    if op == CompareOp.NEQ:
        return col != threshold
    if op == CompareOp.GT:
        return col > threshold
    if op == CompareOp.LT:
        return col < threshold
    if op == CompareOp.GTE:
        return col >= threshold
    if op == CompareOp.LTE:
        return col <= threshold
    raise ValueError(f"Unknown compare op: {op}")


def _match_labels(labels: dict[str, str], matchers: list[LabelMatcher]) -> bool:
    for m in matchers:
        actual = labels.get(m.label, "")
        if m.op == LabelMatchOp.EQ:
            if actual != m.value:
                return False
        elif m.op == LabelMatchOp.NEQ:
            if actual == m.value:
                return False
        elif m.op == LabelMatchOp.RE:
            if not re.fullmatch(m.value, actual):
                return False
    return True


def _apply_range_function(
    func_name: str,
    samples: list[_TimeSeriesSample],
) -> float:
    if func_name == "count_over_time":
        return float(len(samples))

    if func_name == "changes":
        if len(samples) < 2:
            return 0.0
        changes = sum(
            1
            for i in range(1, len(samples))
            if samples[i].value != samples[i - 1].value
        )
        return float(changes)

    if func_name == "min_over_time":
        return min(s.value for s in samples)

    if func_name == "max_over_time":
        return max(s.value for s in samples)

    if func_name == "avg_over_time":
        return sum(s.value for s in samples) / len(samples)

    raise ValueError(f"Unknown range function: {func_name}")


def _parse_prometheus_text(text: str) -> list[MetricSample]:
    samples: list[MetricSample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        label_match = re.match(r"^(\w+)(\{([^}]*)\})?\s+(.+?)(\s+\d+)?$", line)
        if not label_match:
            continue

        name = label_match.group(1)
        labels_str = label_match.group(3) or ""
        value_str = label_match.group(4)

        labels: dict[str, str] = {}
        if labels_str:
            for pair in labels_str.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    labels[k.strip()] = v.strip().strip('"')

        try:
            value = float(value_str)
        except ValueError:
            continue

        samples.append(MetricSample(name=name, labels=labels, value=value))

    return samples
