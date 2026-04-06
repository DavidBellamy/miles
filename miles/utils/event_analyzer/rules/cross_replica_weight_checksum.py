from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from miles.utils.event_logger.models import Event, LocalWeightChecksumEvent
from miles.utils.process_identity import TrainProcessIdentity
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ChecksumMismatchIssue(FrozenStrictBaseModel):
    key: str
    label_a: str
    label_b: str
    value_a: str
    value_b: str


def check(events: list[Event]) -> list[ChecksumMismatchIssue]:
    """
    Check: weight checksum across replicas should be exactly the same
    """

    checksum_events = [e for e in events if isinstance(e, LocalWeightChecksumEvent)]
    if not checksum_events:
        return []

    all_mismatches: list[ChecksumMismatchIssue] = []

    events_by_key: dict[tuple[int, int], list[LocalWeightChecksumEvent]] = {}
    for event in checksum_events:
        key = (event.rollout_id, event.attempt)
        events_by_key.setdefault(key, []).append(event)

    for key in sorted(events_by_key.keys()):
        all_mismatches += list(_check_one_step(events=events_by_key[key]))

    return all_mismatches


def _get_rank_key(event: LocalWeightChecksumEvent) -> int:
    if isinstance(event.source, TrainProcessIdentity):
        return event.source.rank_within_cell
    return -1


def _check_one_step(events: list[LocalWeightChecksumEvent]) -> Iterable[ChecksumMismatchIssue]:
    # Group events by rank_within_cell so we only compare across replicas (cell_index),
    # not across TP/PP/EP ranks within the same cell (which have different param shards).
    # TODO: group by (component, rank_within_cell) once critic checksum events are supported.
    #  Currently only actor emits LocalWeightChecksumEvent.
    by_rank: dict[int, list[LocalWeightChecksumEvent]] = defaultdict(list)
    for event in events:
        by_rank[_get_rank_key(event)].append(event)

    for rank_events in by_rank.values():
        first = rank_events[0]
        first_flat = _flatten_event(first)
        for other in rank_events[1:]:
            yield from _compare_flat_dicts(
                a=first_flat,
                b=_flatten_event(other),
                label_a=_compute_label(first),
                label_b=_compute_label(other),
            )


def _compute_label(event: LocalWeightChecksumEvent) -> str:
    return f"rollout_{event.rollout_id}/{event.source.to_name()}"


def _flatten_event(event: LocalWeightChecksumEvent) -> dict[str, Any]:
    """Flatten all fields of an event into a flat dict with dot-separated keys."""
    return _flatten_nested(event.state.model_dump(), prefix="")


def _compare_flat_dicts(
    a: dict[str, Any],
    b: dict[str, Any],
    label_a: str,
    label_b: str,
) -> Iterable[ChecksumMismatchIssue]:
    """Compare two flat dicts and yield mismatches."""
    all_keys = sorted(set(a.keys()) | set(b.keys()))

    for key in all_keys:
        value_a = a.get(key, "<missing>")
        value_b = b.get(key, "<missing>")
        if value_a != value_b:
            yield ChecksumMismatchIssue(
                key=key,
                label_a=label_a,
                label_b=label_b,
                value_a=str(value_a),
                value_b=str(value_b),
            )


def _flatten_nested(obj: Any, *, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict/list into a flat dict with dot-separated keys. Keeps all primitive leaf values."""
    result: dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda x: str(x[0])):
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            result.update(_flatten_nested(v, prefix=child_prefix))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            result.update(_flatten_nested(v, prefix=f"{prefix}[{i}]"))
    else:
        result[prefix] = obj

    return result
