"""Tests for event_analyzer rules/witness."""

from datetime import datetime, timezone

from pydantic import TypeAdapter

from miles.utils.event_analyzer.rules.witness import check
from miles.utils.event_logger.models import Event, WitnessSnapshotParamEvent
from miles.utils.process_identity import TrainProcessIdentity

_event_adapter = TypeAdapter(Event)

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_source(cell_index: int = 0, rank_within_cell: int = 0) -> TrainProcessIdentity:
    return TrainProcessIdentity(component="actor", cell_index=cell_index, rank_within_cell=rank_within_cell)


def _make_event(
    rollout_id: int,
    nonzero_ids: list[int],
    position: str = "head_witness",
    cell_index: int = 0,
    rank_within_cell: int = 0,
) -> WitnessSnapshotParamEvent:
    return WitnessSnapshotParamEvent(
        timestamp=_FIXED_TS,
        source=_make_source(cell_index=cell_index, rank_within_cell=rank_within_cell),
        rollout_id=rollout_id,
        position=position,
        nonzero_ids=nonzero_ids,
    )


class TestWitnessCheck:
    def test_happy_path_all_ranks_agree(self) -> None:
        events = [
            _make_event(rollout_id=0, nonzero_ids=[1, 2, 3], cell_index=0),
            _make_event(rollout_id=0, nonzero_ids=[1, 2, 3], cell_index=1),
            _make_event(rollout_id=0, nonzero_ids=[1, 2, 3], cell_index=2),
        ]
        assert check(events) == []

    def test_cross_rank_mismatch(self) -> None:
        events = [
            _make_event(rollout_id=0, nonzero_ids=[1, 2, 3], cell_index=0),
            _make_event(rollout_id=0, nonzero_ids=[1, 2, 4], cell_index=1),
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].rollout_id == 0

    def test_groups_by_rollout_id(self) -> None:
        events = [
            # Rollout 0: match
            _make_event(rollout_id=0, nonzero_ids=[1, 2], cell_index=0),
            _make_event(rollout_id=0, nonzero_ids=[1, 2], cell_index=1),
            # Rollout 1: mismatch
            _make_event(rollout_id=1, nonzero_ids=[1, 2], cell_index=0),
            _make_event(rollout_id=1, nonzero_ids=[1, 3], cell_index=1),
        ]
        mismatches = check(events)
        assert len(mismatches) == 1
        assert mismatches[0].rollout_id == 1

    def test_empty_events(self) -> None:
        assert check([]) == []

    def test_single_rank_no_comparison(self) -> None:
        events = [_make_event(rollout_id=0, nonzero_ids=[1, 2])]
        assert check(events) == []


class TestWitnessEventSerialization:
    def test_roundtrip(self) -> None:
        event = _make_event(rollout_id=5, nonzero_ids=[10, 20], position="tail_witness", cell_index=1)
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, WitnessSnapshotParamEvent)
        assert parsed.rollout_id == 5
        assert parsed.position == "tail_witness"
        assert parsed.nonzero_ids == [10, 20]
