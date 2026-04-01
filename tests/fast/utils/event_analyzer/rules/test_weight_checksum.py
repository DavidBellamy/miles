"""Tests for event_analyzer rules/weight_checksum."""

from pathlib import Path

from miles.utils.event_analyzer.rules.weight_checksum import _flatten_nested, check
from miles.utils.event_logger.logger import EventLogger, read_events
from miles.utils.event_logger.models import LocalWeightChecksumEvent, OptimizerStateInfo
from miles.utils.process_identity import MainProcessIdentity


def _make_event(
    step: int,
    rank: int,
    param_hashes: dict[str, str] | None = None,
    buffer_hashes: dict[str, str] | None = None,
    optimizer_state_dict: dict | None = None,
) -> LocalWeightChecksumEvent:
    return LocalWeightChecksumEvent(
        step=step,
        rank=rank,
        param_hashes=param_hashes or {},
        buffer_hashes=buffer_hashes or {},
        optimizer_hashes=[
            OptimizerStateInfo(
                param_names={0: "pp0.weight"},
                state_dict=optimizer_state_dict or {},
            ),
        ] if optimizer_state_dict is not None else [],
    )


def _read_checksum_events(tmp_path: Path) -> list:
    return read_events(tmp_path)


class TestCheckWeightChecksums:
    def test_matching_checksums_across_replicas_passes(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        event_logger.log(_make_event(step=0, rank=0, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"}))
        event_logger.log(_make_event(step=0, rank=1, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"}))
        event_logger.log(_make_event(step=0, rank=2, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"}))
        event_logger.close()

        mismatches = check(_read_checksum_events(tmp_path))
        assert mismatches == []

    def test_param_hash_mismatch_reports_correct_details(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        event_logger.log(_make_event(step=5, rank=0, param_hashes={"pp0.weight": "aaa"}))
        event_logger.log(_make_event(step=5, rank=1, param_hashes={"pp0.weight": "zzz"}))
        event_logger.log(_make_event(step=5, rank=2, param_hashes={"pp0.weight": "aaa"}))
        event_logger.close()

        mismatches = check(_read_checksum_events(tmp_path))

        assert len(mismatches) == 1
        m = mismatches[0]
        assert m.step == 5
        assert m.category == "param"
        assert m.key == "pp0.weight"
        assert 0 in m.cell_indices
        assert 1 in m.cell_indices

    def test_missing_tensor_in_one_replica_reports_mismatch(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        event_logger.log(_make_event(step=0, rank=0, param_hashes={"pp0.weight": "aaa", "pp0.bias": "bbb"}))
        event_logger.log(_make_event(step=0, rank=1, param_hashes={"pp0.weight": "aaa"}))
        event_logger.close()

        mismatches = check(_read_checksum_events(tmp_path))

        assert len(mismatches) == 1
        assert mismatches[0].key == "pp0.bias"
        assert "<missing>" in mismatches[0].values

    def test_multiple_steps_only_reports_mismatched_step(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())

        # Step 0: all match
        event_logger.log(_make_event(step=0, rank=0, param_hashes={"pp0.weight": "aaa"}))
        event_logger.log(_make_event(step=0, rank=1, param_hashes={"pp0.weight": "aaa"}))

        # Step 1: mismatch
        event_logger.log(_make_event(step=1, rank=0, param_hashes={"pp0.weight": "aaa"}))
        event_logger.log(_make_event(step=1, rank=1, param_hashes={"pp0.weight": "zzz"}))

        # Step 2: all match
        event_logger.log(_make_event(step=2, rank=0, param_hashes={"pp0.weight": "aaa"}))
        event_logger.log(_make_event(step=2, rank=1, param_hashes={"pp0.weight": "aaa"}))
        event_logger.close()

        mismatches = check(_read_checksum_events(tmp_path))

        assert len(mismatches) == 1
        assert mismatches[0].step == 1

    def test_empty_events_returns_no_mismatches(self) -> None:
        mismatches = check([])
        assert mismatches == []

    def test_buffer_mismatch_detected(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        event_logger.log(_make_event(step=0, rank=0, buffer_hashes={"pp0.running_mean": "aaa"}))
        event_logger.log(_make_event(step=0, rank=1, buffer_hashes={"pp0.running_mean": "bbb"}))
        event_logger.close()

        mismatches = check(_read_checksum_events(tmp_path))

        assert len(mismatches) == 1
        assert mismatches[0].category == "buffer"

    def test_optimizer_state_mismatch_detected(self, tmp_path: Path) -> None:
        event_logger = EventLogger(log_dir=tmp_path, source=MainProcessIdentity())
        event_logger.log(_make_event(
            step=3, rank=0,
            optimizer_state_dict={"state": {0: {"exp_avg": "aaa"}}},
        ))
        event_logger.log(_make_event(
            step=3, rank=1,
            optimizer_state_dict={"state": {0: {"exp_avg": "bbb"}}},
        ))
        event_logger.close()

        mismatches = check(_read_checksum_events(tmp_path))

        assert len(mismatches) == 1
        assert mismatches[0].category == "optimizer"
        assert "exp_avg" in mismatches[0].key


class TestFlattenNested:
    def test_flat_dict_with_string_values(self) -> None:
        result = _flatten_nested({"a": "hash1", "b": "hash2"}, prefix="root")
        assert result == {"root.a": "hash1", "root.b": "hash2"}

    def test_nested_dict(self) -> None:
        result = _flatten_nested({"state": {0: {"exp_avg": "h1"}}}, prefix="opt0")
        assert result == {"opt0.state.0.exp_avg": "h1"}

    def test_list_values(self) -> None:
        result = _flatten_nested({"params": ["a", "b"]}, prefix="opt0")
        assert result == {"opt0.params[0]": "a", "opt0.params[1]": "b"}

    def test_ignores_non_string_leaves(self) -> None:
        result = _flatten_nested({"lr": 0.001, "hash": "abc"}, prefix="root")
        assert result == {"root.hash": "abc"}

    def test_empty_dict(self) -> None:
        result = _flatten_nested({}, prefix="root")
        assert result == {}
