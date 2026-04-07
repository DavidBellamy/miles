"""Comprehensive parameterized tests for RecordLogger.

Tests cover: single-session writes, concurrent multi-session writes,
ordering guarantees, close-then-reopen, drain-on-close, exception
resilience, no-op close of non-existent sessions, and mixed concurrent
lifecycle operations.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import pytest

from miles.rollout.session.record_logger import RecordLogger
from miles.rollout.session.session_types import SessionRecord


# ---------------------------------------------------------------------------
# Helpers (module-level, self-contained)
# ---------------------------------------------------------------------------


def _make_record(index: int, session_tag: str = "default") -> SessionRecord:
    """Deterministic factory for SessionRecord."""
    return SessionRecord(
        timestamp=1000.0 + index,
        method="POST",
        path=f"/api/{session_tag}/{index}",
        request={"index": index, "session_tag": session_tag},
        response={"ok": True, "index": index},
        status_code=200,
    )


def _read_jsonl(path: Path) -> list[dict]:
    """Read a .jsonl file and return a list of parsed dicts."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleSessionWriteReadback:
    @pytest.mark.parametrize(
        "record_count",
        [
            pytest.param(1, id="1-record"),
            pytest.param(10, id="10-records"),
            pytest.param(100, id="100-records"),
            pytest.param(500, id="500-records"),
        ],
    )
    def test_single_session_write_readback(self, tmp_path: Path, record_count: int):
        log_dir = str(tmp_path / "logs")
        rl = RecordLogger(log_dir)
        try:
            for i in range(record_count):
                rl.log_record("sess-0", _make_record(i, session_tag="sess-0"))
        finally:
            rl.close_all()

        jsonl_files = list(Path(log_dir).glob("*.jsonl"))
        assert len(jsonl_files) == 1, f"Expected 1 .jsonl file, got {len(jsonl_files)}"

        records = _read_jsonl(jsonl_files[0])
        assert len(records) == record_count

        for i, rec in enumerate(records):
            assert rec["timestamp"] == 1000.0 + i
            assert rec["path"] == f"/api/sess-0/{i}"
            assert rec["request"]["index"] == i
            assert rec["status_code"] == 200

        # Timestamps strictly increasing
        timestamps = [r["timestamp"] for r in records]
        assert timestamps == sorted(timestamps)
        assert len(set(timestamps)) == len(timestamps)


class TestConcurrentMultiSession:
    @pytest.mark.parametrize(
        "num_sessions, records_per_session",
        [
            pytest.param(2, 5, id="2-sessions-5-records"),
            pytest.param(5, 20, id="5-sessions-20-records"),
            pytest.param(10, 50, id="10-sessions-50-records"),
            pytest.param(20, 100, id="20-sessions-100-records"),
        ],
    )
    def test_concurrent_multi_session(self, tmp_path: Path, num_sessions: int, records_per_session: int):
        log_dir = str(tmp_path / "logs")
        rl = RecordLogger(log_dir)
        try:

            def _worker(session_idx: int) -> None:
                tag = f"session-{session_idx}"
                for i in range(records_per_session):
                    rl.log_record(tag, _make_record(i, session_tag=tag))

            max_workers = min(num_sessions, 8)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_worker, idx) for idx in range(num_sessions)]
                for fut in as_completed(futures):
                    fut.result()  # raise if worker failed
        finally:
            rl.close_all()

        jsonl_files = list(Path(log_dir).glob("*.jsonl"))
        assert len(jsonl_files) == num_sessions

        for session_idx in range(num_sessions):
            tag = f"session-{session_idx}"
            fpath = Path(log_dir) / f"{tag}.jsonl"
            assert fpath.exists(), f"Missing file for {tag}"

            records = _read_jsonl(fpath)
            assert len(records) == records_per_session

            # No cross-session contamination
            for rec in records:
                assert rec["request"]["session_tag"] == tag


class TestOrderingGuarantee:
    @pytest.mark.parametrize(
        "record_count",
        [
            pytest.param(50, id="50-records"),
            pytest.param(200, id="200-records"),
        ],
    )
    def test_ordering_guarantee(self, tmp_path: Path, record_count: int):
        log_dir = str(tmp_path / "logs")
        rl = RecordLogger(log_dir)
        try:
            for i in range(record_count):
                rl.log_record("ordered", _make_record(i))
        finally:
            rl.close_all()

        records = _read_jsonl(Path(log_dir) / "ordered.jsonl")
        assert len(records) == record_count

        timestamps = [r["timestamp"] for r in records]
        expected = [1000.0 + i for i in range(record_count)]
        assert timestamps == expected


class TestCloseSessionThenReopen:
    @pytest.mark.parametrize(
        "records_before, records_after",
        [
            pytest.param(3, 2, id="3-before-2-after"),
            pytest.param(10, 10, id="10-before-10-after"),
            pytest.param(1, 1, id="1-before-1-after"),
        ],
    )
    def test_close_session_then_reopen(self, tmp_path: Path, records_before: int, records_after: int):
        log_dir = str(tmp_path / "logs")
        rl = RecordLogger(log_dir)
        try:
            for i in range(records_before):
                rl.log_record("reopen", _make_record(i))

            rl.close_session("reopen")

            for i in range(records_before, records_before + records_after):
                rl.log_record("reopen", _make_record(i))
        finally:
            rl.close_all()

        records = _read_jsonl(Path(log_dir) / "reopen.jsonl")
        total = records_before + records_after
        assert len(records) == total

        timestamps = [r["timestamp"] for r in records]
        expected = [1000.0 + i for i in range(total)]
        assert timestamps == expected


class TestCloseAllDrainsPending:
    @pytest.mark.parametrize(
        "flood_count",
        [
            pytest.param(100, id="100-flood"),
            pytest.param(500, id="500-flood"),
            pytest.param(1000, id="1000-flood"),
        ],
    )
    def test_close_all_drains_pending(self, tmp_path: Path, flood_count: int):
        log_dir = str(tmp_path / "logs")
        rl = RecordLogger(log_dir)
        try:
            for i in range(flood_count):
                rl.log_record("flood", _make_record(i))
        finally:
            rl.close_all()

        records = _read_jsonl(Path(log_dir) / "flood.jsonl")
        assert len(records) == flood_count

        timestamps = [r["timestamp"] for r in records]
        expected = [1000.0 + i for i in range(flood_count)]
        assert timestamps == expected


class TestExceptionResilience:
    @pytest.mark.parametrize(
        "num_healthy, records_per",
        [
            pytest.param(1, 5, id="1-healthy-5-records"),
            pytest.param(3, 10, id="3-healthy-10-records"),
        ],
    )
    def test_exception_resilience(self, tmp_path: Path, num_healthy: int, records_per: int):
        log_dir = str(tmp_path / "logs")

        _real_open = open

        def _patched_open(path, *args, **kwargs):
            if "bad-sess" in str(path):
                raise OSError("Simulated disk failure")
            return _real_open(path, *args, **kwargs)

        rl = RecordLogger(log_dir)
        try:
            with patch("builtins.open", side_effect=_patched_open):
                # Write to bad session
                for i in range(records_per):
                    rl.log_record("bad-sess", _make_record(i, session_tag="bad-sess"))

                # Write to healthy sessions
                for h in range(num_healthy):
                    tag = f"healthy-{h}"
                    for i in range(records_per):
                        rl.log_record(tag, _make_record(i, session_tag=tag))

                # close_all() must happen inside the patch so that the
                # background writer thread processes every queued item while
                # builtins.open is still intercepted.
                rl.close_all()
        except Exception:
            rl.close_all()
            raise

        # Bad session file should not exist
        assert not (Path(log_dir) / "bad-sess.jsonl").exists()

        # Healthy files should all be correct
        for h in range(num_healthy):
            tag = f"healthy-{h}"
            fpath = Path(log_dir) / f"{tag}.jsonl"
            assert fpath.exists(), f"Missing file for {tag}"
            records = _read_jsonl(fpath)
            assert len(records) == records_per
            for rec in records:
                assert rec["request"]["session_tag"] == tag


class TestCloseNonexistentSessionNoop:
    def test_close_nonexistent_session_noop(self, tmp_path: Path):
        log_dir = str(tmp_path / "logs")
        rl = RecordLogger(log_dir)
        try:
            rl.log_record("real", _make_record(0, session_tag="real"))
            rl.log_record("real", _make_record(1, session_tag="real"))

            # Closing a session that was never opened should be a no-op
            rl.close_session("never-opened")

            rl.log_record("real", _make_record(2, session_tag="real"))
        finally:
            rl.close_all()

        assert not (Path(log_dir) / "never-opened.jsonl").exists()

        records = _read_jsonl(Path(log_dir) / "real.jsonl")
        assert len(records) == 3
        timestamps = [r["timestamp"] for r in records]
        assert timestamps == [1000.0, 1001.0, 1002.0]


class TestMixedConcurrentLifecycle:
    @pytest.mark.parametrize(
        "num_sessions, records_per_session",
        [
            pytest.param(4, 10, id="4-sessions-10-records"),
            pytest.param(8, 25, id="8-sessions-25-records"),
        ],
    )
    def test_mixed_concurrent_lifecycle(self, tmp_path: Path, num_sessions: int, records_per_session: int):
        log_dir = str(tmp_path / "logs")
        rl = RecordLogger(log_dir)
        half = records_per_session // 2
        try:

            def _worker(session_idx: int) -> None:
                tag = f"mixed-{session_idx}"
                # Write first half
                for i in range(half):
                    rl.log_record(tag, _make_record(i, session_tag=tag))

                # Close session mid-way
                rl.close_session(tag)

                # Write second half (re-opens the file)
                for i in range(half, records_per_session):
                    rl.log_record(tag, _make_record(i, session_tag=tag))

            max_workers = min(num_sessions, 8)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_worker, idx) for idx in range(num_sessions)]
                for fut in as_completed(futures):
                    fut.result()
        finally:
            rl.close_all()

        for session_idx in range(num_sessions):
            tag = f"mixed-{session_idx}"
            fpath = Path(log_dir) / f"{tag}.jsonl"
            assert fpath.exists(), f"Missing file for {tag}"

            records = _read_jsonl(fpath)
            assert len(records) == records_per_session

            # First half should come before second half
            first_half_timestamps = [r["timestamp"] for r in records[:half]]
            second_half_timestamps = [r["timestamp"] for r in records[half:]]

            # Each half should be in order
            assert first_half_timestamps == sorted(first_half_timestamps)
            assert second_half_timestamps == sorted(second_half_timestamps)

            # All first-half timestamps should be < all second-half timestamps
            if first_half_timestamps and second_half_timestamps:
                assert max(first_half_timestamps) < min(second_half_timestamps)

            # No cross-session contamination
            for rec in records:
                assert rec["request"]["session_tag"] == tag
