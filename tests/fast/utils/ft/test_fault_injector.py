from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.ft.e2e.fault_injector import (
    FaultInjectorActor,
    _TRAINING_CMDLINE_PATTERNS,
)


class TestFindTrainingProcesses:
    def _make_proc_info(
        self, pid: int, cmdline: list[str], name: str = "python",
    ) -> MagicMock:
        proc = MagicMock()
        proc.info = {"pid": pid, "cmdline": cmdline, "name": name}
        return proc

    def test_matches_megatron_process(self) -> None:
        procs = [
            self._make_proc_info(100, ["python", "-m", "megatron.training"]),
            self._make_proc_info(200, ["bash", "-c", "echo hello"]),
        ]
        with patch("psutil.process_iter", return_value=procs):
            actor = FaultInjectorActor.__new__(FaultInjectorActor)
            actor.__init__()
            results = actor.find_training_processes()

        assert len(results) == 1
        assert results[0]["pid"] == 100

    def test_matches_torchrun(self) -> None:
        procs = [
            self._make_proc_info(300, ["torchrun", "--nproc_per_node=8", "train.py"]),
        ]
        with patch("psutil.process_iter", return_value=procs):
            actor = FaultInjectorActor.__new__(FaultInjectorActor)
            actor.__init__()
            results = actor.find_training_processes()

        assert len(results) == 1
        assert results[0]["pid"] == 300

    def test_matches_run_deepseek(self) -> None:
        procs = [
            self._make_proc_info(400, ["python", "miles/run_deepseek_v3.py"]),
        ]
        with patch("psutil.process_iter", return_value=procs):
            actor = FaultInjectorActor.__new__(FaultInjectorActor)
            actor.__init__()
            results = actor.find_training_processes()

        assert len(results) == 1

    def test_no_match_returns_empty(self) -> None:
        procs = [
            self._make_proc_info(500, ["vim", "somefile.txt"]),
        ]
        with patch("psutil.process_iter", return_value=procs):
            actor = FaultInjectorActor.__new__(FaultInjectorActor)
            actor.__init__()
            results = actor.find_training_processes()

        assert results == []

    def test_handles_access_denied(self) -> None:
        import psutil as psutil_mod

        proc = MagicMock()
        proc.info.__getitem__ = MagicMock(side_effect=psutil_mod.AccessDenied(pid=1))
        proc.info.get = MagicMock(side_effect=psutil_mod.AccessDenied(pid=1))

        with patch("psutil.process_iter", return_value=[proc]):
            actor = FaultInjectorActor.__new__(FaultInjectorActor)
            actor.__init__()
            results = actor.find_training_processes()

        assert results == []


class TestDiskOperations:
    def test_fill_and_cleanup_disk(self, tmp_path: Path) -> None:
        actor = FaultInjectorActor.__new__(FaultInjectorActor)
        actor.__init__()

        fill_path = str(tmp_path / "fill_file")
        size_bytes = 1024 * 1024  # 1 MB
        actor.fill_disk(path=fill_path, size_bytes=size_bytes)

        assert os.path.exists(fill_path)
        assert os.path.getsize(fill_path) == size_bytes
        assert fill_path in actor._filled_paths

        actor.cleanup_disk(path=fill_path)
        assert not os.path.exists(fill_path)
        assert fill_path not in actor._filled_paths

    def test_cleanup_disk_nonexistent_file(self, tmp_path: Path) -> None:
        actor = FaultInjectorActor.__new__(FaultInjectorActor)
        actor.__init__()

        actor.cleanup_disk(path=str(tmp_path / "nonexistent"))

    def test_cleanup_all(self, tmp_path: Path) -> None:
        actor = FaultInjectorActor.__new__(FaultInjectorActor)
        actor.__init__()

        path1 = str(tmp_path / "file1")
        path2 = str(tmp_path / "file2")
        actor.fill_disk(path=path1, size_bytes=1024)
        actor.fill_disk(path=path2, size_bytes=1024)

        assert os.path.exists(path1)
        assert os.path.exists(path2)

        actor.cleanup_all()

        assert not os.path.exists(path1)
        assert not os.path.exists(path2)
        assert actor._filled_paths == []
        assert actor._stress_pids == []


class TestTrainingCmdlinePatterns:
    def test_all_patterns_are_lowercase(self) -> None:
        for pattern in _TRAINING_CMDLINE_PATTERNS:
            assert pattern == pattern.lower()
