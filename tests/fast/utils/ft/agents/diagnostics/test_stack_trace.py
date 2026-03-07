"""Tests for StackTraceDiagnostic."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from miles.utils.ft.agents.diagnostics.stack_trace import StackTraceDiagnostic
from tests.fast.utils.ft.helpers import make_mock_subprocess


class TestStackTraceDiagnosticEmptyPids:
    async def test_empty_pids_returns_failed(self) -> None:
        diag = StackTraceDiagnostic(pids=[])
        result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "no PIDs provided" in result.details

    async def test_none_pids_returns_failed(self) -> None:
        diag = StackTraceDiagnostic(pids=None)
        result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "no PIDs provided" in result.details


class TestStackTraceDiagnosticSinglePid:
    async def test_single_pid_success(self) -> None:
        mock_proc = make_mock_subprocess(stdout=b"stack trace here")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            diag = StackTraceDiagnostic(pids=[1234])
            result = await diag.run(node_id="node-0")

        assert result.passed is True
        assert "PID 1234" in result.details
        assert "stack trace here" in result.details

    async def test_single_pid_pyspy_failure(self) -> None:
        mock_proc = make_mock_subprocess(
            stdout=b"",
            stderr=b"process not found",
            returncode=1,
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            diag = StackTraceDiagnostic(pids=[1234])
            result = await diag.run(node_id="node-0")

        assert result.passed is False
        assert "FAILED" in result.details


class TestStackTraceDiagnosticMultiplePids:
    async def test_partial_failure_still_passes(self) -> None:
        good_proc = make_mock_subprocess(stdout=b"good trace")
        bad_proc = make_mock_subprocess(
            stdout=b"",
            stderr=b"error",
            returncode=1,
        )

        call_count = 0

        async def mock_subprocess(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return good_proc if call_count == 1 else bad_proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_subprocess):
            diag = StackTraceDiagnostic(pids=[100, 200])
            result = await diag.run(node_id="node-0")

        assert result.passed is True
        assert "PID 100" in result.details
        assert "PID 200" in result.details
        assert "FAILED" in result.details

    async def test_all_pids_fail_returns_not_passed(self) -> None:
        bad_proc = make_mock_subprocess(
            stdout=b"",
            stderr=b"error",
            returncode=1,
        )

        with patch("asyncio.create_subprocess_exec", return_value=bad_proc):
            diag = StackTraceDiagnostic(pids=[100, 200, 300])
            result = await diag.run(node_id="node-0")

        assert result.passed is False

    async def test_timeout_treated_as_failure(self) -> None:
        mock_proc = make_mock_subprocess()
        mock_proc.communicate.side_effect = asyncio.TimeoutError()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            diag = StackTraceDiagnostic(pids=[1234])
            result = await diag.run(node_id="node-0", timeout_seconds=10)

        assert result.passed is False
        assert "FAILED" in result.details
