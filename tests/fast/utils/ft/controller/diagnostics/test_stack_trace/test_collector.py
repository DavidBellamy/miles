"""Tests for miles.utils.ft.controller.diagnostics.stack_trace.collector."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

from tests.fast.utils.ft.conftest import FakeNodeAgent

from miles.utils.ft.agents.diagnostics.executors.stack_trace import PySpyThread
from miles.utils.ft.agents.types import DiagnosticResult
from miles.utils.ft.controller.diagnostics.stack_trace.collector import collect_stack_trace_suspects


def _make_agent(node_id: str) -> FakeNodeAgent:
    return FakeNodeAgent(node_id=node_id)


def _normal_threads() -> list[dict[str, object]]:
    return [
        PySpyThread(
            id=1,
            name="MainThread",
            active=True,
            owns_gil=False,
            frames=[{"name": "train", "filename": "train.py", "line": 10}],
        ).model_dump()
    ]


class TestCollectStackTraceSuspects:
    def test_empty_agents_returns_empty(self) -> None:
        result = asyncio.run(
            collect_stack_trace_suspects(
                agents={},
                rank_pids_provider=lambda nid: {},
                default_timeout_seconds=30,
            )
        )

        assert result == []

    def test_rank_pids_provider_exception_marks_node_as_suspect(self) -> None:
        agents = {"node-0": _make_agent("node-0")}

        def _failing_provider(node_id: str) -> dict[int, int]:
            raise RuntimeError("cannot get pids")

        result = asyncio.run(
            collect_stack_trace_suspects(
                agents=agents,
                rank_pids_provider=_failing_provider,
                default_timeout_seconds=30,
            )
        )

        assert "node-0" in result

    def test_diagnostic_execution_failure_marks_node_as_suspect(self) -> None:
        agents = {"node-0": _make_agent("node-0")}

        with patch(
            "miles.utils.ft.controller.diagnostics.stack_trace.collector.StackTraceNodeExecutor"
        ) as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(
                return_value=DiagnosticResult(
                    diagnostic_type="stack_trace",
                    node_id="node-0",
                    passed=False,
                    details="py-spy failed",
                )
            )
            mock_cls.return_value = mock_instance

            result = asyncio.run(
                collect_stack_trace_suspects(
                    agents=agents,
                    rank_pids_provider=lambda nid: {0: 1234},
                    default_timeout_seconds=30,
                )
            )

        assert "node-0" in result

    def test_successful_collection_returns_aggregation_suspects(self) -> None:
        agents = {
            "node-0": _make_agent("node-0"),
            "node-1": _make_agent("node-1"),
            "node-2": _make_agent("node-2"),
        }

        threads_json = json.dumps(_normal_threads())
        pass_result = DiagnosticResult(
            diagnostic_type="stack_trace",
            node_id="placeholder",
            passed=True,
            details=threads_json,
        )

        with patch(
            "miles.utils.ft.controller.diagnostics.stack_trace.collector.StackTraceNodeExecutor"
        ) as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.run = AsyncMock(return_value=pass_result)
            mock_cls.return_value = mock_instance

            result = asyncio.run(
                collect_stack_trace_suspects(
                    agents=agents,
                    rank_pids_provider=lambda nid: {0: 1234},
                    default_timeout_seconds=30,
                )
            )

        assert isinstance(result, list)
