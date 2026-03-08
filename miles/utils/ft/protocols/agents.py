from __future__ import annotations

from typing import Protocol, runtime_checkable

from miles.utils.ft.models.diagnostics import DiagnosticResult

DIAGNOSTIC_TIMEOUT_SECONDS: int = 120


@runtime_checkable
class NodeAgentProtocol(Protocol):
    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
        **kwargs: object,
    ) -> DiagnosticResult: ...


@runtime_checkable
class DiagnosticProtocol(Protocol):
    diagnostic_type: str

    async def run(
        self,
        node_id: str,
        timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
    ) -> DiagnosticResult: ...


@runtime_checkable
class DiagnosticExecutor(Protocol):
    """Strategy for executing one diagnostic step within the pipeline.

    Returns (bad_node_ids, remaining_agents_without_bad_nodes).
    """

    async def execute(
        self,
        agents: dict[str, NodeAgentProtocol],
        timeout_seconds: int,
    ) -> tuple[list[str], dict[str, NodeAgentProtocol]]: ...
