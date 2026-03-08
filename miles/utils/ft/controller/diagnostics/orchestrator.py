from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from miles.utils.ft.controller.diagnostics.stack_trace import collect_stack_trace_suspects
from miles.utils.ft.models.diagnostic import DiagnosticPipelineResult
from miles.utils.ft.models.fault import TriggerType
from miles.utils.ft.protocols.agents import DIAGNOSTIC_TIMEOUT_SECONDS, DiagnosticExecutor, NodeAgentProtocol
from miles.utils.ft.protocols.platform import DiagnosticOrchestratorProtocol

logger = logging.getLogger(__name__)


class DiagnosticOrchestrator(DiagnosticOrchestratorProtocol):
    """Layered progressive diagnostic pipeline.

    Runs registered diagnostic executors in order on all agents (nodes)
    in parallel. Failed nodes are excluded from subsequent steps.
    """

    def __init__(
        self,
        agents: dict[str, NodeAgentProtocol],
        pipeline: list[DiagnosticExecutor] | None = None,
        default_timeout_seconds: int = DIAGNOSTIC_TIMEOUT_SECONDS,
        pipeline_timeout_seconds: int = 900,
    ) -> None:
        self._agents = agents
        self._pipeline = pipeline or []
        self._default_timeout_seconds = default_timeout_seconds
        self._pipeline_timeout_seconds = pipeline_timeout_seconds

    async def run_diagnostic_pipeline(
        self,
        trigger_reason: TriggerType,
        suspect_node_ids: list[str] | None = None,
        rank_pids_provider: Callable[[str], dict[int, int]] | None = None,
    ) -> DiagnosticPipelineResult:
        logger.info(
            "diagnostic_pipeline_start trigger=%s suspect_nodes=%s pipeline_steps=%d",
            trigger_reason,
            suspect_node_ids,
            len(self._pipeline),
        )

        try:
            return await asyncio.wait_for(
                self._run_diagnostic_pipeline_inner(
                    trigger_reason=trigger_reason,
                    suspect_node_ids=suspect_node_ids,
                    rank_pids_provider=rank_pids_provider,
                ),
                timeout=self._pipeline_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "diagnostic_pipeline_timeout timeout=%d trigger=%s",
                self._pipeline_timeout_seconds,
                trigger_reason,
            )
            return DiagnosticPipelineResult(
                bad_node_ids=[],
                reason=f"diagnostic pipeline timed out after {self._pipeline_timeout_seconds}s",
            )

    async def _run_diagnostic_pipeline_inner(
        self,
        trigger_reason: TriggerType,
        suspect_node_ids: list[str] | None = None,
        rank_pids_provider: Callable[[str], dict[int, int]] | None = None,
    ) -> DiagnosticPipelineResult:
        if trigger_reason == TriggerType.HANG and rank_pids_provider is not None:
            suspect_from_trace = await collect_stack_trace_suspects(
                agents=self._agents,
                rank_pids_provider=rank_pids_provider,
                default_timeout_seconds=self._default_timeout_seconds,
            )
            if suspect_from_trace:
                if suspect_node_ids is not None:
                    suspect_node_ids = sorted(set(suspect_node_ids) | set(suspect_from_trace))
                else:
                    suspect_node_ids = suspect_from_trace

        if not self._pipeline:
            logger.info("diagnostic_pipeline_empty — no diagnostics configured")
            return DiagnosticPipelineResult(
                bad_node_ids=[],
                reason="no diagnostics configured (empty pipeline)",
            )

        if suspect_node_ids is not None:
            suspect_set = set(suspect_node_ids)
            remaining_agents: dict[str, NodeAgentProtocol] = {
                nid: agent for nid, agent in self._agents.items() if nid in suspect_set
            }
        else:
            remaining_agents = dict(self._agents)

        for executor in self._pipeline:
            if not remaining_agents:
                break

            bad_node_ids, remaining_agents = await executor.execute(
                agents=remaining_agents,
                timeout_seconds=self._default_timeout_seconds,
            )

            if bad_node_ids:
                logger.info(
                    "diagnostic_step_found_bad bad_nodes=%s",
                    bad_node_ids,
                )
                return DiagnosticPipelineResult(
                    bad_node_ids=sorted(bad_node_ids),
                    reason=f"diagnostic failed on nodes: {bad_node_ids}",
                )

        logger.info("diagnostic_pipeline_all_passed trigger=%s", trigger_reason)
        return DiagnosticPipelineResult(
            bad_node_ids=[],
            reason="all diagnostics passed — no bad nodes found",
        )
