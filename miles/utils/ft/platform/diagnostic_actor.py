"""Temporary Ray actor for running diagnostics on a target node.

The DiagnosticActor is spawned by ``RayDiagnosticAgentFactory`` when the
DiagnosticOrchestrator needs to run a diagnostic pipeline. It is pinned
to a specific node via ``NodeAffinitySchedulingStrategy`` and killed
when the pipeline completes.

This replaces the old pattern of embedding diagnostics inside FtNodeAgent.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.ft.agents.diagnostics.gpu_diagnostic import GpuDiagnostic
from miles.utils.ft.agents.diagnostics.nccl.inter_machine import InterMachineCommDiagnostic
from miles.utils.ft.agents.diagnostics.nccl.intra_machine import IntraMachineCommDiagnostic
from miles.utils.ft.agents.diagnostics.stack_trace import StackTraceDiagnostic
from miles.utils.ft.models.diagnostics import DiagnosticResult, UnknownDiagnosticError
from miles.utils.ft.platform.ray_node_agent_proxy import RayNodeAgentProxy
from miles.utils.ft.protocols.agents import DiagnosticProtocol, NodeAgentProtocol
from miles.utils.ft.protocols.diagnostics import DiagnosticAgentFactoryProtocol

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=0)
class _DiagnosticActorCls:
    """Lightweight Ray actor pinned to a node for running diagnostics.

    Created on demand by ``RayDiagnosticAgentFactory``, killed after use.
    """

    def __init__(self, node_id: str, num_gpus: int = 8) -> None:
        self._node_id = node_id
        self._num_gpus = num_gpus
        self._diagnostics: dict[str, DiagnosticProtocol] = {
            "gpu": GpuDiagnostic(),
            "intra_machine": IntraMachineCommDiagnostic(num_gpus=num_gpus),
            "inter_machine": InterMachineCommDiagnostic(num_gpus=num_gpus),
        }

    async def run_diagnostic(
        self,
        diagnostic_type: str,
        timeout_seconds: int = 120,
        **kwargs: object,
    ) -> DiagnosticResult:
        if diagnostic_type == "stack_trace":
            return await self._run_stack_trace(
                timeout_seconds=timeout_seconds, **kwargs,
            )

        diagnostic = self._diagnostics.get(diagnostic_type)
        if diagnostic is None:
            raise UnknownDiagnosticError(
                f"node {self._node_id}: unknown diagnostic type '{diagnostic_type}', "
                f"registered types: {sorted(self._diagnostics.keys())}"
            )

        try:
            return await asyncio.wait_for(
                diagnostic.run(
                    node_id=self._node_id,
                    timeout_seconds=timeout_seconds,
                    **kwargs,
                ),
                timeout=timeout_seconds + 5,
            )
        except asyncio.TimeoutError:
            return DiagnosticResult.fail_result(
                diagnostic_type=diagnostic_type,
                node_id=self._node_id,
                details=f"diagnostic timed out after {timeout_seconds}s",
            )
        except Exception:
            logger.warning(
                "diagnostic_error type=%s node=%s",
                diagnostic_type, self._node_id,
                exc_info=True,
            )
            return DiagnosticResult.fail_result(
                diagnostic_type=diagnostic_type,
                node_id=self._node_id,
                details="diagnostic raised exception",
            )

    async def _run_stack_trace(
        self,
        timeout_seconds: int = 30,
        **kwargs: object,
    ) -> DiagnosticResult:
        raw_pids = kwargs.get("pids", [])
        pids = [int(p) for p in raw_pids] if raw_pids else []  # type: ignore[union-attr]
        diag = StackTraceDiagnostic(pids=pids)
        return await diag.run(
            node_id=self._node_id,
            timeout_seconds=timeout_seconds,
        )


class RayDiagnosticAgentFactory(DiagnosticAgentFactoryProtocol):
    """Factory that spawns temporary DiagnosticActor Ray actors on target nodes.

    Usage::

        factory = RayDiagnosticAgentFactory(node_ray_ids={"host-0": "ray-node-id-abc"})
        async with factory.create_agents(["host-0", "host-1"]) as agents:
            result = await agents["host-0"].run_diagnostic("gpu")
    """

    def __init__(
        self,
        node_ray_ids: dict[str, str],
        num_gpus: int = 8,
    ) -> None:
        self._node_ray_ids = node_ray_ids
        self._num_gpus = num_gpus

    @contextlib.asynccontextmanager
    async def create_agents(
        self, node_ids: list[str],
    ) -> AsyncIterator[dict[str, NodeAgentProtocol]]:
        actors: dict[str, ray.actor.ActorHandle] = {}
        agents: dict[str, NodeAgentProtocol] = {}

        for node_id in node_ids:
            ray_node_id = self._node_ray_ids.get(node_id)
            if ray_node_id is None:
                logger.warning(
                    "diagnostic_factory_skip node=%s reason=no_ray_node_id",
                    node_id,
                )
                continue

            actor = _DiagnosticActorCls.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray_node_id, soft=False,
                ),
            ).remote(node_id=node_id, num_gpus=self._num_gpus)
            actors[node_id] = actor
            agents[node_id] = RayNodeAgentProxy(handle=actor)

        try:
            yield agents
        finally:
            for node_id, actor in actors.items():
                try:
                    ray.kill(actor)
                except Exception:
                    logger.warning(
                        "diagnostic_actor_kill_failed node=%s",
                        node_id,
                        exc_info=True,
                    )


class InProcessDiagnosticAgentFactory(DiagnosticAgentFactoryProtocol):
    """In-process factory for testing without Ray.

    Wraps pre-built agents and filters to the requested node_ids.
    """

    def __init__(self, agents: dict[str, NodeAgentProtocol]) -> None:
        self._agents = agents

    @contextlib.asynccontextmanager
    async def create_agents(
        self, node_ids: list[str],
    ) -> AsyncIterator[dict[str, NodeAgentProtocol]]:
        yield {nid: self._agents[nid] for nid in node_ids if nid in self._agents}
