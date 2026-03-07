from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol, runtime_checkable

from miles.utils.ft.protocols.agents import NodeAgentProtocol


@runtime_checkable
class DiagnosticAgentFactoryProtocol(Protocol):
    """Protocol for factories that create temporary diagnostic agents."""

    @asynccontextmanager
    async def create_agents(
        self, node_ids: list[str],
    ) -> AsyncIterator[dict[str, NodeAgentProtocol]]:
        ...
        yield {}  # pragma: no cover
