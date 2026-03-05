"""Unit tests for ControllerHandleMixin.

Uses a minimal concrete subclass so the mixin is tested in isolation
rather than indirectly through each agent class.
"""

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from miles.utils.ft.agents.utils.controller_handle import (
    ActorResolverProtocol,
    ControllerHandleMixin,
)


class _FakeResolver:
    """Test resolver that records calls and returns a configurable handle."""

    def __init__(self, handle: Any = None, *, should_fail: bool = False) -> None:
        self.handle = handle
        self.should_fail = should_fail
        self.calls: list[str] = []

    def get_actor(self, name: str) -> Any:
        self.calls.append(name)
        if self.should_fail:
            raise RuntimeError("resolver failure")
        return self.handle


class _StubAgent(ControllerHandleMixin):
    def __init__(
        self,
        ft_id: str = "",
        actor_resolver: ActorResolverProtocol | None = None,
    ) -> None:
        super().__init__(ft_id=ft_id, actor_resolver=actor_resolver)


class TestControllerHandleMixin:
    def test_caches_result(self) -> None:
        resolver = _FakeResolver()
        agent = _StubAgent(actor_resolver=resolver)
        mock_handle = MagicMock()
        agent._controller_handle = mock_handle

        result = agent._get_controller_handle()

        assert result is mock_handle
        assert len(resolver.calls) == 0

    def test_negative_cache_within_cooldown(self) -> None:
        resolver = _FakeResolver()
        agent = _StubAgent(actor_resolver=resolver)
        agent._last_lookup_failure_time = time.monotonic()

        result = agent._get_controller_handle()

        assert result is None
        assert len(resolver.calls) == 0

    def test_retries_after_cooldown(self) -> None:
        mock_handle = MagicMock()
        resolver = _FakeResolver(handle=mock_handle)
        agent = _StubAgent(actor_resolver=resolver)
        agent._last_lookup_failure_time = time.monotonic() - 60.0

        result = agent._get_controller_handle()

        assert result is mock_handle

    def test_reset(self) -> None:
        agent = _StubAgent(actor_resolver=_FakeResolver())
        agent._controller_handle = MagicMock()
        agent._last_lookup_failure_time = time.monotonic()

        agent._reset_controller_handle()

        assert agent._controller_handle is None
        assert agent._last_lookup_failure_time is None

    def test_ft_id_scopes_actor_name(self) -> None:
        resolver = _FakeResolver(handle=MagicMock())
        agent = _StubAgent(ft_id="abc123", actor_resolver=resolver)

        result = agent._get_controller_handle()

        assert result is not None
        assert resolver.calls == ["ft_controller_abc123"]

    def test_empty_ft_id_uses_default_name(self) -> None:
        resolver = _FakeResolver(handle=MagicMock())
        agent = _StubAgent(ft_id="", actor_resolver=resolver)

        result = agent._get_controller_handle()

        assert result is not None
        assert resolver.calls == ["ft_controller"]

    def test_ft_id_from_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MILES_FT_ID", "env789")
        resolver = _FakeResolver(handle=MagicMock())
        agent = _StubAgent(actor_resolver=resolver)

        agent._get_controller_handle()
        assert resolver.calls == ["ft_controller_env789"]

    def test_failure_sets_cooldown(self) -> None:
        resolver = _FakeResolver(should_fail=True)
        agent = _StubAgent(actor_resolver=resolver)

        result = agent._get_controller_handle()

        assert result is None
        assert agent._last_lookup_failure_time is not None
