from __future__ import annotations

import logging
import os
import time
from typing import Any, Protocol

from miles.utils.ft.protocols.platform import ft_controller_actor_name

logger = logging.getLogger(__name__)

_RETRY_COOLDOWN_SECONDS: float = 30.0


class ActorResolverProtocol(Protocol):
    def get_actor(self, name: str) -> Any: ...


class RayActorResolver:
    def get_actor(self, name: str) -> Any:
        import ray

        return ray.get_actor(name)


class ControllerHandleMixin:
    """Lazy-caching lookup of the ft_controller actor handle.

    Mixed into agent classes that need to communicate with FtController.
    Provides _get_controller_handle() and _reset_controller_handle().

    After a failed lookup, retries are suppressed for _RETRY_COOLDOWN_SECONDS
    to avoid spamming logs, then the next call will attempt lookup again.

    Pass a custom ``actor_resolver`` to decouple from Ray (useful for testing).
    """

    def __init__(
        self,
        ft_id: str = "",
        actor_resolver: ActorResolverProtocol | None = None,
    ) -> None:
        self._ft_id = ft_id or os.environ.get("MILES_FT_ID", "")
        self._actor_resolver = actor_resolver or RayActorResolver()
        self._controller_handle: Any | None = None
        self._last_lookup_failure_time: float | None = None

    def _get_controller_handle(self) -> Any | None:
        if self._controller_handle is not None:
            return self._controller_handle

        if self._last_lookup_failure_time is not None:
            elapsed = time.monotonic() - self._last_lookup_failure_time
            if elapsed < _RETRY_COOLDOWN_SECONDS:
                return None

        try:
            actor_name = ft_controller_actor_name(self._ft_id)
            self._controller_handle = self._actor_resolver.get_actor(actor_name)
            self._last_lookup_failure_time = None
        except Exception:
            self._last_lookup_failure_time = time.monotonic()
            logger.warning("Failed to get ft_controller actor handle", exc_info=True)
            return None

        return self._controller_handle

    def _reset_controller_handle(self) -> None:
        self._controller_handle = None
        self._last_lookup_failure_time = None
