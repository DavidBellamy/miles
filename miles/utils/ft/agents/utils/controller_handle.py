from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from miles.utils.ft.protocols.platform import ft_controller_actor_name

logger = logging.getLogger(__name__)


class ActorResolverProtocol(Protocol):
    def get_actor(self, name: str) -> Any: ...


class RayActorResolver:
    def get_actor(self, name: str) -> Any:
        import ray

        return ray.get_actor(name)


class ControllerHandleMixin:
    """Lazy-caching lookup of the ft_controller actor handle.

    Mixed into agent classes that need to communicate with FtController.
    On cache miss, attempts ``ray.get_actor()``; caches on success,
    returns None on failure.

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

    def _get_controller_handle(self) -> Any | None:
        if self._controller_handle is not None:
            return self._controller_handle

        try:
            actor_name = ft_controller_actor_name(self._ft_id)
            self._controller_handle = self._actor_resolver.get_actor(actor_name)
        except Exception:
            logger.warning("Failed to get ft_controller actor handle", exc_info=True)
            return None

        return self._controller_handle
