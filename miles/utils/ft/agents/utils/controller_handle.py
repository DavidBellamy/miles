from __future__ import annotations

from typing import Any

from miles.utils.ft.protocols.platform import ft_controller_actor_name
from miles.utils.ft.utils.graceful_degrade import graceful_degrade


@graceful_degrade(msg="Failed to get ft_controller actor handle")
def get_controller_handle(ft_id: str) -> Any | None:
    """Look up the ft_controller Ray actor by *ft_id*. Returns None on failure."""
    import ray

    return ray.get_actor(ft_controller_actor_name(ft_id))
