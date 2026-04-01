import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


async def retry(
    fn: Callable[[], Awaitable[Any]],
) -> None:
    """Retry until `fn` does not throw"""
    TODO
