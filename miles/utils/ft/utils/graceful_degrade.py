"""Decorator for graceful degradation: catch Exception, log, return default."""
from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable, TypeVar, overload

_T = TypeVar("_T")


@overload
def graceful_degrade(
    *,
    default: _T,
    msg: str | None = ...,
    log_level: int = ...,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]: ...


@overload
def graceful_degrade(
    *,
    msg: str | None = ...,
    log_level: int = ...,
) -> Callable[[Callable[..., _T]], Callable[..., _T | None]]: ...


def graceful_degrade(
    *,
    default: Any = None,
    msg: str | None = None,
    log_level: int = logging.WARNING,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a function so that any ``Exception`` is logged and *default* is returned.

    Supports both sync and async callables.  The log message defaults to
    ``"{func.__qualname__} failed"`` and always includes ``exc_info=True``.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        resolved_msg = msg if msg is not None else f"{func.__qualname__} failed"
        func_logger = logging.getLogger(func.__module__)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    func_logger.log(log_level, resolved_msg, exc_info=True)
                    return default

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception:
                func_logger.log(log_level, resolved_msg, exc_info=True)
                return default

        return sync_wrapper

    return decorator
