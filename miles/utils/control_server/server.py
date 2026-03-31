from __future__ import annotations

import asyncio
import logging
import threading

import ray
import uvicorn
from fastapi import FastAPI, HTTPException

from miles.ray.train.group import RayTrainGroup
from miles.utils.control_server.handles import _ActorCellHandle, _CellHandle, _RolloutCellHandle
from miles.utils.control_server.models import _CellInfo, _OkResponse, _StopRequest
from miles.utils.control_server.registry import _CellRegistry

logger = logging.getLogger(__name__)


# -------------------------- entrypoint ------------------------------


def start_control_server(
    *,
    actor_model: RayTrainGroup,
    rollout_manager: object,
    port: int,
    ft_components: frozenset[str],
) -> None:
    registry = _CellRegistry()

    if "train" in ft_components:
        for i in range(len(actor_model._cells)):
            registry.register(_ActorCellHandle(group=actor_model, cell_index=i))

    if "rollout" in ft_components:
        # TODO the code will NOT work before implementing rollout ft
        num_rollout_cells = ray.get(rollout_manager.get_cell_count.remote())
        for i in range(num_rollout_cells):
            registry.register(
                _RolloutCellHandle(
                    rollout_manager=rollout_manager,
                    cell_index=i,
                )
            )

    _start_control_server_raw(registry=registry, port=port)


def _start_control_server_raw(registry: _CellRegistry, port: int) -> None:
    app = _create_control_app(registry)

    def _run() -> None:
        uvicorn.run(app, host="0.0.0.0", port=port)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info("Control server started on port %d", port)


# -------------------------- main app ------------------------------


def _create_control_app(registry: _CellRegistry) -> FastAPI:
    app = FastAPI()

    # -------------------------- APIs ------------------------------

    @app.get("/api/v1/health")
    async def health() -> _OkResponse:
        return _OkResponse()

    @app.get("/api/v1/cells")
    async def get_cells() -> list[_CellInfo]:
        handles = registry.get_all()

        async def _fetch(handle: _CellHandle) -> _CellInfo:
            status, node_ids = await asyncio.gather(handle.get_status(), handle.get_node_ids())
            return _CellInfo(
                cell_id=handle.cell_id,
                cell_type=handle.cell_type,
                status=status,
                node_ids=node_ids,
            )

        return list(await asyncio.gather(*(_fetch(h) for h in handles)))

    @app.post("/api/v1/cells/{cell_id}/stop")
    async def stop_cell(cell_id: str, body: _StopRequest | None = None) -> _OkResponse:
        if body is None:
            body = _StopRequest()
        handle = _get_handle(cell_id)
        return await _call_handle(cell_id, "stop", handle.stop(timeout_seconds=body.timeout_seconds))

    @app.post("/api/v1/cells/{cell_id}/start")
    async def start_cell(cell_id: str) -> _OkResponse:
        handle = _get_handle(cell_id)
        return await _call_handle(cell_id, "start", handle.start())

    # -------------------------- utils ------------------------------

    def _get_handle(cell_id: str) -> _CellHandle:
        try:
            return registry.get(cell_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Cell '{cell_id}' not found") from None

    async def _call_handle(cell_id: str, action: str, coro) -> _OkResponse:
        try:
            await coro
        except Exception:
            logger.error("Failed to %s cell %s", action, cell_id, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to {action} cell '{cell_id}'") from None
        return _OkResponse()

    return app
