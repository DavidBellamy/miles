import asyncio
from typing import Annotated

import typer

from miles.utils.ft.controller.controller import FtController
from miles.utils.ft.controller.mini_prometheus import MiniPrometheus, MiniPrometheusConfig
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.platform.stubs import StubNodeManager, StubTrainingJob

app = typer.Typer()


@app.command()
def main(
    tick_interval: Annotated[
        float, typer.Option(help="Controller main loop interval (seconds)")
    ] = 30.0,
) -> None:
    """FT Controller entry point (stub mode)."""
    node_manager = StubNodeManager()
    training_job = StubTrainingJob()
    metric_store = MiniPrometheus(config=MiniPrometheusConfig())
    mini_wandb = MiniWandb()

    controller = FtController(
        node_manager=node_manager,
        training_job=training_job,
        metric_store=metric_store,
        mini_wandb=mini_wandb,
        tick_interval=tick_interval,
    )

    asyncio.run(controller.run())


if __name__ == "__main__":
    app()
