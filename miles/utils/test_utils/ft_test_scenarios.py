import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miles.ray.train.group import RayTrainGroup

logger = logging.getLogger(__name__)

_SCENARIOS: dict[str, "type[FTTestScenarioBase]"] = {}


def _register_scenario(name: str):
    def _decorator(cls: "type[FTTestScenarioBase]") -> type:
        _SCENARIOS[name] = cls
        return cls
    return _decorator


@dataclass
class FTTestContext:
    group: "RayTrainGroup"
    num_cells: int
    current_step: int = 0


class FTTestScenarioBase:
    def __init__(self, ctx: FTTestContext) -> None:
        self.ctx = ctx
        self._target_cell_index: int = ctx.num_cells - 1

    def before_step(self, step: int) -> None:
        """Called before each train() invocation."""

    def after_step(self, step: int) -> None:
        """Called after each train() invocation."""


@_register_scenario("with_failure")
class _WithFailureScenario(FTTestScenarioBase):
    def after_step(self, step: int) -> None:
        if step == 0:
            logger.info(
                "WithFailureScenario: stopping cell %d after step %d",
                self._target_cell_index, step,
            )
            self.ctx.group.stop_cell(self._target_cell_index)

        elif step == 1:
            logger.info(
                "WithFailureScenario: starting cell %d after step %d",
                self._target_cell_index, step,
            )
            self.ctx.group.start_cell(self._target_cell_index)

    def on_complete(self) -> None:
        logger.info("WithFailureScenario: completed successfully")


@_register_scenario("deterministic")
class _DeterministicScenario(FTTestScenarioBase):
    def after_step(self, step: int) -> None:
        if step == 1:
            logger.info(
                "DeterministicScenario: stop+start cell %d after step %d (trigger healing)",
                self._target_cell_index, step,
            )
            self.ctx.group.stop_cell(self._target_cell_index)
            self.ctx.group.start_cell(self._target_cell_index)

        elif step == 2:
            logger.info(
                "DeterministicScenario: stopping cell %d after step %d (create degraded state)",
                self._target_cell_index, step,
            )
            self.ctx.group.stop_cell(self._target_cell_index)

        elif step == 3:
            logger.info(
                "DeterministicScenario: starting cell %d after step %d (restore for healing)",
                self._target_cell_index, step,
            )
            self.ctx.group.start_cell(self._target_cell_index)

    def on_complete(self) -> None:
        logger.info("DeterministicScenario: completed successfully")


def get_scenario(name: str, ctx: FTTestContext) -> FTTestScenarioBase:
    if name not in _SCENARIOS:
        raise ValueError(
            f"Unknown FT test scenario: {name!r}. Available: {list(_SCENARIOS.keys())}"
        )
    return _SCENARIOS[name](ctx)
