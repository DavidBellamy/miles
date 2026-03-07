from enum import Enum

from pydantic import ConfigDict

from miles.utils.ft.models.base import FtBaseModel


class RecoveryPhase(str, Enum):
    # Entry point: inspect collected alerts for known hardware/network faults
    CHECK_ALERTS = "check_alerts"
    # Stop current run and resubmit without evicting nodes; poll until job starts
    REATTEMPTING = "reattempting"
    # Watch the reattempted run for iteration progress to confirm recovery
    MONITORING = "monitoring"
    # Run diagnostic pipeline (GPU checks, NCCL tests, etc.) to locate bad nodes
    DIAGNOSING = "diagnosing"
    # Evict confirmed bad nodes from the cluster and resubmit training
    EVICT_AND_RESTART = "evict_and_restart"
    # Escalate to humans — automated recovery could not resolve the issue
    NOTIFY = "notify"
    # Terminal state: recovery workflow complete
    DONE = "done"


class ControllerMode(str, Enum):
    MONITORING = "monitoring"
    RECOVERY = "recovery"


class RecoverySnapshot(FtBaseModel):
    model_config = ConfigDict(frozen=True)

    in_progress: bool
    phase: RecoveryPhase | None
    phase_history: list[RecoveryPhase] | None
    diagnosing_nodes: list[str]
    bad_nodes_confirmed: bool


class ControllerStatus(FtBaseModel):
    mode: ControllerMode
    recovery_phase: str | None
    phase_history: list[str] | None
    tick_count: int
    active_run_id: str | None
    bad_nodes: list[str]
    recovery_in_progress: bool
    bad_nodes_confirmed: bool
    latest_iteration: int | None


