from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FtBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricSample(FtBaseModel):
    name: str
    labels: dict[str, str]
    value: float
    metric_type: Literal["gauge", "counter"] = "gauge"


class CollectorOutput(FtBaseModel):
    metrics: list[MetricSample]


class ActionType(str, Enum):
    NONE = "none"
    MARK_BAD_AND_RESTART = "mark_bad_and_restart"
    ENTER_RECOVERY = "enter_recovery"
    NOTIFY_HUMAN = "notify_human"


class TriggerType(str, Enum):
    NONE = ""
    HANG = "hang"
    NAN_LOSS = "nan_loss"
    CRASH = "crash"


class NodeFault(FtBaseModel):
    node_id: str
    reason: str


def unique_node_ids(faults: list["NodeFault"]) -> list[str]:
    """Return deduplicated node IDs from faults, preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for fault in faults:
        if fault.node_id not in seen:
            seen.add(fault.node_id)
            result.append(fault.node_id)
    return result


class Decision(FtBaseModel):
    action: ActionType
    bad_node_ids: list[str] = Field(default_factory=list)
    reason: str
    trigger: TriggerType = TriggerType.NONE

    @classmethod
    def from_node_faults(
        cls,
        faults: "list[NodeFault]",
        *,
        fallback_reason: str,
    ) -> "Decision":
        if not faults:
            return cls(action=ActionType.NONE, reason=fallback_reason)

        return cls(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=sorted(unique_node_ids(faults)),
            reason="; ".join(f.reason for f in faults),
        )


class DiagnosticResult(FtBaseModel):
    diagnostic_type: str
    node_id: str
    passed: bool
    details: str

    @classmethod
    def pass_result(
        cls, *, diagnostic_type: str, node_id: str, details: str,
    ) -> "DiagnosticResult":
        return cls(diagnostic_type=diagnostic_type, node_id=node_id, passed=True, details=details)

    @classmethod
    def fail_result(
        cls, *, diagnostic_type: str, node_id: str, details: str,
    ) -> "DiagnosticResult":
        return cls(diagnostic_type=diagnostic_type, node_id=node_id, passed=False, details=details)


class UnknownDiagnosticError(Exception):
    """Raised when a node agent is asked to run a diagnostic type it does not have."""


class RecoveryPhase(str, Enum):
    CHECK_ALERTS = "check_alerts"
    REATTEMPTING = "reattempting"
    MONITORING = "monitoring"
    DIAGNOSING = "diagnosing"
    EVICT_AND_RESTART = "evict_and_restart"
    NOTIFY = "notify"
    DONE = "done"


class ControllerMode(str, Enum):
    MONITORING = "monitoring"
    RECOVERY = "recovery"


_BAD_NODES_CONFIRMED_PHASES: frozenset[RecoveryPhase] = frozenset({
    RecoveryPhase.EVICT_AND_RESTART,
    RecoveryPhase.NOTIFY,
    RecoveryPhase.DONE,
})


class ControllerStatus(FtBaseModel):
    mode: ControllerMode
    recovery_phase: RecoveryPhase | None
    phase_history: list[RecoveryPhase] | None
    tick_count: int
    active_run_id: str | None
    bad_nodes: list[str]
    recovery_in_progress: bool
    bad_nodes_confirmed: bool


RECOVERY_PHASE_TO_INT: dict[RecoveryPhase, int] = {
    RecoveryPhase.CHECK_ALERTS: 1,
    RecoveryPhase.REATTEMPTING: 2,
    RecoveryPhase.MONITORING: 3,
    RecoveryPhase.DIAGNOSING: 4,
    RecoveryPhase.EVICT_AND_RESTART: 5,
    RecoveryPhase.NOTIFY: 6,
    RecoveryPhase.DONE: 7,
}
