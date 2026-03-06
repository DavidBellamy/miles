from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from miles.utils.ft.models.base import FtBaseModel
from miles.utils.ft.models.fault import TriggerType
from miles.utils.ft.models.recovery import RecoveryPhase

PENDING_TIMEOUT_SECONDS: int = 300


class ReattemptState(FtBaseModel):
    submitted: bool = False
    submit_time: datetime | None = None
    start_time: datetime | None = None
    base_iteration: int | None = None


class RecoveryContext(FtBaseModel):
    trigger: TriggerType
    phase: RecoveryPhase = RecoveryPhase.CHECK_ALERTS
    recovery_start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    phase_before_notify: RecoveryPhase | None = None
    bad_node_ids: list[str] = Field(default_factory=list)
    phase_history: list[RecoveryPhase] = Field(default_factory=lambda: [RecoveryPhase.CHECK_ALERTS])

    reattempt: ReattemptState = Field(default_factory=ReattemptState)

    # Configuration
    global_timeout_seconds: int = 1800
    monitoring_success_iterations: int = 10
    monitoring_timeout_seconds: int = 600
