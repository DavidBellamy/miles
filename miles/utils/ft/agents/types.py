"""Agent-layer pure data types (no Protocols)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from miles.utils.ft.utils.base_model import FtBaseModel


class DiagnosticPipelineResult(FtBaseModel):
    bad_node_ids: list[str] = []
    reason: str = ""


class DiagnosticResult(FtBaseModel):
    diagnostic_type: str
    node_id: str
    passed: bool
    details: str
    metadata: dict[str, Any] | None = None

    @classmethod
    def pass_result(
        cls,
        *,
        diagnostic_type: str,
        node_id: str,
        details: str,
        metadata: dict[str, Any] | None = None,
    ) -> DiagnosticResult:
        return cls(
            diagnostic_type=diagnostic_type,
            node_id=node_id,
            passed=True,
            details=details,
            metadata=metadata,
        )

    @classmethod
    def fail_result(
        cls,
        *,
        diagnostic_type: str,
        node_id: str,
        details: str,
        metadata: dict[str, Any] | None = None,
    ) -> DiagnosticResult:
        return cls(
            diagnostic_type=diagnostic_type,
            node_id=node_id,
            passed=False,
            details=details,
            metadata=metadata,
        )


class UnknownDiagnosticError(Exception):
    """Raised when a node agent is asked to run a diagnostic type it does not have."""


class _MetricSampleBase(FtBaseModel):
    name: str
    labels: dict[str, str]


class GaugeSample(_MetricSampleBase):
    value: float
    metric_type: Literal["gauge"] = "gauge"


class CounterSample(_MetricSampleBase):
    delta: float
    metric_type: Literal["counter"] = "counter"


MetricSample = Annotated[
    GaugeSample | CounterSample,
    Field(discriminator="metric_type"),
]
