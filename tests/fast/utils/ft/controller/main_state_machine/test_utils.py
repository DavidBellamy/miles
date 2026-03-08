"""Tests for main_state_machine/utils.py — specifically _run_detectors_raw behavior on crashes."""

from __future__ import annotations

from unittest.mock import MagicMock

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.main_state_machine.utils import run_detectors
from miles.utils.ft.models.fault import ActionType, Decision, TriggerType
from miles.utils.ft.protocols.platform import JobStatus


def _make_detector_context() -> DetectorContext:
    return DetectorContext(
        metric_store=MagicMock(),
        mini_wandb=MagicMock(),
        rank_placement={0: "node-0"},
        job_status=JobStatus.RUNNING,
    )


class _PassingDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        return Decision.no_fault(reason="all good")


class _CrashingDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        raise RuntimeError("detector internal error")


class _RecoveryDetector(BaseFaultDetector):
    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        return Decision(
            action=ActionType.ENTER_RECOVERY,
            bad_node_ids=["node-0"],
            reason="found bad node",
            trigger=TriggerType.HARDWARE,
        )


class TestRunDetectorsCrashYieldsNotifyHuman:
    def test_single_crashing_detector_returns_notify_human(self) -> None:
        ctx = _make_detector_context()
        decision = run_detectors(detectors=[_CrashingDetector()], ctx=ctx)

        assert decision.action == ActionType.NOTIFY_HUMAN
        assert "crashed" in decision.reason

    def test_all_detectors_crash_does_not_return_no_fault(self) -> None:
        ctx = _make_detector_context()
        decision = run_detectors(
            detectors=[_CrashingDetector(), _CrashingDetector()],
            ctx=ctx,
        )

        assert decision.action != ActionType.NONE
        assert decision.action == ActionType.NOTIFY_HUMAN

    def test_crashing_detector_before_passing_detector(self) -> None:
        ctx = _make_detector_context()
        decision = run_detectors(
            detectors=[_CrashingDetector(), _PassingDetector()],
            ctx=ctx,
        )

        assert decision.action == ActionType.NOTIFY_HUMAN

    def test_passing_then_crashing_returns_notify_human(self) -> None:
        ctx = _make_detector_context()
        decision = run_detectors(
            detectors=[_PassingDetector(), _CrashingDetector()],
            ctx=ctx,
        )

        assert decision.action == ActionType.NOTIFY_HUMAN

    def test_recovery_detector_before_crash_returns_recovery(self) -> None:
        ctx = _make_detector_context()
        decision = run_detectors(
            detectors=[_RecoveryDetector(), _CrashingDetector()],
            ctx=ctx,
        )

        assert decision.action == ActionType.ENTER_RECOVERY

    def test_all_passing_returns_no_fault(self) -> None:
        ctx = _make_detector_context()
        decision = run_detectors(
            detectors=[_PassingDetector(), _PassingDetector()],
            ctx=ctx,
        )

        assert decision.action == ActionType.NONE
