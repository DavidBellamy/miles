from __future__ import annotations

from miles.utils.ft.controller.detectors.base import BaseFaultDetector
from miles.utils.ft.controller.detectors.hang import HangDetector
from miles.utils.ft.controller.detectors.hardware import HighConfidenceHardwareDetector
from miles.utils.ft.controller.detectors.mfu_decline import MfuDeclineDetector
from miles.utils.ft.controller.detectors.nan_loss import NanLossDetector
from miles.utils.ft.controller.detectors.network import NetworkAlertDetector
from miles.utils.ft.controller.detectors.training_crash import TrainingCrashDetector


def build_detector_chain(
    config: dict[str, object] | None = None,
) -> list[BaseFaultDetector]:
    """Build the default detector chain in priority order (highest first)."""
    cfg = config or {}
    return [
        HighConfidenceHardwareDetector(),
        NetworkAlertDetector(**cfg.get("network", {})),  # type: ignore[arg-type]
        TrainingCrashDetector(),
        HangDetector(**cfg.get("hang", {})),  # type: ignore[arg-type]
        NanLossDetector(),
        MfuDeclineDetector(**cfg.get("mfu", {})),  # type: ignore[arg-type]
    ]
