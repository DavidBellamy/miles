from datetime import timedelta

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.hardware_checks import check_nic_down_in_window
from miles.utils.ft.models.fault import Decision

_DEFAULT_ALERT_WINDOW = timedelta(minutes=5)
_DEFAULT_ALERT_THRESHOLD = 2


class NetworkAlertDetector(BaseFaultDetector):
    is_critical = True

    def __init__(
        self,
        alert_window: timedelta = _DEFAULT_ALERT_WINDOW,
        alert_threshold: int = _DEFAULT_ALERT_THRESHOLD,
    ) -> None:
        if alert_window.total_seconds() <= 0:
            raise ValueError(f"alert_window must be positive, got {alert_window}")
        if alert_threshold < 1:
            raise ValueError(f"alert_threshold must be >= 1, got {alert_threshold}")

        self._alert_window = alert_window
        self._alert_threshold = alert_threshold

    def evaluate(self, ctx: DetectorContext) -> Decision:
        faults = check_nic_down_in_window(
            ctx.metric_store,
            window=self._alert_window,
            threshold=self._alert_threshold,
        )
        return Decision.from_node_faults(faults, fallback_reason="NIC alerts below threshold")
