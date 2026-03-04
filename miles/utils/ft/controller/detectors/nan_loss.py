from miles.utils.ft.controller.detectors.base import BaseFaultDetector, _get_non_finite_loss
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision


class NanLossDetector(BaseFaultDetector):
    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        rank_placement: dict[int, str],
    ) -> Decision:
        bad_loss = _get_non_finite_loss(mini_wandb)

        if bad_loss is not None:
            return Decision(
                action=ActionType.ENTER_RECOVERY,
                reason=f"loss is {bad_loss}",
                trigger="nan_loss",
            )

        return Decision(action=ActionType.NONE, reason="loss is normal")
