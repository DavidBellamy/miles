from miles.utils.ft.controller.detectors._metric_names import TRAINING_JOB_STATUS
from miles.utils.ft.controller.detectors.base import BaseFaultDetector, _get_non_finite_loss
from miles.utils.ft.controller.mini_prometheus.protocol import MetricStoreProtocol
from miles.utils.ft.controller.mini_wandb import MiniWandb
from miles.utils.ft.models import ActionType, Decision

_JOB_STATUS_FAILED = -1


class TrainingCrashDetector(BaseFaultDetector):
    def evaluate(
        self,
        metric_store: MetricStoreProtocol,
        mini_wandb: MiniWandb,
        rank_placement: dict[int, str],
    ) -> Decision:
        df = metric_store.instant_query(f"{TRAINING_JOB_STATUS} == {_JOB_STATUS_FAILED}")
        if df.is_empty():
            return Decision(action=ActionType.NONE, reason="training job not failed")

        trigger = "nan_loss" if _get_non_finite_loss(mini_wandb) is not None else "crash"

        return Decision(
            action=ActionType.ENTER_RECOVERY,
            reason=f"training job failed (trigger={trigger})",
            trigger=trigger,
        )
