from miles.utils.event_logger.models import Event


def check(events: list[Event]) -> list[TODO]:
    """
    Related events:
    * RolloutDataEvent: when a rollout is executed and some data are obtained
    * WitnessAllocateIdEvent: when allocating `witness_id` to `sample_id`
    * WitnessSnapshotParamEvent: near the end of each train() step in MegatronTrainRayActor
        * If a witness_id appears in the weight, it means the corresponding data is consumed at least once.
    * TrainGroupStepEndEvent: after each train() step in RayTrainGroup

    Check:
    1. For each (rollout_id, cell_index),
       if TrainGroupStepEndEvent claims the cell ends with TrainStepOutcome.NORMAL,
       then its WitnessSnapshotParamEvent should observe *EXACTLY* the training data in rollout_id=0~curr.

    Remarks:
    * To correlate witness_id vs sample_id utilize WitnessAllocateIdEvent.
    * To get *all* samples used in a step, must use RolloutDataEvent as source of truth.
    * Witness' ring buffer will remove old data, thus we need to ignore the appearance/disappearance of
      all values in the range of 0..`WitnessSnapshotParamEvent.threshold_removed (TODO: better name)`
    """
    return TODO
