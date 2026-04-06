import itertools
import logging

from miles.ray.rollout.debug_data import load_debug_rollout_data
from miles.rollout.base_types import (
    RolloutFnTrainInput,
    call_rollout_fn,
)
from miles.rollout.inference_rollout.compatibility import call_rollout_function

logger = logging.getLogger(__name__)


def get_rollout_data(self, rollout_id):
    if self.args.load_debug_rollout_data:
        data = load_debug_rollout_data(self.args, rollout_id=rollout_id)
        metadata = {}  # save/load metadata into debug rollout data as well
        metrics = None
    else:
        if self.use_experimental_refactor:
            data = call_rollout_function(self.generate_rollout, RolloutFnTrainInput(rollout_id=rollout_id))
        else:
            data = call_rollout_fn(
                self.generate_rollout, self.args, rollout_id, self.data_source, evaluation=False
            )
        metrics = data.metrics
        data = data.samples
        metadata = {}
        # flatten the data if it is a list of lists
        while isinstance(data[0], list):
            data = list(itertools.chain.from_iterable(data))

        if not self.args.disable_rollout_trim_samples:
            global_batch_size = self.args.global_batch_size
            if self.args.use_dynamic_global_batch_size:
                logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
                dynamic_global_batch_size = self._compute_dynamic_global_batch_size(len(data))
                metadata["dynamic_global_batch_size"] = dynamic_global_batch_size
                global_batch_size = dynamic_global_batch_size

            if len(data) % global_batch_size != 0:
                trim_len = (len(data) // global_batch_size) * global_batch_size
                if trim_len == 0:
                    raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
                origin_data_length = len(data)
                data = data[:trim_len]
                logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
            logger.info(f"Final collected {len(data)} samples from rollout to train")

    return data, metadata, metrics

def _compute_dynamic_global_batch_size(self, num_samples: int) -> int:
    """Calculate dynamic global_batch_size to ensure only one training step.

    Strategy: global_batch_size = num_samples rounded down to a multiple of dp_size
    This ensures num_steps_per_rollout = num_samples // global_batch_size = 1
    """
    dp_size = self.train_parallel_config["dp_size"]
    original_gbs = self.args.global_batch_size

    # Round down to a multiple of dp_size to ensure only one training step
    dynamic_gbs = (num_samples // dp_size) * dp_size

    if dynamic_gbs == 0:
        # Too few samples, use at least dp_size
        dynamic_gbs = dp_size
        logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

    # Calculate how many samples will be discarded
    wasted = num_samples - dynamic_gbs

    if dynamic_gbs != original_gbs or wasted > 0:
        logger.info(
            f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
            f"(num_samples={num_samples}, dp_size={dp_size}, "
            f"num_steps=1, wasted={wasted})"
        )

    return dynamic_gbs
