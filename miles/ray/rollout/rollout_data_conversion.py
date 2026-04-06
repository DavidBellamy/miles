import logging


logger = logging.getLogger(__name__)


def compute_dynamic_global_batch_size(num_samples: int) -> int:
    """Calculate dynamic global_batch_size to ensure only one training step.

    Strategy: global_batch_size = num_samples rounded down to a multiple of dp_size
    This ensures num_steps_per_rollout = num_samples // global_batch_size = 1
    """
    dp_size = train_parallel_config["dp_size"]
    original_gbs = args.global_batch_size

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

