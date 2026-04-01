"""Centralized event analyzer that reads events and runs all rules."""

import logging
from pathlib import Path

from miles.utils.event_analyzer.rules.weight_checksum import ChecksumMismatch, check_weight_checksums
from miles.utils.event_logger.logger import read_events

logger = logging.getLogger(__name__)


def run_analysis(event_dir: Path) -> list[ChecksumMismatch]:
    """Read all events from event_dir and run all analysis rules.

    Currently runs:
    - Weight checksum consistency check across replicas

    Logs errors for any findings and returns them.
    """
    events = read_events(event_dir)
    if not events:
        return []

    mismatches = check_weight_checksums(events)

    for m in mismatches:
        logger.error(
            "Weight checksum mismatch at step %d: %s/%s diverged across ranks %s",
            m.step,
            m.tensor_category,
            m.tensor_name,
            m.cell_indices,
        )

    return mismatches
