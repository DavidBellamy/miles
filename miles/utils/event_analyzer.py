"""Post-hoc checker for cross-replica weight checksum consistency."""

import logging
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from miles.utils.event_logger.logger import read_events
from miles.utils.event_logger.models import WeightChecksumDumped
from miles.utils.pydantic_utils import StrictBaseModel

logger = logging.getLogger(__name__)


class ChecksumMismatch(StrictBaseModel):
    step: int
    tensor_category: str
    tensor_name: str
    cell_indices: list[int]
    hashes: list[str]


def _find_mismatches_in_group(
    step: int,
    category: str,
    entries: list[tuple[int, dict[str, str]]],
) -> list[ChecksumMismatch]:
    """Compare hash dicts across replicas for a single (step, category) group."""
    mismatches: list[ChecksumMismatch] = []

    all_keys: set[str] = set()
    for _, hashes in entries:
        all_keys.update(hashes.keys())

    for key in sorted(all_keys):
        hash_by_rank: dict[str, list[int]] = defaultdict(list)
        for rank, hashes in entries:
            h = hashes.get(key, "<missing>")
            hash_by_rank[h].append(rank)

        if len(hash_by_rank) > 1:
            cell_indices: list[int] = []
            hash_values: list[str] = []
            for h, ranks in sorted(hash_by_rank.items(), key=lambda x: x[1][0]):
                for r in ranks:
                    cell_indices.append(r)
                    hash_values.append(h)

            mismatches.append(
                ChecksumMismatch(
                    step=step,
                    tensor_category=category,
                    tensor_name=key,
                    cell_indices=cell_indices,
                    hashes=hash_values,
                )
            )

    return mismatches


def check_weight_checksums(event_dir: Path) -> list[ChecksumMismatch]:
    """Read all event JSONL files and verify cross-replica weight checksum consistency.

    Args:
        event_dir: Path to the event log directory containing *.jsonl files.

    Returns:
        List of mismatches found. Empty list means all replicas match.
    """
    events = read_events(event_dir)
    checksum_events = [e for e in events if isinstance(e, WeightChecksumDumped)]

    if not checksum_events:
        logger.warning("No weight checksum events found in %s", event_dir)
        return []

    entries_by_step: dict[int, list[tuple[int, WeightChecksumDumped]]] = defaultdict(list)
    for event in checksum_events:
        entries_by_step[event.step].append((event.rank, event))

    all_mismatches: list[ChecksumMismatch] = []

    for step in sorted(entries_by_step.keys()):
        step_entries = entries_by_step[step]

        categories: list[tuple[str, Callable[[WeightChecksumDumped], dict[str, str]]]] = [
            ("param", lambda e: e.param_hashes),
            ("buffer", lambda e: e.buffer_hashes),
            ("master_param", lambda e: e.master_param_hashes),
            ("optimizer_state", lambda e: e.optimizer_state_hashes),
        ]

        for category_name, accessor in categories:
            group = [(rank, accessor(entry)) for rank, entry in step_entries]
            mismatches = _find_mismatches_in_group(
                step=step,
                category=category_name,
                entries=group,
            )
            all_mismatches.extend(mismatches)

    return all_mismatches


def main(event_dir: Path) -> int:
    """Run the checker and print results. Returns 0 if all match, 1 if mismatches found."""
    mismatches = check_weight_checksums(event_dir)

    if not mismatches:
        print("All replicas match across all steps.")
        return 0

    print(f"Found {len(mismatches)} mismatch(es):\n")
    for m in mismatches:
        print(f"  Step {m.step} | {m.tensor_category} | {m.tensor_name}")
        for idx, h in zip(m.cell_indices, m.hashes, strict=True):
            print(f"    rank {idx}: {h}")
        print()

    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <event_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(event_dir=Path(sys.argv[1])))
