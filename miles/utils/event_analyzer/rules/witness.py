"""Rule: cross-rank witness param consistency."""

from collections import defaultdict

from miles.utils.event_logger.models import Event, WitnessEvent
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class WitnessMismatch(FrozenStrictBaseModel):
    rollout_id: int
    description: str


def check(events: list[Event]) -> list[WitnessMismatch]:
    """Verify that all ranks within the same (rollout_id, position) see identical nonzero witness IDs.

    Returns:
        List of mismatches found. Empty list means all ranks agree.
    """
    witness_events = [e for e in events if isinstance(e, WitnessEvent)]
    if not witness_events:
        return []

    grouped: dict[tuple[int, str], list[WitnessEvent]] = defaultdict(list)
    for event in witness_events:
        grouped[(event.rollout_id, event.position)].append(event)

    mismatches: list[WitnessMismatch] = []

    for (rollout_id, position), group in sorted(grouped.items()):
        reference = set(group[0].nonzero_ids)
        ref_source = group[0].source

        for event in group[1:]:
            current = set(event.nonzero_ids)
            if current != reference:
                only_in_ref = reference - current
                only_in_cur = current - reference
                mismatches.append(
                    WitnessMismatch(
                        rollout_id=rollout_id,
                        description=(
                            f"{position}: {ref_source.to_name()} vs {event.source.to_name()}: "
                            f"only_in_first={sorted(only_in_ref)}, "
                            f"only_in_second={sorted(only_in_cur)}"
                        ),
                    )
                )

    return mismatches
