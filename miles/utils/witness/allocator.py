# TODO: move from module.py
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class WitnessIdAllocator:
    def allocate(self, sample_indices: list[int]) -> "WitnessInfo":
        do_allocate_things
        log_event(WitnessAllocateIdEvent())
        return TODO


class WitnessInfo(FrozenStrictBaseModel):
    witness_id_of_sample_index: dict[int, int]
