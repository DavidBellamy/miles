from typing import NamedTuple

from miles.ray.rollout.rollout_server import RolloutServer


class CellIndexer(NamedTuple):
    srv_key: str
    group_index: int
    engine_indices: list[int]


def get_cell_indexer_from_id(servers: dict[str, RolloutServer], cell_id: int) -> CellIndexer:
    assert 0 <= cell_id < get_num_cells(servers)
    return TODO


def get_num_cells(servers: dict[str, RolloutServer]) -> int:
    return TODO
