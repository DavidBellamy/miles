from typing import NamedTuple

from miles.ray.rollout.rollout_server import RolloutServer


class CellIndexer(NamedTuple):
    srv_key: str
    group_index: int
    engine_indices: list[int]


def get_cell_indexer_from_id(servers: dict[str, RolloutServer], cell_id: int) -> CellIndexer:
    assert 0 <= cell_id < get_num_cells(servers)
    offset = 0
    for srv_key, srv in servers.items():
        for group_index, group in enumerate(srv.server_groups):
            num_cells_in_group = len(group.engines)
            if cell_id < offset + num_cells_in_group:
                local_cell = cell_id - offset
                npe = group.nodes_per_engine
                engine_indices = list(range(local_cell * npe, (local_cell + 1) * npe))
                return CellIndexer(
                    srv_key=srv_key,
                    group_index=group_index,
                    engine_indices=engine_indices,
                )
            offset += num_cells_in_group
    raise AssertionError("unreachable")


def get_num_cells(servers: dict[str, RolloutServer]) -> int:
    return sum(
        len(group.engines)
        for srv in servers.values()
        for group in srv.server_groups
    )
