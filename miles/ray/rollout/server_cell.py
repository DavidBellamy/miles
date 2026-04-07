from typing import NamedTuple

from miles.ray.rollout.rollout_server import RolloutServer


class CellIndexer(NamedTuple):
    srv_key: str
    group_index: int
    engine_indices: list[int]


def _get_cell_id_to_indexer_map(servers: dict[str, RolloutServer]) -> list[CellIndexer]:
    result: list[CellIndexer] = []
    for srv_key, srv in servers.items():
        for group_index, group in enumerate(srv.server_groups):
            for local_cell in range(len(group.engines)):
                result.append(CellIndexer(
                    srv_key=srv_key,
                    group_index=group_index,
                    engine_indices=list(
                        range(local_cell * group.nodes_per_engine, (local_cell + 1) * group.nodes_per_engine)),
                ))
    return result


def get_cell_indexer_from_id(servers: dict[str, RolloutServer], cell_id: int) -> CellIndexer:
    mapping = _get_cell_id_to_indexer_map(servers)
    assert 0 <= cell_id < len(mapping)
    return mapping[cell_id]


def get_num_cells(servers: dict[str, RolloutServer]) -> int:
    return len(_get_cell_id_to_indexer_map(servers))
