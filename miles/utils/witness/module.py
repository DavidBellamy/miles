import logging
from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from miles.utils.event_logger.logger import get_event_logger
from miles.utils.event_logger.models import WitnessSnapshotParamEvent
from miles.utils.witness.allocator import WitnessInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_witness(
    model: nn.Module,
    *,
    buffer_size: int,
) -> None:
    model.local_head_witness = _DataWitness(buffer_size=buffer_size)
    model.local_tail_witness = _DataWitness(buffer_size=buffer_size)


def witness_dump_and_clear_stale(
    *,
    model: Sequence[nn.Module],
    witness_info: WitnessInfo,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Log nonzero witness param rows, then clear stale ring buffer entries."""
    for chunk_index, chunk in enumerate(model):
        inner = _unwrap_to_witness_owner(chunk)
        for attr in _WITNESS_ATTRS:
            assert hasattr(inner, attr), f"chunk {chunk_index} missing {attr}"
            witness: _DataWitness = getattr(inner, attr)
            _record_and_log_witness_param(
                witness=witness,
                instance_id=f"pp{chunk_index}." + attr.replace("_witness", ""),
                stale_ids=witness_info.stale_ids,
            )

    _clear_witness_stale_rows(model=model, stale_ids=witness_info.stale_ids, optimizer=optimizer)


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class _DataWitness(nn.Module):
    def __init__(self, buffer_size: int) -> None:
        super().__init__()
        self.buffer_size = buffer_size
        self.witness = nn.Embedding(num_embeddings=buffer_size, embedding_dim=1)
        self.witness.weight._is_witness_param = True
        nn.init.zeros_(self.witness.weight)

    def forward(self, input_ids: Tensor, witness_ids: Tensor) -> Tensor:
        assert input_ids.shape == witness_ids.shape
        w = self.witness(witness_ids)  # (*, 1)
        out = w - w.detach()  # forward: bitwise 0 (for finite w), backward: d/dw = I
        return out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


_WITNESS_ATTRS = ("local_head_witness", "local_tail_witness")


def _has_any_witness(module: nn.Module) -> bool:
    return any(hasattr(module, attr) for attr in _WITNESS_ATTRS)


def _unwrap_to_witness_owner(chunk: nn.Module) -> nn.Module:
    """Navigate through wrapping layers (DDP → Float16Module → GPTModel) to find the module with witness attrs."""
    inner = chunk.module
    while not _has_any_witness(inner) and hasattr(inner, "module"):
        inner = inner.module
    return inner


def _clear_witness_stale_rows(
    *,
    model: Sequence[nn.Module],
    stale_ids: list[int],
    optimizer: torch.optim.Optimizer,
) -> None:
    if not stale_ids:
        return

    witnesses = list(_get_all_witnesses_in_model(model))
    for witness in witnesses:
        idx = torch.tensor(stale_ids, dtype=torch.long, device=witness.witness.weight.device)
        _zero_witness_rows(witness=witness, idx=idx, optimizer=optimizer)


def _get_all_witnesses_in_model(model_chunks: Sequence[nn.Module]) -> list[_DataWitness]:
    witnesses: list[_DataWitness] = []
    for chunk in model_chunks:
        inner = _unwrap_to_witness_owner(chunk)
        for attr in _WITNESS_ATTRS:
            assert hasattr(inner, attr), f"model chunk missing {attr}"
            witnesses.append(getattr(inner, attr))
    return witnesses


def _zero_witness_rows(*, witness: _DataWitness, idx: Tensor, optimizer: torch.optim.Optimizer) -> None:
    model_weight = witness.witness.weight
    model_weight.data[idx] = 0.0

    main_param = getattr(model_weight, "main_param", None)
    if main_param is not None:
        assert main_param is not model_weight
        main_param.data[idx] = 0.0

    # Distributed optimizer keys state by main_param (fp32 copy);
    # non-distributed optimizer keys by model_weight directly.
    optimizer_key = main_param if main_param is not None else model_weight
    if optimizer_key in optimizer.state:
        state = optimizer.state[optimizer_key]
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                state[key][idx] = 0.0


def _record_and_log_witness_param(
    *,
    witness: _DataWitness,
    instance_id: str,
    stale_ids: list[int],
) -> None:
    model_weight = witness.witness.weight
    main_param = getattr(model_weight, "main_param", None)
    check_weight = main_param.data if main_param is not None else model_weight.data
    nonzero_witness_ids: list[int] = check_weight.squeeze(-1).nonzero(as_tuple=True)[0].tolist()

    get_event_logger().log(
        WitnessSnapshotParamEvent,
        dict(
            instance_id=instance_id,
            nonzero_witness_ids=nonzero_witness_ids,
            stale_ids=stale_ids,
        ),
        print_log=False,
    )
