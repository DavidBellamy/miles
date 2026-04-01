"""Per-rank per-step weight checksum dumper for cross-replica consistency verification.

Design principle: fail fast. This module must never silently produce partial results.
If any parameter, buffer, master weight, or optimizer state cannot be hashed, it should
raise an error rather than skip it — incomplete checksums defeat the purpose of
cross-replica consistency verification.
"""

import hashlib
import logging
from argparse import Namespace
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer.optimizer import MegatronOptimizer

from miles.backends.megatron_utils.ci_utils import _hash_tensor_bytes
from miles.utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.event_logger.models import LocalWeightChecksumEvent, OptimizerStateInfo

logger = logging.getLogger(__name__)


def dump_local_weight_checksums(
    args: Namespace,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    step: int,
) -> None:
    """Compute and dump weight checksums if enabled."""

    if not args.save_local_weight_checksum:
        return

    assert is_event_logger_initialized(), (
        "save_local_weight_checksum is enabled but EventLogger is not initialized"
    )

    info = _compute_weight_checksums(model=model, optimizer=optimizer, step=step, rank=torch.distributed.get_rank())
    get_event_logger().log(info, print_log=False)


def _compute_weight_checksums(
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    step: int,
    rank: int,
) -> LocalWeightChecksumEvent:
    param_hashes = _hash_named_tensors(model, accessor="named_parameters")
    assert param_hashes, "No parameters found in model"
    buffer_hashes = _hash_named_tensors(model, accessor="named_buffers")

    optimizer_hashes = _collect_optimizer_hashes(model=model, optimizer=optimizer)
    assert optimizer_hashes, "No sub-optimizers found"

    return LocalWeightChecksumEvent(
        step=step,
        rank=rank,
        param_hashes=param_hashes,
        buffer_hashes=buffer_hashes,
        optimizer_hashes=optimizer_hashes,
    )


def _hash_named_tensors(model: Sequence[DDP], *, accessor: str) -> dict[str, str]:
    """Hash all named tensors from model chunks using the given accessor method."""
    hashes: dict[str, str] = {}
    for pp_idx, model_chunk in enumerate(model):
        for name, tensor in sorted(getattr(model_chunk, accessor)(), key=lambda x: x[0]):
            assert tensor is not None, f"pp{pp_idx}.{name}: tensor is None"
            hashes[f"pp{pp_idx}.{name}"] = _hash_tensor_sha256(tensor)
    return hashes


def _collect_optimizer_hashes(
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
) -> list[OptimizerStateInfo]:
    """Collect optimizer state snapshots with tensors replaced by hashes."""
    param_names_by_index = _build_param_names_by_index(model)
    result: list[OptimizerStateInfo] = []

    for sub_opt in _iter_sub_optimizers(optimizer):
        inner = sub_opt.optimizer
        assert isinstance(inner, torch.optim.Optimizer), (
            f"Expected torch.optim.Optimizer, got {type(inner)}"
        )

        sd = inner.state_dict()
        hashed_sd = _transform_tensor_to_hash(sd)

        result.append(OptimizerStateInfo(
            param_names=param_names_by_index,
            state_dict=hashed_sd,
        ))

    return result


def _build_param_names_by_index(model: Sequence[DDP]) -> dict[int, str]:
    """Build param index → name mapping matching torch optimizer's state_dict indexing."""
    names: dict[int, str] = {}
    idx = 0
    for pp_idx, model_chunk in enumerate(model):
        for name, param in model_chunk.named_parameters():
            assert param is not None, f"pp{pp_idx}.{name}: param is None"
            names[idx] = f"pp{pp_idx}.{name}"
            idx += 1
    return names


def _transform_tensor_to_hash(obj: Any) -> Any:
    """Recursively replace all tensors in a nested structure with their SHA-256 hashes."""
    if isinstance(obj, torch.Tensor):
        return _hash_tensor_sha256(obj)
    if isinstance(obj, dict):
        return {k: _transform_tensor_to_hash(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_transform_tensor_to_hash(v) for v in obj)
    return obj


def _iter_sub_optimizers(optimizer: MegatronOptimizer) -> Iterator[MegatronOptimizer]:
    """Flatten ChainedOptimizer into individual sub-optimizers."""
    if hasattr(optimizer, "chained_optimizers"):
        for sub in optimizer.chained_optimizers:
            yield from _iter_sub_optimizers(sub)
    else:
        yield optimizer


def _hash_tensor_sha256(tensor: torch.Tensor) -> str:
    raw_bytes = _hash_tensor_bytes(tensor)
    return hashlib.sha256(raw_bytes).hexdigest()
