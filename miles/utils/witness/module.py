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
    from megatron.core import parallel_state as mpu

    pp_rank = mpu.get_pipeline_model_parallel_rank()

    for chunk_index, chunk in enumerate(model):
        inner = _unwrap_to_witness_owner(chunk)
        for attr in _WITNESS_ATTRS:
            assert hasattr(inner, attr), f"chunk {chunk_index} missing {attr}"
            witness: _DataWitness = getattr(inner, attr)
            _record_and_log_witness_param(
                witness=witness,
                instance_id=f"pp{pp_rank}_chunk{chunk_index}." + attr.replace("_witness", ""),
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

        # DEBUG: gradient hook to diagnose missing witness_id=104
        debug_wid = 104
        mask_104 = (witness_ids == debug_wid)
        has_104 = mask_104.any().item()
        num_104_tokens = int(mask_104.sum().item())

        def _grad_hook(grad: Tensor) -> None:
            try:
                grad_sum = grad.sum().item()
                grad_max = grad.abs().max().item()
                if has_104:
                    grad_at_104 = grad[mask_104]
                    logger.info(
                        f"[WITNESS_GRAD_DEBUG] has_wid104=True n_tok={num_104_tokens} "
                        f"grad_104_sum={grad_at_104.sum().item():.6e} grad_104_absmax={grad_at_104.abs().max().item():.6e} "
                        f"grad_all_sum={grad_sum:.6e} grad_all_max={grad_max:.6e} shape={list(grad.shape)}"
                    )
                else:
                    logger.info(
                        f"[WITNESS_GRAD_DEBUG] has_wid104=False "
                        f"grad_all_sum={grad_sum:.6e} grad_all_max={grad_max:.6e} shape={list(grad.shape)}"
                    )
            except Exception as e:
                logger.warning(f"[WITNESS_GRAD_DEBUG] hook error: {e}")

        if w.requires_grad:
            w.register_hook(_grad_hook)

        # Register weight grad hook only once
        if not getattr(self, "_debug_weight_hook_registered", False):
            def _weight_grad_hook(grad: Tensor) -> None:
                try:
                    row_104 = grad[debug_wid].item()
                    nz = int((grad.abs() > 0).sum().item())
                    logger.info(
                        f"[WITNESS_WEIGHT_GRAD] weight.grad[{debug_wid}]={row_104:.6e} "
                        f"nonzero={nz}/{grad.shape[0]} absmax={grad.abs().max().item():.6e}"
                    )
                except Exception as e:
                    logger.warning(f"[WITNESS_WEIGHT_GRAD] hook error: {e}")

            self.witness.weight.register_hook(_weight_grad_hook)
            self._debug_weight_hook_registered = True

        return out

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: object = None
    ) -> dict:
        from megatron.core import parallel_state as mpu
        from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        # Embed PP rank in the checkpoint key so each pipeline stage has a unique
        # key (e.g. local_head_witness_pp0.witness.weight vs _pp1.witness.weight).
        # Without this, PP>1 causes a sharding validation error because multiple
        # stages register the same key with identical replica_id.
        pp_prefix = f"{prefix.rstrip('.')}_pp{pp_rank}."
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        module_sd = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            module_sd, pp_prefix, {}, sharded_offsets, dp_cp_group=metadata["dp_cp_group"]
        )


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
