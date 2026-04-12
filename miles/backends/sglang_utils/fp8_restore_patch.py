"""Monkey-patch CompressedTensorsW8A8Fp8 to add restore_weights_before_loading.

SGLang's FP8 compressed-tensors scheme lacks restore_weights_before_loading,
which is needed for RL training weight sync. After process_weights_after_loading
replaces ModelWeightParameter (has weight_loader) with plain Parameter, subsequent
load_weights calls crash because param.weight_loader is missing.

This patch:
1. Wraps create_weights to save weight_loader and original shape on each layer
2. Adds restore_weights_before_loading to recreate typed parameters with weight_loader
"""

import logging

logger = logging.getLogger(__name__)

_patched = False


def apply_fp8_restore_patch() -> None:
    global _patched
    if _patched:
        return
    _patched = True

    try:
        import torch
        from compressed_tensors.quantization import QuantizationStrategy
        from sglang.srt.layers.parameter import (
            BlockQuantScaleParameter,
            ChannelQuantScaleParameter,
            ModelWeightParameter,
            PerTensorScaleParameter,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
            CompressedTensorsW8A8Fp8,
        )
    except ImportError:
        logger.warning("fp8_restore_patch: required modules not available, skipping")
        return

    if hasattr(CompressedTensorsW8A8Fp8, "_miles_patched"):
        return

    # 1. Wrap create_weights to save weight_loader and shape
    _original_create_weights = CompressedTensorsW8A8Fp8.create_weights

    def _patched_create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                                input_size, output_size, params_dtype, weight_loader, **kwargs):
        _original_create_weights(
            self, layer, input_size_per_partition, output_partition_sizes,
            input_size, output_size, params_dtype, weight_loader, **kwargs,
        )
        layer._saved_weight_loader = weight_loader
        layer._original_weight_shape = (sum(output_partition_sizes), input_size_per_partition)

    CompressedTensorsW8A8Fp8.create_weights = _patched_create_weights

    # 2. Add restore_weights_before_loading
    def _restore_weights_before_loading(self, layer) -> None:
        weight_loader = getattr(layer, "_saved_weight_loader", None)
        if weight_loader is None:
            return
        out_size, in_size = layer._original_weight_shape
        device = layer.weight.device

        weight = ModelWeightParameter(
            data=torch.empty(out_size, in_size, dtype=torch.float8_e4m3fn, device=device),
            input_dim=1, output_dim=0, weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.strategy == QuantizationStrategy.CHANNEL:
            ws = ChannelQuantScaleParameter(
                data=torch.empty(out_size, 1, dtype=torch.float32, device=device),
                output_dim=0, weight_loader=weight_loader,
            )
        elif self.strategy == QuantizationStrategy.TENSOR:
            ws = PerTensorScaleParameter(
                data=torch.empty(len(layer.logical_widths), dtype=torch.float32, device=device),
                weight_loader=weight_loader,
            )
        elif self.strategy == QuantizationStrategy.BLOCK:
            bn, bk = self.weight_block_size
            ws = BlockQuantScaleParameter(
                data=torch.empty(
                    (out_size + bn - 1) // bn, (in_size + bk - 1) // bk,
                    dtype=torch.float32, device=device,
                ),
                input_dim=1, output_dim=0, weight_loader=weight_loader,
            )
        else:
            return

        ws[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", ws)

        if self.is_static_input_scheme and hasattr(layer, "input_scale") and layer.input_scale is not None:
            isc = PerTensorScaleParameter(
                data=torch.empty(len(layer.logical_widths), dtype=torch.float32, device=device),
                weight_loader=weight_loader,
            )
            isc[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", isc)

    CompressedTensorsW8A8Fp8.restore_weights_before_loading = _restore_weights_before_loading
    CompressedTensorsW8A8Fp8._miles_patched = True
    logger.info("fp8_restore_patch: patched CompressedTensorsW8A8Fp8 with restore_weights_before_loading")
