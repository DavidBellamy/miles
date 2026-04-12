"""Source-level patch for CompressedTensorsW8A8Fp8.restore_weights_before_loading.

SGLang's FP8 compressed-tensors scheme lacks restore_weights_before_loading,
which is needed for RL training weight sync. After process_weights_after_loading
replaces ModelWeightParameter (has weight_loader) with plain Parameter, subsequent
load_weights calls crash because param.weight_loader is missing.

This module provides generate_patch_code() which returns Python source to append
to the SGLang file. The source-level approach is needed because SGLang spawns
TP workers with multiprocessing "spawn" mode, so runtime monkey-patches in the
parent process are NOT inherited by TP workers.
"""

import textwrap


def generate_patch_code() -> str:
    """Return Python source code to append to compressed_tensors_w8a8_fp8.py.

    The code wraps create_weights to save weight_loader/shape, and adds
    restore_weights_before_loading to recreate typed parameters.
    """
    return textwrap.dedent('''

    # --- BEGIN miles FP8 restore patch ---
    _orig_create_weights = CompressedTensorsW8A8Fp8.create_weights

    def _patched_create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                                input_size, output_size, params_dtype, weight_loader, **kwargs):
        _orig_create_weights(
            self, layer, input_size_per_partition, output_partition_sizes,
            input_size, output_size, params_dtype, weight_loader, **kwargs,
        )
        layer._saved_weight_loader = weight_loader
        layer._original_weight_shape = (sum(output_partition_sizes), input_size_per_partition)

    CompressedTensorsW8A8Fp8.create_weights = _patched_create_weights


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
    # --- END miles FP8 restore patch ---
    ''')
