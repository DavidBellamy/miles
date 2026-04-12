from .padding_remover import remove_padding
from .quantizer_compressed_tensors import quantize_params_compressed_tensors
from .quantizer_fp8 import quantize_params_fp8
from .quantizer_mxfp8 import quantize_params_mxfp8

__all__ = [
    "remove_padding",
    "quantize_param",
    "quantize_params_fp8",
    "quantize_params_mxfp8",
    "quantize_params_compressed_tensors",
]


def quantize_params(args, megatron_name, converted_named_params, quantization_config):
    if quantization_config is None:
        return converted_named_params
    elif quantization_config["quant_method"] == "fp8":
        return quantize_params_fp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "mxfp8":
        return quantize_params_mxfp8(args, megatron_name, converted_named_params, quantization_config)
    elif quantization_config["quant_method"] == "compressed-tensors":
        # INT4 compressed-tensors: quantize inline via fake_int4_quant_cuda.
        # FP8 compressed-tensors (e.g. official zai-org/*-FP8 checkpoints): skip inline
        # quantization — post_process_weights handles FP8 re-quantization on the SGLang side.
        w_cfg = quantization_config.get("config_groups", {}).get("group_0", {}).get("weights", {})
        if w_cfg.get("group_size") is None:
            return converted_named_params
        return quantize_params_compressed_tensors(converted_named_params, quantization_config)
