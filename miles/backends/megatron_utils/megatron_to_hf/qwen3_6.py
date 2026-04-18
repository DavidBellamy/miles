"""Megatron → HuggingFace conversion for Qwen3.6.

Qwen3.6-35B-A3B shares Qwen3.5's MoE-A3B architecture and naming scheme, so
this module delegates to the Qwen3.5 converter. Keep it as a named entry
point so a future Qwen3.6 variant (different key/projection layout) can
fork here without touching ``qwen3_5.py``.
"""

from .qwen3_5 import convert_qwen3_5_to_hf


def convert_qwen3_6_to_hf(args, name, param):
    return convert_qwen3_5_to_hf(args, name, param)
