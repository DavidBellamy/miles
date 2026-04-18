"""Qwen3.6 model spec.

Qwen3.6-35B-A3B shares Qwen3.5's architecture (``model_type: qwen3_5_moe``)
with the same 40-layer MoE-A3B layout, MTP head, linear+full attention
interleaving, attention output gate, and shared-expert gate. This module
re-exports Qwen3.5's components so that the qwen3_6 family has parallel
entry points and can diverge cleanly if Qwen3.6 introduces new variants.
"""

from .qwen3_5 import (
    Attention,
    Qwen3_5GatedDeltaNet as Qwen3_6GatedDeltaNet,
    get_qwen3_5_spec as get_qwen3_6_spec,
)

__all__ = [
    "Attention",
    "Qwen3_6GatedDeltaNet",
    "get_qwen3_6_spec",
]
