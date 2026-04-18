"""Qwen3.6 weight bridge.

Qwen3.6-35B-A3B reuses Qwen3.5's ``qwen3_5_moe`` HF config schema. The only
structural difference (MTP-expert packing: fused 3-D tensor vs per-expert
unfused tensors) is detected automatically by :class:`Qwen3_5Bridge`, so
this subclass exists only as a named alias registered under
``qwen3_6`` / ``qwen3_6_moe`` for future variants that may carry a
dedicated ``model_type``.
"""

from mbridge.core import register_model

from .qwen3_5 import Qwen3_5Bridge


@register_model(["qwen3_6", "qwen3_6_moe"])
class Qwen3_6Bridge(Qwen3_5Bridge):
    pass
