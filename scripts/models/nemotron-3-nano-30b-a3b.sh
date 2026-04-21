# NVIDIA Nemotron-3-Nano-30B-A3B (BF16, MoE nemotron_h = hybrid Mamba + Attention + MoE).
# HF config (verified 2026-04-21):
#   num_hidden_layers=52  hidden_size=2688  num_attention_heads=32  num_key_value_heads=2
#   head_dim=128  intermediate_size=1856  moe_intermediate_size=1856
#   n_routed_experts=128  num_experts_per_tok=6  n_shared_experts=1
#   moe_shared_expert_intermediate_size=3712  sigmoid routing + aux-free expert bias
# The AutoBridge path (--megatron-to-hf-mode bridge) + miles NemotronHBridge MoE shim
# (see miles/backends/megatron_utils/__init__.py) construct the provider and
# HF↔Megatron mapping_registry at load time. Attention-side structural args go
# in MODEL_ARGS for miles' arg parser.

MODEL_ARGS=(
   --disable-bias-linear
   --group-query-attention
   --num-attention-heads 32
   --num-query-groups 2
   --kv-channels 128
   --num-layers 52
   --hidden-size 2688
   --ffn-hidden-size 1856
   --normalization RMSNorm
   --position-embedding-type none
   --vocab-size 131072
   --make-vocab-size-divisible-by 128
   --untie-embeddings-and-output-weights

   # MoE specifics
   --num-experts 128
   --moe-router-topk 6
   --moe-ffn-hidden-size 1856
   --moe-shared-expert-intermediate-size 3712
   --moe-router-score-function sigmoid
   --moe-router-enable-expert-bias
   --moe-grouped-gemm
   --moe-router-dtype fp32
   # Group-limited routing (DeepSeek-style): config has n_groups=8, topk_group=1,
   # routed_scaling_factor=2.5. Missing these diverges training routing from
   # sglang's routing → huge train/rollout logprob mismatch.
   --moe-router-num-groups 8
   --moe-router-group-topk 1
   --moe-router-topk-scaling-factor 2.5
   --moe-router-pre-softmax
)
