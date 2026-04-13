---
name: debug-logprob
description: Use when debugging log-probability differences between training and inference engines, KL divergence issues, PPO loss anomalies, or training-inference mismatch in RLHF/GRPO. Trigger on logprob mismatch, KL too high/low, ppo_kl, pg_clipfrac, rollout_log_probs vs log_probs difference, importance sampling issues, true on-policy mode problems, or when reward/advantage computation seems wrong. Also use for R3 (Rollout Routing Replay) issues in MoE models.
---

# Debug Log-Probability & Training-Inference Mismatch

Log-prob consistency between inference (SGLang) and training (Megatron) engines is critical for RLHF/GRPO stability.

## Quick Decision Tree

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| `ppo_kl > 1e-8` at step 0 | Training-inference logprob mismatch | Check CI thresholds, see Mismatch Sources below |
| `ppo_kl > 0.03` at step 0 | Major mismatch: weight version, MoE routing, or precision | Verify weight sync, check R3 |
| `pg_clipfrac` too high | Policy update too aggressive or stale rollout | Check learning rate, rollout freshness |
| `kl_loss` explodes during training | Reference model diverged or wrong ref logprobs | Verify ref model weights match |
| Training loss oscillates wildly | IS weights have high variance | Enable TIS correction, tune bounds |
| Exact match needed but fails | Not in true on-policy mode | Enable `--true-on-policy-mode` |
| MoE model logprobs differ more | Expert routing differs between engines | Enable R3 (`--use-rollout-routing-replay`) |
| NaN in KL computation | Log-ratio overflow | Check `low_var_kl` clamping |

## Three Types of Log-Probs in Miles

```
┌──────────────────────────────────────────────────────────────┐
│  1. rollout_log_probs  ← SGLang inference engine (generation) │
│  2. log_probs          ← Megatron training engine (recompute)  │
│  3. ref_log_probs      ← Reference model (frozen, for KL)      │
└──────────────────────────────────────────────────────────────┘
```

**Expected differences** (from Miles CI in `log_utils.py:171-202`):
| Comparison | Expected Tolerance | Why |
|-----------|-------------------|-----|
| `log_probs` vs `ref_log_probs` | < 1e-9 (same engine) | Same Megatron model, should be identical |
| `log_probs` vs `rollout_log_probs` | < 0.03 | Different engines (Megatron vs SGLang), different attention impls |
| With R3 (routing replay) | < 1e-5 | Relaxed due to expert routing replay precision |
| With LoRA | < 1e-8 for kl | Weight conversion introduces FP differences |
| True on-policy mode | exact (== 0) | Same code path, must be identical |

## Mismatch Sources (Why SGLang != Megatron)

### 1. Attention Implementation Differences

SGLang uses FlashInfer/FlashAttention; Megatron uses its own attention kernels. Numerical differences in:
- Softmax computation (order of operations, FP32 vs BF16)
- Flash attention tile sizes and accumulation order
- RoPE implementation details

### 2. MoE Expert Routing (Biggest Source for MoE Models)

The router produces slightly different logits → different top-K expert selection → completely different outputs for affected tokens.

**Solution: R3 (Rollout Routing Replay)**
```bash
--use-rollout-routing-replay   # Replay SGLang routing decisions in Megatron
--use-miles-router              # Use Miles router (required for R3)
```

R3 stores the expert routing decisions from SGLang and replays them during Megatron's forward pass. This eliminates routing divergence but still has small numerical differences from weight conversion.

### 3. Token Packing / Padding Differences (Megatron #1809)

Logprobs for the same chunk differ depending on whether padding or concatenation fills the context. This is a known Megatron issue where numerical precision in the processing pipeline differs between padded and packed inputs.

**Key insight**: For RL pipelines, ensure reference logprobs and policy logprobs use **identical input formatting** (same padding/packing strategy).

### 4. Weight Quantization During Sync

When FP8/INT4 quantization is used for SGLang inference, the quantized weights produce different logits than the BF16 training weights:

```
Megatron (BF16 weights) → quantize → SGLang (FP8 weights)
         └── log_probs                    └── rollout_log_probs
         Different due to quantization error!
```

## KL Divergence Computation

Miles supports multiple KL estimators (`miles/utils/ppo_utils.py:11-51`):

```python
log_ratio = log_probs.float() - log_probs_base.float()

# k1: Simple log ratio (can be negative)
kl = log_ratio

# k2: Squared log ratio / 2 (always >= 0)
kl = log_ratio ** 2 / 2

# k3 / low_var_kl: Schulman's unbiased estimator (always >= 0, lower variance)
kl = torch.exp(-log_ratio) - 1 + log_ratio
# Plus clamping to [-10, 10] for numerical stability
```

**When to use which**:
- `k1`: Fastest, but can be negative (bad for optimization)
- `k3/low_var_kl`: Recommended for GRPO/PPO, always non-negative, lower variance

## Debugging Steps

### Step 1: Check CI Values at Step 0

After first rollout, Miles logs these CI values:
```python
# In log_utils.py
assert abs(log_probs - rollout_log_probs) < 0.03  # training vs inference
assert abs(log_probs - ref_log_probs) < 1e-9       # training vs reference
assert 0 < entropy < 0.7                            # sanity check
```

If these fail, the mismatch is beyond acceptable bounds.

### Step 2: Dump Logits for Comparison

Use the dumper to capture logits from both engines:

```yaml
# For Megatron training (source patcher):
patches:
  - target: miles.backends.megatron_utils.model_provider.LinearForLastLayer.forward
    edits:
      - match: "logits = logits.float()"
        append: "dumper.dump('final_logits', logits, dims='t 1 vocab[tp]')"
```

For SGLang, use non-intrusive mode which captures logits automatically.

Then compare:
```bash
python -m sglang.srt.debug_utils.comparator \
  --baseline /tmp/dumper/engines/ \
  --target /tmp/dumper/fwd_only/ \
  --preset sglang_megatron \
  --filter "final_logits|logits"
```

### Step 3: Check Per-Token Log-Prob Divergence

```python
import torch

# Load rollout and recomputed log-probs
rollout_lp = sample.rollout_log_probs  # from SGLang
train_lp = sample.log_probs            # from Megatron recompute

diff = (rollout_lp - train_lp).abs()
print(f"Mean diff: {diff.mean():.6f}")
print(f"Max diff: {diff.max():.6f} at token {diff.argmax()}")
print(f"Tokens with diff > 0.01: {(diff > 0.01).sum()} / {len(diff)}")

# Check if mismatch correlates with specific tokens
for i in range(len(diff)):
    if diff[i] > 0.01:
        token_id = sample.tokens[i+1]  # log_prob is for predicting next token
        print(f"  Token {i}: id={token_id} diff={diff[i]:.6f}")
```

### Step 4: Vocab-Parallel Entropy Cross-Check

Miles computes entropy in a numerically stable way across TP ranks (`ppo_utils.py:162-194`). If entropy is NaN or negative, the vocab-parallel computation has a bug:

```python
# VocabParallelEntropy does:
# 1. logits_max = allreduce(MAX) for stability
# 2. exp_logits = exp(logits - logits_max)
# 3. sum_exp = allreduce(SUM)
# 4. entropy = log(sum_exp) + logits_max - (logits * softmax).sum()
```

### Step 5: True On-Policy Mode

If exact match is required (e.g., for validation), enable true on-policy mode:

```bash
--true-on-policy-mode
```

This mode:
1. Uses `torch.log_softmax` instead of fused TP-aware kernels
2. Enables `batch_invariant_mode` in SGLang (disables bmm for RoPE)
3. Patches MoE layers for deterministic routing
4. CI enforces `log_probs == rollout_log_probs` (exact equality)

**Files**: `miles/backends/experimental/fsdp_utils/actor.py:180-193`, `miles/utils/ppo_utils.py:687-717`

## Training-Inference Mismatch Correction

When mismatch is unavoidable (e.g., MoE models, FP8 inference), Miles provides correction mechanisms:

### TIS (Training-Inference Separation) / MIS (Mismatch Importance Sampling)

**File**: `examples/train_infer_mismatch_helper/mis.py`

Three approaches:
1. **Standard PPO** (no correction): Uses Megatron log-probs as old policy
2. **Bypass** (`--use-rollout-logprobs`): Uses SGLang log-probs directly, skips recompute forward
3. **Decoupled 3-policy** (`--use-tis`): IS weights from `pi_megatron / pi_sglang` multiplied by PPO loss

IS weight computation:
```python
is_weight = exp(train_log_prob - rollout_log_prob)  # per-token
# Aggregation: token-level, sequence-level, or geometric mean
# Bounding: truncate (upper only), clip (lower+upper), or mask (reject)
# Safety: clamp log-ratios to [-20, 20]
```

## Common Issues from Miles Issues Tracker

### Issue #574: Weight Conversion NCCL Deadlock with EP
With EP, different ranks hold different expert subsets. During weight sync, `all_gather_param()` iteration order must be identical across TP ranks even for empty expert subsets.

### Issue #555: cos_sin_cache Weight Mismatch
RoPE's `cos_sin_cache` is a non-trainable buffer that diverges between engines. This is expected and should be excluded from weight comparison:
```bash
--check-weight-update-equal  # This flag will show cos_sin_cache mismatch — it's expected
```

### Issue #861: rollout_routed_experts Not Truncated
`_truncate_sample_output()` misses `rollout_routed_experts`, causing shape validation failures in R3 mode.

## Related Skills

- `/debug-precision`: For numerical precision issues affecting log-probs
- `/debug-distributed`: For parallelism-related log-prob divergence
- `/dumper-usage`: For capturing logits and activations for comparison

## References

- @references/logprob_patterns.md - Detailed log-prob computation flow, KL estimators, IS weight math
