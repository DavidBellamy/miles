# Log-Prob Computation Patterns

Detailed reference for how log-probs flow through the Miles training pipeline.

## 1. Log-Prob Computation Pipeline

### 1.1 Rollout Phase (SGLang)

```
SGLang Engine → Generate tokens → Compute logits → log_softmax → rollout_log_probs
                                                                        │
                                                                  Store in Sample
```

SGLang computes log-probs during token generation using its own attention kernels (FlashInfer), operating on FP8 or BF16 weights.

### 1.2 Recompute Phase (Megatron Forward-Only)

```
Megatron Engine → forward_only(model, data) → Collect logits → FP32 cast → 
                                                                    │
                    get_log_probs_and_entropy() ← loss.py:143-217  │
                              │                                     │
                    calculate_log_probs_and_entropy() ← ppo_utils  │
                              │                                     │
                    fused_vocab_parallel_cross_entropy()             │
                              │                                     │
                         log_probs (stored)                         │
```

### 1.3 Reference Model Phase

Same as recompute but on frozen weights → `ref_log_probs`

### 1.4 Training Phase (Megatron Forward-Backward)

```
Train step → forward(model, data) → logits → loss computation using:
    - log_probs (from recompute)
    - ref_log_probs (from reference model)
    - rollout_log_probs (from SGLang)
    - advantages (from reward model)
```

## 2. Key Functions

### 2.1 get_responses() — Extract Response Tokens

**File**: `miles/backends/training_utils/loss.py:26-95`

Extracts per-sample (logits, tokens) pairs from the batch. Handles:
- Loss mask application
- Context length offset
- Temperature scaling
- FP32 enforcement (`assert logits.dtype == torch.float32`)

### 2.2 calculate_log_probs_and_entropy() — Core Computation

**File**: `miles/utils/ppo_utils.py:151-158`

Two paths:
1. **Standard (TP-aware)**: `fused_vocab_parallel_cross_entropy` from Megatron
2. **True on-policy**: `torch.log_softmax(logits, dim=-1)` (simple, no TP)

The fused kernel handles vocabulary sharding across TP ranks internally.

### 2.3 VocabParallelEntropy — Custom Autograd

**File**: `miles/utils/ppo_utils.py:162-194`

Custom `torch.autograd.Function` for numerically stable entropy across TP-sharded vocabulary:

```python
# Forward:
logits_max = allreduce(MAX, logits.max(dim=-1))  # numerical stability
shifted = logits - logits_max
exp_logits = shifted.exp()
sum_exp = allreduce(SUM, exp_logits.sum(dim=-1))
log_sum_exp = sum_exp.log() + logits_max
entropy = log_sum_exp - (logits * (exp_logits / sum_exp)).sum(dim=-1)

# Backward:
# Reuses saved softmax (exp_logits / sum_exp) — memory efficient
# Warning: modifies saved tensors in-place
```

### 2.4 compute_approx_kl() — KL Estimators

**File**: `miles/utils/ppo_utils.py:11-51`

```python
def compute_approx_kl(log_probs, log_probs_base, action_mask, kl_estimator):
    log_ratio = log_probs.float() - log_probs_base.float()
    
    if kl_estimator == "k1":
        return log_ratio  # can be negative
    elif kl_estimator == "k2":
        return 0.5 * log_ratio.pow(2)  # always >= 0
    elif kl_estimator in ("k3", "low_var_kl"):
        # Schulman's unbiased estimator
        kl = torch.exp(-log_ratio) - 1 + log_ratio  # always >= 0
        if kl_estimator == "low_var_kl":
            kl = kl.clamp(-10, 10)  # numerical stability
        return kl
```

**Optional importance ratio** (DeepSeek-V3.2 style):
```python
if importance_ratio is not None:
    kl = kl * importance_ratio  # per-token IS correction
```

## 3. Policy Loss Computation

### 3.1 Standard PPO

**File**: `loss.py:580-594`

```python
ppo_kl = old_log_probs - log_probs  # per-token
ratio = torch.exp(-ppo_kl)          # importance sampling ratio
# Clipped surrogate:
loss1 = -advantages * ratio
loss2 = -advantages * torch.clamp(ratio, 1-eps, 1+eps)
policy_loss = torch.max(loss1, loss2)
```

### 3.2 GSPO (Group-level Sequence PPO)

```python
# Sequence-level KL instead of token-level
seq_kl = compute_gspo_kl(old_log_probs, log_probs, action_mask)
# Expand to per-token
token_kl = seq_kl.unsqueeze(-1).expand_as(action_mask)
```

### 3.3 Loss Scaling

```python
# loss.py:902-913
if megatron_scaling:
    loss_scale = num_microbatches / global_batch_size * dp_cp_size
else:
    loss_scale = 1 / global_batch_size * dp_size

# CP-specific: ensure backward triggers on all ranks
loss = loss + 0 * logits.sum()  # zero-valued dependency trick
```

## 4. CI Verification Thresholds

### 4.1 At Rollout ID 0 (First Rollout)

**File**: `log_utils.py:171-202`

| Check | Condition | Tolerance | Notes |
|-------|-----------|-----------|-------|
| Megatron == Reference | `abs(log_probs - ref_log_probs)` | < 1e-9 | Same engine, same weights |
| Megatron ≈ SGLang | `abs(log_probs - rollout_log_probs)` | < 0.03 | Different engines |
| R3 mode | `abs(log_probs - ref_log_probs)` | < 1e-5 | Relaxed for routing replay |
| True on-policy | `log_probs == rollout_log_probs` | exact | Must be identical |
| Multi-model | tolerance | < 1e-8 | Slightly relaxed |
| Entropy range | `0 < entropy < 0.7` | - | Sanity check |

### 4.2 At Training Step 0

**File**: `ci_utils.py:12-24`

| Check | Condition | Notes |
|-------|-----------|-------|
| PPO KL | `ppo_kl < 1e-10` | Standard mode |
| PPO KL (MLA) | `ppo_kl < 1e-8` | MLA has larger mismatch (TODO: investigate) |
| PPO KL (LoRA) | `ppo_kl < 1e-8` | Weight conversion FP diff |
| Clip fraction | `pg_clipfrac < 1e-10` | No clipping at step 0 |
| KL loss | `kl_loss < 1e-9` | At first accumulated step |

## 5. Training-Inference Mismatch Correction (TIS/MIS)

### 5.1 Importance Sampling Framework

**File**: `examples/train_infer_mismatch_helper/mis.py`

```python
# Per-token IS weight:
is_weight = exp(train_log_prob - rollout_log_prob)

# Aggregation modes:
"token"     → per-token weights directly
"sequence"  → product of per-token weights (= sequence probability ratio)
"geometric" → geometric mean of per-token weights

# Bounding modes:
"truncate"  → clamp(is_weight, max=upper_bound)
"clip"      → clamp(is_weight, min=lower_bound, max=upper_bound)
"mask"      → zero out tokens where is_weight outside bounds

# Rejection sampling:
if seq_is_weight < rs_lower or seq_is_weight > rs_upper:
    reject entire sequence

# Veto: reject if any token has catastrophically low ratio
if any(per_token_ratio < veto_threshold):
    reject entire sequence

# Safety bound: clamp log-ratios to [-20, 20] to prevent exp overflow
```

### 5.2 Metrics Tracked

```python
# Per-batch metrics from MIS:
"train_ppl"          # exp(mean negative log-prob) from training engine
"rollout_ppl"        # same from inference engine
"kl_k1"              # mean log_ratio
"kl_k3"              # mean Schulman estimator
"chi_sq_token"       # mean (is_weight - 1)^2 per token
"chi_sq_seq"         # mean (seq_is_weight - 1)^2 per sequence
"rejection_rate"     # fraction of rejected sequences
"veto_rate"          # fraction vetoed
"is_weight_mean"     # should be ~1.0
"is_weight_std"      # lower is better
```

## 6. R3 (Rollout Routing Replay)

### How R3 Works

1. During SGLang rollout: store `topk_ids` (expert routing decisions) per token per layer
2. During Megatron recompute: load stored `topk_ids` and force the router to use them instead of recomputing
3. This eliminates MoE routing divergence, the largest source of mismatch in MoE models

### R3 Data Flow

```
SGLang rollout → moe_topk_ids[layer][token] → stored in Sample
                                                     │
Megatron recompute → router → OVERRIDE with stored topk_ids
                               instead of recomputing
```

### R3 Limitations

- Only replays top-K expert assignment, not the routing probabilities
- Weight conversion (BF16 → FP8) still causes small numerical differences
- `rollout_routed_experts` must be properly truncated (Miles #861)
- When `expert-model-parallel-size != actor-num-gpus-per-node`, R3 data may be corrupted (Miles #599)

### R3 Fix for Data Corruption (Miles #599)

```python
# WRONG: np.frombuffer returns read-only array
routing_data = np.frombuffer(buffer, dtype=np.int32)
# torch.from_numpy creates non-writable tensor → undefined behavior

# RIGHT: copy to make writable
routing_data = np.frombuffer(buffer, dtype=np.int32).copy()
```

## 7. True On-Policy Mode

### What It Changes

| Component | Normal Mode | True On-Policy Mode |
|-----------|-------------|-------------------|
| Log-softmax | Fused vocab-parallel kernel | `torch.log_softmax(logits, dim=-1)` |
| RoPE | SGLang's optimized bmm | Batch-invariant mode (no bmm) |
| MoE | Standard routing | Deterministic routing patch |
| Expected mismatch | up to 0.03 | exactly 0 |

### How to Enable

```bash
--true-on-policy-mode
```

### Caveats

- Slower than normal mode (no fused kernels)
- Only validated with FSDP backend
- Must use `--sglang-disable-cuda-graph`
- Cannot use FP8 weight quantization
