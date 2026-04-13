---
name: debug-precision
description: Use when debugging numerical precision issues in distributed training. Trigger on NaN loss, Inf gradients, BF16/FP16 drift, FP8 quantization errors, mixed precision problems, attention softmax precision, gradient accumulation FP32, loss divergence, or training instability. Also use when comparing numerical results across different hardware (H100 vs H200 vs Blackwell), different attention backends (FlashAttention vs cuDNN), or different precision settings.
---

# Debug Numerical Precision Issues

Systematic approach to diagnosing NaN, Inf, numerical drift, and precision-related training failures.

## Quick Decision Tree

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| NaN at iteration 1-2 | CUDA stream sync race during init | Check DDP param copy synchronization |
| NaN after checkpoint resume | Stale optimizer state or dtype mismatch | Verify checkpoint precision matches config |
| NaN only with DP > 1 | Gradient allreduce in BF16 | Add `--accumulate-allreduce-grads-in-fp32` |
| NaN with FP8 + specific flags | Incompatible FP8 config combo | See @references/precision_patterns.md Section 3 |
| NaN only on Blackwell/B200 | TE kernel scheduling race | Update TransformerEngine; see Megatron #2597 |
| Loss slowly diverges (not NaN) | BF16 accumulation drift over layers | Use FP32 softmax and gradient accumulation |
| Loss 6x higher than expected | FSDP/optimizer incompatibility | Check optimizer wrapper parameter mapping |
| rel_diff > 0.01 in comparator | Expected BF16 drift for deep layers | Relax threshold; see threshold guide below |
| Different results on H100 vs H200 | Hardware-specific kernel behavior | Use non-intrusive dumper to compare |

## Precision Hierarchy in Miles

Miles enforces FP32 at critical computation points:

```
                    BF16 (default training precision)
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   Attention              MLP/MoE              Gradient
        │                     │                     │
   softmax → FP32         BF16                allreduce → FP32
   (--attention-           standard            (--accumulate-allreduce-
    softmax-in-fp32)                            grads-in-fp32)
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    Logits → FP32 (enforced by assertion)
                              │
                    Log-probs → FP32
                              │
                    KL / Loss → FP32
```

**Key code locations**:
- `miles/backends/training_utils/loss.py:65` — `assert logits.dtype == torch.float32`
- `miles/utils/ppo_utils.py:28` — `log_ratio = log_probs.float() - log_probs_base.float()`
- `miles/backends/megatron_utils/model_provider.py:52` — `logits = logits.float()`

## NaN Debugging Steps

### Step 1: Identify Where NaN First Appears

```python
# Add to training loop or use source patcher:
def check_nan(name, tensor):
    if tensor is not None and torch.is_tensor(tensor) and tensor.is_floating_point():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            print(f"[RANK {torch.distributed.get_rank()}] "
                  f"NaN/Inf in {name}: nan={nan_count} inf={inf_count} "
                  f"shape={tensor.shape} dtype={tensor.dtype}")
```

### Step 2: Check CUDA Stream Synchronization

NaN at iteration 1-2 often means parameter copies happen on separate CUDA streams without sync:

```python
# WRONG (Megatron #2301): parameter copy without stream sync
model_copy = copy_model(model)  # runs on separate stream
output = model_copy(input)       # reads uninitialized params → NaN

# RIGHT: explicit stream synchronization  
torch.cuda.current_stream().wait_stream(copy_stream)
model_copy = copy_model(model)
torch.cuda.synchronize()
output = model_copy(input)
```

### Step 3: Gradient NaN Detection (Miles Built-in)

Miles already handles NaN gradients in `miles/backends/megatron_utils/model.py:451-461`:
```python
found_inf_flag = optimizer.prepare_grads()
if found_inf_flag:
    valid_step = False  # skip this step
else:
    grad_norm = optimizer.get_grad_norm()
    valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
```

If NaN persists despite this, the NaN is in the **forward pass**, not gradients.

### Step 4: Use Dumper for Layer-by-Layer NaN Detection

```yaml
# nan_debug_patcher.yaml
patches:
  - target: megatron.core.transformer.transformer_layer.TransformerLayer.forward
    preamble: |
      import torch
      def _check(name, t, layer_num):
          if torch.is_tensor(t) and t.is_floating_point():
              has_nan = torch.isnan(t).any().item()
              has_inf = torch.isinf(t).any().item()
              if has_nan or has_inf:
                  dumper.dump(f'NAN_{name}', t, layer_id=layer_num)
    edits:
      - match: "hidden_states = self._forward_attention("
        prepend: "_check('pre_attn', hidden_states, self.layer_number)"
      - match: "output = hidden_states + bias_dropout_add_exec_handler"
        append: "_check('post_layer', output, self.layer_number)"
```

## Expected Precision Thresholds

When comparing tensors with the dumper comparator, use these thresholds:

| Comparison Type | Layers 0-3 | Layers 10-20 | Layers 20+ | Notes |
|----------------|-----------|------------|----------|-------|
| Same code, same hardware | < 1e-6 | < 1e-6 | < 1e-6 | Should be identical |
| Same code, diff hardware (H100↔H200) | < 0.001 | < 0.003 | < 0.005 | Kernel implementation differences |
| SGLang vs Megatron (same precision) | < 0.001 | < 0.005 | < 0.008 | BF16 drift across layers |
| FlashAttention vs cuDNN attention | < 0.001 | < 0.003 | < 0.005 | Algorithm differences |
| FP8 vs BF16 inference | < 0.01 | < 0.02 | < 0.05 | Expected FP8 quantization error |
| INT4 vs BF16 | < 0.05 | < 0.1 | < 0.15 | Significant quantization error |

## Quantization Debugging

### FP8 Weight Sync Issues

Miles quantizes weights during Megatron→SGLang sync (not during training itself):

```
miles/backends/megatron_utils/megatron_to_hf/processors/
├── quantizer_fp8.py       # Per-tensor and blockwise FP8
├── quantizer_mxfp8.py     # Microscaling FP8 (MXFP8)
├── quantizer_nvfp4.py     # NVIDIA FP4
└── quantizer_compressed_tensors.py  # INT4/AWQ
```

**Common FP8 issues**:
1. **Scale overflow**: `abs_max / FP8_MAX` can overflow if weights have outlier values
2. **Block size mismatch**: MXFP8 requires last dim divisible by 32
3. **Hardware-specific**: FP8 param gather may increase memory on Blackwell (Megatron #2063)
4. **Incompatible combos**: MXFP8 + no-dp-comm-overlap causes NaN (Megatron #2272)

**Debug FP8 quantization quality**:
```python
# Compare pre and post quantization
import torch
original = model.state_dict()['layers.0.self_attn.q_proj.weight']  # BF16
# After FP8 round-trip:
quantized = quantize_to_fp8(original)
dequantized = dequantize_from_fp8(quantized)
rel_error = (original.float() - dequantized.float()).abs() / (original.float().abs() + 1e-8)
print(f"FP8 quantization error: mean={rel_error.mean():.6f} max={rel_error.max():.6f}")
```

### INT4/AWQ Debugging

```python
# Verify INT4 packing is correct
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_compressed_tensors import (
    WQLinear_GEMM
)
# Check qweight is properly packed int32
assert qweight.dtype == torch.int32
# Check scales are float16
assert scales.dtype == torch.float16
# Verify round-trip
unpacked = unpack_int4(qweight)  # Should give int4 values in [0, 15]
```

## Hardware-Specific Precision Issues

### Blackwell/GB200 Kernel Race (Megatron #2597)

**Symptom**: Sporadic NaN in gradient norms, "found NaN in local grad norm for bucket #0".

**Root cause**: TransformerEngine's NvJet GEMM + cuDNN RMSNorm kernel overlap without proper synchronization. RMSNorm reads input before upstream GEMM finishes writing.

**Workaround**: Update TransformerEngine to fixed version, or disable NvJet: `NVTE_NVJET=0`.

### H100 vs H200 Numerical Differences

H200 has identical compute to H100 but more HBM. Numerical differences should be negligible, but different NCCL versions or driver versions can cause subtle kernel implementation changes.

**Debug**: Use the non-intrusive dumper to capture identical workloads on both:
```bash
# On H100:
DUMPER_ENABLE=1 DUMPER_DIR=/tmp/h100_dump python ...
# On H200:
DUMPER_ENABLE=1 DUMPER_DIR=/tmp/h200_dump python ...
# Compare:
python -m sglang.srt.debug_utils.comparator \
  --baseline /tmp/h100_dump --target /tmp/h200_dump \
  --preset sglang_dev
```

## SGLang-Specific Precision Issues

### DP Attention NaN/Inf (SGLang #21460)

**Symptom**: NaN/Inf in probability tensor during rollout with DP attention on H800.

**Root cause**: Tensor shape vs metadata mismatch (e.g., tensor shape [1,35] but metadata says [1,36]) before `deepseek_layer_hidden_post_self_attn`. The crash path is `torch.multinomial(probs)`.

**Impact**: Crashes RLHF rollouts silently. High severity for Miles.

### FP8 KV Cache Accuracy Degradation (SGLang #22671)

**Symptom**: 19.6-point accuracy drop on AIME26 when using fp8_e4m3 KV cache on B300 GPUs.

**Root cause**: FP8 KV cache quantization introduces unacceptable precision loss for certain models (GLM-5.1-FP8).

**Fix**: Revert to BF16 KV cache: `--kv-cache-dtype auto` (default).

### Degenerate Output at Temperature=1.0 (SGLang #21238)

**Symptom**: SGLang generates completely wrong outputs at temperature=1.0 (works at 0.5). Blocks GRPO training which requires temperature=1.0.

**Root cause**: Hypothesized FlashInfer sampling kernel BF16 precision loss during softmax with high-entropy logits.

### FP8 lm_head Bypass (SGLang #21148)

**Symptom**: `LogitsProcessor._compute_lm_head()` falls through to generic `torch.matmul` for FP8 weights instead of using `quant_method.apply()`. Causes unnecessary BF16 cast and potential logit differences.

### Quality Regression Across SGLang Versions (SGLang #21696)

**Symptom**: ~20% decline in LLM-as-judge scores after upgrading 0.5.9 → 0.5.10rc0.

**Root cause**: Backend selection changes (`flashinfer_cutlass` vs `auto`), FP8 scale format mismatch warnings, missing KV cache scaling factors defaulting to 1.0.

**Lesson**: Always verify output quality after SGLang version upgrades, especially with FP8 models.

## Related Skills

- `/dumper-usage`: How to capture tensors for precision analysis
- `/debug-distributed`: For precision issues combined with parallelism bugs
- `/debug-logprob`: For log-probability precision issues specifically
- `/debug-moe`: For MoE-specific precision issues

## References

- @references/precision_patterns.md - Detailed precision patterns, FP8 configs, hardware-specific issues
