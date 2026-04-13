---
name: debug-weight-sync
description: Use when debugging weight update/sync issues between Megatron training and SGLang inference engines. Trigger on weight update timeout, stale weights, weight version mismatch, checkpoint conversion errors, cos_sin_cache mismatch, FP8/INT4 weight quantization during sync, megatron_to_hf conversion bugs, or weight update freezing inference. Also trigger on OOM during weight update, or NCCL deadlock during all_gather_param with Expert Parallelism.
---

# Debug Weight Update & Sync Issues

Weight sync between Megatron (training) and SGLang (inference) is a critical and error-prone pipeline.

## Weight Sync Architecture

```
Megatron Training Engine (BF16 weights)
         │
         ├── 1. Extract state_dict
         │
         ├── 2. Convert Megatron format → HuggingFace format
         │      (miles/backends/megatron_utils/megatron_to_hf/)
         │
         ├── 3. Optional: Quantize (FP8/MXFP8/INT4/NVFP4)
         │      (megatron_to_hf/processors/quantizer_*.py)
         │
         ├── 4. Distribute to SGLang engines
         │      (all_gather_param across TP ranks)
         │
         └── 5. SGLang loads updated weights
               (engine frozen during update ~20-30s)
```

## Quick Decision Tree

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Weight update hangs with EP | all_gather_param iteration order mismatch | See Miles #574 below |
| OOM during weight update | Buffer size too large | Reduce `--update-weight-buffer-size` or remove flag |
| Inference frozen 20-30s during update | Normal behavior (engine pause) | See Miles #921 for mitigation |
| `cos_sin_cache` mismatch warning | Expected: RoPE buffer not synced | Exclude from comparison |
| Weight version warning in logs | Stale weights in inference engine | Check sync completed before next rollout |
| Gibberish output after fine-tuning | Weight conversion bug or tokenizer mismatch | See Miles #650 |
| `Unknown parameter name` error | Model type not supported in converter | Add parameter mapping |
| NCCL "Bad address" during sync | torch_memory_saver buffer relocation | See Miles #806 |

## Common Weight Sync Bugs

### Bug 1: EP All-Gather Deadlock (Miles #574)

**Symptom**: Training hangs after ~74 steps during weight update with EP.

**Root cause**: With EP, different ranks hold different expert subsets. In `_update_weight_from_distributed()`, `all_gather_param()` is a TP-group collective. If iteration order or parameter availability differs across ranks, some ranks skip the all_gather while others block.

**Fix**: Ensure all TP ranks iterate parameters in identical order, even for empty expert subsets. Add explicit barriers.

### Bug 2: Inference Freeze During Update (Miles #921)

**Symptom**: 119 timeout errors immediately after weight update. SGLang engines frozen for ~22 seconds.

**Root cause**: Weight checkpoint loading blocks SGLang engines from processing requests. In-flight requests at 580+ seconds get pushed past the 600s timeout.

**Mitigation**: Three-phase dispatch gate:
1. Pre-update: stop routing new requests
2. During update: queue requests locally
3. Post-update: resume after engine confirms readiness

### Bug 3: torch_memory_saver Buffer Invalidation (Miles #806)

**Symptom**: NCCL "Bad address" during cross-node weight sync on GB200/Blackwell.

**Root cause**: `torch_memory_saver`'s LD_PRELOAD-based GPU memory offload relocates GPU memory buffers, invalidating NCCL's registered addresses.

**Workaround**: Disable `torch_memory_saver` for weight sync operations, or use a different memory offload mechanism.

### Bug 4: Weight Conversion Missing Parameters

**Symptom**: `ValueError: Unknown parameter name: module.module.vision_model.patch_embed.proj.weight`

**Root cause**: The Megatron→HF converter (`megatron_to_hf/`) doesn't have mappings for all model types (e.g., vision tower in multimodal models).

**Fix**: Add parameter mapping rules for the new model type.

### Bug 5: Gibberish After SFT (Miles #650)

**Symptom**: Good training loss but model outputs nonsense.

**Possible causes**:
1. Tokenizer mismatch between training and inference
2. Chat template not applied correctly
3. Embedding/output layer weight tied but converted separately
4. Special tokens not preserved during conversion

**Debug**:
```python
# Verify tokenizer produces same output
train_tokenizer = AutoTokenizer.from_pretrained(train_model_path)
infer_tokenizer = AutoTokenizer.from_pretrained(infer_model_path)
test_text = "Hello, world!"
assert train_tokenizer.encode(test_text) == infer_tokenizer.encode(test_text)
```

## Weight Version Tracking

Miles tracks weight versions to detect staleness:

```python
# miles/backends/megatron_utils/actor.py:538-541
# miles/backends/experimental/fsdp_utils/actor.py:574-577
if engine_weight_version != updater_weight_version:
    logger.warning(f"Weight version mismatch: engine={engine_version} updater={updater_version}")
```

**When this warning fires**: The inference engine is serving with old weights. This can happen if:
- Weight update failed silently
- Network timeout during distribution
- Engine crashed and restarted with stale checkpoint

## FP8 Quantization During Sync

### Quantization Pipeline

**File**: `miles/backends/megatron_utils/megatron_to_hf/processors/__init__.py`

```python
def quantize_params(params, quant_method, **kwargs):
    if quant_method == "fp8":
        return quantizer_fp8.quantize(params, **kwargs)
    elif quant_method == "mxfp8":
        return quantizer_mxfp8.quantize(params, **kwargs)
    elif quant_method == "compressed-tensors":
        return quantizer_compressed_tensors.quantize(params, **kwargs)
```

### Which Weights Get Quantized

Only specific weight patterns are quantized:
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj` (or fused `qkv_proj`)
- MLP: `gate_proj`, `up_proj`, `down_proj` (or fused `gate_up_proj`)
- MoE experts: same as MLP but per-expert

**NOT quantized**: embeddings, LayerNorm, router gates, biases

### Debugging Quantization Quality

```python
import torch
from miles.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import quantize_to_fp8

# Before sync:
original_weight = model.state_dict()['layers.0.self_attn.q_proj.weight']

# After FP8 round-trip:
fp8_weight, scale = quantize_to_fp8(original_weight)
dequantized = fp8_weight.float() * scale

# Measure error:
rel_error = (original_weight.float() - dequantized).abs() / (original_weight.float().abs() + 1e-8)
print(f"FP8 error: mean={rel_error.mean():.6f} max={rel_error.max():.6f}")
# Expected: mean ~0.005, max ~0.05
```

## cos_sin_cache Mismatch (Miles #555)

The RoPE `cos_sin_cache` is a non-trainable buffer that diverges between engines. This is expected:

```python
# When using --check-weight-update-equal:
# Expected output:
# WARNING: cos_sin_cache max_abs_error=2.0, mean_abs_error=0.645
# This is NORMAL - cos_sin_cache is recomputed by each engine
```

**Do not** try to sync `cos_sin_cache`. It's recomputed from position embeddings and should be identical if the model config is the same.

## Debugging Weight Sync

### Quick Verification

```bash
# Check weight versions match
grep "weight_version" /tmp/miles_logs/*.log

# Dump weights from both sides for comparison
python3 -c "
import torch
train_sd = torch.load('megatron_checkpoint/model_state.pt')
infer_sd = torch.load('sglang_weights/model.safetensors')
for key in train_sd:
    if key in infer_sd:
        diff = (train_sd[key].float() - infer_sd[key].float()).abs().max()
        if diff > 1e-6:
            print(f'{key}: max_diff={diff:.6e}')
    else:
        print(f'{key}: MISSING in inference model')
"
```

### Monitor Weight Update Duration

```python
import time
start = time.time()
# ... weight update ...
duration = time.time() - start
print(f"Weight update took {duration:.1f}s")
# Expected: 5-30s depending on model size and quantization
# If > 60s: investigate network or memory issues
```

## SGLang Weight Sync Issues

### Hardcoded src=0 Deadlock (SGLang #19251)

**Symptom**: Weight sync deadlock when using multi-GPU torchrun training.

**Root cause**: `update_weights_from_distributed` hardcodes broadcast source to rank 0. With torchrun, NCCL prevents a process from being rank 0 in multiple independent groups simultaneously.

**Miles workaround**: Miles uses its own NCCL group management (`broadcast.py`, `p2p.py`), but new deployments using default path may hit this.

### NCCL Timeout Not Propagated to Subgroups (SGLang #21911)

**Symptom**: Weight sync times out even with `--dist-timeout` set high.

**Root cause**: `--dist-timeout` only applies to `init_process_group()`, not to `new_group()` calls. Custom weight-sync groups default to 600s regardless.

**Fix**: Set `NCCL_TIMEOUT` environment variable (applies globally).

### RDMA Weight Transfer (SGLang #17311, PR #21278 merged)

P2P weight update with zero-copy cross-host transfers. Key points:
- Parallelism mirroring between Megatron naming and SGLang naming
- `register_memory_region_v2` iterates CUDA memory snapshots — any PR changing memory layout may break this
- Already used by Miles in `p2p.py`

### Layer-wise Broadcasting (SGLang #21677)

Potential optimization: overlap weight transfers with inference by exploiting transformer's sequential layer processing. Working PoC exists.

## Related Skills

- `/debug-precision`: For FP8/INT4 quantization precision issues
- `/debug-logprob`: For logprob mismatch caused by weight quantization
- `/debug-hang`: For NCCL deadlocks during weight sync
- `/debug-distributed`: For TP/EP issues during weight distribution
