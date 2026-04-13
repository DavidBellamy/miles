# Precision Patterns Reference

Detailed reference for numerical precision issues in Miles/Megatron distributed training.

## 1. FP32 Enforcement Points in Miles

Miles enforces FP32 at these critical locations to prevent BF16 drift:

| Location | File | What It Does |
|----------|------|-------------|
| Logits to float | `model_provider.py:52` | `logits = logits.float()` in LinearForLastLayer |
| FSDP logits to float | `fsdp_utils/actor.py:361,532` | `logits = model(**args).logits.float()` |
| Logits assertion | `loss.py:65` | `assert logits.dtype == torch.float32` |
| KL log ratio | `ppo_utils.py:28` | `log_ratio = log_probs.float() - log_probs_base.float()` |
| Vocab-parallel entropy | `ppo_utils.py:162-194` | Custom autograd function in FP32 |
| Fused cross entropy | `ppo_utils.py:151-158` | Megatron's fused kernel handles precision internally |

## 2. BF16 vs FP16 Characteristics

| Property | BF16 | FP16 |
|----------|------|------|
| Exponent bits | 8 | 5 |
| Mantissa bits | 7 | 10 |
| Range | ~1e-38 to ~3e38 | ~6e-5 to ~6e4 |
| Precision | ~3 decimal digits | ~3.3 decimal digits |
| Overflow risk | Low (same range as FP32) | High (narrow range) |
| Underflow risk | Higher than FP16 | Lower |
| Typical error | ~0.8% per operation | ~0.05% per operation |

**Miles default**: BF16 (`miles/backends/megatron_utils/arguments.py:17: args.bf16 = not args.fp16`)

**BF16 compound error**: After N operations, expected relative error is ~0.008*sqrt(N). For a 24-layer transformer with ~200 operations per layer: ~0.008*sqrt(4800) ≈ 0.55. This explains why deep layers can show rel_diff > 0.5% even with correct implementations.

## 3. FP8 Quantization Details

### 3.1 FP8 Formats

| Format | Exponent | Mantissa | Range | Precision | Use Case |
|--------|----------|----------|-------|-----------|----------|
| E4M3 (float8_e4m3fn) | 4 | 3 | ±448 | ~1 decimal digit | Weights, activations |
| E5M2 (float8_e5m2) | 5 | 2 | ±57344 | ~0.5 decimal digits | Gradients |

### 3.2 Miles FP8 Weight Sync Pipeline

```
Megatron BF16 weights
     │
     ├── Per-tensor quantization (quantizer_fp8.py)
     │   scale = weight.abs().max() / FP8_MAX
     │   fp8_weight = (weight / scale).clamp(-FP8_MAX, FP8_MAX).to(float8_e4m3fn)
     │
     ├── Blockwise quantization (fp8_kernel.py)
     │   For each block (e.g., 128 elements):
     │     block_scale = block.abs().max() / FP8_MAX
     │     fp8_block = (block / block_scale).to(float8_e4m3fn)
     │
     └── MXFP8 quantization (quantizer_mxfp8.py)
         Uses sglang's mxfp8_group_quantize()
         Requires last dim divisible by 32
```

### 3.3 FP8 Common Issues

**Issue: FP8 param gather increases memory on Blackwell (Megatron #2063)**
- Expected to decrease memory, but increases by ~4GB when loading from checkpoint
- Hardware-generation-specific; validate on target hardware

**Issue: MXFP8 + no-dp-comm-overlap causes NaN (Megatron #2272)**
- `--fp8-param-gather` + `--reuse-grad-buf-for-mxfp8-param-ag` without comm overlap → NaN
- Guard: assertion prevents this config combo

**Issue: FP8 dispatch not supported with MoE training (Megatron #3578)**
- FP8 dispatch during MoE token routing needs special handling

### 3.4 INT4 Quantization (AWQ Style)

```
Miles INT4 pipeline (quantizer_compressed_tensors.py):
  Weight → Group quantize (group_size=128) → Pack 8 int4 into 1 int32
  Scales: FP16, per-group
  Zero points: INT4, packed into INT32
```

Expected INT4 error: ~2-5% relative error per weight element. Cumulative effect on model quality depends on model and task.

### 3.5 NVFP4 Quantization

4-bit NVIDIA format with global + block-level scales. Even lower precision than INT4 but with hardware-accelerated dequantization.

## 4. Hardware-Specific Issues

### 4.1 Blackwell/GB200 Kernel Race (Megatron #2597)

TransformerEngine's NvJet GEMM combined with cuDNN RMSNorm causes kernel overlap without proper CUDA stream synchronization on GB200/B200 GPUs.

**Symptom**: Sporadic NaN in gradient norms
**Root cause**: RMSNorm kernel reads input before upstream GEMM finishes writing
**Fix**: Update TE or set `NVTE_NVJET=0`

### 4.2 CUDA Stream Synchronization (Megatron #2301)

**Symptom**: NaN at iteration 1-2 with DP only
**Root cause**: Parameter copy during DDP module creation runs on separate CUDA stream without `wait_stream()`. Forward pass reads uninitialized/partially-copied parameters.

```python
# The fix pattern:
torch.cuda.current_stream().wait_stream(param_copy_stream)
```

### 4.3 Multi-Node NCCL + torch_memory_saver (Miles #806)

On GB200, `torch_memory_saver`'s LD_PRELOAD GPU memory offload relocates GPU buffers, invalidating NCCL's registered addresses for cross-node communication.

## 5. Gradient Precision

### 5.1 FP32 Gradient Accumulation

```bash
--accumulate-allreduce-grads-in-fp32  # ALWAYS use this
```

Without this flag, gradient allreduce in BF16 causes:
- Gradients near zero get rounded to exactly zero
- Large gradient magnitudes lose precision
- DP ranks accumulate different rounding errors → divergence

### 5.2 Gradient NaN Detection

Miles (`model.py:451-461`):
```python
found_inf_flag = optimizer.prepare_grads()
if found_inf_flag:
    valid_step = False  # skip optimizer update
else:
    grad_norm = optimizer.get_grad_norm()
    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        valid_step = False
```

### 5.3 Memory-Aware Gradient Finalization

Miles has a workaround for OOM during gradient finalization (`model.py:492-498`):
```python
free, total = torch.cuda.mem_get_info(device)
if free / total < 0.1:
    clear_memory()  # emergency cache cleanup
```

## 6. Loss Scaling with Parallelism

Different loss types scale differently with parallelism dimensions:

```python
# Standard loss scaling (loss.py:902-913):
loss_scale = num_microbatches / global_batch_size * dp_cp_size

# Per-token loss with CP:
loss = loss * cp_size  # compensate for token splitting

# CP allgather deadlock prevention (loss.py:900):
loss = loss + 0 * logits.sum()  # forces autograd traversal on all CP ranks
```

**Why the `0 * logits.sum()` trick**: Without this, the backward pass for the CP gather's reduce-scatter may not be triggered on all CP ranks, causing a deadlock. Adding a zero-valued dependency on logits ensures all ranks participate.

## 7. Attention Backend Precision Comparison

| Backend | Precision | Notes |
|---------|-----------|-------|
| FlashAttention v2 | ~0.001 rel_diff from reference | Tiled computation, BF16 accumulation |
| FlashInfer | ~0.001 rel_diff | SGLang default, slightly different tiling |
| cuDNN attention | ~0.0005 rel_diff | More precise but slower |
| Vanilla PyTorch | ~0 (reference) | Full FP32, no tiling |
| Megatron core attention | ~0.001 rel_diff | Depends on `--attention-backend` flag |

Switching between backends can cause rel_diff of 0.001-0.005 even with identical inputs.

## 8. Dynamic Filtering Precision Bug (Miles #570)

**Symptom**: `torch.tensor([0.25]*16, dtype=torch.float).std() > 0.0` returns `False` (should be True in theory, but FP32 rounding makes std exactly 0.0).

**Fix**: Use `torch.float64` for statistical computations and compare against epsilon instead of zero.

```python
# WRONG
if rewards.std() > 0.0:  # fails for some "identical" values in FP32

# RIGHT  
if rewards.to(torch.float64).std() > 1e-8:
```

## 9. Debugging Precision Issues with Dumper

### Compare FP32 vs BF16 Outputs

```bash
# Run 1: Force FP32 where possible
DUMPER_ENABLE=1 DUMPER_DIR=/tmp/fp32_run python ... --attention-softmax-in-fp32

# Run 2: Default BF16
DUMPER_ENABLE=1 DUMPER_DIR=/tmp/bf16_run python ...

# Compare
python -m sglang.srt.debug_utils.comparator \
  --baseline /tmp/fp32_run/ --target /tmp/bf16_run/ \
  --preset sglang_dev --verbosity verbose
```

### Layer-by-Layer Drift Analysis

```python
import json
with open('comparator_report.jsonl') as f:
    results = [json.loads(l) for l in f]

# Group by layer and plot drift
layer_diffs = {}
for r in results:
    if r.get('type') == 'comparison_tensor':
        name = r['name']
        layer = r.get('layer_id', 'unknown')
        diff = r.get('diff', {}).get('rel_diff', 0)
        layer_diffs.setdefault(layer, []).append((name, diff))

for layer in sorted(layer_diffs.keys()):
    diffs = layer_diffs[layer]
    avg_diff = sum(d for _, d in diffs) / len(diffs)
    print(f"Layer {layer}: avg_rel_diff={avg_diff:.6f} ({len(diffs)} tensors)")
```
