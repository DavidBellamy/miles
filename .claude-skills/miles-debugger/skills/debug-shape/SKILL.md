---
name: debug-shape
description: Use when debugging tensor shape mismatches in distributed training, especially RuntimeError about size/shape mismatch, view size incompatible, or mat1 and mat2 shapes cannot be multiplied. Trigger on shape errors in Megatron, Miles, or any TP/PP/EP/CP configuration. Also use when tracing expected tensor shapes through a transformer model with specific parallelism settings.
---

# Debug Tensor Shape Mismatches

Systematic approach to diagnosing tensor shape errors in distributed Megatron training.

## Quick Diagnosis

### Step 1: Read the Error

Shape errors always tell you the two incompatible shapes. Extract them:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (4096x512 and 1024x4096)
# mat1 is [4096, 512] but weight expects first dim = 1024
# This means hidden_size got divided by TP somewhere it shouldn't have been
```

### Step 2: Identify the Layer

The traceback tells you which module failed. Map it to the parallelism axis:

| Module Pattern | Likely Axis | What to Check |
|---------------|-------------|---------------|
| `self_attention.linear_qkv` | TP | num_heads divisible by TP? |
| `self_attention.linear_proj` | TP | Output projection input size |
| `mlp.linear_fc1` / `mlp.linear_fc2` | TP | FFN intermediate size divisible by TP? |
| `moe.experts` | EP/ETP | num_experts divisible by EP? |
| `embedding` / `output_layer` | TP | vocab_size divisible by TP? |
| `attention` with CP | CP | seq_len handling in packed vs padded format |

### Step 3: Compute Expected Shapes

For a model with hidden_size=H, num_heads=N, head_dim=D, num_kv_heads=K:

**With TP=T:**
| Tensor | Shape per rank |
|--------|---------------|
| Q projection weight | `[H, (N/T)*D]` |
| K projection weight | `[H, (K/T)*D]` if K >= T, else special handling |
| V projection weight | `[H, (K/T)*D]` if K >= T, else special handling |
| O projection weight | `[(N/T)*D, H]` |
| MLP gate/up weight | `[H, FFN/T]` |
| MLP down weight | `[FFN/T, H]` |

**Critical: When num_kv_heads < TP**

This is the #1 source of shape bugs. If K < T, you CANNOT simply divide K by T. Solutions:
1. **GQA replication**: Replicate KV heads so effective K = T (Megatron's approach)
2. **Reduce TP**: Use TP <= K
3. **Handle in attention**: The `attention_output_gate` and similar must account for this

See @references/shape_patterns.md for full shape tables and the KV-head edge case.

## Shape Tracing Methodology

When the error isn't obvious, trace shapes through the model:

### 1. Dump shapes at each layer boundary

```python
# Quick shape debugging (add temporarily)
dumper.dump('debug_shape', tensor, dims=f'shape={list(tensor.shape)}')
```

Or use the source patcher to inject shape logging:
```yaml
patches:
  - target: megatron.core.transformer.attention.Attention.forward
    preamble: "import logging; _log = logging.getLogger(__name__)"
    edits:
      - match: "query = query.reshape(query.size(0)"
        prepend: "_log.warning(f'SHAPE_DEBUG attn query={query.shape} key={key.shape} value={value.shape}')"
```

### 2. Compare shapes across ranks

Different TP ranks should have the same shapes for sharded tensors (just different data). If shapes differ across ranks, something is wrong with the model config distribution.

```bash
# Quick check: dump shapes from all ranks
for f in /tmp/dumper/fwd_only/step=0___rank=*___name=attn_q*.pt; do
  python3 -c "import torch; d=torch.load('$f',weights_only=False); print(f'{f}: {d[\"value\"].shape}')"
done
```

### 3. Check the model config

```python
# In Megatron, these must be consistent:
args.hidden_size           # H
args.num_attention_heads   # N (must be divisible by TP)
args.num_query_groups      # K (num_kv_heads, GQA)
args.ffn_hidden_size       # FFN
args.tensor_model_parallel_size  # TP
```

## Common Shape Bugs and Fixes

| Bug | Symptom | Fix |
|-----|---------|-----|
| num_kv_heads not divisible by TP | Shape error in KV projection | Use GQA replication or reduce TP |
| FFN size not divisible by TP | Shape error in MLP | Pad FFN or adjust TP |
| vocab_size not divisible by TP | Shape error in embedding/output | Pad vocab_size |
| Wrong qkv_format (THD vs BSHD) | Shape error in attention | Check `--qkv-format` flag matches model code |
| CP relayout missing | Shape error when CP > 1 | Ensure `cp_utils.relayout()` is called |
| PP layer assignment wrong | Missing layers or duplicate layers | Check `pipeline_model_parallel_split_rank` |

## Related Skills

- `/debug-distributed`: Full distributed debugging methodology
- `/dumper-usage`: How to capture tensor shapes with the dumper
- `/dumper-dims`: Annotation syntax for describing expected shapes

## References

- @references/shape_patterns.md - Complete shape tables for all Megatron layer types with parallelism
