---
name: dumper-dims
description: Use when writing or debugging dims annotation strings for Dumper or Dump Comparator. Trigger when working with tensor sharding descriptions like "t h[tp]", "s[cp:zigzag,sp] b h", or any dims= parameter in dumper.dump() calls. Also use when diagnosing comparator alignment failures, writing source patcher YAML with dims, or understanding how tensors are sharded across TP/CP/EP/SP axes.
---

# Dumper Dims Annotation

## What is it

The `dims` annotation is a string inside dumper calls that describes the shape and sharding of a tensor:

```python
dumper.dump("my_name", my_tensor, dims="t h[tp]")
```

The comparator uses dims to automatically unshard, reorder, and align tensors from different ranks before comparison.

## How to Write Dims

Read @references/dims_reference.md for the full BNF grammar, modifier reference, and comprehensive real-world examples.

### Quick Syntax Guide

```
dims_string = dims_part [ "#" comment_part ]

# Examples:
"t h"                           # token dim, hidden dim (no sharding)
"t h[tp]"                       # hidden dim sharded across TP
"t h[tp:partial]"               # partial sum before allreduce (needs reduce-sum)
"t[cp:zigzag,sp] 1 h"          # token dim: CP zigzag + SP sharded; singleton batch; hidden
"s[cp:zigzag,sp] b h"          # BSHD: seq dim sharded, batch dim, hidden dim
"t (num_heads*head_dim)[tp]"   # fused dims, TP-sharded
"t h # tp:replicated ep:replicated"  # replicated (identical) across TP and EP
"t h # dp:=attn_dp"            # data parallel group alias
```

### Key Rules

1. **Dimension names** are arbitrary: `h`, `d`, `num_heads`, `head_dim`, `vocab_size`
2. **Special dims**: `t` = num_tokens, `b` = batch_size, `s` = seq_lens
3. **`1`** = singleton dim (squeezed by comparator)
4. **Modifiers** in `[]`: `tp`, `cp`, `ep`, `sp` with qualifiers `partial`, `sharded`, `zigzag`, `natural`
5. **Comment section** after `#`: `tp:replicated`, `dp:=group_name`
6. **Fused dims**: `(num_heads*head_dim)` for dimensions stored as one physical dim

### Common Patterns

| Tensor Type | THD dims | BSHD dims |
|------------|----------|-----------|
| Layer hidden state (replicated) | `t[cp:zigzag,sp] 1 h # tp:replicated` | `s[cp:zigzag,sp] b h # tp:replicated` |
| Attention Q (TP-sharded heads) | `t[cp:zigzag,sp] num_heads[tp] head_dim` | `s[cp:zigzag,sp] b num_heads[tp] head_dim` |
| Attention output (partial sum) | `t h[tp:partial]` | `s b h[tp:partial]` |
| Fused QKV (SGLang) | `t (num_heads*head_dim)[tp]` | - |
| MoE router logits | `t[cp:zigzag,sp] 1 num_experts # tp:replicated` | `s[cp:zigzag,sp] b num_experts # tp:replicated` |

## How to Debug Dims (Do NOT Re-run)

Do NOT re-run expensive training when only dims annotations change. Use `--override-dims`:

```bash
python -m sglang.srt.debug_utils.comparator \
  --baseline ... --target ... \
  --override-dims "attn_pre_o_proj:t (num_heads*head_dim)[tp]" \
  --override-baseline-dims "layer_input:t h # dp:=attn_dp"
```

Or use a YAML override config file:
```yaml
overrides:
  - match: "attn_.*"
    dims: "t (num_heads*head_dim)[tp]"
    side: target
```

## Related Skills

- `/dumper-usage`: Full dumper and comparator reference
- `/debug-distributed`: Using dims for distributed debugging
