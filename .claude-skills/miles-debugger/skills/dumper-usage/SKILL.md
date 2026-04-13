---
name: dumper-usage
description: Must read when working with Dumper or Dump Comparator for capturing and comparing intermediate tensors. Use when instrumenting training/inference code with dump calls, running the comparator CLI, writing source patcher YAML configs, or debugging tensor value mismatches between SGLang inference and Megatron training. Also trigger when user mentions dumper, dump comparator, tensor comparison, activation comparison, or source patching.
---

# Dumper & Dump Comparator Usage Guide

A debugging toolkit for dumping and comparing intermediate tensors across runs, hardware, frameworks, and parallelism configurations.

## Overview

The dumper captures intermediate tensor values during training/inference. The comparator aligns and compares tensors from two runs, handling distributed sharding automatically.

### Three Usage Modes

1. **Non-intrusive**: `DUMPER_ENABLE=1 DUMPER_NON_INTRUSIVE_MODE=all` - dump module inputs/outputs without code changes
2. **One-liner**: Add `dumper.dump("name", tensor)` calls at specific points
3. **Annotated-one-liner**: Add dims annotation `dumper.dump("name", tensor, dims="t h[tp]")` for cross-parallelism comparison

### Miles Integration

Miles wraps the dumper with CLI flags:
```bash
--dumper-enable                              # Master enable switch
--dumper-dir /tmp/dumper                     # Output directory
--dumper-inference 'filter=layer_id < 3'     # SGLang inference phase config
--dumper-fwd-only 'filter=layer_id < 3'     # Megatron forward-only phase config
--dumper-fwd-bwd 'filter=layer_id < 3'      # Megatron forward-backward phase config
--dumper-source-patcher-config-inference patcher.yaml   # SGLang source patcher
--dumper-source-patcher-config-train patcher.yaml       # Megatron source patcher
```

### Quick Recipes

**Compare SGLang inference vs Megatron training:**
```bash
python -m sglang.srt.debug_utils.comparator \
  --baseline-path /tmp/dumper/engines/ \
  --target-path /tmp/dumper/fwd_bwd/ \
  --preset sglang_megatron \
  --diff-threshold 0.001
```

**Override dims without re-running (fast iteration):**
```bash
python -m sglang.srt.debug_utils.comparator \
  --baseline ... --target ... \
  --override-dims "attn_pre_o_proj:t (num_heads*head_dim)[tp]"
```

**Filter dumps to specific layers:**
```bash
DUMPER_FILTER='layer_id is None or layer_id < 3 or layer_id == 24'
```

**Examine raw dump files:**
```python
import torch
data = torch.load("step=0___rank=0___dump_index=1___name=hidden_states.pt", weights_only=False)
print(data["value"].shape)  # tensor shape
print(data["meta"])         # metadata: step, rank, dims, parallel_info, etc.
```

## Related Skills

- `/dumper-dims`: Dims annotation syntax and examples
- `/debug-distributed`: Using dumper to debug TP/PP/EP/CP bugs

## Manuals

Detailed API and CLI references:
- @references/dumper_manual.md - Dumper core API: `dump()`, `step()`, `dump_model()`, filtering, config, HTTP control, output format
- @references/dump_comparator_manual.md - Comparator CLI flags, presets, alignment pipeline (unsharder/reorderer/token aligner/axis aligner), output formats
- @references/source_patcher_manual.md - YAML config format, matching rules, edit modes, auto import injection

## Source Patcher Quick Reference

Inject `dumper.dump()` calls without editing source files:

```yaml
patches:
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention
    edits:
      - match: "inference_context = deprecate_inference_params(inference_context, inference_params)"
        append: "dumper.dump('layer_input', hidden_states, dims='t[cp:zigzag,sp] 1 h')"
```

Run with: `DUMPER_SOURCE_PATCHER_CONFIG=patch.yaml DUMPER_ENABLE=1 python ...`

The `from sglang.srt.debug_utils.dumper import dumper` import is auto-injected by the patcher.

## Reference Source Patcher Configs

Production-tested configs exist in `tests/e2e/conftest_dumper.py`:
- `MEGATRON_SOURCE_PATCHER_CONFIG_YAML` - THD format (packed tokens)
- `MEGATRON_SOURCE_PATCHER_CONFIG_BSHD_YAML` - BSHD format (batch x sequence)
- `SGLANG_SOURCE_PATCHER_CONFIG_YAML` - SGLang Qwen3 MoE model

These cover: `layer_input`, `attn_output`, `attn_q`, `attn_v`, `attn_pre_o_proj`, `pre_mlp_residual`, `pre_mlp_layernorm_output`, `mlp_output`, `moe_router_logits`, `moe_topk_ids`.
