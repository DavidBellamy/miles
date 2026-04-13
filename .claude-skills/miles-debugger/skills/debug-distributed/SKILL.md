---
name: debug-distributed
description: Use when debugging distributed training bugs in Miles/Megatron/SGLang, especially TP/PP/EP/CP issues. Trigger on tensor shape mismatch, numerical divergence between inference and training, parallelism configuration errors, AllReduce/AllGather/AllToAll failures, or when a user mentions debugging Megatron, distributed training, or parallelism correctness. Also use when someone wants to validate a new parallelism configuration or compare activations across different setups.
---

# Distributed Training Debugger

Debug TP (Tensor Parallel), PP (Pipeline Parallel), EP (Expert Parallel), CP (Context Parallel), and SP (Sequence Parallel) issues in Megatron-based training within the Miles framework.

## Quick Decision Tree

Start here. Match the symptom to the diagnosis path:

| Symptom | Likely Cause | Go To |
|---------|-------------|-------|
| `RuntimeError: shape mismatch` or `size mismatch` | Wrong tensor slicing for parallelism config | @references/common_bugs.md Section 1 |
| Training loss NaN/Inf after adding parallelism | Incorrect allreduce/partial-sum handling | @references/common_bugs.md Section 2 |
| Training hangs (no progress, no error) | Collective op deadlock or mismatched process groups | @references/debugging_recipes.md Section 3 |
| Inference-training mismatch (different outputs) | Activation divergence between SGLang and Megatron | @references/debugging_recipes.md Section 1 |
| `NCCL error` or `NCCL timeout` | Process group misconfiguration or OOM | @references/common_bugs.md Section 4 |
| Validation loss much worse than expected | Silent numerical corruption in parallel ops | @references/debugging_recipes.md Section 2 |
| Shape error only at certain TP/EP sizes | num_kv_heads < TP or num_experts % EP != 0 | @references/common_bugs.md Section 1.1 |

## Debugging Methodology

### Step 1: Reproduce Minimally

Before using the dumper, narrow down the issue:

```bash
# Test with minimal parallelism first (TP=1, PP=1, CP=1)
# Then enable one axis at a time to isolate which causes the bug
--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1

# Then try:
--tensor-model-parallel-size 2  # just TP
--context-parallel-size 2       # just CP
# etc.
```

### Step 2: Use the Dumper

The dumper captures intermediate tensors during training/inference. Use `/dumper-usage` for full API reference.

**Quick start for debugging:**

1. Write a source patcher YAML to inject dumps at strategic points (see examples in @references/debugging_recipes.md)
2. Run with `--dumper-enable --dumper-dir /tmp/debug_dumps`
3. Compare dumps between working config (e.g., TP=1) and broken config (e.g., TP=2)

### Step 3: Analyze with Comparator

```bash
python -m sglang.srt.debug_utils.comparator \
  --baseline-path /tmp/debug_dumps/working/ \
  --target-path /tmp/debug_dumps/broken/ \
  --preset sglang_megatron \
  --diff-threshold 0.001 \
  --output-format json
```

### Step 4: Trace to Root Cause

Read the comparator output. The first tensor that shows significant divergence (rel_diff > threshold) is usually near the bug. Trace that tensor's code path.

## Key Files in Miles/Megatron

| File | What It Does |
|------|-------------|
| `miles/backends/megatron_utils/model.py` | Training loop: `forward_only()` and `train_one_step()` |
| `miles/backends/megatron_utils/actor.py` | Actor initialization, source patching |
| `miles/utils/dumper_utils.py` | Dumper integration: `DumperMegatronUtil`, `DumperPhase` |
| `miles/utils/arguments.py` | CLI args including `--dumper-*` flags |
| `miles/ray/rollout.py` | SGLang rollout, passes dumper env vars |
| `miles/ray/actor_group.py` | Actor group, passes source patcher config |
| `miles_plugins/models/hf_attention.py` | HF attention with CP relayout support |
| `miles_plugins/models/cp_utils.py` | CP utilities for relayout |
| `tests/e2e/conftest_dumper.py` | Reference source patcher configs for Megatron + SGLang |

## Megatron Parallelism Axes Quick Reference

| Axis | What It Shards | Key Config | Common Pitfall |
|------|---------------|------------|----------------|
| **TP** | Attention heads, MLP columns/rows | `--tensor-model-parallel-size` | num_kv_heads must be divisible by TP (or handled with replication) |
| **SP** | Sequence dim (allgather before attn, reduce-scatter after) | `--sequence-parallel` | Always paired with TP; affects LayerNorm inputs |
| **PP** | Model layers across stages | `--pipeline-model-parallel-size` | Schedule must match (1F1B, interleaved); loss computed only on last stage |
| **CP** | Sequence split + zigzag interleave | `--context-parallel-size` | Zigzag reorder needed; seq_lens splitting for packed format |
| **EP** | MoE experts across ranks | `--expert-model-parallel-size` | AllToAll token dispatch; expert counts must divide evenly |
| **ETP** | Expert tensor parallel | `--expert-tensor-parallel-size` | Nested within EP; affects MoE MLP sharding |

## Related Skills — Complete Debugging Toolkit

| Skill | When to Use |
|-------|------------|
| `/dumper-usage` | Dumper API, comparator CLI, source patcher format |
| `/dumper-dims` | Dims annotation syntax for tensor sharding |
| `/debug-shape` | Tensor shape mismatch tracing |
| `/debug-hang` | Distributed hang / deadlock diagnosis |
| `/debug-precision` | NaN/Inf, FP8/BF16 drift, quantization errors |
| `/debug-logprob` | Log-prob mismatch, KL divergence, PPO issues |
| `/debug-weight-sync` | Weight update/sync between Megatron and SGLang |
| `/debug-colocate` | Ray actor, colocate mode, GPU contention, timeout cascades |

## References

For detailed debugging procedures and common bug patterns:
- @references/common_bugs.md - Catalog of common TP/PP/EP/CP bugs with symptoms and fixes
- @references/debugging_recipes.md - Step-by-step debugging procedures for specific scenarios
- @references/megatron_internals.md - How Megatron implements each parallelism axis
