---
name: debug-moe
description: Use when debugging Mixture-of-Experts (MoE) specific issues including expert routing, AllToAll dispatch, R3 (Rollout Routing Replay), expert parallelism (EP), expert tensor parallelism (ETP), token dispatch mismatch, expert load imbalance, MoE auxiliary loss, routing logit divergence between SGLang and Megatron, or DeepEP/HybridEP issues. Also trigger on moe_topk_ids mismatch, dispatch deadlock, or expert output shape errors.
---

# Debug MoE (Mixture-of-Experts) Issues

MoE models introduce unique debugging challenges due to discrete routing decisions, AllToAll communication, and expert parallelism.

## MoE Architecture in Miles

```
Hidden states → Router (TopK gating) → Token dispatch (AllToAll) → Expert compute → Combine (AllToAll) → Output
                    │                         │                         │
              routing logits            dispatch to EP ranks       per-expert MLP
              topk expert IDs           variable tokens/rank       potentially ETP-sharded
```

## Quick Decision Tree

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| MoE routing differs between SGLang/Megatron | BF16 router logit precision | Enable R3 (`--use-rollout-routing-replay`) |
| R3 spiky training | Read-only numpy array corruption | Add `.copy()` after `np.frombuffer()` (Miles #599) |
| R3 shape validation fails after truncation | `rollout_routed_experts` not truncated | Fix `_truncate_sample_output()` (Miles #861) |
| AllToAll hang with EP > 1 | Token count mismatch across ranks | Dump dispatch counts, verify symmetry |
| RDMA assertion failure with HybridEP | Token count exceeds RDMA queue depth (65535) | Increase TP/CP to reduce tokens/rank |
| Expert output shape mismatch | num_experts not divisible by EP | Fix config: `num_experts % EP == 0` |
| Loss spikes with MoE aux loss | Aux loss scale too high or EP misconfiguration | Tune `--moe-aux-loss-coeff` |
| Different MoE behavior on different hardware | Kernel implementation differences | Compare with dumper |
| KeyError in HybridDeviceOptimizer | Tensor identity used as key, broken after wrapping | Use stable indices (Megatron #4042) |

## MoE Routing Divergence (The #1 MoE Problem)

### Why SGLang and Megatron Route Differently

The router computes logits and takes top-K. Even tiny numerical differences can flip the routing:

```python
# Example: router logits for 8 experts
sglang_logits = [0.51, 0.49, 0.30, 0.20, ...]  # top-2: experts 0, 1
megatron_logits = [0.49, 0.51, 0.30, 0.20, ...]  # top-2: experts 1, 0 ← DIFFERENT!
```

This is NOT a bug — it's inherent BF16 precision. But it causes completely different outputs for those tokens.

### R3: Rollout Routing Replay

R3 records routing decisions from SGLang and replays them in Megatron:

```bash
--use-rollout-routing-replay   # Replay SGLang routing in Megatron
--use-miles-router              # Required for R3
```

**How R3 works**:
1. SGLang generates rollout → stores `topk_ids` per token per layer
2. Megatron forward pass → loads stored `topk_ids` instead of recomputing
3. Same experts process same tokens → much closer logprobs

**R3 Data flow**:
```
SGLang: router_logits → topk_ids → stored in rollout_routed_experts tensor
                                     shape: [seq_len-1, num_layers, moe_router_topk]
Megatron: router → SKIP topk → use stored topk_ids directly
```

### R3 Known Bugs

**Bug 1: Read-only NumPy arrays (Miles #599)**
```python
# WRONG: np.frombuffer returns read-only
routing_data = np.frombuffer(buffer, dtype=np.int32)
tensor = torch.from_numpy(routing_data)  # non-writable → undefined behavior!

# RIGHT: copy to make writable
routing_data = np.frombuffer(buffer, dtype=np.int32).copy()
tensor = torch.from_numpy(routing_data)
```

**Bug 2: Missing truncation (Miles #861)**
When `_truncate_sample_output()` truncates tokens for max_seq_len, it must also truncate `rollout_routed_experts` along the sequence dimension.

**Bug 3: EP != actor-num-gpus-per-node (Miles #599)**
When `--expert-model-parallel-size` doesn't match `--actor-num-gpus-per-node`, R3 data layout may be incorrect.

## Expert Parallelism (EP) Debugging

### EP Process Group Topology

```python
# With EP=4, TP=2, total 8 GPUs:
# EP groups: {0,2,4,6}, {1,3,5,7}  (each group has 4 members)
# TP groups: {0,1}, {2,3}, {4,5}, {6,7}
# Each EP rank hosts num_experts/EP experts
```

### Common EP Bugs

**AllToAll Token Dispatch Mismatch**:
```python
# Each rank sends tokens to experts on other ranks
# AllToAll requires ALL ranks to participate even with 0 tokens
# Bug: skipping AllToAll when local dispatch count is 0

# Debug: dump dispatch counts
dumper.dump('dispatch_counts', 
            tokens_per_expert,  # shape: [num_experts]
            dims='num_experts # ep:replicated')
```

**RDMA Queue Depth Overflow (Megatron #3999)**:
```
With large batch + long seq: max_tokens_per_rank can exceed 65535
RDMA queue depth = 3 * max_tokens + 1 must be < 65536
Fix: increase TP or CP to reduce tokens per rank
```

**HybridDeviceOptimizer KeyError (Megatron #4042)**:
```python
# Optimizer uses tensor identity as dict key
# MixedPrecisionOptimizer replaces tensor objects → keys become stale
# Fix: use parameter indices instead of tensor identity
```

## MoE Token Dispatcher Types

| Dispatcher | Communication | Memory | Use Case |
|-----------|--------------|--------|----------|
| `alltoall` | AllToAll | Lower | Default, general purpose |
| `allgather` | AllGather | Higher (full copy) | Simpler, for small models |

```bash
--moe-token-dispatcher-type alltoall  # Default
```

## MoE Dims Annotations

### THD Format (Megatron)
```
# Router logits (replicated across all axes except CP/SP):
dims='t[cp:zigzag,sp] 1 num_experts # tp:replicated ep:replicated'

# Top-K IDs:
dims='t[cp:zigzag,sp] topk # tp:replicated ep:replicated'

# Expert output (partial sum from ETP):
dims='t h[tp:partial] # ep:replicated etp:replicated'
```

### SGLang
```
# Router logits:
dims='t num_experts # tp:replicated'

# Top-K IDs:
dims='t topk # tp:replicated'
```

### Important: ETP == TP orthogonality
When `etp == tp`, do NOT declare both `tp:replicated` and `etp:replicated` — the comparator treats them as the same axis and throws an orthogonality error.

## MoE Source Patcher Config

For debugging MoE internals in Megatron:

```yaml
patches:
  - target: megatron.core.transformer.moe.router.TopKRouter.forward
    edits:
      - match: "logits = self.gating(input)"
        append: "dumper.dump('moe_router_logits', logits, dims='t[cp:zigzag,sp] 1 num_experts # tp:replicated ep:replicated')"
      - match: "return probs, routing_map"
        prepend: "dumper.dump('moe_topk_ids', routing_map.int().topk(k=self.topk, dim=-1).indices.sort(dim=-1).values, dims='t[cp:zigzag,sp] topk # tp:replicated ep:replicated')"
```

## MoE Auxiliary Loss Debugging

MoE uses auxiliary losses for load balancing:

```bash
--moe-aux-loss-coeff 0.01  # Scale of aux loss
```

**Issue**: Aux loss scales differently with CP. With CP, `num_microbatches` is multiplied by CP size, but gradient scaling uses `1.0/data_parallel_world_size`. The MTP computes per-token mean before scaling, which is correct (Megatron #3943).

## Related Skills

- `/debug-distributed`: For EP/TP interaction issues
- `/debug-logprob`: For routing-induced logprob mismatch
- `/debug-hang`: For AllToAll deadlocks
- `/dumper-usage`: For capturing MoE tensor dumps
