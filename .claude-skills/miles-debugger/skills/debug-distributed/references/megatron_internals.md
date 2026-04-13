# Megatron Parallelism Internals

How Megatron implements each parallelism axis, and what to know when debugging.

## 1. Tensor Parallelism (TP)

### Implementation

Megatron TP shards linear layers into `ColumnParallelLinear` and `RowParallelLinear`:

```
ColumnParallelLinear: Weight [H, H_out/TP] — each rank computes partial output
RowParallelLinear:    Weight [H_in/TP, H] — each rank has partial input, allreduce output
```

**Attention TP sharding**:
- QKV projection: ColumnParallel → each rank gets `num_heads/TP` heads
- Output projection: RowParallel → allreduce combines partial attention outputs
- With GQA: `num_kv_heads` may be < TP → Megatron replicates KV heads via `adjust_key_value_for_gqa()`

**MLP TP sharding**:
- Gate/Up: ColumnParallel → each rank gets `ffn_hidden_size/TP` columns
- Down: RowParallel → allreduce combines partial MLP outputs

### Key Code Paths

```
megatron/core/tensor_parallel/layers.py
  ├── ColumnParallelLinear.forward()  → scatter input to TP ranks
  ├── RowParallelLinear.forward()     → allreduce/reduce-scatter output
  └── VocabParallelEmbedding.forward() → vocab sharded across TP

megatron/core/transformer/attention.py
  ├── Attention.__init__()  → creates QKV (ColumnParallel) + O (RowParallel)
  ├── adjust_key_value_for_gqa() → replicates KV heads when num_kv_heads < TP
  └── Attention.forward()  → QKV split, RoPE, core attention, O projection

megatron/core/transformer/mlp.py
  ├── MLP.__init__()  → gate/up (ColumnParallel) + down (RowParallel)
  └── MLP.forward()   → SwiGLU/GELU activation between column and row
```

### Communication Pattern

Per transformer layer with TP (no SP):
1. AllGather input for ColumnParallel (QKV, gate/up)
2. AllReduce output from RowParallel (O proj, down proj)

Per transformer layer with TP + SP:
1. AllGather before attention/MLP (gather sequence dim)
2. ReduceScatter after attention/MLP (scatter sequence dim)

## 2. Sequence Parallelism (SP)

### Implementation

SP splits the **token/sequence dimension** across TP ranks between attention and MLP operations. This reduces memory for activations that don't need to be replicated (LayerNorm, dropout).

```
[AllGather seq] → Attention → [ReduceScatter seq] → LayerNorm → [AllGather seq] → MLP → [ReduceScatter seq]
```

Between attention/MLP, activations are `[seq_len/TP, batch, hidden]` (SP-split).
During attention/MLP, activations are `[seq_len, batch, hidden/TP]` (TP-split on hidden).

### Key Detail

SP is always paired with TP. The `--sequence-parallel` flag only works with `--tensor-model-parallel-size > 1`.

### Dims Annotation

With SP, the token dimension has the `sp` modifier:
```
# Between attention/MLP (SP-split):
dims='t[sp] 1 h # tp:replicated'

# With CP too:
dims='t[cp:zigzag,sp] 1 h # tp:replicated'
```

## 3. Pipeline Parallelism (PP)

### Implementation

PP splits model layers across GPU groups. Each PP stage runs a subset of layers.

**Schedule types**:
- **1F1B** (One Forward One Backward): Default schedule. Steady state alternates F and B micro-batches.
- **Interleaved**: Each stage has multiple non-contiguous layer chunks (virtual pipeline parallelism).

### Layer Assignment

```python
# Default even split:
layers_per_stage = num_layers // PP

# Stage i gets layers:
start = i * layers_per_stage
end = (i + 1) * layers_per_stage

# Special: embedding on stage 0, output_layer on last stage
```

### Communication

PP stages communicate via point-to-point send/recv:
- Forward: stage i sends activations to stage i+1
- Backward: stage i+1 sends gradients to stage i

**Critical**: All PP stages must execute the same number of micro-batches. This means:
```
num_microbatches = global_batch_size / (DP * micro_batch_size)
# Must be an integer >= PP (for 1F1B warmup)
```

### Key Code Paths

```
megatron/core/pipeline_parallel/schedules.py
  ├── forward_backward_pipelining_with_interleaving()
  ├── forward_backward_pipelining_without_interleaving()  → 1F1B
  └── forward_backward_no_pipelining()  → PP=1

megatron/core/pipeline_parallel/p2p_communication.py
  ├── send_forward() / recv_forward()
  └── send_backward() / recv_backward()
```

### Debugging PP

- Loss is only computed on the last PP stage
- Intermediate activations between stages are communicated tensors — dump them to verify
- PP hangs are usually micro-batch count mismatches

## 4. Context Parallelism (CP)

### Implementation

CP splits the **sequence dimension** across ranks, allowing very long sequences. Unlike SP (which uses AllGather/ReduceScatter), CP uses **ring attention** patterns.

**Zigzag interleaving**: For load balancing, sequences are split into 2*CP chunks and assigned in zigzag order:
```
Natural:  [chunk0, chunk1, chunk2, chunk3]  (CP=2)
Zigzag:   rank0=[chunk0, chunk3], rank1=[chunk1, chunk2]
```

This balances compute because causal attention has more work for later chunks.

### Communication

- Before attention: AllGather KV across CP ranks (or ring-style send/recv)
- After attention: ReduceScatter or reduce results

### Key Code Paths

```
megatron/core/transformer/attention.py
  └── Attention.forward() → handles CP via context_parallel groups

megatron/core/models/gpt/gpt_layer_specs.py
  └── Defines which layers participate in CP

miles_plugins/models/cp_utils.py
  └── relayout() → handles THD format CP splitting

miles_plugins/models/hf_attention.py
  └── CP relayout for HuggingFace attention implementations
```

### THD vs BSHD with CP

- **THD (packed tokens)**: `t[cp:zigzag,sp] 1 h` — tokens are packed, `seq_lens` tracks per-sequence boundaries. CP splitting must respect sequence boundaries.
- **BSHD (batch x sequence)**: `s[cp:zigzag,sp] b h` — sequence dimension directly split. Simpler but wastes memory on padding.

### CP + Causal Attention

Causal mask must be adjusted for zigzag ordering. Megatron's `cp_algo='megatron_cp_algo'` handles this.

## 5. Expert Parallelism (EP)

### Implementation

EP distributes MoE experts across ranks. Each EP rank hosts `num_experts / EP` experts.

**Token dispatch**: Router computes expert assignment → tokens are sent to the rank hosting that expert via AllToAll → experts process tokens → results are sent back via AllToAll.

### Dispatcher Types

```
--moe-token-dispatcher-type alltoall     # Standard AllToAll (Megatron default)
--moe-token-dispatcher-type allgather    # AllGather (simpler, more memory)
```

### Communication

Per MoE layer:
1. AllToAll: send tokens to expert-hosting ranks
2. Expert compute
3. AllToAll: send results back

### Key Code Paths

```
megatron/core/transformer/moe/router.py
  └── TopKRouter.forward() → computes routing probabilities and top-K assignment

megatron/core/transformer/moe/token_dispatcher.py
  ├── MoEAlltoAllTokenDispatcher → AllToAll dispatch
  └── MoEAllGatherTokenDispatcher → AllGather dispatch

megatron/core/transformer/moe/experts.py
  └── GroupedMLP / SequentialMLP → expert compute
```

### EP + ETP (Expert Tensor Parallelism)

ETP shards each expert's weights like TP shards regular layers. This is orthogonal to EP:
- EP: different experts on different ranks
- ETP: same expert's weights split across ranks

Process group topology: `world_size = DP * TP * PP * CP` and EP groups are formed within the TP dimension or orthogonally.

## 6. Process Group Topology

Megatron creates multiple process groups for different parallelism axes:

```python
from megatron.core import parallel_state as mpu

mpu.get_tensor_model_parallel_group()      # TP communication
mpu.get_pipeline_model_parallel_group()     # PP communication  
mpu.get_data_parallel_group()               # DP communication (gradient sync)
mpu.get_context_parallel_group()            # CP communication
mpu.get_expert_model_parallel_group()       # EP communication
```

**World size decomposition**:
```
world_size = TP * PP * CP * DP
# where DP = world_size / (TP * PP * CP)
# EP is formed within the DP or TP dimension
```

### Debugging Process Groups

```python
import torch.distributed as dist
from megatron.core import parallel_state as mpu

rank = dist.get_rank()
print(f"[Rank {rank}] "
      f"TP={mpu.get_tensor_model_parallel_rank()}/{mpu.get_tensor_model_parallel_world_size()} "
      f"PP={mpu.get_pipeline_model_parallel_rank()}/{mpu.get_pipeline_model_parallel_world_size()} "
      f"DP={mpu.get_data_parallel_rank()}/{mpu.get_data_parallel_world_size()} "
      f"CP={mpu.get_context_parallel_rank()}/{mpu.get_context_parallel_world_size()}")
```

## 7. Data Flow Through a Transformer Layer

For a single transformer layer with TP+SP+CP:

```
Input: [T_local, 1, H]  (SP-split tokens, full hidden)
  │
  ├── LayerNorm (local, no communication)
  │
  ├── AllGather seq dim (SP → full sequence for this CP rank)
  │   Result: [T_cp_chunk, 1, H]
  │
  ├── QKV Projection (ColumnParallel)
  │   Q: [T, num_heads/TP, D]
  │   K: [T, num_kv_heads/TP, D]  
  │   V: [T, num_kv_heads/TP, D]
  │
  ├── RoPE (position encoding, adjusted for CP chunk offset)
  │
  ├── Core Attention (with CP ring attention)
  │   AllGather/Ring KV across CP ranks
  │   Result: [T_cp_chunk, num_heads/TP, D]
  │
  ├── Output Projection (RowParallel)
  │   ReduceScatter (TP allreduce + SP scatter)
  │   Result: [T_local, 1, H]
  │
  ├── Residual Add (local)
  │
  ├── LayerNorm (local)
  │
  ├── AllGather seq dim (SP → full for MLP)
  │
  ├── MLP / MoE
  │   If MoE: Router → AllToAll → Experts → AllToAll
  │   If MLP: Gate/Up (ColumnParallel) → Act → Down (RowParallel)
  │
  ├── ReduceScatter (TP + SP)
  │   Result: [T_local, 1, H]
  │
  └── Residual Add → Output: [T_local, 1, H]
```
