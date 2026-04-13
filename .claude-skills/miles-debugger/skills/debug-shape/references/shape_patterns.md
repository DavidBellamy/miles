# Tensor Shape Patterns in Megatron

Complete shape reference for all tensor types across parallelism configurations.

## Notation

- H = hidden_size
- N = num_attention_heads
- K = num_kv_heads (GQA groups)
- D = head_dim = H / N
- FFN = ffn_hidden_size
- E = num_experts
- V = vocab_size (padded to be divisible by TP)
- T = total tokens (THD packed format)
- S = sequence length (BSHD format)
- B = batch size (BSHD format)
- TP = tensor_model_parallel_size
- PP = pipeline_model_parallel_size
- CP = context_parallel_size
- EP = expert_model_parallel_size
- ETP = expert_tensor_parallel_size

## 1. Weight Shapes (per rank)

### 1.1 Attention Weights

| Weight | Shape per TP rank | Notes |
|--------|-------------------|-------|
| QKV fused (Megatron default) | `[H, (N/TP + 2*K_eff/TP) * D]` | K_eff = max(K, TP) for GQA replication |
| Q only | `[H, (N/TP) * D]` | |
| K only | `[H, (K_eff/TP) * D]` | K_eff handles GQA |
| V only | `[H, (K_eff/TP) * D]` | Same as K |
| O projection | `[(N/TP) * D, H]` | RowParallel |

### 1.2 MLP Weights

| Weight | Shape per TP rank | Notes |
|--------|-------------------|-------|
| Gate (SwiGLU) | `[H, FFN/TP]` | ColumnParallel |
| Up (SwiGLU) | `[H, FFN/TP]` | ColumnParallel |
| Gate+Up fused | `[H, 2*FFN/TP]` | When fused |
| Down | `[FFN/TP, H]` | RowParallel |

### 1.3 MoE Expert Weights (per EP rank)

Each EP rank hosts `E/EP` experts. Each expert has MLP weights:

| Weight | Shape per EP+ETP rank | Notes |
|--------|----------------------|-------|
| Gate per expert | `[H, FFN/ETP]` | Sharded by ETP |
| Up per expert | `[H, FFN/ETP]` | Sharded by ETP |
| Down per expert | `[FFN/ETP, H]` | Sharded by ETP |
| GroupedMLP gate | `[E/EP, H, FFN/ETP]` | Batched across local experts |

### 1.4 Embedding / Output

| Weight | Shape per TP rank | Notes |
|--------|-------------------|-------|
| Word embedding | `[V/TP, H]` | VocabParallel |
| Output layer | `[V/TP, H]` | ColumnParallel (or tied to embedding) |
| Position embedding | `[max_seq_len, H]` | Not TP-sharded (replicated) |

## 2. Activation Shapes (per rank)

### 2.1 THD Format (Packed Tokens)

With `--use-dynamic-batch-size` or packed format. T_local = tokens on this rank after SP split.

| Activation | Shape | Dims annotation |
|-----------|-------|-----------------|
| Layer input/output | `[T_local, 1, H]` | `t[cp:zigzag,sp] 1 h # tp:replicated` |
| After AllGather (pre-attn) | `[T_cp_chunk, 1, H]` | `t[cp:zigzag] 1 h # tp:replicated sp:replicated` |
| Q after projection | `[T_cp_chunk, N/TP, D]` | `t[cp:zigzag] num_heads[tp] head_dim` |
| K after projection | `[T_cp_chunk, K_eff/TP, D]` | `t[cp:zigzag] num_kv_heads[tp] head_dim` |
| V after projection | `[T_cp_chunk, K_eff/TP, D]` | `t[cp:zigzag] num_kv_heads[tp] head_dim` |
| Attention output | `[T_cp_chunk, 1, (N/TP)*D]` | `t[cp:zigzag] 1 (num_heads*head_dim)[tp]` |
| After O proj + ReduceScatter | `[T_local, 1, H]` | `t[cp:zigzag,sp] 1 h # tp:replicated` |
| Pre-MLP (after LN) | `[T_cp_chunk, 1, H]` | `t[cp:zigzag] 1 h # tp:replicated` |
| MLP gate/up output | `[T_cp_chunk, 1, FFN/TP]` | `t[cp:zigzag] 1 ffn[tp]` |
| MLP down output (pre-reduce) | `[T_cp_chunk, 1, H]` | `t[cp:zigzag] 1 h[tp:partial]` |
| After ReduceScatter | `[T_local, 1, H]` | `t[cp:zigzag,sp] 1 h # tp:replicated` |

### 2.2 BSHD Format (Batch x Sequence)

| Activation | Shape | Dims annotation |
|-----------|-------|-----------------|
| Layer input/output | `[S_local, B, H]` | `s[cp:zigzag,sp] b h # tp:replicated` |
| Q after projection | `[S_cp, B, N/TP, D]` | `s[cp:zigzag] b num_heads[tp] head_dim` |
| K after projection | `[S_cp, B, K_eff/TP, D]` | `s[cp:zigzag] b num_kv_heads[tp] head_dim` |
| V after projection | `[S_cp, B, K_eff/TP, D]` | `s[cp:zigzag] b num_kv_heads[tp] head_dim` |
| Attention output | `[S_cp, B, (N/TP)*D]` | `s[cp:zigzag] b (num_heads*head_dim)[tp]` |

### 2.3 MoE-Specific Activations

| Activation | Shape | Dims annotation |
|-----------|-------|-----------------|
| Router logits | `[T, 1, E]` | `t[cp:zigzag,sp] 1 num_experts # tp:replicated` |
| Top-K indices | `[T, top_k]` | `t[cp:zigzag,sp] topk # tp:replicated` |
| Dispatched tokens (AllToAll) | `[T_dispatched, H]` | Variable per rank |
| Expert output | `[T_dispatched, H]` | `t h[etp:partial] # ep:replicated` |

## 3. GQA (Grouped Query Attention) Edge Cases

### When K >= TP (Normal Case)

```
K=8, TP=4 → each rank: K/TP = 2 KV heads
K=4, TP=4 → each rank: K/TP = 1 KV head
K=4, TP=2 → each rank: K/TP = 2 KV heads
```

### When K < TP (Edge Case — Most Common Shape Bug Source)

```
K=4, TP=8 → K/TP = 0 → CANNOT divide!
```

**Megatron's solution**: Replicate KV heads.
```python
# In adjust_key_value_for_gqa():
if num_kv_heads < tp_size:
    # Replicate KV heads so each TP rank has at least 1
    num_kv_heads_per_rank = 1  # or ceil(K / TP)
    # Actual replication happens in the attention computation
```

**Shape after GQA adjustment**:
```
K_effective = max(K, TP)
K_per_rank = K_effective / TP
# So K=4, TP=8 → K_eff=8, K_per_rank=1 (each KV head replicated to 2 ranks)
```

### What Breaks

Code that assumes `K_per_rank = K / TP` will fail when K < TP:

```python
# WRONG
kv_heads_per_rank = num_kv_heads // tp_size  # = 0 when K < TP!
gate = gate[:, :, :kv_heads_per_rank, :]     # empty tensor!

# RIGHT  
kv_heads_per_rank = max(1, num_kv_heads // tp_size)
# or
kv_heads_per_rank = num_query_groups_per_partition  # Megatron's computed value
```

**The attention_output_gate fix** (Miles PR example):
The gate tensor corresponds to attention output (N heads), not KV (K heads). Slicing by KV heads is wrong:
```python
# WRONG: gate is [batch, seq, N, D], slicing by K/TP
gate = gate[:, :, :num_kv_heads // tp, :]

# RIGHT: gate is [batch, seq, N, D], slicing by N/TP  
gate = gate[:, :, :num_attention_heads // tp, :]
```

## 4. CP Sequence Splitting

### BSHD Format

```
Full sequence: S tokens
CP=2 → each rank: S/2 tokens (but zigzag interleaved)
CP=4 → each rank: S/4 tokens
Constraint: S % (2*CP) == 0 (zigzag needs 2*CP chunks)
```

### THD Format (Packed)

```
Total tokens: T (sum of all sequence lengths)
seq_lens = [s1, s2, s3, ...]  # per-sequence lengths
CP splitting: each sequence is split independently
  s1 → rank0 gets s1/(2*CP) chunks (zigzag), rank1 gets s1/(2*CP) chunks, ...
Constraint: each si % (2*CP) == 0
```

### Relayout for CP

Miles handles CP relayout in `miles_plugins/models/cp_utils.py`:
```python
def relayout(hidden_states, seq_lens, cp_size, cp_rank):
    # Split each sequence into 2*cp_size chunks
    # Assign chunks to ranks in zigzag order
    # Return the chunks for this rank + updated seq_lens
```

## 5. Shape Debugging Checklist

When you encounter a shape error, verify:

1. **Model config consistency**:
   ```python
   assert hidden_size % num_attention_heads == 0, "H must be divisible by N"
   assert num_attention_heads % tp_size == 0, "N must be divisible by TP"
   assert ffn_hidden_size % tp_size == 0, "FFN must be divisible by TP"
   assert num_layers % pp_size == 0, "layers must be divisible by PP"
   ```

2. **GQA handling**:
   ```python
   if num_kv_heads < tp_size:
       # Verify GQA replication is enabled
       # Verify downstream code uses effective kv_heads, not raw kv_heads
   ```

3. **CP sequence lengths**:
   ```python
   for seq_len in seq_lens:
       assert seq_len % (2 * cp_size) == 0, f"seq_len={seq_len} not divisible by 2*CP={2*cp_size}"
   ```

4. **MoE expert count**:
   ```python
   assert num_experts % ep_size == 0, "experts must be divisible by EP"
   ```

5. **Vocabulary padding**:
   ```python
   padded_vocab = math.ceil(vocab_size / tp_size) * tp_size
   ```
