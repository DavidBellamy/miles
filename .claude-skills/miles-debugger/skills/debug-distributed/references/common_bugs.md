# Common Distributed Training Bugs

Catalog of frequently encountered bugs in TP/PP/EP/CP configurations, with symptoms, root causes, and fixes.

## 1. Tensor Parallel (TP) Bugs

### 1.1 num_kv_heads < TP (GQA Edge Case)

**Symptom**: `RuntimeError: shape mismatch` in KV projection or attention output gate when TP > num_kv_heads.

**Root cause**: With GQA (Grouped Query Attention), the number of KV heads can be much smaller than the number of query heads. If TP > num_kv_heads, naive division `num_kv_heads // TP` gives 0, causing shape errors.

**Example**: Qwen3-30B-A3B has `num_kv_heads=4`. With TP=8, `4 // 8 = 0`.

**Fix patterns**:
1. Megatron handles this with KV head replication in `adjust_key_value_for_gqa()` — verify it's being called
2. Check `attention_output_gate` slicing: it must slice by `num_heads // TP`, not `num_kv_heads // TP`
3. In Miles, see `miles_plugins/models/hf_attention.py` for the correct CP relayout handling

**Relevant code**:
```python
# WRONG: slicing gate by kv_heads
gate = gate[:, :, :num_kv_heads // tp_size, :]

# RIGHT: slicing gate by query heads (gate corresponds to attention output, not KV)
gate = gate[:, :, :num_heads // tp_size, :]
```

### 1.2 TP Allreduce Missing After Column-Parallel Linear

**Symptom**: Training produces NaN/garbage after TP > 1; loss diverges but doesn't crash.

**Root cause**: Column-parallel linear outputs are partial sums across TP ranks. If the subsequent allreduce (or reduce-scatter with SP) is skipped, each rank trains on incomplete data.

**Debug**: Dump the tensor before and after the suspected allreduce:
```python
dumper.dump('pre_allreduce', tensor, dims='t h[tp:partial]')
# ... allreduce happens ...
dumper.dump('post_allreduce', tensor, dims='t h # tp:replicated')
```

**Fix**: Verify `RowParallelLinear` or explicit `allreduce` follows every `ColumnParallelLinear`.

### 1.3 Sequence Parallel Dimension Mismatch

**Symptom**: Shape error in LayerNorm or dropout when `--sequence-parallel` is enabled.

**Root cause**: With SP, the sequence/token dimension is split across TP ranks between attention and MLP. LayerNorm input should be the SP-split size, but some custom layers may expect the full size.

**Debug**: Check shapes at LayerNorm boundaries:
```python
# With SP, hidden_states shape should be [seq_len/TP, batch, hidden]
# Without SP, it should be [seq_len, batch, hidden]
```

### 1.4 Embedding / Output Layer Vocab Padding

**Symptom**: Shape error in `word_embeddings` or `output_layer` with certain TP sizes.

**Root cause**: `vocab_size` must be divisible by TP. Megatron auto-pads, but custom code may not.

**Fix**: Ensure vocab_size is padded: `padded_vocab_size = math.ceil(vocab_size / TP) * TP`

## 2. Numerical Divergence Bugs

### 2.1 Attention Softmax Precision

**Symptom**: Comparator shows rel_diff > 0.01 in attention outputs but inputs match perfectly.

**Root cause**: Attention softmax in BF16 loses precision for long sequences. Different implementations (FlashAttention vs vanilla) may diverge.

**Fix**: Use `--attention-softmax-in-fp32` to compute softmax in FP32. This is especially important for comparisons.

### 2.2 BF16 Accumulation Drift

**Symptom**: Early layers match well (rel_diff < 0.001) but deeper layers show increasing divergence (rel_diff > 0.005).

**Root cause**: BF16 has limited mantissa bits. Small numerical differences compound across layers.

**Fix**: This is expected behavior. Relax `--diff-threshold` for deep layers:
```bash
# Layer 0-3: threshold 0.001
# Layer 24+: threshold 0.008-0.01
--diff-threshold 0.0085
```

### 2.3 MoE Router Logit Divergence Causing Different Expert Assignment

**Symptom**: `moe_topk_ids` differ between baseline and target even though `moe_router_logits` are close.

**Root cause**: Small numerical differences in router logits can flip the top-K selection when multiple experts have similar scores.

**Fix**: Compare `moe_router_logits` values (should be very close). If logits match but topk_ids differ, it's a precision issue, not a bug. Use `--allow-failed-pattern moe_topk_ids` to exclude from pass/fail.

### 2.4 Gradient Accumulation FP32 Mismatch

**Symptom**: Forward pass matches but backward pass diverges significantly.

**Root cause**: Gradient accumulation without `--accumulate-allreduce-grads-in-fp32` uses BF16, amplifying errors.

**Fix**: Always use `--accumulate-allreduce-grads-in-fp32` for reliable comparisons.

## 3. Context Parallel (CP) Bugs

### 3.1 Zigzag Reorder Missing or Wrong

**Symptom**: Comparator shows `attn_output` diverges with CP > 1 even though `layer_input` matches after unsharding.

**Root cause**: CP uses zigzag interleaving for load balancing. If the reorder step is missing or wrong, token positions don't align.

**How zigzag works**: With CP=2, a sequence [A0, A1, B0, B1] becomes:
- Rank 0: [A0, B1] (first and last chunks)
- Rank 1: [A1, B0] (middle chunks)

The comparator handles this via `dims='t[cp:zigzag]'`, but the annotation must be correct.

**Debug**: Check if `zigzag` qualifier is present in dims for all CP-sharded tensors:
```python
dims='t[cp:zigzag,sp] 1 h'   # CORRECT for Megatron THD
dims='t[cp,sp] 1 h'           # WRONG: missing zigzag, default is natural order
```

### 3.2 CP Relayout for Packed Format (THD)

**Symptom**: Shape or content errors when switching between CP=1 and CP>1 with packed token format.

**Root cause**: In THD format, tokens from multiple sequences are packed together. CP splitting must respect per-sequence boundaries (via `seq_lens`), not just split the flat token stream.

**Debug**: Check `seq_lens` tensor — each CP rank should have the right portion of each sequence.

**In Miles**: See `miles_plugins/models/cp_utils.py` and `hf_attention.py` for relayout logic.

### 3.3 RoPE Position ID Mismatch with CP

**Symptom**: Attention output diverges but Q/K/V match before RoPE.

**Root cause**: Position IDs must be adjusted for each CP rank's chunk of the sequence. If positions aren't offset correctly, RoPE embeddings are wrong.

### 3.4 Causal Mask Handling with CP

**Symptom**: Attention values wrong with CP > 1 for causal (autoregressive) models.

**Root cause**: The causal mask must account for the zigzag ordering. Standard causal mask assumes natural order and will mask wrong tokens.

## 4. NCCL / Communication Bugs

### 4.1 NCCL Timeout with Asymmetric Computation

**Symptom**: `NCCL watchdog timeout` after a variable amount of time.

**Root cause**: One rank takes significantly longer (e.g., more tokens in MoE routing), and the other ranks time out waiting.

**Fix**: Increase timeout: `export NCCL_TIMEOUT=1800` (default is often 300-600s). If the asymmetry is extreme, the routing or load balancing is likely wrong.

### 4.2 NCCL Error with Process Group Mismatch

**Symptom**: `NCCL error: invalid usage` or `NCCL error: unhandled system error`.

**Root cause**: A collective is called on the wrong process group, or ranks are not in the expected group.

**Debug**: Print process group info:
```python
import torch.distributed as dist
from megatron.core import parallel_state as mpu

print(f"Rank {dist.get_rank()}: "
      f"TP={mpu.get_tensor_model_parallel_rank()}/{mpu.get_tensor_model_parallel_world_size()} "
      f"PP={mpu.get_pipeline_model_parallel_rank()}/{mpu.get_pipeline_model_parallel_world_size()} "
      f"DP={mpu.get_data_parallel_rank()}/{mpu.get_data_parallel_world_size()}")
```

### 4.3 OOM During Collective Operations

**Symptom**: CUDA OOM during AllGather or AllToAll, not during forward/backward.

**Root cause**: Collectives allocate temporary buffers. AllGather with SP allocates `TP * local_size`. AllToAll with EP allocates proportional to total dispatch.

**Fix**: Reduce batch size or sequence length. For SP, the peak memory includes the full (non-split) tensor during the gather phase.

## 5. Pipeline Parallel (PP) Bugs

### 5.1 Layer Assignment Imbalance

**Symptom**: Some PP stages run out of memory while others are underutilized.

**Root cause**: Default even split doesn't account for embedding/output layers on first/last stages.

**Fix**: Use explicit layer assignment or `--pipeline-model-parallel-split-rank`.

### 5.2 Micro-batch Count Mismatch

**Symptom**: PP hang or shape error at pipeline boundaries.

**Root cause**: `global_batch_size / (DP * micro_batch_size)` must be an integer. If not, some stages process more micro-batches than others, breaking the send/recv schedule.

### 5.3 Loss Computed on Wrong Stage

**Symptom**: Training proceeds but loss is always 0 or NaN.

**Root cause**: In PP, loss is computed only on the last stage. If custom loss logic runs on all stages, it gets wrong inputs on non-last stages.

## 6. Expert Parallel (EP) Bugs

### 6.1 num_experts Not Divisible by EP

**Symptom**: Shape error in MoE layer initialization.

**Root cause**: Each EP rank gets `num_experts / EP` experts. Must divide evenly.

### 6.2 Token Dispatch Mismatch in AllToAll

**Symptom**: Hang or wrong results with EP + AllToAll dispatcher.

**Root cause**: AllToAll expects each rank to send the right number of tokens to each other rank. If the routing produces inconsistent dispatch counts, the communication deadlocks or produces garbage.

**Debug**: Dump router output and check token counts:
```python
dumper.dump('dispatch_counts', tokens_per_expert, dims='num_experts # ep:replicated')
```

### 6.3 ETP + EP Interaction

**Symptom**: Wrong results with both EP and ETP (Expert Tensor Parallel) enabled.

**Root cause**: ETP shards each expert's weights across TP-like groups. When combined with EP, the process group topology becomes complex. Ensure `EP * ETP <= world_size`.

**Note on dims**: When `etp == tp`, don't declare both `tp:replicated` and `etp:replicated` — it causes an orthogonality error in the comparator.

## 7. Real-World Bugs from Issue Trackers

### 7.1 CUDA Stream Sync Race → NaN at Iteration 1 (Megatron #2301)

**Symptom**: NaN in loss at iteration 2 with DP-only and `use-distributed-optimizer`.

**Root cause**: Parameter copy during DDP module creation runs on separate CUDA stream without `wait_stream()`. Forward pass reads uninitialized/partially-copied parameters → NaN.

**Fix**: Explicit `torch.cuda.Stream()` synchronization with `wait_stream()` before/after DDP module creation.

**Meta-pattern**: When NaN appears at iteration 1-2 and parameters appear initialized, suspect CUDA stream races.

### 7.2 NCCL Collective Ordering Deadlock (Megatron #1810)

**Symptom**: Training hangs at beginning of backward pass on 128+ node clusters with `--overlap-moe-expert-parallel-comm`.

**Root cause**: A single shared CUDA event used as "baton" for compute and comm streams allows NCCL collectives to be launched in different orders on different ranks. NCCL requires same collectives in same order per communicator.

**Fix**: Replace single event with two CUDA events per model chunk ensuring deterministic ordering.

**Meta-pattern**: NCCL collective ordering must be identical across all ranks in a communicator. Shared sync primitives between compute/comm streams break this invariant.

### 7.3 Packed Sequence Padding Mismatch with CP+SP (Megatron #4194)

**Symptom**: `RuntimeError: The size of tensor a (974) must match the size of tensor b (971)` during residual connection.

**Root cause**: Alignment padding in packed sequences causes packed tensor length (3896 tokens) to differ from `cu_seqlens[-1]` (3884 tokens). GDN's all-to-all uses `cu_seqlens` boundaries but `hidden_states` uses padded length.

**Fix**: Handle alignment padding explicitly in packed all-to-all path; extend `cu_seqlens` boundaries for trailing padding.

**Meta-pattern**: When combining packed sequences with multiple parallelism dimensions, padding accounting must be consistent across ALL code paths.

### 7.4 Missing cu_seqlens Broadcast in PP Intermediate Stages (Megatron #4092)

**Symptom**: `RuntimeError: Tensors must have same number of dimensions: got 4 and 3` in RoPE with PP>2, TP>1, and SFT packing.

**Root cause**: `get_batch_on_this_tp_rank()` doesn't broadcast `cu_seqlens` and `max_seqlen` to non-zero TP ranks in intermediate pipeline stages. These remain `None`, causing fallback to non-packed logic.

**Meta-pattern**: Metadata tensors (cu_seqlens, max_seqlen) must be broadcast to ALL TP ranks at EVERY pipeline stage.

### 7.5 DistributedOptimizer Stale Index Map (Megatron #2777)

**Symptom**: `RuntimeError` during checkpoint save with size mismatch; `.copy_()` failures during load.

**Root cause**: Parameter reordering by dtype (FP32 first, then BF16) occurs AFTER `model_param_group_index_map` is built. The mapping becomes stale.

**Fix**: Rebuild index map after reordering; use `.data.copy_()` to bypass autograd.

**Meta-pattern**: Index maps / lookup tables must be rebuilt after any parameter reordering or reorganization.

### 7.6 Empty PP Stages Deadlock (Megatron #1852)

**Symptom**: Hangs with custom `--pipeline-parallel-layout` containing stages with no transformer layers.

**Root cause**: Pipeline scheduling assumes non-empty stages. Empty stages (embedding-only or loss-only) break send/recv patterns.

### 7.7 HybridDeviceOptimizer KeyError with MoE + CPU Offload (Megatron #4042)

**Symptom**: `KeyError` at first optimizer step with MoE + CPU optimizer offload.

**Root cause**: `param_to_inner_param` uses tensor object identity as keys. MixedPrecisionOptimizer replaces tensor objects after init, breaking the mapping.

**Fix**: Use stable identifiers (parameter indices) instead of tensor identity.

**Meta-pattern**: Tensor object identity is unreliable across optimizer wrapping layers. Always use stable indices.

### 7.8 RDMA QP Assertion Failure with HybridEP (Megatron #3999)

**Symptom**: `Assertion 'tx_depth > 0 && tx_depth < 65536' failed` with SIGABRT during HybridEP init.

**Root cause**: Token count per rank exceeds RDMA queue depth limit. `3 * max_tokens + 1 > 65535`.

**Workaround**: Increase TP/CP to reduce tokens per rank below threshold.

**Meta-pattern**: Hardware-level constraints (RDMA queue depth, NIC buffer sizes) surface as opaque assertion failures.

### 7.9 MTP Loss Scale Bug with CP (Megatron #3943)

**Root cause analysis**: `num_microbatches` is multiplied by CP size, but gradient scaling uses `1.0/data_parallel_world_size`. Analysis confirmed the implementation is correct because MTP computes per-token mean before scaling.

**Meta-pattern**: Loss scaling with CP requires careful reasoning about which dimensions are sharded vs replicated. Different loss types (main, MoE aux, MTP) scale differently.

### 7.10 EP Weight Sync Deadlock in Miles (Miles #574)

**Symptom**: Training hangs after 74 steps during weight update with EP=4.

**Root cause**: With EP, different ranks hold different expert subsets. `all_gather_param()` is a TP-group collective. If iteration order differs across ranks, deadlock.

**Fix**: All TP ranks must iterate parameters in identical order, even for empty expert subsets.

### 7.11 rollout_routed_experts Truncation Missing (Miles #861)

**Symptom**: `Sample.validate()` fails with shape mismatch after truncation in R3 mode.

**Root cause**: `_truncate_sample_output()` handles tokens, log_probs, loss_mask but misses `rollout_routed_experts`.

### 7.12 R3 Data Corruption from Read-Only NumPy Arrays (Miles #599)

**Symptom**: Spiky training with UserWarning about non-writable tensors.

**Root cause**: `np.frombuffer()` returns read-only arrays. `torch.from_numpy()` creates non-writable tensors causing undefined behavior.

**Fix**: Add `.copy()` after `np.frombuffer()`.
