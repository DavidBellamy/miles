# Hang Pattern Case Studies

Real-world distributed training hang cases with diagnosis and resolution.

## Case 1: Pipeline Parallel Micro-batch Deadlock

### Setup
- PP=4, DP=2, micro_batch_size=2, global_batch_size=12
- 1F1B schedule

### Symptom
Training hangs after first warmup phase. All ranks stuck in NCCL send/recv.

### Root Cause
```
num_microbatches = global_batch_size / (DP * micro_batch_size) = 12 / (2 * 2) = 3
```
With PP=4, the 1F1B schedule needs at least PP micro-batches for warmup. 3 < 4, causing a deadlock where stage 3 never receives a micro-batch during warmup.

### Fix
```
global_batch_size must >= DP * micro_batch_size * PP
12 >= 2 * 2 * 4 = 16  → FAIL
# Fix: global_batch_size=16 or micro_batch_size=1
```

### How to Detect
```python
num_microbatches = global_batch_size // (dp_size * micro_batch_size)
assert num_microbatches >= pp_size, (
    f"num_microbatches ({num_microbatches}) must be >= PP ({pp_size}). "
    f"Increase global_batch_size to at least {dp_size * micro_batch_size * pp_size}"
)
```

---

## Case 2: Expert Parallel AllToAll Token Count Mismatch

### Setup
- EP=4, num_experts=16, top_k=2
- AllToAll token dispatcher

### Symptom
Hang after ~50 iterations. Non-deterministic — sometimes works, sometimes hangs.

### Root Cause
MoE router produces different top-K expert assignments depending on input. In rare cases, one EP rank's experts receive zero tokens while another rank expects to send tokens to it. The AllToAll communication requires all ranks to participate even if they send/receive zero tokens, but the dispatcher code had a bug where it skipped the AllToAll when local dispatch count was zero.

### Diagnosis Steps
1. Set `NCCL_DEBUG=INFO` → last log shows AllToAll on some ranks, nothing on others
2. Dump `moe_topk_ids` before the hang:
   ```python
   dumper.dump('moe_topk_ids', topk_ids, dims='t topk # tp:replicated ep:replicated')
   ```
3. Analyze dispatch distribution:
   ```python
   # Check if any expert gets 0 tokens
   for expert_id in range(num_experts):
       count = (topk_ids == expert_id).sum().item()
       if count == 0:
           print(f"WARNING: expert {expert_id} receives 0 tokens")
   ```

### Fix
Ensure AllToAll is always called on all ranks, even with zero dispatch. Most modern Megatron versions handle this correctly.

---

## Case 3: Context Parallel + Sequence Parallel Collective Mismatch

### Setup
- TP=2, CP=2, SP=True
- THD format with dynamic batch size

### Symptom
Hang when CP > 1 with SP. Works fine with either CP=1+SP or CP=2+no-SP.

### Root Cause
With both CP and SP, the token dimension is split across both axes. The AllGather for SP must happen within the TP group (not the CP group), but the sequence lengths per CP rank may differ due to uneven token distribution, causing AllGather size mismatch.

### Diagnosis Steps
1. Add logging before AllGather:
   ```python
   print(f"[rank={rank}] pre-AllGather shape={tensor.shape} "
         f"tp_rank={tp_rank} cp_rank={cp_rank}")
   ```
2. Check if shapes differ within the same TP group but different CP ranks — they shouldn't, but with dynamic batching they might.

### Fix
Ensure token counts are padded to be identical across all ranks within a TP group before AllGather. Miles handles this in the batch construction.

---

## Case 4: Gradient AllReduce with Parameter Mismatch

### Setup
- DP=4, model with optional layers (e.g., adapter layers only on some ranks)

### Symptom
Hang during first backward pass. Forward pass works fine.

### Root Cause
DDP (DistributedDataParallel) expects all DP ranks to have identical parameter sets. If some ranks have extra parameters (e.g., adapters loaded only on some ranks), the gradient allreduce bucket counts don't match, causing a hang.

### Diagnosis
```python
# Print parameter count per rank
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[rank={rank}] trainable params: {total_params}")
# If these differ across DP ranks → that's the bug
```

### Fix
Ensure all DP ranks have identical model architecture. If using adapters, all ranks must load them (even if some ranks don't use them during forward).

---

## Case 5: NCCL Timeout from GPU Memory Pressure

### Setup
- Large model near GPU memory limit
- OOM happens during collective operation

### Symptom
`NCCL watchdog timeout` with no prior error. Sometimes preceded by `CUDA error: out of memory` on stderr of one rank.

### Root Cause
NCCL collectives allocate temporary GPU memory. When GPU memory is near capacity, the allocation fails silently, and the collective never completes on that rank. Other ranks timeout waiting.

### Diagnosis
1. Monitor GPU memory before collectives:
   ```python
   print(f"[rank={rank}] GPU memory: "
         f"{torch.cuda.memory_allocated()/1e9:.1f}GB / "
         f"{torch.cuda.max_memory_allocated()/1e9:.1f}GB peak")
   ```
2. Check `nvidia-smi` across all ranks during training

### Fix
- Reduce batch size or sequence length
- Enable activation checkpointing (`--recompute-activations`)
- Enable CPU offloading for optimizer states

---

## Case 6: Hang After Checkpoint Load with Mismatched Barrier

### Setup
- Training with periodic checkpointing
- Resumed from checkpoint

### Symptom
Hang when resuming from checkpoint. Fresh training works fine.

### Root Cause
Checkpoint loading code has a `dist.barrier()` that is inside a conditional branch that only executes on rank 0 (or the loading rank). Other ranks skip the barrier, causing a permanent hang.

### Diagnosis
Search for `dist.barrier()` calls in checkpoint loading code:
```bash
grep -n "barrier" miles/backends/megatron_utils/checkpoint*.py
```
Verify every barrier is reached by ALL ranks.

### Fix
Move barrier outside conditional blocks, or ensure all ranks enter the same code path.

---

## Case 7: SP ReduceScatter Size Mismatch with Odd Token Count

### Setup
- TP=4, SP=True
- Dynamic batch with variable token counts

### Symptom
Hang or crash at ReduceScatter after attention. Error: `input tensor must be divisible by world size`.

### Root Cause
ReduceScatter requires the input tensor's scattered dimension to be divisible by TP. With dynamic batching, the token count may not be divisible by TP.

### Diagnosis
```python
print(f"[rank={rank}] pre-ReduceScatter shape={tensor.shape}, TP={tp_size}")
assert tensor.shape[0] % tp_size == 0, f"Tokens {tensor.shape[0]} not divisible by TP {tp_size}"
```

### Fix
Pad token count to be divisible by TP before AllGather/ReduceScatter. Miles handles this in the data loader.

---

## Hang Debugging Toolkit Summary

### NCCL Debug Environment Variables
```bash
export NCCL_DEBUG=INFO                    # Log all NCCL operations
export NCCL_DEBUG_SUBSYS=COLL,NET        # Focus on collectives and network
export NCCL_TIMEOUT=120                   # Faster timeout for debugging
export NCCL_ASYNC_ERROR_HANDLING=1        # Don't hang on errors
export TORCH_DISTRIBUTED_DEBUG=DETAIL     # Torch-level collective logging
export TORCH_NCCL_BLOCKING_WAIT=1        # Make NCCL ops blocking (easier to trace)
```

### Stack Trace Collection
```bash
# py-spy for all python processes
for pid in $(pgrep -f "miles.train\|megatron"); do
  echo "=== PID $pid ===" 
  py-spy dump --pid $pid 2>/dev/null
done

# Or send SIGUSR1 (if faulthandler.enable() was called)
for pid in $(pgrep -f "miles.train"); do
  kill -USR1 $pid
done
```

### Process Group Inspection
```python
import torch.distributed as dist
from megatron.core import parallel_state as mpu

rank = dist.get_rank()
for name, getter in [
    ("TP", mpu.get_tensor_model_parallel_group),
    ("PP", mpu.get_pipeline_model_parallel_group),
    ("DP", mpu.get_data_parallel_group),
]:
    group = getter()
    ranks_in_group = dist.get_process_group_ranks(group)
    print(f"[Rank {rank}] {name} group: ranks={ranks_in_group}")
```
