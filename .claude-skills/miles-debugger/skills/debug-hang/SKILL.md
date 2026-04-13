---
name: debug-hang
description: Use when distributed training hangs with no error message, no progress, or NCCL timeout. Trigger on training stuck, hang, deadlock, NCCL watchdog timeout, collective operation timeout, or process not responding. Also use when some ranks finish but others don't, or when training freezes at a specific step.
---

# Debug Distributed Training Hangs

Systematic approach to diagnosing why distributed training stops making progress.

## Quick Decision Tree

| Observation | Likely Cause | Action |
|------------|-------------|--------|
| All ranks hang at same collective | Shape mismatch in collective inputs | Check tensor shapes before the collective |
| Some ranks finish, others don't | Asymmetric code path (e.g., PP stage skip) | Check if all ranks enter the same collective |
| Hang only with CP > 1 | CP collective misconfiguration | Verify CP process groups and seq_lens |
| Hang only with EP > 1 | AllToAll token dispatch mismatch | Check expert routing produces valid dispatch |
| Hang after first iteration | Gradient allreduce mismatch | Check parameter counts match across ranks |
| Hang during model init | Weight loading deadlock | Check checkpoint loading has proper barriers |
| NCCL watchdog timeout after ~30min | Slow collective or actual deadlock | Enable NCCL debug logging |

## Diagnosis Steps

### Step 1: Enable NCCL Debug Logging

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL    # Focus on collective operations
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

This shows which collective each rank is executing. If rank 0 shows `AllReduce` but rank 1 shows `AllGather`, you have a collective mismatch.

### Step 2: Get Stack Traces from Hung Processes

```bash
# Method 1: py-spy (non-invasive, recommended)
pip install py-spy
py-spy dump --pid <PID>

# Method 2: faulthandler (requires code change)
import faulthandler; faulthandler.enable()
# Then send SIGUSR1: kill -USR1 <PID>

# Method 3: torch distributed debug
# Set TORCH_DISTRIBUTED_DEBUG=DETAIL before launch
# It will print the collective name and tensor shapes on timeout
```

### Step 3: Identify the Hanging Collective

Common hanging points in Megatron:

| Location | Collective | Typical Cause |
|----------|-----------|---------------|
| `forward_step` / `backward_step` | PP send/recv | Pipeline schedule mismatch |
| `DistributedDataParallel.allreduce` | AllReduce | Parameter count mismatch across DP ranks |
| `SequenceParallel` regions | AllGather/ReduceScatter | SP tensor size mismatch |
| `ContextParallel` attention | AllGather/ReduceScatter | Sequence length not divisible by CP |
| MoE `AllToAll` | AllToAll | Token dispatch count mismatch |
| `dist.barrier()` | Barrier | One rank skipped the barrier call |

### Step 4: Use Dumper to Isolate

Add dumps before the hanging collective to verify tensor shapes match:

```yaml
patches:
  - target: megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward
    edits:
      - match: "output_parallel = linear_with_grad_accumulation"
        prepend: |
          dumper.dump('pre_column_parallel', input_, 
                      dims=f'shape={list(input_.shape)}',
                      layer_info=str(self))
```

Then check if all ranks produce the same shapes:
```bash
for f in /tmp/dumper/*/step=0___rank=*___name=pre_column_parallel*.pt; do
  python3 -c "
import torch
d = torch.load('$f', weights_only=False)
print(f'rank={d[\"meta\"][\"rank\"]} shape={d[\"value\"].shape}')
"
done
```

## Common Hang Patterns

### Pattern 1: Pipeline Parallel Schedule Mismatch

**Symptom**: Hang with PP > 1, works with PP = 1.

**Cause**: Different PP stages execute different numbers of micro-batches, so send/recv pairs don't match.

**Debug**: Check `num_microbatches` is consistent. Verify `global_batch_size % (DP * micro_batch_size) == 0`.

### Pattern 2: Expert Parallel AllToAll Deadlock

**Symptom**: Hang with EP > 1, works with EP = 1.

**Cause**: Token routing produces different dispatch counts per rank, and AllToAll expects symmetric communication.

**Debug**: Dump `moe_topk_ids` and verify total tokens dispatched to each expert sums correctly across ranks.

### Pattern 3: Context Parallel Sequence Mismatch

**Symptom**: Hang with CP > 1, works with CP = 1.

**Cause**: Sequence lengths not properly split across CP ranks, causing collective size mismatch.

**Debug**: Check `seq_lens` tensor. Total tokens across CP ranks should equal the full sequence.

### Pattern 4: Gradient AllReduce with Frozen Layers

**Symptom**: Hang during backward pass.

**Cause**: Some ranks freeze layers that others don't, causing param count mismatch in DDP allreduce.

**Debug**: Verify `requires_grad` is identical across all DP ranks for every parameter.

## Prevention Checklist

Before running with new parallelism configs:

- [ ] `global_batch_size % (DP_size * micro_batch_size) == 0`
- [ ] `num_attention_heads % TP == 0`
- [ ] `num_kv_heads % TP == 0` (or GQA replication is enabled)
- [ ] `ffn_hidden_size % TP == 0`
- [ ] `num_layers % PP == 0` (or explicit layer assignment)
- [ ] `num_experts % EP == 0`
- [ ] `seq_len % (CP * 2) == 0` (CP uses 2*CP chunks for zigzag)
- [ ] `vocab_size % TP == 0` (or padded)
- [ ] All ranks load the same model architecture (no conditional skips)

## Related Skills

- `/debug-distributed`: Full distributed debugging methodology
- `/debug-shape`: Shape mismatch diagnosis (often the root cause of hangs)
- `/dumper-usage`: Using the dumper for pre-collective inspection

## References

- @references/hang_patterns.md - Detailed case studies of real hang bugs and their resolutions
