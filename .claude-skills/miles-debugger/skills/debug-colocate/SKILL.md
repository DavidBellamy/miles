---
name: debug-colocate
description: Use when debugging Ray actor issues, colocate mode problems, SGLang engine lifecycle, multi-node setup failures, or resource contention between training and inference. Trigger on Ray actor crash, engine initialization failure, colocate OOM, GPU contention, request timeout during training, session eviction, prefill storm, dist_init_addr wrong assignment, or process group initialization errors. Also trigger on torch_memory_saver issues or memory release failures.
---

# Debug Colocate Mode & Ray Actor Issues

Miles colocates Megatron training and SGLang inference on the same GPUs via Ray actors. This creates unique resource contention and lifecycle issues.

## Colocate Architecture

```
┌─────────────────────────────── Node 0 ───────────────────────────────┐
│                                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  GPU 0      │  │  GPU 1      │  │  GPU 2      │  │  GPU 3      │ │
│  │             │  │             │  │             │  │             │ │
│  │ SGLang Eng  │  │ SGLang Eng  │  │ SGLang Eng  │  │ SGLang Eng  │ │
│  │ (inference) │  │ (inference) │  │ (inference) │  │ (inference) │ │
│  │      +      │  │      +      │  │      +      │  │      +      │ │
│  │ Megatron    │  │ Megatron    │  │ Megatron    │  │ Megatron    │ │
│  │ (training)  │  │ (training)  │  │ (training)  │  │ (training)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                                       │
│  Ray Head + Driver                                                    │
└───────────────────────────────────────────────────────────────────────┘
```

## Quick Decision Tree

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| OOM during Megatron init after rollout | SGLang KV cache not released | Check `/release_memory_occupation` works |
| OOM only in colocate, not standalone | GPU memory shared between engines | Reduce `--sglang-mem-fraction-static` |
| Timeout errors during weight update | Engine frozen 20-30s during load | Implement dispatch gate |
| Prefill storms causing decode starvation | Long TITO sessions monopolize GPU | Limit prefill batch size |
| Session 404 errors | LRU eviction under memory pressure | Pin active sessions |
| NCCL timeout after engine switch | Process group stale after memory release | Reinitialize process groups |
| Wrong node IP in dist_init_addr | Ray scheduling assumption | Use dynamic IP discovery |
| `tms_interesting_region` flag corrupted | deep_ep monkey-patch side effect | Save/restore flag value |
| Single-GPU KeyError in process groups | ReloadableProcessGroup not created | Make cleanup idempotent |

## Memory Management in Colocate Mode

### Memory Release Protocol

When switching from inference to training:
```
1. SGLang releases KV cache: POST /release_memory_occupation
2. torch_memory_saver offloads inference model (if enabled)
3. Megatron initializes and loads model
4. Training runs forward/backward
5. Megatron releases memory
6. SGLang re-allocates KV cache
7. Weight sync happens
8. Inference resumes
```

### SWA KV Pool Memory Bug (Miles #860)

**Symptom**: `/release_memory_occupation` silently fails for sliding-window attention models. Full KV cache stays allocated.

**Root cause**: Two bugs:
1. `swa_memory_pool.py` forces `enable_memory_saver = False` unconditionally
2. `SWAKVPool` doesn't forward `enable_memory_saver` from server config

**Fix**: Use `setdefault()` instead of direct assignment; forward the parameter.

### Memory Monitoring

```python
import torch

def log_gpu_memory(tag):
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        print(f"[{tag}] GPU {i}: "
              f"free={free/1e9:.1f}GB "
              f"allocated={allocated/1e9:.1f}GB "
              f"reserved={reserved/1e9:.1f}GB "
              f"total={total/1e9:.1f}GB")
```

## Request Timeout Cascade (Miles #920, #921, #955, #956)

This is a cascade failure pattern unique to colocate mode:

```
1. Long TITO prefill (50K-100K tokens, 30-40s)
   ↓
2. GPU monopolized by prefill → decode starved (3-8 tok/s)
   ↓
3. In-flight requests approach timeout (600s)
   ↓
4. Weight update freezes engines for 20-30s
   ↓
5. Requests exceed timeout → timeout errors
   ↓
6. Client retries but server already appended partial response
   ↓
7. Message count mismatch → rollback failure
   ↓
8. Memory pressure from abandoned sessions → LRU eviction
   ↓
9. Client retries target evicted sessions → 404 errors
```

**Mitigation strategies**:
1. Track engine processing time, not wall-clock time for timeouts
2. Implement three-phase dispatch gate for weight updates
3. Pin active sessions to prevent eviction
4. Detect timeout-induced corruption and reset to valid state

## Multi-Node Setup Issues

### Wrong Node IP Assignment (Miles #807 Bug 1)

**Symptom**: TCP timeout for engines on node 1 — all engines receive node 0's IP.

**Root cause**: `_allocate_rollout_engine_addr_and_ports_normal` assumes `node_index = local_rank // num_engines_per_node`, but Ray may schedule engines differently.

**Fix**: Query each engine's actual IP dynamically:
```python
ip = ray.get(engine._get_current_node_ip_and_free_port.remote())
```

### torch_memory_saver + Multi-Node NCCL (Miles #806)

**Symptom**: NCCL "Bad address" during cross-node operations.

**Root cause**: LD_PRELOAD-based memory offload relocates GPU buffers, invalidating NCCL registered addresses.

**Affected**: GB200/Blackwell multi-node with colocate mode.

## Process Group Issues

### ReloadableProcessGroup KeyError (Miles #354)

**Symptom**: `KeyError` during `destroy_process_groups()` on single-GPU.

**Fix**: Make cleanup idempotent:
```python
if current_pid in ReloadableProcessGroup.GROUPS:
    del ReloadableProcessGroup.GROUPS[current_pid]
```

### deep_ep Flag Corruption (Miles #807 Bug 3)

**Symptom**: `tms_interesting_region` flag corrupted after deep_ep initialization.

**Root cause**: deep_ep's `Buffer.__init__` wrapper unconditionally sets the flag to True.

**Fix**: Save and restore the previous flag value.

## GPU Contention Debugging

### Monitoring GPU Utilization

```bash
# Watch GPU utilization across all GPUs
watch -n 1 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader'

# Check if one GPU is monopolized
nvidia-smi dmon -s u -d 1  # utilization monitoring every 1 second
```

### Identifying Contention

```python
# In training code:
import time
start = time.time()
output = model.forward(input)
forward_time = time.time() - start

if forward_time > expected_time * 2:
    print(f"WARNING: Forward pass took {forward_time:.1f}s (expected {expected_time:.1f}s)")
    print(f"Possible GPU contention with inference engine")
    # Check if inference is doing a large prefill:
    # curl http://localhost:30000/get_server_info
```

## Ray Actor Lifecycle

### Actor Crash Recovery

```python
# Check actor status
ray.get(actor.ping.remote())  # should return quickly

# If actor is dead:
# 1. Check Ray logs: /tmp/ray/session_latest/logs/
# 2. Check for CUDA errors: grep "CUDA" /tmp/ray/session_latest/logs/*
# 3. Check for OOM: grep "OutOfMemory" /tmp/ray/session_latest/logs/*
```

### Common Actor Crash Causes

1. **CUDA OOM**: Reduce batch size or model precision
2. **NCCL timeout**: Check network connectivity between nodes
3. **Segfault in NCCL**: Driver version mismatch
4. **Python exception**: Check actor logs for traceback

## Related Skills

- `/debug-hang`: For NCCL hangs in colocate mode
- `/debug-weight-sync`: For weight update issues
- `/debug-precision`: For memory-related precision issues (FP8 param gather)
