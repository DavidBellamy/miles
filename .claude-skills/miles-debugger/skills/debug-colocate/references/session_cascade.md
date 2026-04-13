# Session Lifecycle Cascade Failure

The most severe interconnected bug cluster in Miles colocate mode. A single prefill storm can cascade into timeout, eviction, rollback failure, and session corruption — wasting dozens of GPU-hours.

## The Cascade

```
Step 1: TITO prefill storm (Miles #920)
   Long agentic sessions (50K-100K+ tokens) force full-context re-prefill
   GPU monopolized for 30-40s per request (80+ consecutive 512-token batches)
   Decode throughput drops: ~68 tok/s → 3-8 tok/s
                    │
Step 2: Timeout metric wrong (Miles #936)
   httpx timeout counts from submission, not processing start
   Queue wait + GPU stalls consume timeout budget
   625 timeouts in 1h25m test, 57 GPU-hours wasted
                    │
Step 3: Weight update freeze (Miles #921)
   Weight checkpoint loading freezes engines 20-30s
   In-flight requests at 580+ seconds pushed past 600s timeout
   119 timeouts immediately after single weight update
                    │
Step 4: Session corruption (Miles #955)
   Engine appends assistant response to stored messages
   Client timeout means client never receives it
   Client retries with original messages → stored vs request mismatch
   Rollback fails: "no assistant message found in the first 1 matched messages"
   127 rollback failures in 1h25m
                    │
Step 5: Session eviction (Miles #956)
   Timeout storms create abandoned sessions
   Abandoned sessions retain KV cache → memory pressure
   LRU eviction removes active sessions → 404 errors
   99 "session not found" errors in 9-minute window
   Each failure yields reward=0.0
```

## Mitigation Strategies

### For Prefill Storms (#920)

1. **Chunked prefill interleaving**: Allow decode batches to run between prefill chunks
2. **Batch staggering**: Distribute long prefills across DP shards
3. **Prefill priority queue**: Don't let one long prefill block all other requests
4. **Max prefill length**: Cap single-request prefill to prevent monopolization

### For Timeout Metric (#936)

1. **Processing-start timeout**: Reset timer when engine actually starts processing
2. **Client concurrency control**: Limit outstanding requests per session
3. **Server-side processing signal**: Engine reports when it starts/finishes each request

### For Weight Update Freeze (#921)

1. **Three-phase dispatch gate**:
   - Phase 1 (pre-update): Stop routing new requests to this engine
   - Phase 2 (during update): Queue requests locally, engine loads weights
   - Phase 3 (post-update): Resume after engine confirms readiness

2. **Staggered updates**: Update engines one at a time, not all simultaneously

### For Session Corruption (#955)

1. **Detect timeout pattern**: If stored messages > request messages and last stored is assistant, it's a timeout-induced corruption
2. **Reset to checkpoint**: Roll back to last valid state instead of failing
3. **Return 409 Conflict**: Retryable error instead of 400 Bad Request

### For Session Eviction (#956)

1. **Session pinning**: Pin active sessions from LRU eviction
2. **Return 410 Gone**: Distinguishable from 404 Not Found
3. **Eviction logging**: Log which sessions are evicted and why
4. **Proactive cleanup**: Release KV cache for timed-out sessions faster

## Monitoring Checklist

```python
# Key metrics to monitor in colocate mode:
metrics = {
    "timeout_rate": "percentage of requests that time out",
    "rollback_failure_rate": "percentage of rollbacks that fail",
    "session_eviction_rate": "sessions evicted per minute",
    "prefill_duration_p99": "99th percentile prefill time",
    "decode_throughput": "tokens per second during decode",
    "weight_update_duration": "time for weight loading",
    "gpu_utilization_variance": "difference across GPUs (contention indicator)",
}
```

## Relevant Environment Variables

```bash
# Increase NCCL timeout for weight updates
export NCCL_TIMEOUT=1800

# Increase httpx timeout (client side)
export MILES_HTTP_TIMEOUT=900

# SGLang memory fraction (reduce to leave room for training)
--sglang-mem-fraction-static 0.6

# Disable CUDA graph (required for dumper, also helps with memory)
--sglang-disable-cuda-graph
```
