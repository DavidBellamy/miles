# Debugging Recipes

Step-by-step procedures for specific distributed training debugging scenarios.

## 1. Diagnosing Inference-Training Mismatch (SGLang vs Megatron)

**Goal**: Find where SGLang inference and Megatron training produce different activation values.

### Prerequisites
- Model checkpoint converted for both SGLang and Megatron
- The dumper system from PR #769 merged
- Source patcher YAML configs for both SGLang and Megatron

### Step-by-Step

**1.1 Prepare source patcher configs**

Use the reference configs in `tests/e2e/conftest_dumper.py` as a starting point. For a Qwen3 MoE model in THD format:

```yaml
# megatron_patcher.yaml
patches:
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention
    edits:
      - match: |
          inference_context = deprecate_inference_params(inference_context, inference_params)
        append: "dumper.dump('layer_input', hidden_states, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
      - match: "nvtx_range_pop(suffix=\"self_attention\")"
        append: "dumper.dump('attn_output', attention_output_with_bias[0], dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
  - target: megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp
    edits:
      - match: "residual = hidden_states"
        append: "dumper.dump('pre_mlp_residual', residual, dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
      - match: "return self._forward_post_mlp(mlp_output_with_bias, residual)"
        prepend: "dumper.dump('mlp_output', mlp_output_with_bias[0], dims='t[cp:zigzag,sp] 1 h # tp:replicated ep:replicated')"
```

```yaml
# sglang_patcher.yaml (for Qwen3 MoE)
patches:
  - target: sglang.srt.models.qwen3_moe.Qwen3MoeDecoderLayer.forward
    edits:
      - match: |
          hidden_states, residual = (
              self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
        append: "dumper.dump('layer_input', residual, dims='t h # tp:replicated dp:=attn_dp')"
      # ... (see conftest_dumper.py for full config)
```

**1.2 Run Miles with dumper enabled**

```bash
python -m miles.train \
  --hf-checkpoint /root/models/Qwen3-30B-A3B \
  --ref-load /root/Qwen3-30B-A3B_torch_dist \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 2 \
  --context-parallel-size 2 \
  --expert-model-parallel-size 2 \
  --dumper-enable \
  --dumper-dir /tmp/debug_dumps \
  --dumper-inference 'filter=layer_id is None or layer_id < 3' \
  --dumper-fwd-only 'filter=layer_id is None or layer_id < 3' \
  --dumper-fwd-bwd 'filter=layer_id is None or layer_id < 3' \
  --dumper-source-patcher-config-train megatron_patcher.yaml \
  --dumper-source-patcher-config-inference sglang_patcher.yaml \
  --sglang-disable-cuda-graph \
  --num-rollout 1 --rollout-batch-size 1 --global-batch-size 1 \
  # ... other args
```

**Key flags**:
- `--sglang-disable-cuda-graph`: Required for dumping in SGLang
- `--num-rollout 1 --rollout-batch-size 1 --global-batch-size 1`: Minimal run for debugging
- `filter=layer_id is None or layer_id < 3`: Only dump first 3 layers (reduces data volume)

**1.3 Verify dumps were created**

```bash
# Should see three directories:
ls /tmp/debug_dumps/
# engines/   fwd_only/   fwd_bwd/

# Check .pt files exist:
find /tmp/debug_dumps/ -name "*.pt" | head -20

# Verify a sample file:
python3 -c "
import torch
d = torch.load('/tmp/debug_dumps/engines/engine_0/step=0___rank=0___dump_index=0___name=layer_input.pt', weights_only=False)
print('Shape:', d['value'].shape)
print('Dims:', d['meta'].get('dims'))
print('Parallel info:', {k:v for k,v in d['meta'].items() if 'parallel' in k})
"
```

**1.4 Run comparator**

```bash
# Compare SGLang inference (engines) vs Megatron forward-backward (fwd_bwd)
python -m sglang.srt.debug_utils.comparator \
  --baseline-path /tmp/debug_dumps/engines/ \
  --target-path /tmp/debug_dumps/fwd_bwd/ \
  --preset sglang_megatron \
  --diff-threshold 0.001 \
  --output-format text \
  --verbosity verbose \
  --allow-skipped-pattern "input_ids|positions|cu_seqlens_q|cu_seqlens_kv|qkv_format"
```

**1.5 Interpret results**

The comparator output shows each tensor pair with:
- `rel_diff`: Relative difference (should be < threshold)
- `PASSED` / `FAILED`: Whether it's within threshold
- Shape transformations: Shows how unsharding/reordering was applied

**Finding the bug**: The FIRST tensor that FAILS is your starting point. If `layer_input` passes but `attn_output` fails at layer 2, the bug is in the attention computation of layer 2.

**1.6 Fix dims if comparator errors out**

If you get alignment errors, fix dims without re-running:
```bash
python -m sglang.srt.debug_utils.comparator \
  --baseline-path /tmp/debug_dumps/engines/ \
  --target-path /tmp/debug_dumps/fwd_bwd/ \
  --preset sglang_megatron \
  --override-target-dims "attn_pre_o_proj:t[cp:zigzag,sp] 1 (num_heads*head_dim)[tp]" \
  --diff-threshold 0.001
```

## 2. Validating New Parallelism Config

**Goal**: Verify that a new parallelism configuration (e.g., adding CP=2) produces correct results.

### Step-by-Step

**2.1 Establish baseline with simple config**

Run with TP=1, PP=1, CP=1 (single GPU or minimal parallelism):
```bash
DUMPER_ENABLE=1 DUMPER_DIR=/tmp/baseline python -m miles.train \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --dumper-fwd-only 'filter=layer_id is None or layer_id < 3' \
  --num-rollout 1 --global-batch-size 1
```

**2.2 Run with new config**

```bash
DUMPER_ENABLE=1 DUMPER_DIR=/tmp/target python -m miles.train \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 2 \
  --sequence-parallel \
  --dumper-fwd-only 'filter=layer_id is None or layer_id < 3' \
  --dumper-source-patcher-config-train megatron_patcher.yaml \
  --num-rollout 1 --global-batch-size 1
```

**2.3 Compare**

```bash
python -m sglang.srt.debug_utils.comparator \
  --baseline-path /tmp/baseline/fwd_only/ \
  --target-path /tmp/target/fwd_only/ \
  --preset sglang_dev \
  --diff-threshold 0.001
```

Use `sglang_dev` preset (not `sglang_megatron`) since both are Megatron runs.

## 3. Debugging a Hang

**Goal**: Find exactly where and why training hangs.

### Step-by-Step

**3.1 Quick timeout test**

Set a short NCCL timeout to get error messages faster:
```bash
export NCCL_TIMEOUT=120  # 2 minutes instead of default 30
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

**3.2 Add strategic dump points before collectives**

```yaml
# hang_debug_patcher.yaml
patches:
  - target: megatron.core.tensor_parallel.layers.ColumnParallelLinear.forward
    preamble: "import sys"
    edits:
      - match: "output_parallel = linear_with_grad_accumulation"
        prepend: |
          sys.stderr.write(f"[RANK {dumper._get_rank()}] ColumnParallel input shape={input_.shape}\n")
          sys.stderr.flush()
  - target: megatron.core.tensor_parallel.layers.RowParallelLinear.forward
    preamble: "import sys"  
    edits:
      - match: "output_ = linear_with_grad_accumulation"
        prepend: |
          sys.stderr.write(f"[RANK {dumper._get_rank()}] RowParallel input shape={input_.shape}\n")
          sys.stderr.flush()
```

**3.3 Run and observe output**

```bash
# Run with hang debug patcher
--dumper-enable --dumper-source-patcher-config-train hang_debug_patcher.yaml
```

Watch stderr. The last log line before the hang tells you which collective is stuck. If rank 0 prints `ColumnParallel input shape=[4096, 512]` but rank 1 prints `ColumnParallel input shape=[4096, 1024]`, you have a shape mismatch.

**3.4 Get stack traces from hung processes**

```bash
# Find the hung python processes
ps aux | grep python | grep miles

# Get stack traces
for pid in $(pgrep -f "miles.train"); do
  echo "=== PID $pid ==="
  py-spy dump --pid $pid 2>/dev/null || python3 -c "
import os, signal
os.kill($pid, signal.SIGUSR1)
"
done
```

**3.5 Common resolution patterns**

After identifying the hanging collective:
- **AllReduce hang**: Check parameter counts match across DP ranks
- **Send/Recv hang**: Check PP schedule and micro-batch count
- **AllToAll hang**: Check MoE token dispatch symmetry
- **AllGather hang**: Check SP/CP tensor sizes match across ranks
- **Barrier hang**: Find which rank skips the barrier (conditional code path)

## 4. Writing Source Patcher Configs for New Models

**Goal**: Create source patcher YAML for a model not yet supported.

### Step-by-Step

**4.1 Identify target functions**

For Megatron models, the key injection points are usually:
```
megatron.core.transformer.transformer_layer.TransformerLayer._forward_attention
megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp
megatron.core.transformer.attention.Attention.forward
megatron.core.transformer.moe.router.TopKRouter.forward
```

For SGLang models, find the decoder layer:
```
sglang.srt.models.<model_name>.<ModelName>DecoderLayer.forward
sglang.srt.models.<model_name>.<ModelName>Attention.forward_core
```

**4.2 Find match lines**

Read the source code and find unique lines to match:
```bash
# For Megatron core:
python3 -c "import inspect; import megatron.core.transformer.transformer_layer as m; print(inspect.getsource(m.TransformerLayer._forward_attention))"
```

Pick lines that are:
- Unique within the function
- Stable across versions
- Near the tensor you want to dump

**4.3 Write dims annotations**

For each dump, determine:
1. What dimensions does this tensor have? (Check the shape)
2. Which dimensions are sharded? (Check the parallelism config)
3. What ordering is used? (Check if zigzag for CP)
4. What's replicated? (Hidden state after allreduce is replicated across TP)

Use `/dumper-dims` skill for the full annotation syntax.

**4.4 Test the patcher**

```bash
# Quick test: just check the patcher works without running training
python3 -c "
from sglang.srt.debug_utils.source_patcher import apply_patches_from_config
import yaml
config = open('my_patcher.yaml').read()
states = apply_patches_from_config(config)
print('Patcher applied successfully')
for s in states:
    s.restore()
"
```

## 5. Analyzing Comparator JSON Output Programmatically

**Goal**: Write scripts to automatically analyze comparator results.

```python
import json

results = []
with open('/tmp/debug_dumps/fwd_bwd/comparator_report.jsonl') as f:
    for line in f:
        results.append(json.loads(line))

# Find failed comparisons
failed = [r for r in results if r.get('type') == 'comparison_tensor' and not r.get('diff', {}).get('passed', True)]
for f in failed:
    print(f"FAILED: {f['name']} rel_diff={f['diff']['rel_diff']:.6f} shapes={f.get('shape')}")

# Find skipped tensors
skipped = [r for r in results if r.get('type') == 'comparison_skip']
for s in skipped:
    print(f"SKIPPED: {s['name']} reason={s.get('reason')}")

# Summary
summary = next((r for r in results if r.get('type') == 'summary'), None)
if summary:
    print(f"\nTotal: {summary['total']} Passed: {summary['passed']} Failed: {summary['failed']}")
```

## 6. Quick Smoke Test for Dumper Setup

**Goal**: Verify the dumper pipeline works end-to-end before a long training run.

```bash
# 1. Simple non-intrusive test (no source patcher needed)
DUMPER_ENABLE=1 DUMPER_DIR=/tmp/smoke_test DUMPER_NON_INTRUSIVE_MODE=core \
python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --port 30000 &

# Wait for server to start, then trigger a request
curl -s http://localhost:30000/v1/completions \
  -d '{"prompt":"Hello","max_tokens":5}'

# 2. Check dumps were created
ls /tmp/smoke_test/
find /tmp/smoke_test/ -name "*.pt" | head -5

# 3. Verify dump content
python3 -c "
import torch
import glob
files = glob.glob('/tmp/smoke_test/**/*.pt', recursive=True)
print(f'Found {len(files)} dump files')
if files:
    d = torch.load(files[0], weights_only=False)
    print(f'Keys: {list(d.keys())}')
    print(f'Value type: {type(d[\"value\"])}')
    if hasattr(d['value'], 'shape'):
        print(f'Shape: {d[\"value\"].shape}')
    print(f'Meta keys: {list(d[\"meta\"].keys())}')
"

# 4. Kill the server
pkill -f "sglang.launch_server"
```
