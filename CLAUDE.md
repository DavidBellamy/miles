# Miles

Miles is a distributed RLHF/GRPO training framework that colocates Megatron-LM training with SGLang inference on the same GPU cluster. It supports tensor parallelism (TP), pipeline parallelism (PP), context parallelism (CP), expert parallelism (EP), and sequence parallelism (SP).

## Architecture Overview

```
SGLang (inference) ←──weight sync──→ Megatron (training)
       │                                    │
       └── rollout_log_probs               └── log_probs, ref_log_probs
                    │                              │
                    └──────── PPO/GRPO loss ────────┘
```

- **SGLang**: Generates rollouts (token sequences + log-probs)
- **Megatron**: Recomputes log-probs, computes policy loss, updates weights
- **Weight sync**: Megatron→HF conversion, optional FP8/INT4 quantization, distribution to SGLang
- **Colocate mode**: Both engines share the same GPUs via Ray actors

## Key Directories

| Path | Description |
|------|-------------|
| `miles/backends/megatron_utils/` | Megatron training backend (model.py has training loop) |
| `miles/backends/experimental/fsdp_utils/` | FSDP training backend |
| `miles/rollout/` | SGLang rollout integration |
| `miles/ray/` | Ray actor management, weight distribution |
| `miles/utils/` | Utilities: dumper, arguments, loss, PPO |
| `miles/backends/megatron_utils/megatron_to_hf/` | Weight conversion + quantization |
| `miles_plugins/models/` | Model-specific plugins (attention, CP utils) |
| `tests/e2e/` | End-to-end tests including dumper tests |
| `tests/fast/` | Unit tests |

## Debugging Skills

This project ships with Claude Code debugging skills in `.claude-skills/miles-debugger/`. Available skills:

| Skill | Purpose |
|-------|---------|
| `/debug-distributed` | Hub for all distributed training debugging (TP/PP/EP/CP) |
| `/debug-shape` | Tensor shape mismatch diagnosis |
| `/debug-hang` | Distributed hang / deadlock diagnosis |
| `/debug-precision` | NaN/Inf, FP8/BF16, quantization issues |
| `/debug-logprob` | Log-prob mismatch, KL divergence, training-inference gap |
| `/debug-weight-sync` | Weight update/sync bugs |
| `/debug-colocate` | Colocate mode, Ray actor, GPU contention |
| `/dumper-usage` | Dumper API, comparator CLI, source patcher |
| `/dumper-dims` | Dims annotation syntax for tensor sharding |

## Dumper System

The dumper (from SGLang) captures intermediate tensors during training/inference for comparison:

```bash
# Enable dumper in Miles:
--dumper-enable --dumper-dir /tmp/debug_dumps \
--dumper-source-patcher-config-train megatron_patcher.yaml \
--dumper-source-patcher-config-inference sglang_patcher.yaml

# Compare results:
python -m sglang.srt.debug_utils.comparator \
  --baseline-path /tmp/debug_dumps/engines/ \
  --target-path /tmp/debug_dumps/fwd_bwd/ \
  --preset sglang_megatron
```

Reference source patcher configs are in `tests/e2e/conftest_dumper.py`.

## Common Commands

```bash
# Run unit tests
pytest tests/fast/ -x

# Run e2e dumper test
python tests/e2e/short/test_dumper.py run --mode tp2_pp2_cp2_ep2_etp2

# Convert checkpoint
python tools/convert_hf_to_torch_dist.py --model Qwen/Qwen3-30B-A3B --num-gpus 8

# Lint
ruff check . && black --check . && isort --check .
```

## CI Thresholds

At step 0, these invariants should hold:
- `ppo_kl < 1e-10` (standard) or `< 1e-8` (MLA/LoRA)
- `pg_clipfrac < 1e-10`
- `abs(log_probs - rollout_log_probs) < 0.03`
- `abs(log_probs - ref_log_probs) < 1e-9`
