# CI Invariants and Validation Patterns

Complete reference for what Miles CI checks and what thresholds are enforced.

## 1. Log-Prob Parity Checks (log_utils.py:171-202)

Triggered at `rollout_id == 0` when `args.ci_test` and not `args.ci_disable_logprobs_checker`.

| Check | Condition | Tolerance | When |
|-------|-----------|-----------|------|
| Megatron == Reference | `log_probs` vs `ref_log_probs` | 1e-9 (default) | Always at rollout 0 |
| Megatron == Reference (R3) | Same, but R3 adds routing noise | 1e-5 | `use_rollout_routing_replay=True` |
| Megatron == Reference (multi-model) | Offload/onload cycle noise | 1e-8 | `sglang_config is not None` |
| Megatron ≈ SGLang | `log_probs` vs `rollout_log_probs` | 0.03 | Always at rollout 0 |
| Entropy bounds | `0 < entropy < 0.7` | - | Always at rollout 0 |
| True on-policy exact | `log_probs == rollout_log_probs` | exact | `true_on_policy_mode=True` |

## 2. KL / Policy Checks (ci_utils.py:12-24)

Triggered at `step_id == 0`:

| Check | Condition | Threshold | When |
|-------|-----------|-----------|------|
| PPO KL (default) | `abs(ppo_kl)` | < 1e-10 | Standard mode |
| PPO KL (MLA) | `ppo_kl` | < 1e-8 | `multi_latent_attention=True` (non-deterministic) |
| PPO KL (LoRA) | `abs(ppo_kl)` | < 1e-8 | `lora_rank > 0` |
| Clip fraction | `abs(pg_clipfrac)` | < 1e-10 | Standard and LoRA |
| KL loss | `abs(kl_loss)` | < 1e-9 | At `accumulated_step_id == 0`, no R3 |

## 3. Gradient Norm Validation (ci_utils.py:27-59)

- Save mode: `--ci-save-grad-norm path/{role}_{rollout_id}_{step_id}.pt`
- Load+compare mode: `--ci-load-grad-norm path/{role}_{rollout_id}_{step_id}.pt`
- Tolerance: `rel_tol=0.03, abs_tol=0.03`
- Only validates on rank 0

## 4. Dumper Mode Overrides (arguments.py + test_arguments.py)

When `--dumper-enable` is set, Miles forces:
```python
args.use_fault_tolerance = False          # Disable heartbeats
args.router_disable_health_check = True   # Disable router health check
args.rollout_health_check_interval = 1e18 # Effectively disable rollout health check
args.num_rollout = start_rollout_id + 1   # Single rollout only
args.eval_interval = None                 # No evaluation
args.save_interval = None                 # No checkpoint saving
```

This prevents the dumper run from being interrupted by health checks or saving.

## 5. Configuration Validation (arguments.py:1805-2154)

### Mutual Exclusivity
- Only one of `kl_coef` or `kl_loss_coef` can be non-zero
- `debug_rollout_only` and `debug_train_only` are mutually exclusive
- Cannot combine certain PPO modes

### Prerequisites
- `eval_datasets` required when `eval_interval` is set
- `--save` required when `save_interval` is set
- `--target-modules` required for LoRA
- `max_tokens_per_gpu` must be set when `use_dynamic_batch_size=True`

### Warnings (non-fatal)
- Missing chat template fix rules
- Colocate NVLS OOM risk
- `debug_rollout_only` forcing `memory_margin=0`

### Fatal Errors
- `FileNotFoundError`: missing chat template file, missing ref_load checkpoint
- `ValueError`: invalid eval config structure, invalid qkv_format

## 6. Distributed State Invariants (parallel.py:14-37)

After initialization, Miles validates:
```python
assert actual_rank == self.rank, f"{name}: rank mismatch"
assert actual_size == self.size, f"{name}: size mismatch"
```

## 7. PPO Training Assertions (ppo_utils.py)

- `eps_clip_c > 1.0` when dual-clip PPO is enabled
- `full_mask.sum() > 0` — no fully-masked sequences allowed
- `B == len(values_list) == len(rewards_list)` — batch size consistency

## 8. Model Conversion Support (megatron_to_hf/__init__.py:34-54)

Supported models (in matching order):
1. `glm4moelite` / `deepseekv3`
2. `glm4moe`
3. `glm4`
4. `qwen3moe`
5. `qwen3next`
6. `qwen3_5`
7. `qwen2` / `qwen3`
8. `llama`
9. `mimo`

Unsupported models raise `ValueError`.

## 9. Checkpoint Finalization Protocol

```python
# Save checkpoint
torch.save(state_dict, tmp_path)
# Barrier to ensure all ranks finish saving
dist.barrier()
# Rename to final path (atomic on same filesystem)
os.rename(tmp_path, final_path)
# Write tracker file with "release" sentinel
# Tracker file presence signals successful completion
```

## 10. Memory Management Sequence

```python
def clear_memory(clear_host_memory=False):
    torch.cuda.synchronize()     # Wait for all GPU ops
    gc.collect()                  # Python garbage collection
    torch.cuda.empty_cache()     # Release cached GPU memory
    if clear_host_memory:
        torch._C._host_emptyCache()  # Release pinned host memory
```

Called at: initialization, weight sync, rollout transitions, and when GPU free memory < 10%.

## 11. Health Monitoring

- **Rollout health check**: interval-based, configurable timeout
- **First-wait delay**: `rollout_health_check_first_wait` for MoE model readiness after weight update
- **Lifecycle**: start → pause (offload) → resume (onload) → stop
- **Disabled by dumper mode**: set interval to 1e18
