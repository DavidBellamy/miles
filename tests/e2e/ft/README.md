# Fault Tolerance E2E Tests

### How tests consume it

`conftest_ft/execution.py` `prepare()` downloads the data via `U.hf_download_dataset()`. Then `get_common_train_args()` passes `--load-debug-rollout-data` and `--debug-train-only` to the training command (for modes with `rollout_gpus == 0`).

## Test Overview

| Test | Type | What it verifies |
|------|------|-----------------|
| `test_trainer_ft_no_failure.py` | Comparison (baseline vs target) | indep_dp and normal DP produce same results without faults |
| `test_trainer_ft_with_failure.py` | Comparison, multi-phase | Weights/grads roughly match after fault + ckpt resume |
| `test_trainer_ft_deterministic.py` | Non-comparison | Healed cell has bitwise-equal state to source cell |
| `test_ft_random.py` | Non-comparison | System survives random crashes without hanging |

## Mode Variants

Each test runs with `--mode`:

| Mode | DP cells | Parallelism | Rollout GPUs | Coverage |
|------|----------|-------------|-------------|----------|
| `dp2_cp2_tp2_ep2` | 2 | CP2 TP2 EP2 | 0 | TP + EP |
| `dp2_cp2_pp2` | 2 | CP2 PP2 | 0 | PP |
| `dp4_cp2` | 4 | CP2 | 0 | Multi-replica (>=4 cells) |
| `dp2_cp2_real_rollout` | 2 | CP2 | 4 | Real weight update path |

## Running

```bash
# Comparison tests: full pipeline
python tests/e2e/ft/test_trainer_ft_no_failure.py run --mode dp2_cp2_tp2_ep2

# Comparison tests: step by step (for debugging)
python tests/e2e/ft/test_trainer_ft_no_failure.py baseline --mode dp2_cp2_tp2_ep2 --dump-dir /tmp/ft
python tests/e2e/ft/test_trainer_ft_no_failure.py target   --mode dp2_cp2_tp2_ep2 --dump-dir /tmp/ft
python tests/e2e/ft/test_trainer_ft_no_failure.py compare  --mode dp2_cp2_tp2_ep2 --dump-dir /tmp/ft

# Non-comparison tests
python tests/e2e/ft/test_trainer_ft_deterministic.py run --mode dp2_cp2_tp2_ep2
python tests/e2e/ft/test_ft_random.py run --mode dp4_cp2 --seed 42 --num-steps 50
```

## Generate Debug Rollout Data (If not on HuggingFace)

FT tests use pre-recorded rollout data (`--load-debug-rollout-data --debug-train-only`) to skip real rollout generation and save GPU resources.

### How to generate

Any `run_*.py` script with `--mode debug_minimal` already dumps rollout data via `--save-debug-rollout-data` (part of dump details). To regenerate:

```bash
# Step 1: Run with debug_minimal to dump rollout data (use the trimmed model)
python scripts/run_qwen3_30b_a3b.py --mode debug_minimal --num-rollout 10

# Step 2: Locate the dumped rollout data
ls /root/output/<run_id>/dump_details/

# Step 3: Upload to HF
huggingface-cli upload --repo-type dataset fzyzcjy/miles-ft-test-debug-rollout-data <path>
```
