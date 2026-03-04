# FT End-to-End Tests

Integration tests that validate the complete fault-tolerance pipeline:
fault injection → detection → recovery → training continuation.

## Prerequisites

- **Ray cluster**: >= 4 GPU nodes, accessible via `RAY_ADDRESS`
- **Training entrypoint**: small-model training command via `FT_E2E_TRAINING_ENTRYPOINT`
- **K8s access**: Controller uses K8s API for node management

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RAY_ADDRESS` | Ray dashboard URL (e.g. `http://head-node:8265`) |
| `FT_E2E_TRAINING_ENTRYPOINT` | Training job command (e.g. `cd miles && python miles/run_train.py --small-model`) |

## Running

```bash
# Run all E2E tests
pytest tests/e2e/ft/ -m e2e -v

# Run a specific scenario
pytest tests/e2e/ft/test_transient_crash.py -v
```

## Scenarios

| Test | Fault | Expected Path | ~Runtime |
|------|-------|---------------|----------|
| `test_transient_crash` | kill -9 one process | ENTER_RECOVERY → CHECK_ALERTS → REATTEMPTING → MONITORING → DONE | ~3 min |
| `test_repeated_crash` | kill twice | → DIAGNOSING → NOTIFY | ~5 min |
| `test_hang` | SIGSTOP | HangDetector → ENTER_RECOVERY | ~10 min |
| `test_mfu_decline` | GPU stress | MfuDeclineDetector → MARK_BAD or NOTIFY | ~10 min |
| `test_disk_full` | fill disk | HighConfidenceHardwareDetector → MARK_BAD → eviction | ~3 min |

**Total expected runtime**: ~30-60 min
