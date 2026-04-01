# Generate Hub + Rollout Design (Current Implementation)

This document describes the current rollout stack and the fully-async behavior
as implemented in code.

## 1. Layer Boundaries

There are two different abstractions:

1. `GenerateFn` (in `generate_hub/`)
- Per-sample generation semantics.
- Input: one `Sample`, generation state, sampling params.
- Output: one `Sample` (or list for multi-sample variants).

2. `RolloutFn` (in `inference_rollout/`)
- Per-rollout scheduling semantics.
- Input: rollout id + data source + rollout config.
- Output: grouped samples for one train step.

Rule of thumb:
- "How one sample talks to model" -> `generate_hub`.
- "How many samples are scheduled/filtered/aborted" -> rollout layer.

## 2. Runtime Path

```text
RolloutManager
  -> InferenceRolloutFn._call_train
    -> generate_rollout_async(..., continuous=fully_async_rollout)
      -> submit_generate_tasks
        -> generate_and_rm_group
          -> generate_and_rm
            -> generate_hub.<variant>.generate
```

## 3. Fully-Async and Interrupt Policies

Fully-async is enabled by:
- `--fully-async-rollout`

Interrupt behavior is controlled by:
- `--fully-async-interrupt-policy`
  - `legacy_abort_resume` (default)
  - `no_interrupt`
- `--fully-async-pause-mode` (only used when policy=`no_interrupt`)
  - `retract` (default)
  - `in_place`

### 3.1 Submission policy (`continuous=True`)

In fully-async mode, rollout submission keeps in-flight tasks near
`over_sampling_batch_size` instead of only targeting `rollout_batch_size`.

### 3.2 End-of-rollout behavior (important)

After enough valid groups are collected:

- `legacy_abort_resume`:
  - Calls rollout `abort()`.
  - Sends `/abort_request` to workers.
  - Optionally collects partial groups for buffer reuse (`partial_rollout`).

- `no_interrupt` (only when `continuous=True`):
  - Skips rollout `abort()` at rollout end.
  - Leaves in-flight requests running.
  - Reports metric `rollout/no_interrupt/pending_at_end`.

Current Phase-1 behavior: pending tasks left by `no_interrupt` are not fed back
into this rollout output as reusable buffered groups.

## 4. Weight Update Behavior

Weight update behavior is split across actor/session server/engine:

- Actor (`actor.update_weights`):
  - If `use_session_server` and policy is NOT `no_interrupt`, call
    `rollout_manager.pause_sessions()` before update and
    `rollout_manager.resume_sessions()` after update.
  - If policy is `no_interrupt`, skip session pause/resume.

- Weight updater (`update_weight_from_tensor.py` and
  `update_weight_from_distributed.py`):
  - Selects pause mode via policy:
    - `legacy_abort_resume` -> `pause_generation(mode="abort")`
    - `no_interrupt` -> `pause_generation(mode=fully_async_pause_mode)`
  - Sequence remains:
    `pause_generation -> flush_cache -> sync weights -> continue_generation`

- SGLang engine client:
  - `pause_generation(mode: str = "abort")` forwards mode to
    `/pause_generation`.

## 5. Session Server Scope

Session-server abort/resume is now a policy-dependent mechanism:

- Active path: `legacy_abort_resume`.
- Bypassed path: `no_interrupt`.

In `legacy_abort_resume`, session server gate logic can absorb abort/retry so
agent-side requests do not fail due to mid-update interruption.

## 6. Staleness and Partial Rollout

`max_buffer_staleness` is implemented in the data source/buffer layer for
partial-rollout buffered groups.

Notes:
- Legacy abort path is the main source of reusable partial groups.
- In current no-interrupt Phase-1 behavior, rollout-end pending tasks are not
  converted into buffered partial groups by default.

## 7. Tests to Trust

- Rollout integration:
  - `tests/fast/rollout/inference_rollout/integration/test_fully_async.py`
- Pause mode / policy mapping:
  - `tests/fast/backends/megatron_utils/test_pause_mode.py`
- Mock wait-agent resilience under pause/continue perturbation:
  - `tests/e2e/sglang/test_no_interrupt_mock_wait_weather_loop.py`
- Deterministic logprob equivalence verification utility:
  - `tests/e2e/sglang/test_tito_logprob_equivalence.py`
  - `tests/e2e/sglang/utils/logprob_verify_generate.py`

## 8. Historical Prototype

`examples/fully_async/fully_async_rollout.py` is a prototype and not the
production rollout path.
