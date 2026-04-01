# tito/5.2-abort-resume: Fully-Async No-Explicit-Interrupt Plan (Low-Intrusion)

## 1. Design Constraints

- Do not change `train_async.py` main-loop semantics.
- New behavior must be optional.
- Default behavior must remain backward-compatible.
- Prefer existing extension points (`--rollout-function-path`,
  `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR`).

## 2. Summary

Provide an optional fully-async path that avoids user-visible explicit abort
interruptions:

- Rollout side can skip per-rollout-end `abort_request` (policy-controlled).
- Weight update window uses
  `pause_generation(mode=<user_selected>) -> update -> continue_generation`.
- Default behavior remains unchanged unless users opt in.

## 3. Minimal-Change Plan

### A. Policy switches

- `fully_async_interrupt_policy`
  - `legacy_abort_resume` (default)
  - `no_interrupt`
- `fully_async_pause_mode` (only when policy=`no_interrupt`)
  - `retract` (default)
  - `in_place`

Rollout preference order:
1. Read env vars first (`MILES_FULLY_ASYNC_INTERRUPT_POLICY`,
   `MILES_FULLY_ASYNC_PAUSE_MODE`) to reduce argument-chain changes.
2. Keep CLI flags as stable public interface.

### B. Keep `train_async.py` unchanged

- Keep current training-loop synchronization points.
- Apply changes only inside rollout/update implementations.
- Any aggressive scheduling experiments should stay outside mainline entry.

### C. Rollout path (policy branch only)

Files:
- `miles/rollout/inference_rollout/inference_rollout_common.py`
- `miles/rollout/inference_rollout/inference_rollout_train.py`

Behavior:
- `continuous=True` + `no_interrupt`:
  - Skip rollout-end `abort()`.
  - Phase-1 current behavior: pending requests may finish in background, but
    their results are not reused as a persistent ready queue across rollout
    calls.
  - Phase-2 target: add persistent in-flight/ready reuse across rollouts.
- `legacy_abort_resume`:
  - Keep existing behavior (including rollout-end abort).

Boundaries:
- Keep `max_buffer_staleness` logic.
- Phase-1 current behavior: `start_rollout_id` is mostly written when abort
  path recycles partial samples.
- Phase-2 target: write `start_rollout_id` at submission time for persistent
  queue staleness checks.

### D. Weight update path (mode-only change)

Files:
- `miles/backends/sglang_utils/sglang_engine.py`
- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
- `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py`

Behavior:
- `pause_generation` accepts `mode` (default remains `abort` for compatibility).
- `no_interrupt`:
  - `pause_generation(mode=<fully_async_pause_mode>)`
  - `flush_cache`
  - `update_weights_*`
  - `continue_generation()`
- `legacy_abort_resume` keeps `mode="abort"` behavior.

Note:
- No SGLang server API changes required; use existing
  `PauseGenerationReqInput.mode` contract.

### E. Session server behavior

- Keep `session_server.py` / `sessions.py` unchanged in this phase.
- `legacy_abort_resume` continues using session-server gate.
- `no_interrupt` does not use session-server abort/resume control in actor
  update path.

## 4. Public Interface / Compatibility

- Default remains `legacy_abort_resume` with no behavior change for existing
  jobs.
- `no_interrupt` is opt-in.
- `no_interrupt` exposes user pause mode choice: `retract` or `in_place`.
- `train_async.py` call graph remains unchanged.

## 5. Test Plan

### A. Rollout integration

`tests/fast/rollout/inference_rollout/integration/test_fully_async.py`:
- `test_no_interrupt_strategy_no_rollout_end_abort`
- `test_no_interrupt_strategy_preserves_batch_size`
- `test_no_interrupt_strategy_with_staleness_bound`

### B. Weight-update unit tests

- `legacy_abort_resume` -> `pause_generation(mode="abort")`
- `no_interrupt + retract` -> `pause_generation(mode="retract")`
- `no_interrupt + in_place` -> `pause_generation(mode="in_place")`
- All should call `continue_generation` for cleanup.

### C. Regression

- Keep all existing fully-async and buffer-staleness tests.
- Parameterize policy tests to ensure legacy compatibility.

### D. E2E required: mock wait-agent weather loop

Files:
- `tests/e2e/sglang/test_no_interrupt_mock_wait_weather_loop.py`
- `tests/e2e/sglang/utils/mock_wait_weather_agent.py`

D1. Agent behavior contract:
- Calls `/v1/chat/completions` every turn.
- Expects `tool_calls` each turn.
- Executes mock weather tool and appends tool message.
- Sleeps (`wait_s`) each turn to emulate wait-agent.
- Loops fixed `max_turns`.
- Reports: `turns_completed`, `total_tool_calls`, `wait_calls`,
  `aborted_seen`.

D2. Mock server capability:
- Supports `/pause_generation` and `/continue_generation`.
- Supports paused state to block requests until continue.
- Tracks `pause_calls`, `continue_calls`, `last_pause_mode`.

D3. Minimum case matrix:
1. `test_mock_wait_weather_loop_no_interrupt_retract`
2. `test_mock_wait_weather_loop_no_interrupt_in_place`
3. `test_mock_wait_weather_loop_legacy_abort_resume_control`

D4. Pass criteria:
- Stable multi-turn tool-call + wait loop (>=12 turns).
- Multiple pause/continue perturbations do not break loop.
- In no-interrupt cases, agent side does not observe
  `finish_reason=abort`.

### E. E2E required: deterministic equivalence

Goal: verify deterministic equivalence for
`mode=abort` vs `mode=retract`, and verify internal retract consistency.

Reuse:
- `tests/e2e/sglang/test_tito_logprob_equivalence.py`
- `tests/e2e/sglang/utils/logprob_verify_generate.py`

E1. Deterministic configuration:
- `--sglang-enable-deterministic-inference`
- `temperature=0.0`
- fixed rollout seed

E2. Comparison scopes:
- Scope 1 (within retract): session decode `output_token_logprobs`
  vs full re-prefill `input_token_logprobs`
- Scope 2 (cross mode): final rollout outputs from abort mode vs retract mode

E3. Strict pass criteria:
- token_id exact match
- logprob exact match (engineering tolerance `abs diff <= 1e-8`)
- when routing replay is enabled, `routed_experts` must match

E4. Coverage requirements:
- Must cover both `pause_generation(mode="abort")` and
  `pause_generation(mode="retract")`.
- Use same prompts, same deterministic settings, same seed.
- Keep one baseline run without pause perturbation for diagnosis.

## 6. Phased Delivery

- Phase 1 (current): minimal rollout + update-path changes; no `train_async`
  modification.
- Phase 2 (optional): persistent cross-rollout in-flight/ready reuse.

## 7. Assumptions

- Current target backend: Megatron path.
- `use_fault_tolerance=false` unchanged.
- Keep `use_session_server` enabled in environment; policy controls whether
  actor uses session pause/resume.

## 8. Phase-1 Implementation Record

Completed:
- Args:
  - `--fully-async-interrupt-policy`
  - `--fully-async-pause-mode`
- Engine client mode passthrough:
  - `pause_generation(mode="...")`
- Rollout:
  - `no_interrupt + continuous` skips rollout-end abort
  - metric `rollout/no_interrupt/pending_at_end`
- Weight updater:
  - policy-based pause mode selection in tensor/distributed paths
- Actor:
  - `no_interrupt` skips session pause/resume calls
- Mock server:
  - pause/continue/flush endpoints and counters

Coverage summary:
- Rollout integration: 3 new tests
- Pause-mode unit tests: 10 tests
- Mock wait-agent E2E: 3 tests
- Deterministic strict equivalence: pending real-GPU validation

Phase-2 notes:
- Persist pending-completion reuse across rollout calls.
- Move `start_rollout_id` assignment to submission time.
- Validate `retract` vs `in_place` tradeoffs against off-policy tolerance.
