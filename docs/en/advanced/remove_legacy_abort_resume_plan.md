# Remove `legacy_abort_resume` Completely: Execution Plan (Draft, not current requirement)

> Note: this is a long-term convergence/cleanup direction. For the current
> requirement, use:
> `docs/en/advanced/fully_async_no_explicit_abort_minimal_plan.md`

## 1. Background and Goal

The current fully-async stack maintains two interrupt semantics:

- `legacy_abort_resume`: explicit rollout-end `abort_request`, plus session-server gate during weight update.
- `no_interrupt`: no rollout-end abort, weight update controlled by `pause_generation(mode=<...>) -> continue_generation`.

Goal: **remove `legacy_abort_resume` end-to-end** and keep only the new fully-async semantics to reduce branching and long-term maintenance cost.

## 2. Non-Goals

- No redesign of the training main loop (`train_async.py` semantics stay intact).
- No new rollout algorithm in this plan.
- No boundary refactor between rollout and generate_hub abstractions.

## 3. Scope Definition

### 3.1 Scope A (recommended first)

Refactor path only (`inference_rollout/` + fully-async weight update + session-gate-related runtime paths).

### 3.2 Scope B (“complete” cleanup)

Scope A + remove abort/resume legacy logic in `miles/rollout/sglang_rollout.py`.

## 4. Code Size Baseline (current HEAD, static slice)

> Numbers below are static slice estimates for the fully-async subsystem, not full file LOC.

### 4.1 Refactor-path slice

- Total slice: about **281 LOC**
- Legacy-removable slice: about **160 LOC**
- Remaining with new fully-async only: about **121 LOC**
- Legacy share: about **56.9%**

### 4.2 If old `sglang_rollout` path is included

- Extra legacy abort slice: about **45 LOC**
- Total slice: about **326 LOC**
- Legacy-removable total: about **205 LOC**
- Legacy share: about **62.9%**

## 5. Removable Code Inventory

## 5.1 Rollout-end abort path

- `miles/rollout/inference_rollout/inference_rollout_train.py`
  - `abort(...)`: ~26 LOC
  - `get_worker_urls(...)`: ~7 LOC (needs reuse handling first, see 5.6)
  - `no_interrupt` vs `abort` branch block: ~12 LOC

## 5.2 aborted_samples plumbing

- `miles/rollout/inference_rollout/inference_rollout_common.py`
  - `_call_train` aborted-sample reinjection path: ~7 LOC

## 5.3 Actor-side session gate branch

- `miles/backends/megatron_utils/actor.py`
  - `use_session_server and not no_interrupt` pre/post update gate calls: ~6 LOC

## 5.4 Weight updater policy branches

- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
  - policy-based `_get_pause_mode`: ~5 LOC
- `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py`
  - policy-based `_get_pause_mode`: ~5 LOC

## 5.5 CLI policy entrypoint

- `miles/utils/arguments.py`
  - `--fully-async-interrupt-policy` argument definition: ~11 LOC

## 5.6 Session-server gate surfaces

- `miles/ray/rollout.py`
  - `pause_sessions()` / `resume_sessions()`: ~17 LOC
- `miles/rollout/session/session_server.py`
  - `_resume_event` + pause/resume gate logic: ~44 LOC
- `miles/rollout/session/sessions.py`
  - `/abort_sessions` and `/resume_sessions` endpoints: ~9 LOC
  - paused-wait-and-retry loop in chat proxy: ~11 LOC

## 5.7 Old path (Scope B)

- `miles/rollout/sglang_rollout.py`
  - `abort(...)`: ~42 LOC
  - rollout-end `abort(...)` call: ~3 LOC

## 6. Execution Strategy (Phased)

## Phase 0: Lock the interface decision

1. Decide whether to keep `--fully-async-interrupt-policy` (recommended: remove).
2. Keep `--fully-async-pause-mode` as the single knob (recommended).
3. Decide whether old `sglang_rollout` is in this round (recommended: A first, B second).

Deliverable:

- ADR/design note that fully-async supports no-interrupt semantics only.

## Phase 1: Behavior switch (without aggressive deletion)

Goal: move runtime behavior to a single path first.

Changes:

1. Remove rollout-end `abort_request` behavior.
2. Standardize weight update to `pause_generation(mode=<pause_mode>) -> continue_generation`.
3. Stop actor-side session pause/resume RPC usage.

Exit criteria:

- No regressions in core training indicators.
- No rollout-end `/abort_request` in fully-async runs.

## Phase 2: Runtime dead-code removal

Goal: delete unreachable branches after Phase 1.

Changes:

1. Remove `inference_rollout_train.abort()` and policy branch.
2. Remove aborted-samples reinjection path.
3. Remove rollout manager `pause_sessions/resume_sessions`.
4. Remove session-server abort/resume gate code and endpoints.

Exit criteria:

- Unit/integration tests pass.
- No remaining runtime legacy policy references.

## Phase 3: Old-path cleanup (Scope B)

Goal: repository-wide removal of legacy abort behavior.

Changes:

1. Remove abort logic from `miles/rollout/sglang_rollout.py`.
2. Decide whether to retire old rollout path or keep a minimal compatibility shell.

Exit criteria:

- No runtime references to `legacy_abort_resume`.
- Docs and scripts no longer expose legacy policy.

## Phase 4: Test and documentation convergence

Test updates:

1. Remove legacy control-group tests; keep and strengthen `no_interrupt/retract/in_place`.
2. Keep long-loop perturbation e2e (`pause/continue` injected multiple times).
3. Add regression coverage for session-server path without gate.

Docs updates:

1. `generate_hub_rollout_design.md`: remove dual-policy description.
2. `fully_async_rollout.md`: update to single-policy conclusion.
3. `session_server_abort_resume.md`: remove or archive as historical context.

## 7. Risks and Mitigations

## 7.1 `partial_rollout` behavior shift

Risk:

- Partial sample recovery previously depended on abort path.

Mitigation:

1. Document behavior change explicitly.
2. Add staleness/buffer monitoring.
3. If needed, implement non-abort recovery in a separate scoped feature.

## 7.2 Session-server behavioral change

Risk:

- Removing gate means no paused blocking/retry in session server.

Mitigation:

1. Keep long-loop e2e with pause/continue perturbation.
2. Monitor agent error and retry metrics.

## 7.3 Compatibility risk (CLI/scripts)

Risk:

- Old scripts may fail on removed flags.

Mitigation:

1. Optional deprecation cycle first (warning), then hard remove.
2. Provide migration notes for release consumers.

## 8. Validation Matrix (minimum)

1. Fast integration:
   - `tests/fast/rollout/inference_rollout/integration/test_fully_async.py`
2. Weight update pause-mode behavior:
   - `tests/fast/backends/megatron_utils/test_pause_mode.py` (updated for no legacy branch)
3. Long-loop e2e:
   - `tests/e2e/sglang/test_no_interrupt_mock_wait_weather_loop.py`

Pass criteria:

1. No rollout-end abort requests.
2. Both `retract` and `in_place` complete multi-turn agent loops robustly.
3. No new deadlock/hang behavior.

## 9. Rollback Strategy

1. Keep Phase 1 and Phase 2 in separate PRs for clean rollback boundaries.
2. If deprecation is used, keep one release-window transition period.
3. In production regressions, rollback to pre-switch baseline instead of ad-hoc hot branch logic.

## 10. PR Split Proposal

1. PR-1: behavior switch (single-policy runtime path).
2. PR-2: runtime dead-code cleanup (including session gate).
3. PR-3: tests and docs convergence.
4. PR-4 (optional): old `sglang_rollout` path cleanup.

This split keeps risk isolated and gives each stage clear exit criteria.
