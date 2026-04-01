# Fully-Async Minimal-Change Plan: No Explicit Abort in `no_interrupt` (Draft)

## 1. Goal (corrected requirement)

Goal: in fully-async mode, support `in_place / retract` pause modes **without explicit abort**, with minimal code changes.

Key points:

1. This is **not** a full removal of `legacy_abort_resume`.
2. Keep `legacy_abort_resume` as a compatibility/default path.
3. Use `no_interrupt` as an opt-in strategy.

## 2. Design Principles (minimal intrusion)

1. Do not change `train_async.py` main-loop semantics.
2. Do not refactor generate_hub abstraction boundaries.
3. Restrict changes to rollout-end behavior and weight-update pause behavior.
4. Keep default behavior backward compatible.

## 3. Target Behavior Matrix

| Scenario | rollout end | weight-update window |
|---|---|---|
| `legacy_abort_resume` (default) | keep current abort-capable path | keep `pause_generation(mode=\"abort\") -> continue_generation` |
| `no_interrupt` + `fully_async_rollout` | no explicit `abort_request` | `pause_generation(mode=<retract|in_place>) -> continue_generation` |

Notes:

- `fully_async_pause_mode` applies under `no_interrupt`: `retract` or `in_place`.
- Existing fallback/compat paths remain intact.

## 4. Minimal Code-Change Surface

## 4.1 Config/Arguments (keep dual strategy)

File:

- `miles/utils/arguments.py`

Requirements:

1. Keep `--fully-async-interrupt-policy` (`legacy_abort_resume` / `no_interrupt`).
2. Keep `--fully-async-pause-mode` (`retract` / `in_place`).
3. Preserve current defaults for compatibility.

## 4.2 Rollout-end behavior (policy branch only)

File:

- `miles/rollout/inference_rollout/inference_rollout_train.py`

Requirements:

1. Under `continuous=True` and `policy=no_interrupt`, skip explicit rollout-end abort.
2. Keep existing behavior for all other paths (including legacy).
3. Keep observability metrics (e.g., pending-related metric) for validation.

## 4.3 Weight-update pause behavior (core requirement)

Files:

- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
- `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py`
- `miles/backends/sglang_utils/sglang_engine.py`

Requirements:

1. Under `no_interrupt`, pass `pause_generation(mode=<fully_async_pause_mode>)`.
2. Keep update-window order unchanged:
   - `pause_generation` -> `flush_cache` -> weight sync -> `continue_generation`
3. Keep legacy branch with `mode=\"abort\"` for compatibility.

## 4.4 Session Server (avoid structural change)

Files:

- `miles/backends/megatron_utils/actor.py`
- `miles/ray/rollout.py`
- `miles/rollout/session/session_server.py`
- `miles/rollout/session/sessions.py`

Policy:

1. No structural deletion in this scope.
2. Ensure `no_interrupt` path does not require session gate for correctness.
3. Keep legacy gate behavior to avoid broad churn.

## 5. Explicitly Out of Scope

1. Do not remove all `legacy_abort_resume` code in this iteration.
2. Do not clean old `sglang_rollout.py` path now.
3. Do not do broad dead-code cleanup now.
4. Do not change training/scheduling main flow.

These can be tracked as follow-up technical-debt work, separate from the current requirement.

## 6. Test Plan (requirement-focused)

## 6.1 Fast integration

File:

- `tests/fast/rollout/inference_rollout/integration/test_fully_async.py`

Coverage:

1. Under `no_interrupt`, rollout-end `abort_request` is not sent.
2. `rollout_batch_size` is still fully collected.
3. staleness/buffer behavior does not regress.

## 6.2 Pause-mode unit tests

File:

- `tests/fast/backends/megatron_utils/test_pause_mode.py`

Coverage:

1. `no_interrupt + retract` -> `pause_generation(mode=\"retract\")`
2. `no_interrupt + in_place` -> `pause_generation(mode=\"in_place\")`
3. Legacy remains `mode=\"abort\"` (compat regression check).

## 6.3 Long-loop e2e (required)

File:

- `tests/e2e/sglang/test_no_interrupt_mock_wait_weather_loop.py`

Coverage:

1. `no_interrupt + retract`: multi-turn tool-call + wait loop survives pause/continue perturbations.
2. `no_interrupt + in_place`: same.
3. Keep legacy control group as regression baseline; not a deletion target in this plan.

## 7. Acceptance Criteria

1. No explicit rollout-end abort in `no_interrupt`.
2. Both `retract` and `in_place` run stably end-to-end.
3. Default configuration remains backward compatible.
4. No new deadlock/hang introduced.

## 8. Risks and Mitigations

## 8.1 Scope creep risk

Mitigation:

1. Do not mix in full legacy removal.
2. Move cleanup work into separate follow-up PRs.

## 8.2 `partial_rollout` / buffer interaction risk

Mitigation:

1. Keep existing behavior and metrics intact.
2. Only switch policy-level behavior for `no_interrupt`.

## 8.3 Session-server coupling risk

Mitigation:

1. Do not remove session gate in this scope.
2. Validate `no_interrupt` correctness independently through e2e.

## 9. PR Split (small and rollback-safe)

1. PR-1: `no_interrupt` behavior closure (rollout + updater + args + core tests).
2. PR-2: e2e hardening and observability additions.
3. PR-3 (optional follow-up): technical-debt cleanup.

This split keeps risk controlled and aligns with the minimal-change requirement.
