# Session Server Abort/Resume (Legacy Path)

This document describes the **legacy session-server gate path**. It is no
longer the only fully-async behavior.

## 1. When This Path Is Active

Session-server abort/resume is active only when all conditions hold:

- `--fully-async-rollout`
- `--use-session-server`
- `--fully-async-interrupt-policy=legacy_abort_resume`

If `--fully-async-interrupt-policy=no_interrupt`, actor-side session
`pause_sessions()/resume_sessions()` is skipped.

## 2. Policy Matrix (Current)

| Policy | Rollout end | During weight update | Session server gate |
|---|---|---|---|
| `legacy_abort_resume` | Abort remaining in-flight requests | `pause_generation(mode="abort")` then update then continue | Used |
| `no_interrupt` | Skip rollout-end abort in continuous mode | `pause_generation(mode=<retract|in_place>)` then update then continue | Bypassed |

## 3. Legacy Pipeline Overview

```text
actor.update_weights()
  -> rollout_manager.pause_sessions()
     -> POST /abort_sessions (session server)
        -> clear resume_event
        -> POST /abort_request (backend)
  -> weight updater syncs weights
  -> rollout_manager.resume_sessions()
     -> POST /resume_sessions
        -> wait backend /health
        -> set resume_event
```

## 4. Gate Mechanism in `chat_completions`

The session route wraps proxying with a pause gate:

```python
while True:
    await backend.resume_event.wait()
    result = await backend.do_proxy(...)
    if not backend.is_paused():
        break
    # paused while request was in-flight:
    # discard partial response and retry after resume
```

Behavior:
- Request arrives while paused: blocks until resume.
- Request aborted mid-flight: partial result is discarded and retried.
- Agent receives one final response (not an explicit abort failure).

## 5. Scope and Non-Goals

- This mechanism is about transport-level continuity during update windows.
- It does **not** define the new no-interrupt policy behavior.
- For current fully-async behavior across both policies, see:
  `docs/en/advanced/generate_hub_rollout_design.md`.
