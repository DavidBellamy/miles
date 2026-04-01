# Fully-Async Rollout (Bilingual-Mirror Version)

This is the English entry page for fully-async documentation. Chinese and
English docs are maintained as one-to-one same-name pairs.

## Companion Docs (English)

- `docs/en/advanced/generate_hub_rollout_design.md`
- `docs/en/advanced/session_server_abort_resume.md`
- `docs/en/advanced/fully_async_no_interrupt_detailed_plan.md`
- `docs/en/advanced/fully_async_no_explicit_abort_minimal_plan.md`

## Current Implementation Summary

- Fully-async is "scheduling mode + interrupt policy":
  - `legacy_abort_resume` (default): abort at rollout end; use session-server
    gate during weight update.
  - `no_interrupt`: in continuous mode, skip rollout-end abort; control update
    windows via SGLang
    `pause_generation(mode=<retract|in_place>) -> continue_generation`.

## Maintenance Rule

- Chinese docs should point to Chinese docs, English docs to English docs.
- Same-name Chinese/English files should keep aligned structure and semantics.
