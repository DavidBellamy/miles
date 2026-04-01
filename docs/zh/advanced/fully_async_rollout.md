# Fully-Async Rollout（中英文双份同步版）

本文是 fully-async 相关文档的中文入口页。中文与英文文档按同名一一对应维护。

## 配套文档（中文）

- `docs/zh/advanced/generate_hub_rollout_design.md`
- `docs/zh/advanced/session_server_abort_resume.md`
- `docs/zh/advanced/fully_async_no_interrupt_detailed_plan.md`
- `docs/zh/advanced/fully_async_no_explicit_abort_minimal_plan.md`

## 当前实现结论

- fully-async 是“调度模式 + 中断策略”：
  - `legacy_abort_resume`（默认）：rollout 末尾 abort；weight update 期间使用 session server gate。
  - `no_interrupt`：continuous 模式下 rollout 末尾不主动 abort；weight update 期间由 SGLang
    `pause_generation(mode=<retract|in_place>) -> continue_generation` 控制。

## 维护约定

- 中文看中文、英文看英文，不再跨语言互相跳转。
- 同名中英文文件应保持结构和语义一致。
