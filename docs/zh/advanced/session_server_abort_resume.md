# Session Server Abort/Resume（legacy 路径）

本文只描述 `legacy_abort_resume` 策略下的 session server gate，不代表
fully-async 全部行为。

## 1. 生效条件

以下条件同时满足时启用：

- `--fully-async-rollout`
- `--use-session-server`
- `--fully-async-interrupt-policy=legacy_abort_resume`

当策略为 `no_interrupt` 时，actor 侧会跳过
`pause_sessions()/resume_sessions()`。

## 2. 策略矩阵（当前实现）

| 策略 | rollout 结束 | weight update 期间 | session server gate |
|---|---|---|---|
| `legacy_abort_resume` | abort 剩余 in-flight | `pause_generation(mode="abort") -> update -> continue` | 使用 |
| `no_interrupt` | continuous 下跳过 rollout-end abort | `pause_generation(mode=<retract|in_place>) -> update -> continue` | 跳过 |

## 3. legacy 流程

```text
actor.update_weights()
  -> rollout_manager.pause_sessions()
     -> POST /abort_sessions
        -> clear resume_event
        -> POST /abort_request (backend)
  -> weight updater 同步权重
  -> rollout_manager.resume_sessions()
     -> POST /resume_sessions
        -> 等待 backend /health
        -> set resume_event
```

## 4. `chat_completions` gate 机制

```python
while True:
    await backend.resume_event.wait()
    result = await backend.do_proxy(...)
    if not backend.is_paused():
        break
    # 若请求在飞时被暂停：丢弃 partial response，resume 后重试
```

行为：
- 请求在 paused 状态到达：阻塞到 resume。
- 请求在飞时遇到 abort：丢弃 partial 结果并重试。
- agent 侧接收一个最终响应，而不是显式 abort 错误。

## 5. 范围说明

- 该机制的目标是 update 窗口的传输层连续性。
- 它不定义 `no_interrupt` 的完整行为。
- fully-async 全量行为请看中文配套文档：
  - `docs/zh/advanced/fully_async_rollout.md`
  - `docs/zh/advanced/generate_hub_rollout_design.md`
  - `docs/zh/advanced/fully_async_no_interrupt_detailed_plan.md`
