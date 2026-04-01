# Fully-Async 最小改动方案：`no_interrupt` 下不显式 abort（Draft）

## 1. 目标（修正后的需求）

本方案目标是：在 fully-async 场景下，实现 `in_place / retract` 两种 pause mode，**不依赖显式 abort**，并且保持改动尽可能小。

关键点：

1. **不是**全面移除 `legacy_abort_resume`。
2. `legacy_abort_resume` 继续保留，作为默认/兼容路径。
3. `no_interrupt` 作为可选策略，只在需要时启用。

## 2. 设计原则（最小侵入）

1. 不修改 `train_async.py` 主循环语义。
2. 不改 generate_hub 抽象边界。
3. 仅在 rollout 末尾行为和 weight update 暂停行为上加策略分支。
4. 默认行为不变（即未显式开启时与当前一致）。

## 3. 目标行为矩阵

| 场景 | rollout 末尾 | weight update 窗口 |
|---|---|---|
| `legacy_abort_resume`（默认） | 维持现状：可走 abort 路径 | 维持现状：`pause_generation(mode=\"abort\") -> continue_generation` |
| `no_interrupt` + `fully_async_rollout` | 不显式 `abort_request` | `pause_generation(mode=<retract|in_place>) -> continue_generation` |

补充：

- `fully_async_pause_mode` 在 `no_interrupt` 下生效，允许 `retract` / `in_place`。
- 不改变已有 fallback 和兼容分支。

## 4. 最小代码改动范围

## 4.1 配置与参数（保留双策略）

文件：

- `miles/utils/arguments.py`

要求：

1. 保留 `--fully-async-interrupt-policy`（`legacy_abort_resume` / `no_interrupt`）。
2. 保留 `--fully-async-pause-mode`（`retract` / `in_place`）。
3. 默认值继续兼容现有作业。

## 4.2 Rollout 结束行为（仅策略分支）

文件：

- `miles/rollout/inference_rollout/inference_rollout_train.py`

要求：

1. 在 `continuous=True` 且 `policy=no_interrupt` 时，rollout 末尾不显式 abort。
2. 其它路径（含 legacy）保持原行为。
3. 继续输出可观测指标（例如 pending 相关 metric），便于验证和回归比较。

## 4.3 Weight Update 暂停行为（核心需求）

文件：

- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
- `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py`
- `miles/backends/sglang_utils/sglang_engine.py`

要求：

1. `no_interrupt` 时按 `fully_async_pause_mode` 传递 `pause_generation(mode=...)`。
2. 保证更新窗口顺序不变：
   - `pause_generation` -> `flush_cache` -> 权重同步 -> `continue_generation`
3. legacy 分支保持 `mode=\"abort\"` 兼容。

## 4.4 Session Server（尽量不动）

文件：

- `miles/backends/megatron_utils/actor.py`
- `miles/ray/rollout.py`
- `miles/rollout/session/session_server.py`
- `miles/rollout/session/sessions.py`

策略：

1. 不做结构性删除。
2. 只保证 `no_interrupt` 路径不要求依赖 session gate 才能工作。
3. legacy gate 行为保留，避免扩大改动面。

## 5. 明确不做（本次边界）

1. 不删除 `legacy_abort_resume` 相关代码。
2. 不清理旧 `sglang_rollout.py` 路径。
3. 不做大规模 dead-code 清理。
4. 不改训练/调度主干流程。

以上项可作为后续“代码收敛”单独立项，避免与本次需求耦合。

## 6. 测试计划（聚焦需求闭环）

## 6.1 快速集成测试

文件：

- `tests/fast/rollout/inference_rollout/integration/test_fully_async.py`

覆盖：

1. `no_interrupt` 下 rollout 末尾不发 `abort_request`。
2. `no_interrupt` 仍严格收满 `rollout_batch_size`。
3. staleness/buffer 相关行为不回归。

## 6.2 Pause mode 单测

文件：

- `tests/fast/backends/megatron_utils/test_pause_mode.py`

覆盖：

1. `no_interrupt + retract` -> `pause_generation(mode=\"retract\")`
2. `no_interrupt + in_place` -> `pause_generation(mode=\"in_place\")`
3. legacy 路径保持 `mode=\"abort\"`（兼容性回归）。

## 6.3 长链路 e2e（必须）

文件：

- `tests/e2e/sglang/test_no_interrupt_mock_wait_weather_loop.py`

覆盖：

1. `no_interrupt + retract`：多轮 tool-call + wait 在 pause/continue 扰动下持续完成。
2. `no_interrupt + in_place`：同上。
3. legacy control 组保留，用作回归对照，不作为本次删除目标。

## 7. 验收标准

1. `no_interrupt` 路径下，rollout 末尾无显式 abort。
2. `retract` 与 `in_place` 两种模式都可稳定完成完整训练/推理闭环。
3. 默认配置行为不变（兼容 legacy）。
4. 无新增 deadlock、无限等待、训练中断。

## 8. 风险与缓解

## 8.1 风险：需求外扩导致改动失控

缓解：

1. 严格不做“全面移除 legacy”。
2. 将清理型工作拆到后续独立 PR。

## 8.2 风险：`partial_rollout` 与 buffer 行为扰动

缓解：

1. 保留现有逻辑与指标。
2. 对 `no_interrupt` 仅做策略层行为切换，不引入额外回收机制。

## 8.3 风险：session server 链路耦合

缓解：

1. 本次不删 session gate，只验证 `no_interrupt` 可独立工作。
2. 将 session gate 清理放到后续独立阶段。

## 9. PR 切分建议（小步可回滚）

1. PR-1：`no_interrupt` 行为闭环（rollout + updater + 参数 + 基础测试）。
2. PR-2：补充/加固 e2e 与指标观测。
3. PR-3（可选后续）：技术债清理（若后续决定再推进）。

这个切分可以保证每一步都可验证、可回滚、并且与“最小改动”目标一致。
