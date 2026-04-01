# 彻底移除 `legacy_abort_resume`：实施计划（Draft，非当前需求）

> 说明：该文档描述“长期收敛/清理”方向。当前需求请优先参考：
> `docs/zh/advanced/fully_async_no_explicit_abort_minimal_plan.md`

## 1. 背景与目标

当前 fully-async 路径同时维护两套中断语义：

- `legacy_abort_resume`：rollout 末尾显式 `abort_request`，weight update 期间依赖 session server gate。
- `no_interrupt`：rollout 末尾不主动 abort，weight update 通过 `pause_generation(mode=<...>) -> continue_generation`。

本计划目标是：在工程上**彻底移除 `legacy_abort_resume`**，使 fully-async 只保留新版语义，降低分支复杂度和维护成本。

## 2. 非目标（明确不做）

- 不在本计划内重做训练主循环调度（`train_async.py` 语义保持不变）。
- 不在本计划内引入新的 rollout 算法。
- 不在本计划内重构 generate_hub 抽象边界。

## 3. 范围定义

### 3.1 范围 A（建议 Phase 1 先做）

只覆盖 refactor 路径（`inference_rollout/` + fully-async weight update + session gate 相关路径）。

### 3.2 范围 B（“彻底”定义）

范围 A + 旧路径 `miles/rollout/sglang_rollout.py` 中的 abort/resume 实现一起移除。

## 4. 代码量基线（当前 HEAD 静态估算）

> 说明：以下数字是对 fully-async 子系统相关代码片段做的静态切片统计，不是整个文件总行数。

### 4.1 Refactor 路径切片

- 切片总量：约 **281 行**
- 其中 `legacy_abort_resume` 相关可删量：约 **160 行**
- 删除后保留（仅新版 fully-async）：约 **121 行**
- `legacy` 占比：约 **56.9%**

### 4.2 若纳入旧 `sglang_rollout` 路径

- 额外 legacy abort 相关：约 **45 行**
- 合计切片：约 **326 行**
- 其中 legacy 可删：约 **205 行**
- `legacy` 占比：约 **62.9%**

## 5. 可删除代码清单（按模块）

## 5.1 Rollout 末尾 abort 分支

- `miles/rollout/inference_rollout/inference_rollout_train.py`
  - `abort(...)`：约 26 行
  - `get_worker_urls(...)`：约 7 行（需先处理复用方，见 5.6）
  - `no_interrupt` vs `abort` 分支块：约 12 行

## 5.2 aborted_samples 回灌链路

- `miles/rollout/inference_rollout/inference_rollout_common.py`
  - `_call_train` 中 `aborted_samples` 回灌数据源逻辑：约 7 行

## 5.3 Actor 侧 session gate 分支

- `miles/backends/megatron_utils/actor.py`
  - `use_session_server and not no_interrupt` 前后 gate：约 6 行

## 5.4 Weight updater 的策略分支

- `miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
  - `_get_pause_mode` 中 policy 分支：约 5 行
- `miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py`
  - `_get_pause_mode` 中 policy 分支：约 5 行

## 5.5 参数入口（策略开关）

- `miles/utils/arguments.py`
  - `--fully-async-interrupt-policy` 参数定义：约 11 行

## 5.6 Session server 相关接口与 gate

- `miles/ray/rollout.py`
  - `pause_sessions()` / `resume_sessions()`：约 17 行
- `miles/rollout/session/session_server.py`
  - `_resume_event` + pause/resume gate 逻辑：约 44 行
- `miles/rollout/session/sessions.py`
  - `/abort_sessions` 与 `/resume_sessions` endpoint：约 9 行
  - chat proxy 中“paused 等待 + 重试”逻辑：约 11 行

## 5.7 旧路径（若范围 B）

- `miles/rollout/sglang_rollout.py`
  - `abort(...)`：约 42 行
  - rollout 末尾调用 `abort(...)`：约 3 行

## 6. 实施策略（分阶段）

## Phase 0：冻结接口决策（必须先定）

1. 是否保留 `--fully-async-interrupt-policy`（建议删除）。
2. `--fully-async-pause-mode` 是否保留（建议保留，作为唯一策略控制项）。
3. 是否将旧 `sglang_rollout` 路径纳入本次（建议分两阶段，先 A 后 B）。

交付物：

- ADR/设计记录：明确“fully-async 仅支持 no-interrupt 语义”。

## Phase 1：行为切换（不追求一次删净）

目标：先把运行时行为切到单一路径，保证线上语义稳定。

变更点：

1. rollout 结束统一不走 `abort_request`。
2. weight update 统一走 `pause_generation(mode=<pause_mode>) -> continue_generation`。
3. actor 不再调用 session gate pause/resume RPC。

验收：

- 新旧训练任务核心指标无异常回归（吞吐、成功率、loss 曲线基本一致）。
- fully-async 场景下不再出现 rollout-end `/abort_request` 调用。

## Phase 2：删除 dead code（runtime 清理）

目标：把 Phase 1 后不可达分支和冗余接口删除。

变更点：

1. 删除 `inference_rollout_train.abort()` 与策略分支。
2. 删除 `aborted_samples` 回灌链路。
3. 删除 RolloutManager 的 `pause_sessions/resume_sessions` 方法。
4. 删除 session server 的 abort/resume gate 代码与 endpoints。

验收：

- 单元与集成测试全绿。
- 运行日志中无 legacy policy 相关字段。

## Phase 3：旧路径清理（范围 B）

目标：仓库级彻底移除 legacy abort 实现残留。

变更点：

1. 清理 `miles/rollout/sglang_rollout.py` 中 abort 逻辑。
2. 评估是否下线旧 rollout 路径或保持最小兼容壳。

验收：

- 代码检索无 `legacy_abort_resume` 运行时引用。
- 文档与示例脚本无 legacy 策略入口。

## Phase 4：测试与文档收敛

目标：让测试矩阵和文档叙述与单策略实现一致。

测试改造建议：

1. 删除 legacy control 组用例，保留并增强 `no_interrupt`/`retract`/`in_place`。
2. 保留长链路扰动 e2e（pause/continue 多次注入）。
3. 增加“无 session gate”下的工具调用稳定性回归。

文档改造建议：

1. `generate_hub_rollout_design.md`：移除 dual-policy 描述。
2. `fully_async_rollout.md`：更新为单策略结论。
3. `session_server_abort_resume.md`：删除或转历史文档。

## 7. 风险与缓解

## 7.1 `partial_rollout` 语义变化风险

风险：

- 过去部分样本通过 abort 回收进入 buffer；移除后该通路消失。

缓解：

1. 明确文档声明行为变化。
2. 增加 staleness/buffer 指标监控。
3. 必要时补“非 abort 回灌”机制（独立特性，不与本次强绑定）。

## 7.2 Session server 行为变化风险

风险：

- 去掉 gate 后，session 侧不再做“暂停期间阻塞+重试”。

缓解：

1. 增加 e2e 覆盖：长链 tool-call + pause/continue 扰动。
2. 在 rollout 和 agent 侧监控错误率与重试率。

## 7.3 兼容性风险（CLI/脚本）

风险：

- 删除参数后老脚本会因未知参数失败。

缓解：

1. 一阶段可先 deprecate（打印 warning），下一阶段硬删。
2. 提供迁移脚本或 release note 迁移指引。

## 8. 验证矩阵（建议最小集合）

1. 快速集成：
   - `tests/fast/rollout/inference_rollout/integration/test_fully_async.py`
2. 权重更新 pause mode：
   - `tests/fast/backends/megatron_utils/test_pause_mode.py`（更新为无 legacy 分支）
3. 长链路 e2e：
   - `tests/e2e/sglang/test_no_interrupt_mock_wait_weather_loop.py`

通过标准：

1. 无 rollout-end abort 请求。
2. `retract` 与 `in_place` 两种 pause mode 均稳定完成多轮 agent 交互。
3. 无新增 deadlock / 卡住现象。

## 9. 回滚策略

1. Phase 1 与 Phase 2 分开合并，确保可以按 PR 粒度回滚。
2. 在正式删参数前保留一个发布周期的迁移窗口（如采用 deprecate 方案）。
3. 出现系统级异常时，优先回滚到“行为切换前”版本，而不是临时热补分支逻辑。

## 10. PR 切分建议

1. PR-1：行为切换（single-policy runtime path）。
2. PR-2：runtime dead code 清理（含 session gate）。
3. PR-3：测试与文档收敛。
4. PR-4（可选）：旧 `sglang_rollout` 路径清理。

这样切分可以降低单次变更风险，并保证每个阶段都有可验证的退出条件。
