# Generate Hub 与 Rollout 设计（当前实现）

本文说明当前代码中的 rollout 分层、fully-async 行为和策略差异。

## 1. 分层边界

有两个核心抽象：

1. `GenerateFn`（位于 `generate_hub/`）
- 负责单样本生成语义。
- 输入：单个 `Sample`、生成状态、采样参数。
- 输出：单个 `Sample`（或多样本列表）。

2. `RolloutFn`（位于 `inference_rollout/`）
- 负责单次 rollout 的调度语义。
- 输入：rollout id + data source + rollout 配置。
- 输出：训练 step 需要的 sample groups。

判断规则：
- “单样本如何与模型交互” -> `generate_hub`
- “多样本如何调度/过滤/中断” -> rollout 层

## 2. 运行路径

```text
RolloutManager
  -> InferenceRolloutFn._call_train
    -> generate_rollout_async(..., continuous=fully_async_rollout)
      -> submit_generate_tasks
        -> generate_and_rm_group
          -> generate_and_rm
            -> generate_hub.<variant>.generate
```

## 3. Fully-Async 与中断策略

启用 fully-async：
- `--fully-async-rollout`

中断策略参数：
- `--fully-async-interrupt-policy`
  - `legacy_abort_resume`（默认）
  - `no_interrupt`
- `--fully-async-pause-mode`（仅 policy=`no_interrupt` 生效）
  - `retract`（默认）
  - `in_place`

### 3.1 提交策略（`continuous=True`）

fully-async 会将 in-flight 维持在 `over_sampling_batch_size` 附近，而不是只满足 `rollout_batch_size`。

### 3.2 rollout 结束行为

收集到足够有效 group 后：

- `legacy_abort_resume`：
  - 调用 rollout `abort()`；
  - 向 worker 发送 `/abort_request`；
  - 可在 `partial_rollout` 下回收 partial groups。

- `no_interrupt`（且 `continuous=True`）：
  - rollout 末尾不执行 `abort()`；
  - in-flight 请求继续运行；
  - 上报 `rollout/no_interrupt/pending_at_end`。

当前 Phase 1：`no_interrupt` 留下的 pending task 结果不会自动并入下一次 rollout 输出复用。

## 4. Weight Update 行为

- Actor（`actor.update_weights`）：
  - 若 `use_session_server` 且 policy 不是 `no_interrupt`，update 前后分别调用
    `pause_sessions()` / `resume_sessions()`。
  - policy=`no_interrupt` 时跳过 session pause/resume。

- Weight updater（tensor/distributed）：
  - pause mode 选择：
    - `legacy_abort_resume` -> `pause_generation(mode="abort")`
    - `no_interrupt` -> `pause_generation(mode=fully_async_pause_mode)`
  - 顺序保持：
    `pause_generation -> flush_cache -> sync weights -> continue_generation`

- SGLang engine client：
  - `pause_generation(mode: str = "abort")` 会透传到 `/pause_generation`。

## 5. Session Server 范围

session server abort/resume 是策略相关能力：

- 使用路径：`legacy_abort_resume`
- 跳过路径：`no_interrupt`

在 `legacy_abort_resume` 下，session server gate 可以吸收中断重试，避免 agent 侧直接看到 abort 失败。

## 6. Staleness 与 Partial Rollout

`max_buffer_staleness` 在 data source / buffer 层生效，用于 partial-rollout 缓冲样本。

注意：
- 可复用 partial groups 主要来自 legacy abort 路径。
- `no_interrupt` 的 Phase 1 实现里，rollout 末尾 pending 结果默认不转换为可复用 partial groups。

## 7. 关键测试

- Rollout 集成：
  - `tests/fast/rollout/inference_rollout/integration/test_fully_async.py`
- Pause mode / 策略映射：
  - `tests/fast/backends/megatron_utils/test_pause_mode.py`
- Mock wait-agent 扰动闭环：
  - `tests/e2e/sglang/test_no_interrupt_mock_wait_weather_loop.py`
- deterministic logprob 等价验证工具：
  - `tests/e2e/sglang/test_tito_logprob_equivalence.py`
  - `tests/e2e/sglang/utils/logprob_verify_generate.py`

## 8. 历史原型

`examples/fully_async/fully_async_rollout.py` 是历史原型，不是生产 rollout 路径。
