# Session 模块重构方案

## 一、背景与动机

`miles/rollout/session/` 是 TITO（Turn-In-Turn-Out）pretokenized session 的管理模块。当前实现有三个问题：

1. **锁管理混乱**：`sessions.py`（路由层）维护了 `session_locks` dict + `closing_sessions` set + `session_locks_guard` 三个闭包变量做并发控制；`SingleUserTurnTrajectoryManager`（数据层）内部又有一把 `threading.RLock`。两套锁不协调，`threading.RLock` 在 asyncio 单线程事件循环中实际上永远不被竞争，形同虚设。
2. **职责混乱**：manager 的 `prepare_pretokenized` 和 `update_pretokenized_state` 方法直接伸手修改 session 内部字段（`session.messages = ...`, `session.trajectory_token_ids.append(...)`, `session.num_assistant += 1`），"谁在写什么"难以追踪。
3. **锁与被保护对象分离**：per-session 的 `asyncio.Lock` 存在 `sessions.py` 的闭包 dict 中，session 状态存在 manager 的 `sessions` dict 中——两个平行 dict 需要额外的 guard 锁来保证同步。

## 二、设计决策

### 2.1 锁放进 session 对象

**决策**：每个 `SingleUserTurnTrajectory` 自带 `lock: asyncio.Lock` 和 `closing: bool`。

**理由**：
- 锁的生命周期 = session 的生命周期，创建时就在，删除时跟着消失
- 消除 `session_locks` dict，不再需要 `session_locks_guard` 来保证两个 dict 同步
- `closing_sessions` set 被 `session.closing` flag 替代，语义更自然

### 2.2 不需要全局 guard 锁

**决策**：完全去掉 `session_locks_guard`。

**理由**：asyncio 是协作式调度（单线程事件循环），两个 `await` 之间的同步代码是原子的。`dict.get()`、`dict.pop()`、`dict[k]=v` 都是同步操作，不可能被其他协程打断。`session.closing` flag 覆盖所有竞态场景：

| 场景 | 安全性 |
|------|--------|
| chat 在 delete 标记 closing 之前运行 | chat 正常拿锁执行；delete 等锁释放后再删 |
| chat 在 closing 之后、dict 移除之前 | `session.closing=True`，chat 立即 404 |
| chat 在 dict 移除之后 | `sessions.get()` 返回 None，404 |
| chat 等待锁期间被标记 closing | 获取锁后 double-check `session.closing`，404 |

### 2.3 Session 拥有所有状态变更方法

**决策**：`prepare_pretokenized` 和 `update_pretokenized_state` 从 manager 移到 `SingleUserTurnTrajectory` 上，tokenizer 作为参数注入。

**理由**：
- session 内部的 rollback + validate + merge 是不可分割的原子操作，拆开容易漏步骤
- route handler 不需要知道 prepare 的内部流程
- manager 只做 "session ID → session 对象" 的映射，不写 session 状态

### 2.4 Pydantic BaseModel → dataclass

**决策**：`SingleUserTurnTrajectory` 从 Pydantic `BaseModel` 改为 `@dataclass`。

**理由**：`asyncio.Lock` 不能放在 Pydantic model 里（不可序列化）。dataclass 满足需求且更轻量。

### 2.5 Manager 改名为 SessionRegistry

**决策**：`SingleUserTurnTrajectoryManager` → `SessionRegistry`，变为纯 CRUD + 共享资源上的只读计算。

**理由**：重构后 manager 不再有业务逻辑，只剩注册表功能。新名字更准确。同时保留旧名作为别名以减少外部影响。

## 三、重构后的架构

```
sessions.py (route handler)
  │  获取 session → 持有 session.lock → 编排整个流程
  │
  ├── registry.get_session(session_id)     → SingleUserTurnTrajectory
  ├── session.prepare_pretokenized(messages, tools, tito_tokenizer)
  ├── await backend.do_proxy(...)
  ├── session.update_pretokenized_state(messages, assistant_msg, ...)
  └── session.append_record(record)

SingleUserTurnTrajectory (dataclass)
  │  拥有: lock, closing, 所有状态, 所有写方法
  │
  ├── prepare_pretokenized(messages, tools, tito_tokenizer)
  │     内部: _assert_no_user_after_assistant → _try_rollback → assert_append_only → merge_tokens
  ├── update_pretokenized_state(messages, assistant_msg, prompt_ids, completion_ids, max_trim)
  │     内部: validate_prefix → 更新 messages + trajectory_token_ids + num_assistant
  ├── append_record(record)
  └── token_ids (property)

SessionRegistry
  │  纯 CRUD + 只读计算
  │
  ├── create_session() → str
  ├── get_session(session_id) → SingleUserTurnTrajectory（找不到则 raise）
  ├── remove_session(session_id)
  └── compute_session_mismatch(session) → list[dict] | None（用共享 tokenizer 计算）
```

### 每个路由的锁协议

| Route | 锁行为 |
|-------|--------|
| `POST /sessions` | 无锁。`registry.create_session()` 是 sync，原子。 |
| `GET /sessions/{id}` | 无锁。纯读，无 await，原子执行。 |
| `DELETE /sessions/{id}` | `session.closing = True` → `await session.lock.acquire()` → `registry.remove_session()` → `session.lock.release()` |
| `POST .../chat/completions` | 检查 `session.closing` → `async with session.lock:` → double-check closing → 整个处理流程在锁内 |
| `ANY .../session/{id}/{path}` | 无锁。纯转发，不访问 session 状态。 |

## 四、当前源码与目标代码

### 4.1 `miles/rollout/session/single_user_turn_trajectory.py`

#### 当前代码

```python
import logging
import threading
import uuid
from typing import Any

from pydantic import BaseModel, Field, computed_field

from miles.rollout.session.session_errors import MessageValidationError, SessionNotFoundError, TokenizationError
from miles.rollout.session.session_types import SessionRecord
from miles.utils.chat_template_utils import apply_chat_template, assert_messages_append_only, message_matches
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer

logger = logging.getLogger(__name__)


class SingleUserTurnTrajectory(BaseModel):
    """State for a single-user-turn trajectory.

    Tracks the full message history and accumulated token IDs for one session.
    The typical message sequence is: [system?, user, assistant, tool, assistant, tool, …],
    but the agent may retry from an earlier point (e.g. re-running a tool call),
    in which case the session is rolled back to the last matching assistant
    checkpoint and re-extended from there.
    """

    messages: list[dict[str, Any]] = Field(default_factory=list)
    records: list[SessionRecord] = Field(default_factory=list)
    trajectory_token_ids: list[list[int]] = Field(default_factory=list)
    num_assistant: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def token_ids(self) -> list[int]:
        """Current token IDs — the latest assistant checkpoint."""
        return self.trajectory_token_ids[-1] if self.trajectory_token_ids else []

    def append_session_record(self, record: SessionRecord):
        self.records.append(record)

    def _try_detect_and_rollback_to_assistant_checkpoint(
        self,
        request_messages: list[dict[str, Any]],
    ) -> None:
        """Detect if *request_messages* diverges from stored history and roll back.

        In agentic workflows the agent may retry from an earlier point — for
        example, re-running a tool call with different arguments.  When that
        happens the new request shares a common prefix with the stored messages
        but diverges before the end.  This method truncates session state back
        to the last assistant checkpoint within the matching prefix.

        Example — agent retries after the first tool call::

            stored:  [sys, user, assistant₁, tool₁, assistant₂]
                      ───────────────────── ▲
                      checkpoint 0 (assistant₁)   checkpoint 1 (assistant₂)

            request: [sys, user, assistant₁, tool₁_different, ...]
                                             ↑ diverges here (index 3)

            match_len = 3  (sys, user, assistant₁ all match)
            Last assistant in matched prefix → assistant₁ (checkpoint 0)

            After rollback:
              messages           = [sys, user, assistant₁]
              trajectory_token_ids = [checkpoint_0_ids]
              records              = [record_0]
              num_assistant        = 1

        No rollback occurs when:
        - The stored history is empty.
        - *request_messages* is a strict extension of stored messages
          (``match_len >= len(stored)``).
        """
        stored = self.messages
        if not stored or not self.trajectory_token_ids:
            return

        match_len = 0
        for i in range(min(len(request_messages), len(stored))):
            if message_matches(stored[i], request_messages[i]):
                match_len = i + 1
            else:
                break

        if match_len >= len(stored):
            return

        # Find the last assistant message within the matched prefix.
        rollback_msg_end = None
        checkpoint_index = -1
        assistant_count = 0
        for i in range(match_len):
            if stored[i].get("role") == "assistant":
                rollback_msg_end = i + 1
                checkpoint_index = assistant_count
                assistant_count += 1

        if checkpoint_index < 0:
            raise MessageValidationError(
                f"rollback failed: no assistant message found in the first "
                f"{match_len} matched messages (stored has {len(stored)} messages, "
                f"request has {len(request_messages)} messages)"
            )

        logger.info(
            "Rolling back session: stored %d messages / %d checkpoints -> " "checkpoint %d (messages[:%d])",
            len(stored),
            self.num_assistant,
            checkpoint_index,
            rollback_msg_end,
        )

        self.messages = stored[:rollback_msg_end]
        self.trajectory_token_ids = self.trajectory_token_ids[: checkpoint_index + 1]
        self.records = self.records[: checkpoint_index + 1]
        self.num_assistant = checkpoint_index + 1


class SingleUserTurnTrajectoryManager:
    """Lightweight session manager for single-user-turn trajectories.

    Handles session CRUD, message-level validation (append-only, no user
    after assistant), and token ID read/store.  All tokenization computation
    is delegated to ``TITOTokenizer``.
    """

    def __init__(self, args, tokenizer: Any, *, tito_tokenizer: TITOTokenizer):
        self.sessions: dict[str, SingleUserTurnTrajectory] = {}
        self.args = args
        self.tokenizer = tokenizer
        self._lock = threading.RLock()
        self.tito_tokenizer = tito_tokenizer
        self.comparator = tito_tokenizer.create_comparator()

    def create_session(self) -> str:
        with self._lock:
            session_id = uuid.uuid4().hex
            self.sessions[session_id] = SingleUserTurnTrajectory()
            return session_id

    def get_session_records_by_id(self, session_id: str) -> list[SessionRecord]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            return session.records

    def get_session_token_ids(self, session_id: str) -> list[int]:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            return session.token_ids

    def compute_session_mismatch(self, session_id: str) -> list[dict] | None:
        """Compare accumulated token IDs against canonical chat template output.

        Returns a list of mismatch dicts from ``TokenSeqComparator.compare_sequences``,
        each containing ``{position, expected_token, actual_token, context}``,
        or ``None`` if the session has no token IDs yet.
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            if not session.token_ids:
                return None
            try:
                tools = session.records[-1].request.get("tools") if session.records else None
                expected_ids = apply_chat_template(
                    session.messages,
                    tokenizer=self.tokenizer,
                    tools=tools,
                    add_generation_prompt=False,
                    tokenize=True,
                )
                mismatches = self.comparator.compare_sequences(expected_ids, session.token_ids)
                return [m.to_dict() for m in mismatches]
            except Exception as e:
                raise TokenizationError(
                    f"failed to compute tito_session_mismatch for session {session_id}: {e}"
                ) from e

    def delete_session_by_id(self, session_id: str) -> bool:
        with self._lock:
            session = self.sessions.pop(session_id, None)
            if session is None:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            return True

    def append_session_record(self, session_id: str, record: SessionRecord) -> bool:
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            session.append_session_record(record)
            return True

    def prepare_pretokenized(
        self,
        session_id: str,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Compute a merged prompt via ``TITOTokenizer.merge_tokens`` and
        return it as ``input_ids`` for SGLang.

        Returns ``None`` on the first turn (no stored token_ids yet).
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                previews = [
                    f"[{i}] role={m.get('role')}, content={(m.get('content') or '')[:100]!r}"
                    for i, m in enumerate(request_messages)
                ]
                raise SessionNotFoundError(
                    f"session not found: session_id={session_id}, "
                    f"num_messages={len(request_messages)}\n"
                    + "\n".join(previews)
                    + "\nThis usually means a stale agent environment from a previous "
                    "training run is still sending requests after the router restarted. "
                    "Ensure all agent containers are fully stopped before restarting training."
                )

            if not session.token_ids:
                return None

            # Validate and reconcile request_messages against stored session state:
            # 1. Reject multi-turn (user after assistant) — single-user-turn only.
            self._assert_no_user_after_assistant(request_messages)
            # 2. Detect agent retries and roll back to the last matching checkpoint.
            session._try_detect_and_rollback_to_assistant_checkpoint(request_messages)
            # 3. Confirm the (possibly rolled-back) stored messages are a prefix of request.
            try:
                assert_messages_append_only(session.messages, request_messages)
            except ValueError as e:
                raise MessageValidationError(str(e)) from e

            merged = self.tito_tokenizer.merge_tokens(
                old_messages=session.messages,
                new_messages=request_messages,
                pretokenized_token_ids=session.token_ids,
                tools=tools,
            )
            return {
                "input_ids": merged,
            }

    def update_pretokenized_state(
        self,
        session_id: str,
        request_messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
    ) -> None:
        """Store raw token IDs after a successful response.

        Appends ``prompt_token_ids + completion_token_ids`` as-is (no
        stripping or modification) as a new checkpoint in
        ``trajectory_token_ids``.  Validates that the previously stored
        token_ids are a prefix of the new checkpoint, tolerating up to
        ``max_trim_tokens`` trailing tokens that may differ due to
        chat-template boundary re-tokenization.  This confirms SGLang
        actually reused our pretokenized input.
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise SessionNotFoundError(f"update_pretokenized_state: session not found: session_id={session_id}")

            all_token_ids = prompt_token_ids + completion_token_ids
            session.messages = list(request_messages) + [assistant_message]

            max_trim = self.tito_tokenizer.max_trim_tokens
            prev = session.token_ids
            if prev:
                check_len = len(prev) - max_trim
                if check_len > 0 and all_token_ids[:check_len] != prev[:check_len]:
                    first_mismatch = next(
                        (
                            i
                            for i, (a, b) in enumerate(zip(all_token_ids[:check_len], prev[:check_len], strict=True))
                            if a != b
                        ),
                        min(len(all_token_ids), check_len),
                    )
                    raise TokenizationError(
                        f"pretokenized prefix mismatch: "
                        f"stored {len(prev)} tokens (checking first {check_len}, "
                        f"allowing {max_trim} trailing) are not a prefix of "
                        f"prompt_token_ids + completion_token_ids "
                        f"({len(all_token_ids)} tokens), "
                        f"first mismatch at index {first_mismatch}, "
                        f"matched {first_mismatch}/{check_len} prefix tokens\n"
                        f"request_messages={request_messages}\n"
                        f"assistant_message={assistant_message}"
                    )

            session.trajectory_token_ids.append(all_token_ids)
            session.num_assistant += 1

    @staticmethod
    def _assert_no_user_after_assistant(messages: list[dict[str, Any]]) -> None:
        """Assert no user message appears after the first assistant message."""
        seen_assistant = False
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                seen_assistant = True
            elif role == "user" and seen_assistant:
                raise MessageValidationError(
                    f"invalid message structure: user message at index {i} "
                    f"appears after the first assistant message"
                )
```

#### 目标代码

变更要点：
- `SingleUserTurnTrajectory`: Pydantic BaseModel → `@dataclass`，新增 `lock`/`closing` 字段，新增 `prepare_pretokenized`/`update_pretokenized_state`/`append_record` 方法
- `SingleUserTurnTrajectoryManager` → `SessionRegistry`：去掉 `threading.RLock`，去掉所有业务方法，只保留 CRUD + `compute_session_mismatch`
- `_assert_no_user_after_assistant` 变为模块级函数
- 保留 `SingleUserTurnTrajectoryManager` 作为 `SessionRegistry` 的别名

```python
import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from miles.rollout.session.session_errors import MessageValidationError, SessionNotFoundError, TokenizationError
from miles.rollout.session.session_types import SessionRecord
from miles.utils.chat_template_utils import apply_chat_template, assert_messages_append_only, message_matches
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer

logger = logging.getLogger(__name__)


def _assert_no_user_after_assistant(messages: list[dict[str, Any]]) -> None:
    """Assert no user message appears after the first assistant message."""
    seen_assistant = False
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "assistant":
            seen_assistant = True
        elif role == "user" and seen_assistant:
            raise MessageValidationError(
                f"invalid message structure: user message at index {i} "
                f"appears after the first assistant message"
            )


@dataclass
class SingleUserTurnTrajectory:
    """State for a single-user-turn trajectory.

    Tracks the full message history and accumulated token IDs for one session.
    The typical message sequence is: [system?, user, assistant, tool, assistant, tool, ...],
    but the agent may retry from an earlier point (e.g. re-running a tool call),
    in which case the session is rolled back to the last matching assistant
    checkpoint and re-extended from there.

    Concurrency contract: all mutating methods must be called under ``self.lock``.
    """

    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False, compare=False)
    closing: bool = field(default=False, repr=False, compare=False)
    messages: list[dict[str, Any]] = field(default_factory=list)
    records: list[SessionRecord] = field(default_factory=list)
    trajectory_token_ids: list[list[int]] = field(default_factory=list)
    num_assistant: int = 0

    @property
    def token_ids(self) -> list[int]:
        """Current token IDs - the latest assistant checkpoint."""
        return self.trajectory_token_ids[-1] if self.trajectory_token_ids else []

    def append_record(self, record: SessionRecord) -> None:
        self.records.append(record)

    def prepare_pretokenized(
        self,
        request_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        tito_tokenizer: TITOTokenizer,
    ) -> dict[str, Any] | None:
        """Validate messages, rollback if needed, and compute merged input_ids.

        Returns ``None`` on the first turn (no stored token_ids yet).
        Must be called under ``self.lock``.
        """
        if not self.token_ids:
            return None

        # 1. Reject multi-turn (user after assistant) - single-user-turn only.
        _assert_no_user_after_assistant(request_messages)
        # 2. Detect agent retries and roll back to the last matching checkpoint.
        self._try_detect_and_rollback_to_assistant_checkpoint(request_messages)
        # 3. Confirm the (possibly rolled-back) stored messages are a prefix of request.
        try:
            assert_messages_append_only(self.messages, request_messages)
        except ValueError as e:
            raise MessageValidationError(str(e)) from e

        merged = tito_tokenizer.merge_tokens(
            old_messages=self.messages,
            new_messages=request_messages,
            pretokenized_token_ids=self.token_ids,
            tools=tools,
        )
        return {"input_ids": merged}

    def update_pretokenized_state(
        self,
        request_messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        max_trim_tokens: int,
    ) -> None:
        """Store raw token IDs after a successful response.

        Appends ``prompt_token_ids + completion_token_ids`` as a new checkpoint.
        Validates that the previously stored token_ids are a prefix of the new
        checkpoint (tolerating up to ``max_trim_tokens`` trailing differences).
        Must be called under ``self.lock``.
        """
        all_token_ids = prompt_token_ids + completion_token_ids

        prev = self.token_ids
        if prev:
            check_len = len(prev) - max_trim_tokens
            if check_len > 0 and all_token_ids[:check_len] != prev[:check_len]:
                first_mismatch = next(
                    (
                        i
                        for i, (a, b) in enumerate(zip(all_token_ids[:check_len], prev[:check_len], strict=True))
                        if a != b
                    ),
                    min(len(all_token_ids), check_len),
                )
                raise TokenizationError(
                    f"pretokenized prefix mismatch: "
                    f"stored {len(prev)} tokens (checking first {check_len}, "
                    f"allowing {max_trim_tokens} trailing) are not a prefix of "
                    f"prompt_token_ids + completion_token_ids "
                    f"({len(all_token_ids)} tokens), "
                    f"first mismatch at index {first_mismatch}, "
                    f"matched {first_mismatch}/{check_len} prefix tokens\n"
                    f"request_messages={request_messages}\n"
                    f"assistant_message={assistant_message}"
                )

        self.messages = list(request_messages) + [assistant_message]
        self.trajectory_token_ids.append(all_token_ids)
        self.num_assistant += 1

    def _try_detect_and_rollback_to_assistant_checkpoint(
        self,
        request_messages: list[dict[str, Any]],
    ) -> None:
        """Detect if *request_messages* diverges from stored history and roll back.

        In agentic workflows the agent may retry from an earlier point - for
        example, re-running a tool call with different arguments.  When that
        happens the new request shares a common prefix with the stored messages
        but diverges before the end.  This method truncates session state back
        to the last assistant checkpoint within the matching prefix.

        Example - agent retries after the first tool call::

            stored:  [sys, user, assistant1, tool1, assistant2]
            request: [sys, user, assistant1, tool1_different, ...]
                                             ^ diverges here (index 3)

            match_len = 3  (sys, user, assistant1 all match)
            Last assistant in matched prefix -> assistant1 (checkpoint 0)

            After rollback:
              messages           = [sys, user, assistant1]
              trajectory_token_ids = [checkpoint_0_ids]
              records              = [record_0]
              num_assistant        = 1

        No rollback occurs when:
        - The stored history is empty.
        - *request_messages* is a strict extension of stored messages
          (``match_len >= len(stored)``).
        """
        stored = self.messages
        if not stored or not self.trajectory_token_ids:
            return

        match_len = 0
        for i in range(min(len(request_messages), len(stored))):
            if message_matches(stored[i], request_messages[i]):
                match_len = i + 1
            else:
                break

        if match_len >= len(stored):
            return

        # Find the last assistant message within the matched prefix.
        rollback_msg_end = None
        checkpoint_index = -1
        assistant_count = 0
        for i in range(match_len):
            if stored[i].get("role") == "assistant":
                rollback_msg_end = i + 1
                checkpoint_index = assistant_count
                assistant_count += 1

        if checkpoint_index < 0:
            raise MessageValidationError(
                f"rollback failed: no assistant message found in the first "
                f"{match_len} matched messages (stored has {len(stored)} messages, "
                f"request has {len(request_messages)} messages)"
            )

        logger.info(
            "Rolling back session: stored %d messages / %d checkpoints -> checkpoint %d (messages[:%d])",
            len(stored),
            self.num_assistant,
            checkpoint_index,
            rollback_msg_end,
        )

        self.messages = stored[:rollback_msg_end]
        self.trajectory_token_ids = self.trajectory_token_ids[: checkpoint_index + 1]
        self.records = self.records[: checkpoint_index + 1]
        self.num_assistant = checkpoint_index + 1


class SessionRegistry:
    """Session ID -> trajectory mapping with shared tokenizer resources.

    Pure CRUD plus read-only computation (compute_session_mismatch).
    Does NOT mutate session state - all mutations are methods on
    SingleUserTurnTrajectory, called by the route handler under session.lock.
    """

    def __init__(self, args, tokenizer: Any, *, tito_tokenizer: TITOTokenizer):
        self.sessions: dict[str, SingleUserTurnTrajectory] = {}
        self.args = args
        self.tokenizer = tokenizer
        self.tito_tokenizer = tito_tokenizer
        self.comparator = tito_tokenizer.create_comparator()

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = SingleUserTurnTrajectory()
        return session_id

    def get_session(self, session_id: str) -> SingleUserTurnTrajectory:
        session = self.sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        return session

    def remove_session(self, session_id: str) -> None:
        if self.sessions.pop(session_id, None) is None:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")

    def compute_session_mismatch(self, session: SingleUserTurnTrajectory) -> list[dict] | None:
        """Compare accumulated token IDs against canonical chat template output.

        Read-only: does not mutate session state.
        """
        if not session.token_ids:
            return None
        try:
            tools = session.records[-1].request.get("tools") if session.records else None
            expected_ids = apply_chat_template(
                session.messages,
                tokenizer=self.tokenizer,
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
            )
            mismatches = self.comparator.compare_sequences(expected_ids, session.token_ids)
            return [m.to_dict() for m in mismatches]
        except Exception as e:
            raise TokenizationError(
                f"failed to compute tito_session_mismatch: {e}"
            ) from e


# Backward-compatible alias
SingleUserTurnTrajectoryManager = SessionRegistry
```

### 4.2 `miles/rollout/session/sessions.py`

#### 当前代码

```python
import asyncio
import json
import logging
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.rollout.session.session_errors import (
    SessionError,
    SessionNotFoundError,
    TokenizationError,
    UpstreamResponseError,
)
from miles.rollout.session.session_types import GetSessionResponse, SessionRecord
from miles.rollout.session.single_user_turn_trajectory import SingleUserTurnTrajectoryManager
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


def setup_session_routes(app, backend, args):
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        logger.info("[session] Skipping session routes (hf_checkpoint not set).")
        return

    tokenizer = load_tokenizer(
        hf_checkpoint, chat_template_path=getattr(args, "chat_template_path", None), trust_remote_code=True
    )

    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=getattr(args, "tito_model", "default"),
    )

    manager = SingleUserTurnTrajectoryManager(args, tokenizer, tito_tokenizer=tito_tokenizer)

    # Concurrency contract:
    # - same session: strictly serialized (one in-flight request)
    # - different sessions: fully parallel
    session_locks: dict[str, asyncio.Lock] = {}
    closing_sessions: set[str] = set()
    session_locks_guard = asyncio.Lock()

    async def _get_open_session_lock(session_id: str) -> asyncio.Lock:
        async with session_locks_guard:
            lock = session_locks.get(session_id)
            if lock is None or session_id in closing_sessions:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            return lock

    @app.exception_handler(SessionError)
    async def session_error_handler(request: Request, exc: SessionError):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    @app.post("/sessions")
    async def create_session():
        session_id = manager.create_session()
        async with session_locks_guard:
            session_locks[session_id] = asyncio.Lock()
        return {"session_id": session_id}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        records = manager.get_session_records_by_id(session_id)
        metadata = {}
        try:
            mismatch = manager.compute_session_mismatch(session_id)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        metadata["accumulated_token_ids"] = manager.get_session_token_ids(session_id)
        metadata["max_trim_tokens"] = manager.tito_tokenizer.max_trim_tokens
        return GetSessionResponse(
            session_id=session_id,
            records=records,
            metadata=metadata,
        )

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        # Mark closing first so new chat requests fail fast for this session.
        async with session_locks_guard:
            session_lock = session_locks.get(session_id)
            if session_lock is None:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")
            closing_sessions.add(session_id)

        # Wait for in-flight request to finish, then delete under the same lock.
        await session_lock.acquire()
        try:
            manager.delete_session_by_id(session_id)
        finally:
            session_lock.release()
            async with session_locks_guard:
                session_locks.pop(session_id, None)
                closing_sessions.discard(session_id)

        return Response(status_code=204)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        """Proxy a chat completion through SGLang with TITO token tracking.

        Flow: prepare pretokenized input_ids (if not first turn) -> inject
        SGLang flags -> proxy to backend -> validate response -> update
        trajectory checkpoint -> append session record.
        """
        session_lock = await _get_open_session_lock(session_id)
        async with session_lock:
            # Session could be marked closing while this request waits for lock.
            async with session_locks_guard:
                if session_id in closing_sessions:
                    raise SessionNotFoundError(f"session not found: session_id={session_id}")

            body = await request.body()
            request_body = json.loads(body) if body else {}

            # TITO token tracking requires three SGLang flags working together:
            #   logprobs=True            -> populates meta_info.output_token_logprobs
            #   return_prompt_token_ids  -> adds choice.prompt_token_ids
            #   return_meta_info         -> wraps the above in choice.meta_info
            # All three are hardcoded (not setdefault) to prevent agent-side
            # overrides from breaking the token accumulation invariants.
            request_body["logprobs"] = True
            request_body["return_prompt_token_ids"] = True
            request_body["return_meta_info"] = True
            if getattr(args, "use_rollout_routing_replay", False):
                request_body["return_routed_experts"] = True
            # Must be False so stop tokens are trimmed from output: otherwise the
            # agent sees stop-token text in content, and the accumulated checkpoint
            # would duplicate structural delimiters that the chat template also emits.
            request_body["no_stop_trim"] = False

            request_messages = request_body.get("messages", [])
            pretokenized = manager.prepare_pretokenized(session_id, request_messages, tools=request_body.get("tools"))
            if pretokenized is not None:
                request_body["input_ids"] = pretokenized["input_ids"]
                logger.debug(
                    "Using pretokenized input_ids: %d tokens",
                    len(pretokenized["input_ids"]),
                )

            body = json.dumps(request_body).encode()

            result = await backend.do_proxy(request, "v1/chat/completions", body=body)

            # If SGLang returned a non-200 error (e.g. 400 for context too long),
            # pass it through to the agent without recording - the agent can retry
            # or handle the error.
            if result["status_code"] != 200:
                return backend.build_proxy_response(result)

            response = json.loads(result["response_body"])

            choice = response.get("choices", [{}])[0]

            meta_info = choice.get("meta_info")
            if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
                raise UpstreamResponseError(
                    "meta_info and output_token_logprobs must be in choice (requires logprobs=True)"
                )
            assistant_message = choice.get("message", {})
            # SGLang may return content=None when the assistant only emits tool_calls or
            # reasoning tokens. Normalize to empty string to keep session tracking robust.
            if assistant_message.get("content") is None:
                assistant_message["content"] = ""

            prompt_token_ids = choice.get("prompt_token_ids")
            output_token_logprobs = meta_info["output_token_logprobs"]
            completion_tokens = meta_info["completion_tokens"]

            actual_output_logprobs_len = len(output_token_logprobs)
            if actual_output_logprobs_len != completion_tokens:
                raise UpstreamResponseError(
                    "invalid chat completion response: "
                    f"len(output_token_logprobs)={actual_output_logprobs_len} "
                    f"!= completion_tokens={completion_tokens}. "
                    f"Please check whether you use the correct SGLang branch which has fix the tokenizer batch decode issue."
                )

            completion_token_ids = [t[1] for t in output_token_logprobs]

            manager.update_pretokenized_state(
                session_id,
                request_messages,
                assistant_message,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
            )

            record = SessionRecord(
                timestamp=time.time(),
                method=request.method,
                path="/v1/chat/completions",
                status_code=result["status_code"],
                request=request_body,
                response=response,
            )
            manager.append_session_record(session_id, record)
            return backend.build_proxy_response(result)

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        result = await backend.do_proxy(request, path)
        return backend.build_proxy_response(result)
```

#### 目标代码

变更要点：
- 去掉 `session_locks` dict、`closing_sessions` set、`session_locks_guard`、`_get_open_session_lock`
- 去掉 `import asyncio`
- `manager` → `registry`，import 改为 `SessionRegistry`
- `create_session`: 直接调用 `registry.create_session()`，无锁
- `get_session`: 通过 `registry.get_session()` 拿到 session 对象，直接读属性
- `delete_session`: 用 `session.closing` + `session.lock` 两阶段删除
- `chat_completions`: 用 `session.lock` 包裹全流程，调用 session 上的方法

```python
import json
import logging
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.rollout.session.session_errors import (
    SessionError,
    SessionNotFoundError,
    TokenizationError,
    UpstreamResponseError,
)
from miles.rollout.session.session_types import GetSessionResponse, SessionRecord
from miles.rollout.session.single_user_turn_trajectory import SessionRegistry
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


def setup_session_routes(app, backend, args):
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        logger.info("[session] Skipping session routes (hf_checkpoint not set).")
        return

    tokenizer = load_tokenizer(
        hf_checkpoint, chat_template_path=getattr(args, "chat_template_path", None), trust_remote_code=True
    )

    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=getattr(args, "tito_model", "default"),
    )

    registry = SessionRegistry(args, tokenizer, tito_tokenizer=tito_tokenizer)

    @app.exception_handler(SessionError)
    async def session_error_handler(request: Request, exc: SessionError):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    @app.post("/sessions")
    async def create_session():
        session_id = registry.create_session()
        return {"session_id": session_id}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = registry.get_session(session_id)
        metadata = {}
        try:
            mismatch = registry.compute_session_mismatch(session)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        metadata["accumulated_token_ids"] = session.token_ids
        metadata["max_trim_tokens"] = registry.tito_tokenizer.max_trim_tokens
        return GetSessionResponse(
            session_id=session_id,
            records=session.records,
            metadata=metadata,
        )

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        session = registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        session.closing = True
        await session.lock.acquire()
        try:
            registry.remove_session(session_id)
        finally:
            session.lock.release()
        return Response(status_code=204)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        """Proxy a chat completion through SGLang with TITO token tracking.

        Flow: prepare pretokenized input_ids (if not first turn) -> inject
        SGLang flags -> proxy to backend -> validate response -> update
        trajectory checkpoint -> append session record.
        """
        session = registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        async with session.lock:
            # Double-check: session may have been marked closing while waiting for lock.
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")

            body = await request.body()
            request_body = json.loads(body) if body else {}

            # TITO token tracking requires three SGLang flags working together:
            #   logprobs=True            -> populates meta_info.output_token_logprobs
            #   return_prompt_token_ids  -> adds choice.prompt_token_ids
            #   return_meta_info         -> wraps the above in choice.meta_info
            # All three are hardcoded (not setdefault) to prevent agent-side
            # overrides from breaking the token accumulation invariants.
            request_body["logprobs"] = True
            request_body["return_prompt_token_ids"] = True
            request_body["return_meta_info"] = True
            if getattr(args, "use_rollout_routing_replay", False):
                request_body["return_routed_experts"] = True
            # Must be False so stop tokens are trimmed from output: otherwise the
            # agent sees stop-token text in content, and the accumulated checkpoint
            # would duplicate structural delimiters that the chat template also emits.
            request_body["no_stop_trim"] = False

            request_messages = request_body.get("messages", [])
            pretokenized = session.prepare_pretokenized(
                request_messages,
                tools=request_body.get("tools"),
                tito_tokenizer=registry.tito_tokenizer,
            )
            if pretokenized is not None:
                request_body["input_ids"] = pretokenized["input_ids"]
                logger.debug(
                    "Using pretokenized input_ids: %d tokens",
                    len(pretokenized["input_ids"]),
                )

            body = json.dumps(request_body).encode()

            result = await backend.do_proxy(request, "v1/chat/completions", body=body)

            # If SGLang returned a non-200 error (e.g. 400 for context too long),
            # pass it through to the agent without recording - the agent can retry
            # or handle the error.
            if result["status_code"] != 200:
                return backend.build_proxy_response(result)

            response = json.loads(result["response_body"])

            choice = response.get("choices", [{}])[0]

            meta_info = choice.get("meta_info")
            if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
                raise UpstreamResponseError(
                    "meta_info and output_token_logprobs must be in choice (requires logprobs=True)"
                )
            assistant_message = choice.get("message", {})
            # SGLang may return content=None when the assistant only emits tool_calls or
            # reasoning tokens. Normalize to empty string to keep session tracking robust.
            if assistant_message.get("content") is None:
                assistant_message["content"] = ""

            prompt_token_ids = choice.get("prompt_token_ids")
            output_token_logprobs = meta_info["output_token_logprobs"]
            completion_tokens = meta_info["completion_tokens"]

            actual_output_logprobs_len = len(output_token_logprobs)
            if actual_output_logprobs_len != completion_tokens:
                raise UpstreamResponseError(
                    "invalid chat completion response: "
                    f"len(output_token_logprobs)={actual_output_logprobs_len} "
                    f"!= completion_tokens={completion_tokens}. "
                    f"Please check whether you use the correct SGLang branch which has fix the tokenizer batch decode issue."
                )

            completion_token_ids = [t[1] for t in output_token_logprobs]

            session.update_pretokenized_state(
                request_messages,
                assistant_message,
                prompt_token_ids=prompt_token_ids,
                completion_token_ids=completion_token_ids,
                max_trim_tokens=registry.tito_tokenizer.max_trim_tokens,
            )

            record = SessionRecord(
                timestamp=time.time(),
                method=request.method,
                path="/v1/chat/completions",
                status_code=result["status_code"],
                request=request_body,
                response=response,
            )
            session.append_record(record)
            return backend.build_proxy_response(result)

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        result = await backend.do_proxy(request, path)
        return backend.build_proxy_response(result)
```

### 4.3 `tests/fast/router/test_single_user_turn_trajectory.py`

这个文件需要大量修改以适配新 API。以下是完整的调用映射：

#### Import 和 fixture

```python
# before
from miles.rollout.session.single_user_turn_trajectory import SingleUserTurnTrajectoryManager

@pytest.fixture
def manager():
    args = SimpleNamespace()
    mock_tito = _MockTITOTokenizer(tokenizer=None, assistant_start_str="<|im_start|>assistant")
    return SingleUserTurnTrajectoryManager(args, tokenizer=None, tito_tokenizer=mock_tito)

# after
from miles.rollout.session.single_user_turn_trajectory import SessionRegistry

@pytest.fixture
def registry():
    args = SimpleNamespace()
    mock_tito = _MockTITOTokenizer(tokenizer=None, assistant_start_str="<|im_start|>assistant")
    return SessionRegistry(args, tokenizer=None, tito_tokenizer=mock_tito)
```

#### 方法调用映射（全部测试方法中替换）

| 当前调用 | 目标调用 |
|---------|---------|
| `manager.create_session()` | `registry.create_session()` |
| `manager.sessions[sid]` | `registry.sessions[sid]` |
| `manager.get_session_records_by_id(sid)` | `registry.get_session(sid).records` |
| `manager.get_session_token_ids(sid)` | `registry.get_session(sid).token_ids` |
| `manager.delete_session_by_id(sid)` | `registry.remove_session(sid)` |
| `manager.append_session_record(sid, record)` | `registry.get_session(sid).append_record(record)` |
| `manager.prepare_pretokenized(sid, msgs)` | `registry.get_session(sid).prepare_pretokenized(msgs, tito_tokenizer=registry.tito_tokenizer)` |
| `manager.prepare_pretokenized(sid, msgs, tools=t)` | `registry.get_session(sid).prepare_pretokenized(msgs, tools=t, tito_tokenizer=registry.tito_tokenizer)` |
| `manager.update_pretokenized_state(sid, msgs, asst, prompt_ids, comp_ids)` | `registry.get_session(sid).update_pretokenized_state(msgs, asst, prompt_ids, comp_ids, max_trim_tokens=0)` |
| `manager.compute_session_mismatch(sid)` | `registry.compute_session_mismatch(registry.get_session(sid))` |
| `manager.comparator` | `registry.comparator` |

注意：`_MockTITOTokenizer` 继承自 `TITOTokenizer`，其 `max_trim_tokens` 默认为 `0`，所以测试中传 `max_trim_tokens=0` 即可。

#### 特殊情况

**1. `test_session_not_found_raises`（prepare_pretokenized 找不到 session）**

```python
# before
with pytest.raises(SessionNotFoundError, match="session not found"):
    manager.prepare_pretokenized("nonexistent", [SYS_MSG, USER_MSG])

# after — prepare_pretokenized 现在是 session 上的方法，找不到 session 的异常由 get_session 抛出
with pytest.raises(SessionNotFoundError, match="session not found"):
    registry.get_session("nonexistent")
```

**2. `TestUpdatePretokenizedStateMissingSession`**

```python
# before
with pytest.raises(SessionNotFoundError, match="session not found"):
    manager.update_pretokenized_state("nonexistent", ...)

# after
with pytest.raises(SessionNotFoundError, match="session not found"):
    registry.get_session("nonexistent")
```

**3. `TestComputeSessionMismatch.test_raises_for_missing_session`**

```python
# before
with pytest.raises(SessionNotFoundError):
    manager.compute_session_mismatch("nonexistent")

# after
with pytest.raises(SessionNotFoundError):
    registry.get_session("nonexistent")
```

**4. `TestComputeSessionMismatch` 其他测试中的 `@patch` 路径不变**（`apply_chat_template` 仍在同一模块被 import）：

```python
@patch("miles.rollout.session.single_user_turn_trajectory.apply_chat_template")
```

**5. `test_delete_session_by_id` 的返回值**

```python
# before
assert manager.delete_session_by_id(session_id) is True

# after — remove_session 返回 None
registry.remove_session(session_id)  # 不 raise 即为成功
```

## 五、不需要修改的文件

| 文件 | 原因 |
|------|------|
| `miles/rollout/session/session_server.py` | 只调用 `setup_session_routes`，接口不变 |
| `miles/rollout/session/session_types.py` | 数据模型，不变 |
| `miles/rollout/session/session_errors.py` | 错误类型，不变 |
| `tests/fast/router/test_sessions.py` | 通过 HTTP API 测试，不直接依赖内部类 |
| `tests/fast/router/test_session_race_conditions.py` | 同上，通过 HTTP API 测试 |
| `tests/fast/router/test_session_pretokenized_e2e.py` | 同上 |
| `miles/ray/rollout.py` | 只 import `run_session_server`，不变 |

`test_session_race_conditions.py` 中有 `test_delete_waits_for_inflight_then_removes_session` 测试，验证 delete 等待 in-flight 请求完成后才删除 session，重构后通过 `session.closing` + `session.lock` 实现，应自动通过。

## 六、验证命令

```bash
# 单元测试（会有大量改动）
pytest tests/fast/router/test_single_user_turn_trajectory.py -v

# HTTP 集成测试（不应需要改动）
pytest tests/fast/router/test_sessions.py -v

# 并发测试（关键：验证 same-session 串行 + cross-session 并行 + delete-waits-for-inflight）
pytest tests/fast/router/test_session_race_conditions.py -v

# Pretokenized e2e（不应需要改动）
pytest tests/fast/router/test_session_pretokenized_e2e.py -v
```

关键断言：
- `test_same_session_requests_are_serialized`: `max_concurrent == 1`
- `test_different_sessions_can_run_in_parallel`: `max_concurrent >= 3`
- `test_delete_waits_for_inflight_then_removes_session`: delete 耗时 >= 0.2s，之后 session 返回 404
