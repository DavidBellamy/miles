"""Disk-based JSONL logger for session records.

Each session gets its own file: ``<log_dir>/<session_id>.jsonl``.
Records are flushed immediately so that partial sessions are preserved
if the process crashes.
"""

import json
import logging
from pathlib import Path
from typing import IO

from miles.rollout.session.session_types import SessionRecord

logger = logging.getLogger(__name__)


class RecordLogger:
    """Writes ``SessionRecord`` objects as one-JSON-per-line to per-session files."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, IO[str]] = {}
        logger.info("[record-logger] Logging session records to %s", self.log_dir)

    def _get_handle(self, session_id: str) -> IO[str]:
        handle = self._handles.get(session_id)
        if handle is None:
            path = self.log_dir / f"{session_id}.jsonl"
            handle = open(path, "a", encoding="utf-8")
            self._handles[session_id] = handle
        return handle

    def log_record(self, session_id: str, record: SessionRecord) -> None:
        try:
            handle = self._get_handle(session_id)
            handle.write(json.dumps(record.model_dump(), default=str) + "\n")
            handle.flush()
        except Exception:
            logger.exception("[record-logger] Failed to write record for session %s", session_id)

    def close_session(self, session_id: str) -> None:
        handle = self._handles.pop(session_id, None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                logger.exception("[record-logger] Failed to close log for session %s", session_id)

    def close_all(self) -> None:
        for session_id in list(self._handles):
            self.close_session(session_id)
