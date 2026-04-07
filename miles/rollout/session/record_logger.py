"""Disk-based JSONL logger for session records.

Each session gets its own file: ``<log_dir>/<session_id>.jsonl``.
A dedicated background thread handles serialization and disk I/O so that
callers (typically async request handlers) never block on file operations.
Records are flushed immediately so that partial sessions are preserved
if the process crashes.
"""

import json
import logging
import threading
from pathlib import Path
from queue import SimpleQueue
from typing import IO

from miles.rollout.session.session_types import SessionRecord

logger = logging.getLogger(__name__)

# Sentinel object to signal the writer thread to shut down.
_SHUTDOWN = object()


class RecordLogger:
    """Writes ``SessionRecord`` objects as one-JSON-per-line to per-session files.

    All disk I/O runs on a single background daemon thread, making
    :meth:`log_record` and :meth:`close_session` non-blocking.
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._queue: SimpleQueue = SimpleQueue()
        self._thread = threading.Thread(target=self._writer_loop, daemon=True, name="record-logger")
        self._thread.start()
        logger.info("[record-logger] Logging session records to %s", self.log_dir)

    # -- public (non-blocking) API ------------------------------------------

    def log_record(self, session_id: str, record: SessionRecord) -> None:
        """Enqueue a record to be written. Returns immediately."""
        self._queue.put(("write", session_id, record))

    def close_session(self, session_id: str) -> None:
        """Enqueue a session-close event (flushes and closes the file handle)."""
        self._queue.put(("close", session_id, None))

    def close_all(self) -> None:
        """Shut down the writer thread and close every open file handle.

        Blocks until the background thread has drained the queue and exited.
        """
        self._queue.put(_SHUTDOWN)
        self._thread.join()

    # -- background writer --------------------------------------------------

    def _writer_loop(self) -> None:
        handles: dict[str, IO[str]] = {}
        try:
            while True:
                item = self._queue.get()
                if item is _SHUTDOWN:
                    break
                action, session_id, record = item
                if action == "write":
                    self._do_write(handles, session_id, record)
                elif action == "close":
                    self._do_close(handles, session_id)
        finally:
            # Drain any remaining items before shutting down.
            while not self._queue.empty():
                item = self._queue.get()
                if item is _SHUTDOWN:
                    continue
                action, session_id, record = item
                if action == "write":
                    self._do_write(handles, session_id, record)
                elif action == "close":
                    self._do_close(handles, session_id)
            # Close all remaining file handles.
            for sid in list(handles):
                self._do_close(handles, sid)

    def _do_write(self, handles: dict[str, IO[str]], session_id: str, record: SessionRecord) -> None:
        try:
            handle = handles.get(session_id)
            if handle is None:
                path = self.log_dir / f"{session_id}.jsonl"
                handle = open(path, "a", encoding="utf-8")
                handles[session_id] = handle
            handle.write(json.dumps(record.model_dump(), default=str) + "\n")
            handle.flush()
        except Exception:
            logger.exception("[record-logger] Failed to write record for session %s", session_id)

    def _do_close(self, handles: dict[str, IO[str]], session_id: str) -> None:
        handle = handles.pop(session_id, None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                logger.exception("[record-logger] Failed to close log for session %s", session_id)
