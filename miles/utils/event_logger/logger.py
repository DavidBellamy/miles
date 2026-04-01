import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from pydantic import TypeAdapter

from miles.utils.event_logger.models import Event, EventBase
from miles.utils.process_identity import ProcessIdentity

logger = logging.getLogger(__name__)

_event_adapter: TypeAdapter[Event] = TypeAdapter(Event)

_event_logger: "EventLogger | None" = None


def set_event_logger(logger: "EventLogger") -> None:
    global _event_logger
    _event_logger = logger


def get_event_logger() -> "EventLogger":
    if _event_logger is None:
        raise RuntimeError("EventLogger not initialized. Call set_event_logger() first.")
    return _event_logger


def is_event_logger_initialized() -> bool:
    return _event_logger is not None


class EventLogger:
    def __init__(self, *, log_dir: Path | str, file_name: str = "events.jsonl", source: ProcessIdentity) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file: TextIO = open(self._log_dir / file_name, "a", encoding="utf-8")
        self._source = source

    def log(self, event: EventBase) -> None:
        enriched = event.model_copy(update={"timestamp": datetime.now(timezone.utc), "source": self._source})
        line = enriched.model_dump_json() + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()

    def close(self) -> None:
        self._file.close()


def read_events(log_dir: Path) -> list[Event]:
    """Read all JSONL event files from a directory and return parsed events."""
    events: list[Event] = []

    jsonl_files = sorted(log_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        logger.warning("No JSONL files found in %s", log_dir)
        return events

    for jsonl_path in jsonl_files:
        for line_num, line in enumerate(jsonl_path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = _event_adapter.validate_json(line)
                events.append(event)
            except Exception:
                logger.warning(
                    "Failed to parse event at %s:%d",
                    jsonl_path,
                    line_num,
                    exc_info=True,
                )

    return events
