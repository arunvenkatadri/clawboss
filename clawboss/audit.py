"""Audit logging — every supervised action gets recorded."""

import json
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, IO


class AuditPhase(Enum):
    TOOL_CALL = "tool_call"
    BUDGET_CHECK = "budget_check"
    CIRCUIT_BREAKER = "circuit_breaker"
    ITERATION_CHECK = "iteration_check"
    DEAD_MAN_SWITCH = "dead_man_switch"
    POLICY_CHECK = "policy_check"
    REQUEST_START = "request_start"
    REQUEST_END = "request_end"


class AuditOutcome(Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    RETRIED = "retried"
    INFO = "info"


@dataclass
class AuditEntry:
    """A single audit log entry."""
    timestamp: str
    request_id: str
    phase: str
    outcome: str
    target: Optional[str] = None
    detail: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AuditSink(ABC):
    """Interface for audit log destinations."""

    @abstractmethod
    def write(self, entry: AuditEntry) -> None:
        ...


class JsonlAuditSink(AuditSink):
    """Writes audit entries as JSONL to a file or stream."""

    def __init__(self, writer: IO[str]):
        self._writer = writer
        self._lock = threading.Lock()

    @classmethod
    def stdout(cls) -> "JsonlAuditSink":
        return cls(sys.stdout)

    @classmethod
    def file(cls, path: str) -> "JsonlAuditSink":
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return cls(open(p, "a"))

    def write(self, entry: AuditEntry) -> None:
        with self._lock:
            self._writer.write(entry.to_json() + "\n")
            self._writer.flush()


class MemoryAuditSink(AuditSink):
    """In-memory audit sink for testing."""

    def __init__(self):
        self._entries: List[AuditEntry] = []
        self._lock = threading.Lock()

    def write(self, entry: AuditEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    @property
    def entries(self) -> List[AuditEntry]:
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class AuditLog:
    """Records audit entries for a single request."""

    def __init__(self, request_id: str, sinks: Optional[List[AuditSink]] = None):
        self._request_id = request_id
        self._sinks = sinks or []

    @classmethod
    def noop(cls) -> "AuditLog":
        """Audit log that discards all entries."""
        return cls("noop", [])

    @property
    def request_id(self) -> str:
        return self._request_id

    def record(
        self,
        phase: AuditPhase,
        outcome: AuditOutcome,
        target: Optional[str] = None,
        detail: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=self._request_id,
            phase=phase.value,
            outcome=outcome.value,
            target=target,
            detail=detail,
            metadata=metadata,
        )
        for sink in self._sinks:
            try:
                sink.write(entry)
            except Exception:
                pass  # audit must never crash the request
