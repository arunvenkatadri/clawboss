"""Durable state persistence — checkpoint / restore for long-running agents.

Pluggable storage backend via the StateStore protocol.  Ships with two
implementations:

- MemoryStore  — dict-backed, for testing
- SqliteStore  — stdlib sqlite3, production-ready single-process default
"""

import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class SessionStatus(Enum):
    """Lifecycle status of an agent session."""

    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Checkpoint:
    """Snapshot of a session's full state — enough to rehydrate a Supervisor."""

    session_id: str
    agent_id: str
    status: SessionStatus

    # Supervisor state
    iterations: int = 0
    tokens_used: int = 0
    token_limit: Optional[int] = None
    iteration_limit: int = 5
    circuit_breaker_states: Dict[str, Any] = field(default_factory=dict)

    # Policy (so we can rebuild the Supervisor without the original object)
    policy_dict: Dict[str, Any] = field(default_factory=dict)

    # Opaque agent payload — intermediate work the agent wants to stash
    payload: Dict[str, Any] = field(default_factory=dict)

    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if isinstance(self.status, str):
            self.status = SessionStatus(self.status)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Checkpoint":
        return cls(
            session_id=d["session_id"],
            agent_id=d["agent_id"],
            status=SessionStatus(d["status"]) if isinstance(d["status"], str) else d["status"],
            iterations=d.get("iterations", 0),
            tokens_used=d.get("tokens_used", 0),
            token_limit=d.get("token_limit"),
            iteration_limit=d.get("iteration_limit", 5),
            circuit_breaker_states=d.get("circuit_breaker_states", {}),
            policy_dict=d.get("policy_dict", {}),
            payload=d.get("payload", {}),
            timestamp=d.get("timestamp", ""),
        )


def new_session_id() -> str:
    """Generate a new session ID."""
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# StateStore protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class StateStore(Protocol):
    """Pluggable storage backend for session checkpoints."""

    def save_checkpoint(self, checkpoint: Checkpoint) -> None: ...

    def load_checkpoint(self, session_id: str) -> Optional[Checkpoint]: ...

    def list_sessions(self) -> List[Checkpoint]: ...

    def delete_session(self, session_id: str) -> bool: ...


# ---------------------------------------------------------------------------
# MemoryStore — dict-backed, for testing
# ---------------------------------------------------------------------------


class MemoryStore:
    """In-memory checkpoint store. Thread-safe."""

    def __init__(self):
        self._data: Dict[str, Checkpoint] = {}
        self._lock = threading.Lock()

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            self._data[checkpoint.session_id] = checkpoint

    def load_checkpoint(self, session_id: str) -> Optional[Checkpoint]:
        with self._lock:
            return self._data.get(session_id)

    def list_sessions(self) -> List[Checkpoint]:
        with self._lock:
            return list(self._data.values())

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._data:
                del self._data[session_id]
                return True
            return False


# ---------------------------------------------------------------------------
# SqliteStore — stdlib sqlite3, production default
# ---------------------------------------------------------------------------

_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS checkpoints (
    session_id   TEXT PRIMARY KEY,
    agent_id     TEXT NOT NULL,
    status       TEXT NOT NULL,
    iterations   INTEGER NOT NULL DEFAULT 0,
    tokens_used  INTEGER NOT NULL DEFAULT 0,
    token_limit  INTEGER,
    iteration_limit INTEGER NOT NULL DEFAULT 5,
    circuit_breaker_states TEXT NOT NULL DEFAULT '{}',
    policy_dict  TEXT NOT NULL DEFAULT '{}',
    payload      TEXT NOT NULL DEFAULT '{}',
    timestamp    TEXT NOT NULL
)"""


class SqliteStore:
    """SQLite-backed checkpoint store. Thread-safe (serialized via lock).

    Uses stdlib sqlite3 — no extra dependencies.

    Usage:
        store = SqliteStore("checkpoints.db")
        store.save_checkpoint(checkpoint)
    """

    def __init__(self, db_path: str = "clawboss_sessions.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """\
                    INSERT OR REPLACE INTO checkpoints
                        (session_id, agent_id, status, iterations, tokens_used,
                         token_limit, iteration_limit, circuit_breaker_states,
                         policy_dict, payload, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        checkpoint.session_id,
                        checkpoint.agent_id,
                        checkpoint.status.value,
                        checkpoint.iterations,
                        checkpoint.tokens_used,
                        checkpoint.token_limit,
                        checkpoint.iteration_limit,
                        json.dumps(checkpoint.circuit_breaker_states),
                        json.dumps(checkpoint.policy_dict),
                        json.dumps(checkpoint.payload),
                        checkpoint.timestamp,
                    ),
                )

    def load_checkpoint(self, session_id: str) -> Optional[Checkpoint]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM checkpoints WHERE session_id = ?", (session_id,)
                ).fetchone()
                if row is None:
                    return None
                return self._row_to_checkpoint(row)

    def list_sessions(self) -> List[Checkpoint]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM checkpoints ORDER BY timestamp DESC"
                ).fetchall()
                return [self._row_to_checkpoint(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM checkpoints WHERE session_id = ?", (session_id,)
                )
                return cursor.rowcount > 0

    @staticmethod
    def _row_to_checkpoint(row) -> Checkpoint:
        return Checkpoint(
            session_id=row[0],
            agent_id=row[1],
            status=SessionStatus(row[2]),
            iterations=row[3],
            tokens_used=row[4],
            token_limit=row[5],
            iteration_limit=row[6],
            circuit_breaker_states=json.loads(row[7]),
            policy_dict=json.loads(row[8]),
            payload=json.loads(row[9]),
            timestamp=row[10],
        )
