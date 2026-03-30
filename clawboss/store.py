"""Durable state persistence — checkpoint / restore for long-running agents.

Pluggable storage backend via the StateStore protocol.  Ships with two
implementations:

- MemoryStore  — dict-backed, for testing
- SqliteStore  — stdlib sqlite3, production-ready single-process default

Security notes:
- policy_dict is the ORIGINAL immutable policy set at start(). It is never
  updated from agent-controlled data. The Supervisor always rebuilds from this.
- payload is agent-writable and UNTRUSTED. Treat it like user input on resume.
- circuit_breaker_states and budget counters are system-controlled (written only
  by the Supervisor internals, never by agent code).
"""

import json
import os
import secrets
import sqlite3
import stat
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# Maximum payload size in bytes (1 MB). Payloads exceeding this are rejected.
MAX_PAYLOAD_BYTES = 1_048_576


class SessionStatus(Enum):
    """Lifecycle status of an agent session."""

    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Checkpoint:
    """Snapshot of a session's full state — enough to rehydrate a Supervisor.

    Trust levels:
    - SYSTEM-CONTROLLED (written only by Supervisor/SessionManager internals):
      session_id, agent_id, status, iterations, tokens_used, token_limit,
      iteration_limit, circuit_breaker_states, original_policy_dict, timestamp,
      created_at
    - AGENT-WRITABLE (untrusted — treat like user input):
      payload
    """

    session_id: str
    agent_id: str
    status: SessionStatus

    # Supervisor state (system-controlled)
    iterations: int = 0
    tokens_used: int = 0
    token_limit: Optional[int] = None
    iteration_limit: int = 5
    circuit_breaker_states: Dict[str, Any] = field(default_factory=dict)

    # IMMUTABLE original policy set at start() — never updated from agent data
    policy_dict: Dict[str, Any] = field(default_factory=dict)

    # Agent-writable payload — UNTRUSTED, treat like user input on resume
    payload: Dict[str, Any] = field(default_factory=dict)

    timestamp: str = ""
    created_at: str = ""  # set once at creation, used for expiry

    # Persisted audit entries (list of dicts) — survives crashes
    audit_log: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = self.timestamp
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
            created_at=d.get("created_at", d.get("timestamp", "")),
            audit_log=d.get("audit_log", []),
        )


def new_session_id() -> str:
    """Generate a cryptographically random session ID (128 bits of entropy)."""
    return secrets.token_hex(16)


def validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize an agent payload before storage.

    Enforces size limits and ensures the payload is safe to serialize.
    Raises ValueError if the payload is invalid.
    """
    serialized = json.dumps(payload, default=str)
    if len(serialized.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"Payload exceeds maximum size ({MAX_PAYLOAD_BYTES} bytes). "
            f"Got {len(serialized.encode('utf-8'))} bytes."
        )
    # Re-parse to strip any non-JSON-serializable objects that default=str masked
    return json.loads(serialized)


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
    timestamp    TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT '',
    audit_log    TEXT NOT NULL DEFAULT '[]'
)"""

# Migration: add columns if upgrading from Phase 1 schema
_MIGRATIONS = [
    "ALTER TABLE checkpoints ADD COLUMN created_at TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE checkpoints ADD COLUMN audit_log TEXT NOT NULL DEFAULT '[]'",
]


class SqliteStore:
    """SQLite-backed checkpoint store. Thread-safe (serialized via lock).

    Uses stdlib sqlite3 — no extra dependencies.
    Creates the database file with owner-only permissions (0600).

    Usage:
        store = SqliteStore("checkpoints.db")
        store.save_checkpoint(checkpoint)
    """

    def __init__(self, db_path: str = "clawboss_sessions.db"):
        self._db_path = db_path
        self._lock = threading.Lock()

        # Create file with restricted permissions if it doesn't exist
        if not os.path.exists(db_path):
            fd = os.open(db_path, os.O_CREAT | os.O_WRONLY, stat.S_IRUSR | stat.S_IWUSR)
            os.close(fd)

        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)
            # Run migrations for existing databases
            for migration in _MIGRATIONS:
                try:
                    conn.execute(migration)
                except sqlite3.OperationalError:
                    pass  # column already exists

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
                         policy_dict, payload, timestamp, created_at, audit_log)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        checkpoint.created_at,
                        json.dumps(checkpoint.audit_log),
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

    def delete_expired(self, max_age_seconds: float) -> int:
        """Delete sessions older than max_age_seconds. Returns count deleted."""
        cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
        with self._lock:
            with self._connect() as conn:
                # created_at is ISO format — compare as strings (works for ISO 8601)
                cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
                cursor = conn.execute(
                    "DELETE FROM checkpoints WHERE created_at != '' AND created_at < ?",
                    (cutoff_iso,),
                )
                return cursor.rowcount

    @staticmethod
    def _row_to_checkpoint(row) -> Checkpoint:
        # Handle both old (11-column) and new (13-column) schemas
        created_at = row[11] if len(row) > 11 else ""
        audit_log = json.loads(row[12]) if len(row) > 12 else []
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
            created_at=created_at,
            audit_log=audit_log,
        )
