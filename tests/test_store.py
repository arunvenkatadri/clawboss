"""Tests for clawboss.store — MemoryStore and SqliteStore."""

import os
import tempfile

from clawboss.store import (
    Checkpoint,
    MemoryStore,
    SessionStatus,
    SqliteStore,
    new_session_id,
)

# ---------------------------------------------------------------------------
# Checkpoint dataclass
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_round_trip_dict(self):
        cp = Checkpoint(
            session_id="abc123",
            agent_id="agent-1",
            status=SessionStatus.RUNNING,
            iterations=3,
            tokens_used=500,
            token_limit=10000,
            iteration_limit=5,
            circuit_breaker_states={"web_search": {"state": "closed", "consecutive_failures": 0}},
            policy_dict={"max_iterations": 5},
            payload={"task": "research"},
        )
        d = cp.to_dict()
        assert d["status"] == "running"
        restored = Checkpoint.from_dict(d)
        assert restored.session_id == "abc123"
        assert restored.status == SessionStatus.RUNNING
        assert restored.iterations == 3
        assert restored.tokens_used == 500
        assert restored.payload == {"task": "research"}

    def test_status_from_string(self):
        cp = Checkpoint(session_id="x", agent_id="a", status="paused")
        assert cp.status == SessionStatus.PAUSED

    def test_auto_timestamp(self):
        cp = Checkpoint(session_id="x", agent_id="a", status=SessionStatus.RUNNING)
        assert cp.timestamp != ""


class TestNewSessionId:
    def test_returns_string(self):
        sid = new_session_id()
        assert isinstance(sid, str)
        assert len(sid) == 12

    def test_unique(self):
        ids = {new_session_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# Shared store tests — run against both MemoryStore and SqliteStore
# ---------------------------------------------------------------------------


def _make_checkpoint(sid="sess-1", agent_id="agent-1", status=SessionStatus.RUNNING):
    return Checkpoint(
        session_id=sid,
        agent_id=agent_id,
        status=status,
        iterations=2,
        tokens_used=300,
        token_limit=5000,
        iteration_limit=10,
        payload={"step": "research"},
    )


class StoreContractTests:
    """Mixin with tests that every StateStore implementation must pass."""

    def make_store(self):
        raise NotImplementedError

    def test_save_and_load(self):
        store = self.make_store()
        cp = _make_checkpoint()
        store.save_checkpoint(cp)
        loaded = store.load_checkpoint("sess-1")
        assert loaded is not None
        assert loaded.session_id == "sess-1"
        assert loaded.agent_id == "agent-1"
        assert loaded.status == SessionStatus.RUNNING
        assert loaded.iterations == 2
        assert loaded.tokens_used == 300
        assert loaded.payload == {"step": "research"}

    def test_load_missing_returns_none(self):
        store = self.make_store()
        assert store.load_checkpoint("nonexistent") is None

    def test_upsert_overwrites(self):
        store = self.make_store()
        cp = _make_checkpoint()
        store.save_checkpoint(cp)
        cp.iterations = 5
        cp.status = SessionStatus.PAUSED
        store.save_checkpoint(cp)
        loaded = store.load_checkpoint("sess-1")
        assert loaded.iterations == 5
        assert loaded.status == SessionStatus.PAUSED

    def test_list_sessions(self):
        store = self.make_store()
        store.save_checkpoint(_make_checkpoint("s1"))
        store.save_checkpoint(_make_checkpoint("s2"))
        sessions = store.list_sessions()
        ids = {s.session_id for s in sessions}
        assert "s1" in ids
        assert "s2" in ids

    def test_delete_session(self):
        store = self.make_store()
        store.save_checkpoint(_make_checkpoint("s1"))
        assert store.delete_session("s1") is True
        assert store.load_checkpoint("s1") is None

    def test_delete_missing_returns_false(self):
        store = self.make_store()
        assert store.delete_session("nope") is False

    def test_circuit_breaker_states_round_trip(self):
        store = self.make_store()
        cp = _make_checkpoint()
        cp.circuit_breaker_states = {
            "flaky_tool": {"state": "open", "consecutive_failures": 3}
        }
        store.save_checkpoint(cp)
        loaded = store.load_checkpoint("sess-1")
        assert loaded.circuit_breaker_states["flaky_tool"]["state"] == "open"
        assert loaded.circuit_breaker_states["flaky_tool"]["consecutive_failures"] == 3


class TestMemoryStore(StoreContractTests):
    def make_store(self):
        return MemoryStore()


class TestSqliteStore(StoreContractTests):
    def make_store(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self._db_path = path
        return SqliteStore(path)

    def teardown_method(self):
        if hasattr(self, "_db_path") and os.path.exists(self._db_path):
            os.unlink(self._db_path)

    def test_persists_across_instances(self):
        """Data survives creating a new SqliteStore with the same db file."""
        store1 = self.make_store()
        store1.save_checkpoint(_make_checkpoint("persist-1"))
        # New store instance, same file
        store2 = SqliteStore(self._db_path)
        loaded = store2.load_checkpoint("persist-1")
        assert loaded is not None
        assert loaded.session_id == "persist-1"
