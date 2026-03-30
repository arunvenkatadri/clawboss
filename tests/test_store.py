"""Tests for clawboss.store — MemoryStore, SqliteStore, and security invariants."""

import os
import tempfile

from clawboss.store import (
    MAX_PAYLOAD_BYTES,
    Checkpoint,
    MemoryStore,
    SessionStatus,
    SqliteStore,
    new_session_id,
    validate_payload,
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

    def test_created_at_defaults_to_timestamp(self):
        cp = Checkpoint(session_id="x", agent_id="a", status=SessionStatus.RUNNING)
        assert cp.created_at == cp.timestamp

    def test_audit_log_defaults_empty(self):
        cp = Checkpoint(session_id="x", agent_id="a", status=SessionStatus.RUNNING)
        assert cp.audit_log == []

    def test_audit_log_round_trip(self):
        entries = [{"phase": "tool_call", "outcome": "allowed"}]
        cp = Checkpoint(
            session_id="x", agent_id="a", status=SessionStatus.RUNNING, audit_log=entries
        )
        d = cp.to_dict()
        restored = Checkpoint.from_dict(d)
        assert restored.audit_log == entries


class TestNewSessionId:
    def test_returns_string(self):
        sid = new_session_id()
        assert isinstance(sid, str)
        assert len(sid) == 32  # 128 bits = 16 bytes = 32 hex chars

    def test_unique(self):
        ids = {new_session_id() for _ in range(100)}
        assert len(ids) == 100

    def test_cryptographically_random(self):
        """Session IDs should have high entropy — no common prefixes."""
        ids = [new_session_id() for _ in range(100)]
        # Check that first 4 chars vary (would be very unlikely with crypto random)
        prefixes = {s[:4] for s in ids}
        assert len(prefixes) > 50


# ---------------------------------------------------------------------------
# Payload validation
# ---------------------------------------------------------------------------


class TestValidatePayload:
    def test_valid_payload(self):
        result = validate_payload({"key": "value", "number": 42})
        assert result == {"key": "value", "number": 42}

    def test_empty_payload(self):
        assert validate_payload({}) == {}

    def test_oversized_payload_raises(self):
        huge = {"data": "x" * (MAX_PAYLOAD_BYTES + 1)}
        try:
            validate_payload(huge)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "maximum size" in str(e).lower()

    def test_non_serializable_objects_coerced(self):
        """Non-JSON types get coerced to strings via default=str."""
        result = validate_payload({"obj": object()})
        # Should not raise — object gets str()'d
        assert isinstance(result["obj"], str)

    def test_nested_payload(self):
        payload = {"a": {"b": {"c": [1, 2, 3]}}}
        assert validate_payload(payload) == payload


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

    def test_audit_log_round_trip(self):
        store = self.make_store()
        cp = _make_checkpoint()
        cp.audit_log = [
            {"phase": "tool_call", "outcome": "allowed", "target": "search"},
            {"phase": "request_end", "outcome": "info"},
        ]
        store.save_checkpoint(cp)
        loaded = store.load_checkpoint("sess-1")
        assert len(loaded.audit_log) == 2
        assert loaded.audit_log[0]["target"] == "search"

    def test_created_at_preserved(self):
        store = self.make_store()
        cp = _make_checkpoint()
        original_created = cp.created_at
        store.save_checkpoint(cp)
        loaded = store.load_checkpoint("sess-1")
        assert loaded.created_at == original_created


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

    def test_file_permissions(self):
        """SQLite file should be created with owner-only permissions."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        os.unlink(path)  # remove so SqliteStore creates it fresh
        SqliteStore(path)
        mode = os.stat(path).st_mode
        # Owner should have read+write, group and others should have nothing
        assert mode & 0o077 == 0, f"File permissions too open: {oct(mode)}"
        os.unlink(path)

    def test_delete_expired(self):
        """delete_expired removes old sessions."""
        store = self.make_store()
        cp = _make_checkpoint("old-session")
        cp.created_at = "2020-01-01T00:00:00+00:00"
        store.save_checkpoint(cp)

        cp2 = _make_checkpoint("new-session")
        store.save_checkpoint(cp2)

        deleted = store.delete_expired(86400)  # 1 day
        assert deleted == 1
        assert store.load_checkpoint("old-session") is None
        assert store.load_checkpoint("new-session") is not None
