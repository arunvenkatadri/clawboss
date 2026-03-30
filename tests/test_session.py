"""Tests for clawboss.session — SessionManager lifecycle."""


import pytest

from clawboss.errors import ClawbossError
from clawboss.session import SessionManager
from clawboss.store import MemoryStore, SessionStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def good_tool(query: str = "test") -> str:
    return f"result for {query}"


POLICY = {"max_iterations": 5, "tool_timeout": 10, "token_budget": 10000}


# ---------------------------------------------------------------------------
# Lifecycle: start → pause → resume → stop
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def test_start_creates_session(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY, {"task": "research"})
        assert isinstance(sid, str)
        cp = mgr.status(sid)
        assert cp is not None
        assert cp.agent_id == "agent-1"
        assert cp.status == SessionStatus.RUNNING
        assert cp.payload == {"task": "research"}

    def test_start_returns_unique_ids(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        ids = {mgr.start("agent-1", POLICY) for _ in range(50)}
        assert len(ids) == 50

    def test_pause_sets_status(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY)
        mgr.pause(sid)
        cp = mgr.status(sid)
        assert cp.status == SessionStatus.PAUSED

    @pytest.mark.asyncio
    async def test_pause_blocks_tool_calls(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY)
        sv = mgr.get_supervisor(sid)
        mgr.pause(sid)
        result = await sv.call("search", good_tool, query="python")
        assert result.succeeded is False
        assert result.error.kind == "agent_paused"

    def test_resume_clears_pause(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY)
        mgr.pause(sid)
        sv = mgr.resume(sid)
        assert sv.paused is False
        cp = mgr.status(sid)
        assert cp.status == SessionStatus.RUNNING

    @pytest.mark.asyncio
    async def test_resume_restores_budget_state(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY)
        sv = mgr.get_supervisor(sid)
        sv.record_tokens(500)
        sv.record_iteration()
        mgr.pause(sid)
        sv2 = mgr.resume(sid)
        snap = sv2.budget()
        assert snap.tokens_used == 500
        assert snap.iterations == 1

    def test_stop_marks_stopped(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY)
        mgr.stop(sid)
        cp = mgr.status(sid)
        assert cp.status == SessionStatus.STOPPED

    def test_stop_removes_supervisor_from_memory(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY)
        mgr.stop(sid)
        assert mgr.get_supervisor(sid) is None

    def test_list_sessions(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        s1 = mgr.start("agent-1", POLICY)
        s2 = mgr.start("agent-2", POLICY)
        sessions = mgr.list_sessions()
        ids = {s.session_id for s in sessions}
        assert s1 in ids
        assert s2 in ids

    def test_status_missing_returns_none(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        assert mgr.status("nonexistent") is None


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestSessionErrors:
    def test_pause_nonexistent_raises(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        with pytest.raises(ClawbossError) as exc_info:
            mgr.pause("nope")
        assert exc_info.value.kind == "session_not_found"

    def test_resume_nonexistent_raises(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        with pytest.raises(ClawbossError) as exc_info:
            mgr.resume("nope")
        assert exc_info.value.kind == "session_not_found"

    def test_stop_nonexistent_raises(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        with pytest.raises(ClawbossError) as exc_info:
            mgr.stop("nope")
        assert exc_info.value.kind == "session_not_found"


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    @pytest.mark.asyncio
    async def test_recover_after_simulated_crash(self):
        """Create session, use supervisor, discard manager, resume from same store."""
        store = MemoryStore()

        # Phase 1: create and use
        mgr1 = SessionManager(store)
        sid = mgr1.start("agent-1", POLICY)
        sv = mgr1.get_supervisor(sid)
        sv.record_tokens(750)
        sv.record_iteration()
        sv.record_iteration()
        await sv.call("search", good_tool, query="test")

        # "Crash" — discard the manager entirely
        del mgr1

        # Phase 2: new manager, same store
        mgr2 = SessionManager(store)
        sv2 = mgr2.resume(sid)
        snap = sv2.budget()
        assert snap.tokens_used == 750
        assert snap.iterations == 2
        # Can still make calls
        result = await sv2.call("search", good_tool, query="recovered")
        assert result.succeeded is True
        assert result.output == "result for recovered"

    def test_recover_preserves_payload(self):
        store = MemoryStore()
        mgr1 = SessionManager(store)
        sid = mgr1.start("agent-1", POLICY, {"step": 3, "data": [1, 2, 3]})
        del mgr1
        mgr2 = SessionManager(store)
        cp = mgr2.status(sid)
        assert cp.payload == {"step": 3, "data": [1, 2, 3]}


# ---------------------------------------------------------------------------
# Payload management
# ---------------------------------------------------------------------------


class TestPayload:
    def test_update_payload(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY, {"step": 1})
        mgr.update_payload(sid, {"step": 2, "results": ["a", "b"]})
        cp = mgr.status(sid)
        assert cp.payload == {"step": 2, "results": ["a", "b"]}

    def test_update_payload_nonexistent_raises(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        with pytest.raises(ClawbossError):
            mgr.update_payload("nope", {})


# ---------------------------------------------------------------------------
# Audit entries
# ---------------------------------------------------------------------------


class TestAuditEntries:
    def test_get_audit_entries(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY)
        entries = mgr.get_audit_entries(sid)
        # At minimum, REQUEST_START should be recorded
        assert len(entries) >= 1
        assert any(e.get("phase") == "request_start" for e in entries)

    def test_get_audit_entries_missing_session(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        assert mgr.get_audit_entries("nope") == []
