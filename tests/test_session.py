"""Tests for clawboss.session — SessionManager lifecycle and security invariants."""

import pytest

from clawboss.errors import ClawbossError
from clawboss.session import SessionManager
from clawboss.store import MAX_PAYLOAD_BYTES, MemoryStore, SessionStatus

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

    def test_oversized_payload_rejected_on_start(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        huge = {"data": "x" * (MAX_PAYLOAD_BYTES + 1)}
        with pytest.raises(ValueError, match="maximum size"):
            mgr.start("agent-1", POLICY, huge)

    def test_oversized_payload_rejected_on_update(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY)
        huge = {"data": "x" * (MAX_PAYLOAD_BYTES + 1)}
        with pytest.raises(ValueError, match="maximum size"):
            mgr.update_payload(sid, huge)


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

    def test_audit_persisted_on_pause(self):
        """Audit entries should be persisted to the store when pausing."""
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY)
        mgr.pause(sid)
        cp = store.load_checkpoint(sid)
        assert len(cp.audit_log) > 0
        assert any(e.get("phase") == "request_start" for e in cp.audit_log)

    def test_audit_persisted_on_stop(self):
        """Audit entries should be persisted to the store when stopping."""
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY)
        mgr.stop(sid)
        cp = store.load_checkpoint(sid)
        assert len(cp.audit_log) > 0

    def test_audit_survives_crash(self):
        """After pause (persists audit), crash, and resume — audit is recoverable."""
        store = MemoryStore()
        mgr1 = SessionManager(store)
        sid = mgr1.start("agent-1", POLICY)
        mgr1.pause(sid)
        del mgr1

        mgr2 = SessionManager(store)
        entries = mgr2.get_audit_entries(sid)
        assert len(entries) > 0
        assert any(e.get("phase") == "request_start" for e in entries)


# ---------------------------------------------------------------------------
# Security: Policy immutability
# ---------------------------------------------------------------------------


class TestPolicyImmutability:
    def test_resume_uses_original_policy(self):
        """Even if checkpoint policy_dict is tampered with, resume uses the original."""
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", {
            "max_iterations": 3,
            "token_budget": 1000,
            "require_confirm": ["dangerous_tool"],
        })

        # Simulate an attacker tampering with the stored checkpoint's policy
        cp = store.load_checkpoint(sid)
        cp.policy_dict = {
            "max_iterations": 9999,
            "token_budget": 999999999,
            "require_confirm": [],  # attacker removed confirmation gate!
        }
        store.save_checkpoint(cp)

        # Resume should use the ORIGINAL policy, not the tampered one
        sv = mgr.resume(sid)
        assert sv.policy.max_iterations == 3
        assert sv.policy.token_budget == 1000
        assert "dangerous_tool" in sv.policy.require_confirm

    def test_resume_after_crash_uses_stored_original_policy(self):
        """After a process restart, resume uses the checkpoint's policy_dict
        which was set at start() and never overwritten by auto-checkpoint."""
        store = MemoryStore()
        mgr1 = SessionManager(store)
        sid = mgr1.start("agent-1", {
            "max_iterations": 3,
            "require_confirm": ["dangerous_tool"],
        })
        del mgr1

        # New manager — no in-memory cache of original policy
        mgr2 = SessionManager(store)
        sv = mgr2.resume(sid)
        assert sv.policy.max_iterations == 3
        assert "dangerous_tool" in sv.policy.require_confirm

    @pytest.mark.asyncio
    async def test_auto_checkpoint_does_not_overwrite_policy(self):
        """Auto-checkpoint after tool calls should not change the policy in the store."""
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", {
            "max_iterations": 5,
            "require_confirm": ["delete_file"],
        })
        sv = mgr.get_supervisor(sid)
        await sv.call("search", good_tool, query="test")

        # Check the stored checkpoint still has the original policy
        cp = store.load_checkpoint(sid)
        assert cp.policy_dict["max_iterations"] == 5
        assert "delete_file" in cp.policy_dict["require_confirm"]

    @pytest.mark.asyncio
    async def test_auto_checkpoint_preserves_payload(self):
        """Auto-checkpoint should not wipe the payload."""
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY, {"important": "data"})
        sv = mgr.get_supervisor(sid)
        await sv.call("search", good_tool, query="test")

        cp = store.load_checkpoint(sid)
        assert cp.payload == {"important": "data"}


# ---------------------------------------------------------------------------
# Session ID entropy
# ---------------------------------------------------------------------------


class TestSessionIdSecurity:
    def test_session_id_length(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY)
        assert len(sid) == 32  # 128 bits of entropy

    def test_session_ids_not_sequential(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        ids = [mgr.start("agent-1", POLICY) for _ in range(10)]
        # No common prefix — crypto random
        prefixes = {s[:8] for s in ids}
        assert len(prefixes) == 10


# ---------------------------------------------------------------------------
# Stateless sessions
# ---------------------------------------------------------------------------


class TestStatelessSessions:
    @pytest.mark.asyncio
    async def test_stateless_session_works(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY, stateless=True)
        sv = mgr.get_supervisor(sid)
        result = await sv.call("search", good_tool, query="test")
        assert result.succeeded is True

    def test_stateless_flag_stored_in_checkpoint(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY, stateless=True)
        cp = mgr.status(sid)
        assert cp.stateless is True

    def test_stateful_flag_default(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY)
        cp = mgr.status(sid)
        assert cp.stateless is False

    def test_stateless_pause_and_unpause_in_memory(self):
        """Stateless sessions can be paused/resumed while still in memory."""
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY, stateless=True)
        mgr.pause(sid)
        sv = mgr.resume(sid)
        assert sv.paused is False

    def test_stateless_cannot_resume_after_crash(self):
        """Stateless sessions cannot be recovered after the manager is gone."""
        store = MemoryStore()
        mgr1 = SessionManager(store)
        sid = mgr1.start("agent-1", POLICY, stateless=True)
        del mgr1

        mgr2 = SessionManager(store)
        with pytest.raises(ClawbossError) as exc_info:
            mgr2.resume(sid)
        assert exc_info.value.kind == "session_not_recoverable"

    def test_stateless_stop_works(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY, stateless=True)
        mgr.stop(sid)
        cp = mgr.status(sid)
        assert cp.status == SessionStatus.STOPPED

    @pytest.mark.asyncio
    async def test_stateless_no_auto_checkpoint(self):
        """Stateless sessions should not update checkpoint on every tool call."""
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY, stateless=True)
        sv = mgr.get_supervisor(sid)
        sv.record_tokens(500)
        await sv.call("search", good_tool, query="test")
        # The checkpoint should still have 0 tokens (no auto-checkpoint)
        cp = store.load_checkpoint(sid)
        assert cp.tokens_used == 0
        assert cp.iterations == 0


# ---------------------------------------------------------------------------
# Crash loop protection (max_resumes)
# ---------------------------------------------------------------------------


class TestCrashLoopProtection:
    def test_default_max_resumes_is_3(self):
        """Policy defaults to max_resumes=3."""
        from clawboss.policy import Policy
        p = Policy()
        assert p.max_resumes == 3

    def test_resume_increments_count(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY)
        mgr.pause(sid)
        mgr.resume(sid)
        cp = mgr.status(sid)
        assert cp.resume_count == 1

    def test_resume_count_persists_across_crashes(self):
        store = MemoryStore()
        mgr1 = SessionManager(store)
        sid = mgr1.start("agent-1", POLICY)
        mgr1.pause(sid)
        mgr1.resume(sid)
        del mgr1

        mgr2 = SessionManager(store)
        mgr2.pause(sid)
        mgr2.resume(sid)
        cp = mgr2.status(sid)
        assert cp.resume_count == 2

    def test_max_resumes_exceeded_marks_failed(self):
        """After max_resumes, session is marked FAILED and resume raises."""
        store = MemoryStore()
        policy = {"max_iterations": 5, "max_resumes": 2}
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", policy)

        # Resume 1
        mgr.pause(sid)
        mgr.resume(sid)

        # Resume 2
        mgr.pause(sid)
        mgr.resume(sid)

        # Resume 3 — should fail (limit is 2)
        mgr.pause(sid)
        with pytest.raises(ClawbossError) as exc_info:
            mgr.resume(sid)
        assert exc_info.value.kind == "max_resumes_exceeded"

        # Session is marked as failed
        cp = mgr.status(sid)
        assert cp.status == SessionStatus.FAILED

    def test_crash_loop_after_repeated_crashes(self):
        """Simulate an agent that keeps crashing and being resumed."""
        store = MemoryStore()
        policy = {"max_iterations": 5, "max_resumes": 2}

        # Process 1: start, crash
        mgr1 = SessionManager(store)
        sid = mgr1.start("crashy-agent", policy)
        del mgr1

        # Process 2: resume, crash
        mgr2 = SessionManager(store)
        mgr2.resume(sid)
        del mgr2

        # Process 3: resume, crash
        mgr3 = SessionManager(store)
        mgr3.resume(sid)
        del mgr3

        # Process 4: resume should fail — crash loop detected
        mgr4 = SessionManager(store)
        with pytest.raises(ClawbossError) as exc_info:
            mgr4.resume(sid)
        assert exc_info.value.kind == "max_resumes_exceeded"
        assert "crash loop" in str(exc_info.value).lower()

    def test_custom_max_resumes(self):
        store = MemoryStore()
        policy = {"max_iterations": 5, "max_resumes": 10}
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", policy)
        # Should allow 10 resumes
        for _ in range(10):
            mgr.pause(sid)
            mgr.resume(sid)
        # 11th should fail
        mgr.pause(sid)
        with pytest.raises(ClawbossError) as exc_info:
            mgr.resume(sid)
        assert exc_info.value.kind == "max_resumes_exceeded"
