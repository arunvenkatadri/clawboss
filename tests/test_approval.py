"""Tests for clawboss approval flow — human-in-the-loop confirmation."""

import pytest

from clawboss.approval import ApprovalQueue, ApprovalStatus
from clawboss.session import SessionManager
from clawboss.store import MemoryStore

try:
    from fastapi.testclient import TestClient

    from clawboss.server import create_app

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POLICY_WITH_CONFIRM = {
    "max_iterations": 5,
    "tool_timeout": 10,
    "require_confirm": ["delete_file", "send_email"],
}


async def delete_file(path: str = "/tmp/data") -> str:
    return f"Deleted {path}"


async def safe_tool(query: str = "test") -> str:
    return f"result for {query}"


# ---------------------------------------------------------------------------
# ApprovalQueue unit tests
# ---------------------------------------------------------------------------


class TestApprovalQueue:
    def test_submit_creates_pending(self):
        q = ApprovalQueue()
        req = q.submit("delete_file", {"path": "/tmp"}, "session-1")
        assert req.status == ApprovalStatus.PENDING
        assert req.tool_name == "delete_file"
        assert req.session_id == "session-1"

    def test_approve(self):
        q = ApprovalQueue()
        req = q.submit("delete_file", {"path": "/tmp"}, "s1")
        result = q.approve(req.approval_id, approved_by="admin")
        assert result.status == ApprovalStatus.APPROVED
        assert result.resolved_by == "admin"

    def test_deny(self):
        q = ApprovalQueue()
        req = q.submit("delete_file", {"path": "/tmp"}, "s1")
        result = q.deny(req.approval_id, reason="Too risky")
        assert result.status == ApprovalStatus.DENIED
        assert result.deny_reason == "Too risky"

    def test_approve_nonexistent_returns_none(self):
        q = ApprovalQueue()
        assert q.approve("nope") is None

    def test_deny_nonexistent_returns_none(self):
        q = ApprovalQueue()
        assert q.deny("nope") is None

    def test_cannot_approve_already_denied(self):
        q = ApprovalQueue()
        req = q.submit("delete_file", {}, "s1")
        q.deny(req.approval_id)
        assert q.approve(req.approval_id) is None

    def test_list_pending(self):
        q = ApprovalQueue()
        q.submit("tool_a", {}, "s1")
        q.submit("tool_b", {}, "s1")
        q.submit("tool_c", {}, "s2")
        assert len(q.list_pending("s1")) == 2
        assert len(q.list_pending("s2")) == 1
        assert len(q.list_pending()) == 3

    def test_list_pending_excludes_resolved(self):
        q = ApprovalQueue()
        req = q.submit("tool_a", {}, "s1")
        q.approve(req.approval_id)
        assert len(q.list_pending("s1")) == 0
        assert len(q.list_all("s1")) == 1


# ---------------------------------------------------------------------------
# Supervisor integration — require_confirm with approval queue
# ---------------------------------------------------------------------------


class TestSupervisorApproval:
    @pytest.mark.asyncio
    async def test_confirmed_tool_queues_approval(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY_WITH_CONFIRM)
        sv = mgr.get_supervisor(sid)

        result = await sv.call("delete_file", delete_file, path="/tmp/data")
        assert result.succeeded is False
        assert result.error.kind == "approval_pending"
        assert "approval_id" in result.error.details

        # Check it's in the queue
        pending = mgr.approval_queue.list_pending(sid)
        assert len(pending) == 1
        assert pending[0].tool_name == "delete_file"

    @pytest.mark.asyncio
    async def test_safe_tool_not_affected(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY_WITH_CONFIRM)
        sv = mgr.get_supervisor(sid)

        result = await sv.call("search", safe_tool, query="test")
        assert result.succeeded is True

    @pytest.mark.asyncio
    async def test_execute_after_approval(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY_WITH_CONFIRM)
        sv = mgr.get_supervisor(sid)

        # Tool call gets queued
        result = await sv.call("delete_file", delete_file, path="/important")
        approval_id = result.error.details["approval_id"]

        # Approve it
        mgr.approval_queue.approve(approval_id)

        # Execute the approved call
        result = await sv.execute_approved(approval_id, delete_file)
        assert result.succeeded is True
        assert result.output == "Deleted /important"

    @pytest.mark.asyncio
    async def test_execute_denied_returns_error(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY_WITH_CONFIRM)
        sv = mgr.get_supervisor(sid)

        result = await sv.call("delete_file", delete_file, path="/tmp")
        approval_id = result.error.details["approval_id"]

        mgr.approval_queue.deny(approval_id, reason="Not safe")

        result = await sv.execute_approved(approval_id, delete_file)
        assert result.succeeded is False
        assert result.error.kind == "approval_denied"
        assert "Not safe" in str(result.error)

    @pytest.mark.asyncio
    async def test_execute_still_pending_returns_pending(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", POLICY_WITH_CONFIRM)
        sv = mgr.get_supervisor(sid)

        result = await sv.call("delete_file", delete_file, path="/tmp")
        approval_id = result.error.details["approval_id"]

        # Try to execute before approval
        result = await sv.execute_approved(approval_id, delete_file)
        assert result.succeeded is False
        assert result.error.kind == "approval_pending"


# ---------------------------------------------------------------------------
# REST API — approval endpoints
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestApprovalREST:
    def _make_client(self):
        store = MemoryStore()
        manager = SessionManager(store)
        app = create_app(manager)
        return TestClient(app), manager

    def test_list_approvals_empty(self):
        client, mgr = self._make_client()
        resp = client.post("/sessions", json={"agent_id": "a"})
        sid = resp.json()["session_id"]
        resp = client.get(f"/sessions/{sid}/approvals")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_full_approval_flow_via_rest(self):
        client, mgr = self._make_client()
        # Create session with require_confirm
        resp = client.post(
            "/sessions",
            json={"agent_id": "a", "policy": POLICY_WITH_CONFIRM},
        )
        sid = resp.json()["session_id"]

        # Simulate a tool call that needs approval (via the manager directly)
        sv = mgr.get_supervisor(sid)
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            sv.call("delete_file", delete_file, path="/tmp")
        )
        assert result.error.kind == "approval_pending"
        approval_id = result.error.details["approval_id"]

        # List approvals via REST
        resp = client.get(f"/sessions/{sid}/approvals")
        assert resp.status_code == 200
        approvals = resp.json()
        assert len(approvals) == 1
        assert approvals[0]["status"] == "pending"
        assert approvals[0]["tool_name"] == "delete_file"

        # Approve via REST
        resp = client.post(f"/sessions/{sid}/approvals/{approval_id}/approve")
        assert resp.status_code == 200
        assert resp.json()["status"] == "approved"

        # Verify it's no longer pending
        resp = client.get(f"/sessions/{sid}/approvals")
        pending = [a for a in resp.json() if a["status"] == "pending"]
        assert len(pending) == 0

    def test_deny_via_rest(self):
        client, mgr = self._make_client()
        resp = client.post(
            "/sessions",
            json={"agent_id": "a", "policy": POLICY_WITH_CONFIRM},
        )
        sid = resp.json()["session_id"]
        sv = mgr.get_supervisor(sid)

        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            sv.call("send_email", delete_file, path="test")
        )
        approval_id = result.error.details["approval_id"]

        resp = client.post(
            f"/sessions/{sid}/approvals/{approval_id}/deny",
            json={"reason": "Not authorized"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "denied"
        assert resp.json()["deny_reason"] == "Not authorized"

    def test_approve_nonexistent_returns_404(self):
        client, _ = self._make_client()
        resp = client.post("/sessions", json={"agent_id": "a"})
        sid = resp.json()["session_id"]
        resp = client.post(f"/sessions/{sid}/approvals/fake-id/approve")
        assert resp.status_code == 404
