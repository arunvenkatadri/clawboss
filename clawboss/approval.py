"""Approval queue — human-in-the-loop confirmation for dangerous tool calls.

When a tool is in the policy's require_confirm list, the Supervisor queues
the call here instead of blocking it outright. The call stays pending until
a human approves or denies it via the REST API, dashboard, or any client
listening on the WebSocket events stream.

Usage (internal — SessionManager wires this up automatically):

    queue = ApprovalQueue()
    approval_id = queue.submit("delete_file", {"path": "/tmp/data"}, session_id)
    # ... user reviews and approves via REST ...
    queue.approve(approval_id)
    # ... or denies ...
    queue.deny(approval_id, reason="Too risky")
"""

import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


@dataclass
class ApprovalRequest:
    """A pending tool call awaiting human approval."""

    approval_id: str
    session_id: str
    tool_name: str
    tool_args: Dict[str, Any]
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = ""
    resolved_at: str = ""
    resolved_by: str = ""  # who approved/denied (for audit)
    deny_reason: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "tool_args": {k: str(v)[:200] for k, v in self.tool_args.items()},
            "status": self.status.value,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "deny_reason": self.deny_reason,
        }


class ApprovalQueue:
    """Thread-safe queue of pending tool call approvals.

    One queue per SessionManager — holds approvals for all sessions.
    """

    def __init__(self):
        self._requests: Dict[str, ApprovalRequest] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        session_id: str,
    ) -> ApprovalRequest:
        """Queue a tool call for approval. Returns the ApprovalRequest."""
        req = ApprovalRequest(
            approval_id=secrets.token_hex(8),
            session_id=session_id,
            tool_name=tool_name,
            tool_args=tool_args,
        )
        with self._lock:
            self._requests[req.approval_id] = req
        return req

    def get(self, approval_id: str) -> Optional[ApprovalRequest]:
        with self._lock:
            return self._requests.get(approval_id)

    def approve(self, approval_id: str, approved_by: str = "") -> Optional[ApprovalRequest]:
        """Approve a pending request. Returns the request, or None if not found."""
        with self._lock:
            req = self._requests.get(approval_id)
            if req is None or req.status != ApprovalStatus.PENDING:
                return None
            req.status = ApprovalStatus.APPROVED
            req.resolved_at = datetime.now(timezone.utc).isoformat()
            req.resolved_by = approved_by
            return req

    def deny(
        self, approval_id: str, reason: str = "", denied_by: str = ""
    ) -> Optional[ApprovalRequest]:
        """Deny a pending request. Returns the request, or None if not found."""
        with self._lock:
            req = self._requests.get(approval_id)
            if req is None or req.status != ApprovalStatus.PENDING:
                return None
            req.status = ApprovalStatus.DENIED
            req.deny_reason = reason
            req.resolved_at = datetime.now(timezone.utc).isoformat()
            req.resolved_by = denied_by
            return req

    def list_pending(self, session_id: Optional[str] = None) -> List[ApprovalRequest]:
        """List pending approvals, optionally filtered by session."""
        with self._lock:
            reqs = list(self._requests.values())
        if session_id is not None:
            reqs = [r for r in reqs if r.session_id == session_id]
        return [r for r in reqs if r.status == ApprovalStatus.PENDING]

    def list_all(self, session_id: Optional[str] = None) -> List[ApprovalRequest]:
        """List all approvals (any status), optionally filtered by session."""
        with self._lock:
            reqs = list(self._requests.values())
        if session_id is not None:
            reqs = [r for r in reqs if r.session_id == session_id]
        return reqs
