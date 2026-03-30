"""Session manager — lifecycle management for long-running agent sessions.

Wraps a Supervisor + StateStore and adds start / pause / resume / stop.

Usage:
    from clawboss import SessionManager, MemoryStore

    store = MemoryStore()
    mgr = SessionManager(store)

    session_id = mgr.start("my-agent", {"max_iterations": 10, "tool_timeout": 30})
    sv = mgr.get_supervisor(session_id)
    # ... use sv.call() in your agent loop ...

    mgr.pause(session_id)    # agent will raise AgentPaused on next call()
    mgr.resume(session_id)   # rehydrate and continue
    mgr.stop(session_id)     # mark complete
"""

import threading
from typing import Any, Dict, List, Optional

from .audit import AuditLog, MemoryAuditSink
from .errors import ClawbossError
from .policy import Policy
from .store import Checkpoint, SessionStatus, StateStore, new_session_id
from .supervisor import Supervisor


class SessionManager:
    """Lifecycle manager for agent sessions backed by a StateStore.

    Keeps active Supervisors in memory and persists state to the store.
    After a crash, create a new SessionManager with the same store and
    call resume() to pick up where you left off.
    """

    def __init__(self, store: StateStore):
        self._store = store
        self._supervisors: Dict[str, Supervisor] = {}
        self._audit_sinks: Dict[str, MemoryAuditSink] = {}
        self._lock = threading.Lock()

    def start(
        self,
        agent_id: str,
        policy_dict: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new agent session.

        Args:
            agent_id: Identifier for the agent type.
            policy_dict: Policy configuration (passed to Policy.from_dict).
            payload: Opaque JSON the agent can stash intermediate work in.

        Returns:
            session_id for the new session.
        """
        sid = new_session_id()
        policy = Policy.from_dict(policy_dict or {})

        sink = MemoryAuditSink()
        audit = AuditLog(sid, sinks=[sink])

        sv = Supervisor(
            policy,
            audit=audit,
            store=self._store,
            session_id=sid,
            agent_id=agent_id,
        )

        checkpoint = Checkpoint(
            session_id=sid,
            agent_id=agent_id,
            status=SessionStatus.RUNNING,
            token_limit=policy.token_budget,
            iteration_limit=policy.max_iterations,
            policy_dict=policy.to_dict(),
            payload=payload or {},
        )
        self._store.save_checkpoint(checkpoint)

        with self._lock:
            self._supervisors[sid] = sv
            self._audit_sinks[sid] = sink

        return sid

    def pause(self, session_id: str) -> None:
        """Pause an agent session. The Supervisor will raise AgentPaused on next call()."""
        cp = self._store.load_checkpoint(session_id)
        if cp is None:
            raise ClawbossError.session_not_found(session_id)

        cp.status = SessionStatus.PAUSED
        self._store.save_checkpoint(cp)

        with self._lock:
            sv = self._supervisors.get(session_id)
        if sv is not None:
            sv.paused = True

    def resume(self, session_id: str) -> Supervisor:
        """Resume a paused or previously-crashed session.

        Rehydrates a Supervisor from the last checkpoint.

        Returns:
            The restored Supervisor, ready for use.
        """
        cp = self._store.load_checkpoint(session_id)
        if cp is None:
            raise ClawbossError.session_not_found(session_id)

        cp.status = SessionStatus.RUNNING
        self._store.save_checkpoint(cp)

        sink = MemoryAuditSink()
        audit = AuditLog(session_id, sinks=[sink])

        sv = Supervisor.restore_from_checkpoint(cp, audit=audit, store=self._store)
        sv.paused = False

        with self._lock:
            self._supervisors[session_id] = sv
            self._audit_sinks[session_id] = sink

        return sv

    def stop(self, session_id: str) -> None:
        """Stop an agent session. Calls supervisor.finish() and marks stopped."""
        cp = self._store.load_checkpoint(session_id)
        if cp is None:
            raise ClawbossError.session_not_found(session_id)

        with self._lock:
            sv = self._supervisors.pop(session_id, None)
            self._audit_sinks.pop(session_id, None)

        if sv is not None:
            sv.finish()

        cp.status = SessionStatus.STOPPED
        # Update with latest supervisor state if available
        if sv is not None:
            data = sv.to_checkpoint_data()
            cp.iterations = data["iterations"]
            cp.tokens_used = data["tokens_used"]
        self._store.save_checkpoint(cp)

    def status(self, session_id: str) -> Optional[Checkpoint]:
        """Get the current checkpoint for a session."""
        return self._store.load_checkpoint(session_id)

    def list_sessions(self) -> List[Checkpoint]:
        """List all sessions."""
        return self._store.list_sessions()

    def get_supervisor(self, session_id: str) -> Optional[Supervisor]:
        """Get the active Supervisor for a session (None if not in memory)."""
        with self._lock:
            return self._supervisors.get(session_id)

    def get_audit_entries(self, session_id: str) -> list:
        """Get audit log entries for a session."""
        with self._lock:
            sink = self._audit_sinks.get(session_id)
        if sink is None:
            return []
        return [e.to_dict() for e in sink.entries]

    def update_payload(self, session_id: str, payload: Dict[str, Any]) -> None:
        """Update the opaque payload for a session."""
        cp = self._store.load_checkpoint(session_id)
        if cp is None:
            raise ClawbossError.session_not_found(session_id)
        cp.payload = payload
        self._store.save_checkpoint(cp)
