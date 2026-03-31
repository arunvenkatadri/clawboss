"""Session manager — lifecycle management for long-running agent sessions.

Wraps a Supervisor + StateStore and adds start / pause / resume / stop.

Security invariants:
- Policy is IMMUTABLE after start(). resume() always rebuilds the Supervisor
  from the original policy stored at creation time — never from agent-controlled
  checkpoint data. An agent cannot weaken its own supervision.
- Payload is UNTRUSTED. It is validated for size/serializability on write, and
  agents consuming it after resume should treat it like user input.
- Audit entries are persisted to the store so they survive crashes.

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

from .approval import ApprovalQueue
from .audit import AuditLog, MemoryAuditSink
from .errors import ClawbossError
from .observe import Observer
from .policy import Policy
from .store import Checkpoint, SessionStatus, StateStore, new_session_id, validate_payload
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
        self._original_policies: Dict[str, Dict[str, Any]] = {}
        self._stateless_sessions: set = set()
        self._approval_queue = ApprovalQueue()
        self._observer = Observer()
        self._lock = threading.Lock()

    @property
    def approval_queue(self) -> ApprovalQueue:
        """The shared approval queue for all sessions."""
        return self._approval_queue

    @property
    def observer(self) -> Observer:
        """The shared observer for all sessions."""
        return self._observer

    def start(
        self,
        agent_id: str,
        policy_dict: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        stateless: bool = False,
    ) -> str:
        """Start a new agent session.

        Args:
            agent_id: Identifier for the agent type.
            policy_dict: Policy configuration (passed to Policy.from_dict).
                         This is stored immutably — the agent can never change it.
            payload: Opaque JSON the agent can stash intermediate work in.
                     Treated as untrusted data. Validated for size limits.
            stateless: If True, the session lives in memory only — no
                       checkpoints written to the store, no crash recovery.
                       You still get supervision, audit, and pause/stop controls.

        Returns:
            session_id for the new session.
        """
        sid = new_session_id()
        safe_policy_dict = policy_dict or {}
        policy = Policy.from_dict(safe_policy_dict)

        # Validate payload before storing
        safe_payload = validate_payload(payload or {})

        sink = MemoryAuditSink()
        audit = AuditLog(sid, sinks=[sink])

        # Stateless sessions don't wire up the store on the Supervisor,
        # so no auto-checkpointing happens on each call/record.
        sv = Supervisor(
            policy,
            audit=audit,
            store=None if stateless else self._store,
            session_id=sid,
            agent_id=agent_id,
            approval_queue=self._approval_queue,
            observer=self._observer,
        )

        # Store the ORIGINAL policy — this is immutable for the session's lifetime
        frozen_policy = policy.to_dict()

        checkpoint = Checkpoint(
            session_id=sid,
            agent_id=agent_id,
            status=SessionStatus.RUNNING,
            token_limit=policy.token_budget,
            iteration_limit=policy.max_iterations,
            policy_dict=frozen_policy,
            payload=safe_payload,
            stateless=stateless,
        )
        # Always save the initial checkpoint so list/status/pause/stop work,
        # but stateless sessions won't auto-checkpoint after each tool call.
        self._store.save_checkpoint(checkpoint)

        with self._lock:
            self._supervisors[sid] = sv
            self._audit_sinks[sid] = sink
            self._original_policies[sid] = frozen_policy
            if stateless:
                self._stateless_sessions.add(sid)

        return sid

    def pause(self, session_id: str) -> None:
        """Pause an agent session. The Supervisor will raise AgentPaused on next call()."""
        cp = self._store.load_checkpoint(session_id)
        if cp is None:
            raise ClawbossError.session_not_found(session_id)

        cp.status = SessionStatus.PAUSED
        # Persist audit entries before pausing
        self._persist_audit(session_id, cp)
        self._store.save_checkpoint(cp)

        with self._lock:
            sv = self._supervisors.get(session_id)
        if sv is not None:
            sv.paused = True

    def resume(self, session_id: str) -> Supervisor:
        """Resume a paused or previously-crashed session.

        Rehydrates a Supervisor from the last checkpoint. Policy is always
        rebuilt from the ORIGINAL immutable policy stored at start() — never
        from any agent-modified data in the checkpoint.

        Stateless sessions cannot be resumed after a crash — their state
        only exists in memory.

        Returns:
            The restored Supervisor, ready for use.
        """
        cp = self._store.load_checkpoint(session_id)
        if cp is None:
            raise ClawbossError.session_not_found(session_id)

        # Stateless sessions can be unpaused (supervisor still in memory)
        # but cannot be recovered after a crash
        with self._lock:
            in_memory = session_id in self._supervisors
        if cp.stateless and not in_memory:
            raise ClawbossError(
                "session_not_recoverable",
                f"Session {session_id} is stateless and cannot be resumed after a crash",
            )

        # Crash loop protection: check max_resumes from the original policy
        policy = Policy.from_dict(cp.policy_dict)
        if cp.resume_count >= policy.max_resumes:
            cp.status = SessionStatus.FAILED
            cp.failure_reason = (
                f"Crash loop: resumed {cp.resume_count} times (limit: {policy.max_resumes})"
            )
            self._store.save_checkpoint(cp)
            raise ClawbossError.max_resumes_exceeded(
                session_id, cp.resume_count, policy.max_resumes
            )

        cp.resume_count += 1
        cp.status = SessionStatus.RUNNING
        self._store.save_checkpoint(cp)

        sink = MemoryAuditSink()
        audit = AuditLog(session_id, sinks=[sink])

        # SECURITY: always use the original policy, not whatever the checkpoint says.
        # The checkpoint's policy_dict IS the original (set at start()), but we also
        # cache it in _original_policies for sessions started in this process.
        with self._lock:
            original = self._original_policies.get(session_id)
        if original is None:
            # Session was started in a previous process — use the checkpoint's policy,
            # which was set at start() and never mutated by auto-checkpoint.
            original = cp.policy_dict

        sv = Supervisor.restore_from_checkpoint(
            cp,
            audit=audit,
            store=self._store,
            policy_override=original,
            approval_queue=self._approval_queue,
            observer=self._observer,
        )
        sv.paused = False

        with self._lock:
            self._supervisors[session_id] = sv
            self._audit_sinks[session_id] = sink
            self._original_policies[session_id] = original

        return sv

    def stop(self, session_id: str) -> None:
        """Stop an agent session. Calls supervisor.finish() and marks stopped."""
        cp = self._store.load_checkpoint(session_id)
        if cp is None:
            raise ClawbossError.session_not_found(session_id)

        with self._lock:
            sv = self._supervisors.get(session_id)

        if sv is not None:
            sv.finish()

        cp.status = SessionStatus.STOPPED
        # Update with latest supervisor state if available
        if sv is not None:
            data = sv.to_checkpoint_data()
            cp.iterations = data["iterations"]
            cp.tokens_used = data["tokens_used"]

        # Persist audit entries BEFORE removing sink from memory
        self._persist_audit(session_id, cp)

        with self._lock:
            self._supervisors.pop(session_id, None)
            self._audit_sinks.pop(session_id, None)
            self._original_policies.pop(session_id, None)

        self._store.save_checkpoint(cp)

    def restart(self, session_id: str) -> str:
        """Restart a stopped or failed session with the same policy and agent ID.

        Creates a fresh session using the original policy from the old session.
        The old session is preserved for audit purposes.

        Returns:
            The new session_id.
        """
        cp = self._store.load_checkpoint(session_id)
        if cp is None:
            raise ClawbossError.session_not_found(session_id)

        # Clean up in-memory state for the old session if any
        with self._lock:
            self._supervisors.pop(session_id, None)
            self._audit_sinks.pop(session_id, None)
            self._original_policies.pop(session_id, None)
            self._stateless_sessions.discard(session_id)

        return self.start(
            agent_id=cp.agent_id,
            policy_dict=cp.policy_dict,
            stateless=cp.stateless,
        )

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
        """Get audit log entries for a session (in-memory + persisted)."""
        # Start with persisted entries from the store
        cp = self._store.load_checkpoint(session_id)
        persisted = cp.audit_log if cp is not None else []

        # Add in-memory entries from current process
        with self._lock:
            sink = self._audit_sinks.get(session_id)
        in_memory = [e.to_dict() for e in sink.entries] if sink is not None else []

        return persisted + in_memory

    def update_payload(self, session_id: str, payload: Dict[str, Any]) -> None:
        """Update the opaque payload for a session.

        The payload is validated for size limits and serializability.
        Treat payload as UNTRUSTED — it may contain agent-controlled data.
        """
        cp = self._store.load_checkpoint(session_id)
        if cp is None:
            raise ClawbossError.session_not_found(session_id)
        cp.payload = validate_payload(payload)
        self._store.save_checkpoint(cp)

    def delete_expired(self, max_age_seconds: float) -> int:
        """Delete sessions older than max_age_seconds.

        Only works with SqliteStore. For MemoryStore, iterate list_sessions()
        and call delete_session() manually.
        """
        if hasattr(self._store, "delete_expired"):
            return self._store.delete_expired(max_age_seconds)
        return 0

    def _persist_audit(self, session_id: str, cp: Checkpoint) -> None:
        """Flush in-memory audit entries into the checkpoint for persistence."""
        with self._lock:
            sink = self._audit_sinks.get(session_id)
        if sink is not None:
            cp.audit_log = cp.audit_log + [e.to_dict() for e in sink.entries]
