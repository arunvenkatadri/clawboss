"""Supervisor — the core of clawboss.

Wraps tool calls with timeout, budget, circuit breaker, and audit.
Doesn't own the agent loop — supervises whatever loop you're running.

Usage:
    supervisor = Supervisor(policy)

    # In your agent loop:
    iteration = supervisor.record_iteration()  # raises if over limit
    result = await supervisor.call("web_search", search_fn, query="python")

    if result.succeeded:
        process(result.output)
    else:
        handle_error(result.error)

    # When done:
    supervisor.finish()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, List, Optional

from .audit import AuditLog, AuditOutcome, AuditPhase
from .budget import BudgetSnapshot, BudgetTracker
from .circuit_breaker import CircuitBreaker
from .errors import ClawbossError
from .policy import Policy

if TYPE_CHECKING:
    from .approval import ApprovalQueue
    from .store import Checkpoint, StateStore


@dataclass
class SupervisedResult:
    """Result of a supervised tool call."""

    output: Any = None
    error: Optional[ClawbossError] = None
    succeeded: bool = False
    duration_ms: int = 0
    budget: Optional[BudgetSnapshot] = None
    tool_name: Optional[str] = None

    def user_message(self) -> str:
        """Always returns something — either the output or an error message."""
        if self.succeeded and self.output is not None:
            return str(self.output)
        if self.error:
            return self.error.user_message()
        return "No output (tool returned nothing)"


class Supervisor:
    """Supervises tool calls with policy enforcement.

    One Supervisor per skill invocation. Tracks budget across multiple
    tool calls within a single request.
    """

    def __init__(
        self,
        policy: Policy,
        audit: Optional[AuditLog] = None,
        store: Optional[StateStore] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        approval_queue: Optional[ApprovalQueue] = None,
    ):
        self._policy = policy
        self._budget = BudgetTracker.from_policy(policy)
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._tool_call_times: Dict[str, List[float]] = {}
        self._audit = audit or AuditLog.noop()
        self._start_time = time.monotonic()
        self._last_activity = time.monotonic()
        self._paused = False

        # Optional durable state
        self._store = store
        self._session_id = session_id
        self._agent_id = agent_id
        self._approval_queue = approval_queue

        self._audit.record(
            AuditPhase.REQUEST_START,
            AuditOutcome.INFO,
            detail=f"Supervisor started with policy: max_iter={policy.max_iterations}, "
            f"tool_timeout={policy.tool_timeout}s, token_budget={policy.token_budget}",
        )

    @classmethod
    def with_defaults(cls) -> "Supervisor":
        """Create a supervisor with default policy."""
        return cls(Policy())

    @property
    def policy(self) -> Policy:
        return self._policy

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def agent_id(self) -> Optional[str]:
        return self._agent_id

    @property
    def paused(self) -> bool:
        return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        self._paused = value

    def budget(self) -> BudgetSnapshot:
        """Get current budget snapshot."""
        return self._budget.snapshot()

    def record_tokens(self, tokens: int) -> int:
        """Record token usage. Returns new total. Raises ClawbossError if over budget."""
        total = self._budget.record_tokens(tokens)
        self._auto_checkpoint()
        return total

    def record_iteration(self) -> int:
        """Record an iteration of the agent loop. Returns iteration count.
        Raises ClawbossError if max iterations exceeded.
        """
        self._last_activity = time.monotonic()
        try:
            count = self._budget.record_iteration()
            self._audit.record(
                AuditPhase.ITERATION_CHECK,
                AuditOutcome.ALLOWED,
                detail=f"Iteration {count} / {self._policy.max_iterations}",
            )
            self._auto_checkpoint()
            return count
        except ClawbossError as e:
            self._audit.record(
                AuditPhase.ITERATION_CHECK,
                AuditOutcome.DENIED,
                detail=str(e),
            )
            raise

    def circuit_breaker_states(self) -> Dict[str, str]:
        """Get current circuit breaker states for all tracked tools."""
        return {name: cb.state.value for name, cb in self._circuit_breakers.items()}

    def _get_circuit_breaker(self, tool_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a tool."""
        if tool_name not in self._circuit_breakers:
            self._circuit_breakers[tool_name] = CircuitBreaker(
                threshold=self._policy.circuit_breaker_threshold,
                reset_after=self._policy.circuit_breaker_reset,
            )
        return self._circuit_breakers[tool_name]

    def _check_request_timeout(self) -> None:
        """Check if the overall request has timed out."""
        elapsed = time.monotonic() - self._start_time
        if elapsed > self._policy.request_timeout:
            raise ClawbossError.timeout(int(elapsed * 1000))

    def _check_silence(self) -> None:
        """Check the dead man's switch."""
        if self._policy.silence_timeout is None:
            return
        silence = time.monotonic() - self._last_activity
        if silence > self._policy.silence_timeout:
            raise ClawbossError.dead_man_switch(int(silence * 1000))

    def _check_paused(self) -> None:
        """Check if this supervisor is paused."""
        if self._paused and self._session_id:
            raise ClawbossError.agent_paused(self._session_id)

    def _check_confirm(self, tool_name: str, kwargs: Dict[str, Any]) -> Optional[str]:
        """Check if this tool requires confirmation.

        If an approval queue is configured, queues the call and returns an
        approval_id. Otherwise raises ClawbossError.

        Returns:
            approval_id if queued, None if no confirmation needed.
        """
        if tool_name not in self._policy.require_confirm:
            return None
        if self._approval_queue is not None and self._session_id:
            req = self._approval_queue.submit(tool_name, kwargs, self._session_id)
            return req.approval_id
        raise ClawbossError.policy_denied(f"Tool '{tool_name}' requires user confirmation")

    def _check_scopes(self, tool_name: str, kwargs: Dict[str, Any]) -> None:
        """Check if tool arguments satisfy scope rules."""
        now = time.monotonic()
        for scope in self._policy.tool_scopes:
            if scope.tool_name == tool_name:
                # Check argument rules
                error_msg = scope.check_args(kwargs)
                if error_msg:
                    raise ClawbossError.scope_denied(tool_name, error_msg)
                # Check rate limit
                if scope.max_calls_per_minute is not None:
                    times = self._tool_call_times.get(tool_name, [])
                    cutoff = now - 60.0
                    recent = [t for t in times if t > cutoff]
                    if len(recent) >= scope.max_calls_per_minute:
                        raise ClawbossError.rate_limited(tool_name, scope.max_calls_per_minute)
                    recent.append(now)
                    self._tool_call_times[tool_name] = recent

    async def call(
        self,
        tool_name: str,
        fn: Callable[..., Coroutine],
        **kwargs,
    ) -> SupervisedResult:
        """Supervise an async tool call.

        Args:
            tool_name: Name of the tool being called
            fn: Async callable to execute
            **kwargs: Arguments passed to fn

        Returns:
            SupervisedResult with output or error (never raises)
        """
        start = time.monotonic()
        self._last_activity = start

        # Pre-flight checks
        try:
            self._check_paused()
            self._check_request_timeout()
            self._check_silence()
            approval_id = self._check_confirm(tool_name, kwargs)
            if approval_id is not None:
                self._audit.record(
                    AuditPhase.POLICY_CHECK,
                    AuditOutcome.DENIED,
                    target=tool_name,
                    detail=f"Queued for approval: {approval_id}",
                    metadata={"approval_id": approval_id},
                )
                return SupervisedResult(
                    error=ClawbossError.approval_pending(tool_name, approval_id),
                    duration_ms=int((time.monotonic() - start) * 1000),
                    budget=self._budget.snapshot(),
                    tool_name=tool_name,
                )
            self._check_scopes(tool_name, kwargs)
        except ClawbossError as e:
            phase = (
                AuditPhase.SCOPE_CHECK
                if e.kind in ("scope_denied", "rate_limited")
                else AuditPhase.POLICY_CHECK
            )
            self._audit.record(
                phase,
                AuditOutcome.DENIED,
                target=tool_name,
                detail=str(e),
            )
            return SupervisedResult(
                error=e,
                duration_ms=int((time.monotonic() - start) * 1000),
                budget=self._budget.snapshot(),
                tool_name=tool_name,
            )

        # Circuit breaker check
        cb = self._get_circuit_breaker(tool_name)
        try:
            cb.check(tool_name)
        except ClawbossError as e:
            self._audit.record(
                AuditPhase.CIRCUIT_BREAKER,
                AuditOutcome.DENIED,
                target=tool_name,
                detail=str(e),
            )
            return SupervisedResult(
                error=e,
                duration_ms=int((time.monotonic() - start) * 1000),
                budget=self._budget.snapshot(),
                tool_name=tool_name,
            )

        # Execute with timeout
        self._audit.record(
            AuditPhase.TOOL_CALL,
            AuditOutcome.ALLOWED,
            target=tool_name,
            metadata={"args": {k: str(v)[:100] for k, v in kwargs.items()}},
        )

        try:
            output = await asyncio.wait_for(
                fn(**kwargs),
                timeout=self._policy.tool_timeout,
            )
        except asyncio.TimeoutError:
            cb.record_failure()
            error = ClawbossError.timeout(int(self._policy.tool_timeout * 1000))
            self._audit.record(
                AuditPhase.TOOL_CALL,
                AuditOutcome.TIMED_OUT,
                target=tool_name,
                detail=str(error),
            )
            return SupervisedResult(
                error=error,
                duration_ms=int((time.monotonic() - start) * 1000),
                budget=self._budget.snapshot(),
                tool_name=tool_name,
            )
        except Exception as e:
            cb.record_failure()
            error = ClawbossError.tool_error(str(e))
            self._audit.record(
                AuditPhase.TOOL_CALL,
                AuditOutcome.FAILED,
                target=tool_name,
                detail=str(e),
            )
            return SupervisedResult(
                error=error,
                duration_ms=int((time.monotonic() - start) * 1000),
                budget=self._budget.snapshot(),
                tool_name=tool_name,
            )

        # Success
        cb.record_success()
        duration_ms = int((time.monotonic() - start) * 1000)
        self._last_activity = time.monotonic()

        # Record token usage if output includes it
        tokens = 0
        if isinstance(output, dict) and "tokens_used" in output:
            tokens = output["tokens_used"]
        if tokens > 0:
            try:
                self._budget.record_tokens(tokens)
            except ClawbossError as e:
                self._audit.record(
                    AuditPhase.BUDGET_CHECK,
                    AuditOutcome.BUDGET_EXCEEDED,
                    target=tool_name,
                    detail=str(e),
                )
                # Budget exceeded — still return the output (best effort)
                return SupervisedResult(
                    output=output,
                    error=e,
                    succeeded=True,  # tool succeeded, budget is over
                    duration_ms=duration_ms,
                    budget=self._budget.snapshot(),
                    tool_name=tool_name,
                )

        self._audit.record(
            AuditPhase.TOOL_CALL,
            AuditOutcome.INFO,
            target=tool_name,
            detail=f"Completed in {duration_ms}ms",
        )

        self._auto_checkpoint()

        return SupervisedResult(
            output=output,
            succeeded=True,
            duration_ms=duration_ms,
            budget=self._budget.snapshot(),
            tool_name=tool_name,
        )

    async def execute_approved(
        self,
        approval_id: str,
        fn: Callable[..., Coroutine],
    ) -> SupervisedResult:
        """Execute a previously-approved tool call.

        Looks up the approval in the queue, verifies it was approved,
        then executes the tool call bypassing the confirmation check.

        Args:
            approval_id: The approval ID returned by the original call().
            fn: The async callable to execute (same function the agent
                originally tried to call).

        Returns:
            SupervisedResult — same as call().
        """
        if self._approval_queue is None:
            return SupervisedResult(
                error=ClawbossError.policy_denied("No approval queue configured"),
                budget=self._budget.snapshot(),
            )

        from .approval import ApprovalStatus

        req = self._approval_queue.get(approval_id)
        if req is None:
            return SupervisedResult(
                error=ClawbossError.policy_denied(f"Approval {approval_id} not found"),
                budget=self._budget.snapshot(),
            )
        if req.status == ApprovalStatus.DENIED:
            return SupervisedResult(
                error=ClawbossError.approval_denied(req.tool_name, req.deny_reason),
                budget=self._budget.snapshot(),
                tool_name=req.tool_name,
            )
        if req.status != ApprovalStatus.APPROVED:
            return SupervisedResult(
                error=ClawbossError.approval_pending(req.tool_name, approval_id),
                budget=self._budget.snapshot(),
                tool_name=req.tool_name,
            )

        # Approved — execute bypassing _check_confirm
        self._audit.record(
            AuditPhase.POLICY_CHECK,
            AuditOutcome.ALLOWED,
            target=req.tool_name,
            detail=f"Approved (id: {approval_id}, by: {req.resolved_by})",
            metadata={"approval_id": approval_id},
        )

        # Run through the normal execution path but skip confirmation
        # We temporarily remove the tool from require_confirm
        original_confirm = list(self._policy.require_confirm)
        self._policy.require_confirm = [
            t for t in self._policy.require_confirm if t != req.tool_name
        ]
        try:
            result = await self.call(req.tool_name, fn, **req.tool_args)
        finally:
            self._policy.require_confirm = original_confirm

        return result

    def call_sync(
        self,
        tool_name: str,
        fn: Callable,
        **kwargs,
    ) -> SupervisedResult:
        """Supervise a synchronous tool call.

        Wraps the sync function and runs it through the async supervisor.
        For use when you don't have an event loop.
        """

        async def _wrapper(**kw):
            return fn(**kw)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — can't use run()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.call(tool_name, _wrapper, **kwargs))
                    return future.result(timeout=self._policy.tool_timeout + 5)
            else:
                return loop.run_until_complete(self.call(tool_name, _wrapper, **kwargs))
        except RuntimeError:
            return asyncio.run(self.call(tool_name, _wrapper, **kwargs))

    def finish(self) -> BudgetSnapshot:
        """Mark the request as complete. Returns final budget snapshot."""
        snap = self._budget.snapshot()
        elapsed = time.monotonic() - self._start_time
        self._audit.record(
            AuditPhase.REQUEST_END,
            AuditOutcome.INFO,
            detail=f"Request complete: {snap.iterations} iterations, "
            f"{snap.tokens_used} tokens, {int(elapsed * 1000)}ms",
            metadata={
                "tokens_used": snap.tokens_used,
                "iterations": snap.iterations,
                "duration_ms": int(elapsed * 1000),
            },
        )
        return snap

    # ------------------------------------------------------------------
    # Checkpoint support (opt-in via store parameter)
    # ------------------------------------------------------------------

    def to_checkpoint_data(self) -> Dict[str, Any]:
        """Export supervisor state as a dict suitable for a Checkpoint."""
        snap = self._budget.snapshot()
        cb_states = {}
        for name, cb in self._circuit_breakers.items():
            cb_states[name] = {
                "state": cb.state.value,
                "consecutive_failures": cb.consecutive_failures,
            }
        return {
            "iterations": snap.iterations,
            "tokens_used": snap.tokens_used,
            "token_limit": snap.token_limit,
            "iteration_limit": snap.iteration_limit,
            "circuit_breaker_states": cb_states,
        }

    @classmethod
    def restore_from_checkpoint(
        cls,
        checkpoint: Checkpoint,
        audit: Optional[AuditLog] = None,
        store: Optional[StateStore] = None,
        policy_override: Optional[Dict[str, Any]] = None,
        approval_queue: Optional[ApprovalQueue] = None,
    ) -> Supervisor:
        """Rebuild a Supervisor from a checkpoint.

        Restores budget counters and circuit breaker states so the agent
        can continue where it left off.

        Args:
            checkpoint: The checkpoint to restore from.
            audit: Optional audit log for the restored supervisor.
            store: Optional state store for auto-checkpointing.
            policy_override: If provided, use this policy dict instead of the
                           checkpoint's. This is the SECURITY mechanism —
                           SessionManager always passes the original immutable
                           policy here so agents cannot downgrade supervision.
        """
        from .store import SessionStatus

        # SECURITY: prefer the override (original immutable policy) over
        # whatever the checkpoint contains.
        policy_dict = policy_override if policy_override is not None else checkpoint.policy_dict
        policy = Policy.from_dict(policy_dict)
        sv = cls(
            policy,
            audit=audit,
            store=store,
            session_id=checkpoint.session_id,
            agent_id=checkpoint.agent_id,
            approval_queue=approval_queue,
        )
        # Restore budget counters
        if checkpoint.tokens_used > 0:
            sv._budget._tokens_used = checkpoint.tokens_used
        if checkpoint.iterations > 0:
            sv._budget._iterations = checkpoint.iterations
        # Restore circuit breaker states
        for name, cb_data in checkpoint.circuit_breaker_states.items():
            from .circuit_breaker import CircuitState

            cb = sv._get_circuit_breaker(name)
            cb._consecutive_failures = cb_data.get("consecutive_failures", 0)
            state_str = cb_data.get("state", "closed")
            cb._state = CircuitState(state_str)
            if cb._state == CircuitState.OPEN:
                cb._opened_at = time.monotonic()
        # Restore pause state
        sv._paused = checkpoint.status == SessionStatus.PAUSED
        return sv

    def _auto_checkpoint(self) -> None:
        """Save a checkpoint if a store is configured.

        Only updates SYSTEM-CONTROLLED fields (budget counters, circuit breaker
        states, status, timestamp). Does NOT touch payload (agent-writable) or
        policy_dict (immutable). This preserves the trust boundary between
        supervisor-controlled and agent-controlled data.
        """
        if self._store is None or self._session_id is None:
            return
        from .store import Checkpoint, SessionStatus

        # Load existing checkpoint to preserve payload, policy, created_at, audit_log
        existing = self._store.load_checkpoint(self._session_id)

        status = SessionStatus.PAUSED if self._paused else SessionStatus.RUNNING
        data = self.to_checkpoint_data()
        cp = Checkpoint(
            session_id=self._session_id,
            agent_id=self._agent_id or "",
            status=status,
            iterations=data["iterations"],
            tokens_used=data["tokens_used"],
            token_limit=data["token_limit"],
            iteration_limit=data["iteration_limit"],
            circuit_breaker_states=data["circuit_breaker_states"],
            # Preserve immutable fields from the original checkpoint
            policy_dict=existing.policy_dict if existing else self._policy.to_dict(),
            payload=existing.payload if existing else {},
            timestamp=datetime.now(timezone.utc).isoformat(),
            created_at=existing.created_at if existing else "",
            audit_log=existing.audit_log if existing else [],
        )
        try:
            self._store.save_checkpoint(cp)
        except Exception:
            pass  # checkpoint must never crash the request
