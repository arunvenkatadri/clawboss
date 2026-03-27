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

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, Optional

from .audit import AuditLog, AuditOutcome, AuditPhase
from .budget import BudgetSnapshot, BudgetTracker
from .circuit_breaker import CircuitBreaker
from .errors import ClawbossError
from .policy import Policy


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
    ):
        self._policy = policy
        self._budget = BudgetTracker.from_policy(policy)
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._audit = audit or AuditLog.noop()
        self._start_time = time.monotonic()
        self._last_activity = time.monotonic()

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

    def budget(self) -> BudgetSnapshot:
        """Get current budget snapshot."""
        return self._budget.snapshot()

    def record_tokens(self, tokens: int) -> int:
        """Record token usage. Returns new total. Raises ClawbossError if over budget."""
        return self._budget.record_tokens(tokens)

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

    def _check_confirm(self, tool_name: str) -> None:
        """Check if this tool requires confirmation."""
        if tool_name in self._policy.require_confirm:
            raise ClawbossError.policy_denied(f"Tool '{tool_name}' requires user confirmation")

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
            self._check_request_timeout()
            self._check_silence()
            self._check_confirm(tool_name)
        except ClawbossError as e:
            self._audit.record(
                AuditPhase.POLICY_CHECK,
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

        return SupervisedResult(
            output=output,
            succeeded=True,
            duration_ms=duration_ms,
            budget=self._budget.snapshot(),
            tool_name=tool_name,
        )

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
