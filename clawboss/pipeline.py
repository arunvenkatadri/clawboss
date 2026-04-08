"""Pipeline orchestration — supervised sequential tool chains.

Define a series of steps. Each step is a tool call supervised by Clawboss.
Output from one step feeds into the next. The pipeline stops early if a
step fails, the budget is exceeded, or an approval is pending.

Usage:
    from clawboss import Pipeline, SessionManager, MemoryStore

    store = MemoryStore()
    mgr = SessionManager(store)

    pipeline = Pipeline(mgr, "my-agent", policy_dict={...})

    pipeline.add_step("search", search_fn, query="quantum computing")
    pipeline.add_step("summarize", summarize_fn)  # receives previous output
    pipeline.add_step("write_report", write_fn)

    results = await pipeline.run()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .session import SessionManager
from .supervisor import SupervisedResult


@dataclass
class Step:
    """A single pipeline step."""

    name: str
    tool_name: str
    fn: Callable[..., Coroutine]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    # If True, the output of the previous step is passed as the first kwarg
    chain_input: bool = True
    # Name of the kwarg to pass the previous output as (default: "input")
    input_key: str = "input"


@dataclass
class StepResult:
    """Result of a single pipeline step."""

    name: str
    tool_name: str
    result: SupervisedResult
    step_index: int = 0


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""

    session_id: str
    steps: List[StepResult] = field(default_factory=list)
    completed: bool = False
    stopped_at: Optional[str] = None  # step name where it stopped
    error: Optional[str] = None

    @property
    def final_output(self) -> Any:
        """The output of the last successful step."""
        for step in reversed(self.steps):
            if step.result.succeeded:
                return step.result.output
        return None

    @property
    def total_duration_ms(self) -> int:
        return sum(s.result.duration_ms for s in self.steps)


class Pipeline:
    """Supervised sequential pipeline — chain tool calls with full Clawboss supervision.

    Each step runs through the Supervisor with all policy enforcement
    (timeouts, budgets, circuit breakers, PII redaction, approvals).
    Output flows from one step to the next.

    Args:
        manager: SessionManager to create the session with.
        agent_id: Agent identifier for the session.
        policy_dict: Policy configuration for the session.
        payload: Optional initial payload.
        stateless: If True, session is in-memory only.
    """

    def __init__(
        self,
        manager: SessionManager,
        agent_id: str,
        policy_dict: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        stateless: bool = False,
    ):
        self._manager = manager
        self._agent_id = agent_id
        self._policy_dict = policy_dict
        self._payload = payload
        self._stateless = stateless
        self._steps: List[Step] = []

    def add_step(
        self,
        tool_name: str,
        fn: Callable[..., Coroutine],
        name: Optional[str] = None,
        chain_input: bool = True,
        input_key: str = "input",
        **kwargs: Any,
    ) -> "Pipeline":
        """Add a step to the pipeline.

        Args:
            tool_name: Name of the tool (for supervision and audit).
            fn: Async callable to execute.
            name: Human-readable step name (defaults to tool_name).
            chain_input: If True, pass previous step's output as a kwarg.
            input_key: Name of the kwarg for the chained input.
            **kwargs: Additional arguments passed to fn.

        Returns:
            self, for chaining: pipeline.add_step(...).add_step(...)
        """
        self._steps.append(
            Step(
                name=name or f"{len(self._steps) + 1}_{tool_name}",
                tool_name=tool_name,
                fn=fn,
                kwargs=kwargs,
                chain_input=chain_input,
                input_key=input_key,
            )
        )
        return self

    async def run(self) -> PipelineResult:
        """Execute all steps in sequence.

        Stops early if:
        - A step fails (error, timeout, circuit breaker)
        - Budget is exceeded
        - An approval is pending (returns so caller can handle it)

        Returns:
            PipelineResult with all step results and the final output.
        """
        sid = self._manager.start(
            self._agent_id,
            self._policy_dict,
            self._payload,
            stateless=self._stateless,
        )
        sv = self._manager.get_supervisor(sid)
        if sv is None:
            return PipelineResult(
                session_id=sid,
                error="Failed to create supervisor",
            )

        pipeline_result = PipelineResult(session_id=sid)
        previous_output: Any = None

        for i, step in enumerate(self._steps):
            sv.record_iteration()

            # Build kwargs — chain previous output if configured
            call_kwargs = dict(step.kwargs)
            if step.chain_input and previous_output is not None:
                call_kwargs[step.input_key] = previous_output

            # Execute the step through the supervisor
            result = await sv.call(step.tool_name, step.fn, **call_kwargs)

            step_result = StepResult(
                name=step.name,
                tool_name=step.tool_name,
                result=result,
                step_index=i,
            )
            pipeline_result.steps.append(step_result)

            # Update payload with progress
            self._manager.update_payload(
                sid,
                {
                    "pipeline_step": i,
                    "pipeline_step_name": step.name,
                    "pipeline_total_steps": len(self._steps),
                },
            )

            if not result.succeeded:
                pipeline_result.stopped_at = step.name
                error_kind = result.error.kind if result.error else "unknown"
                pipeline_result.error = f"Step '{step.name}' failed: {error_kind}"

                # If approval is pending, don't stop the session — caller handles it
                if error_kind == "approval_pending":
                    err = result.error
                    approval_id = err.details.get("approval_id", "") if err else ""
                    pipeline_result.error = f"Step '{step.name}' awaiting approval: {approval_id}"
                    return pipeline_result

                # For other failures, stop the session
                self._manager.stop(sid)
                return pipeline_result

            previous_output = result.output

        # All steps completed
        pipeline_result.completed = True
        self._manager.stop(sid)
        return pipeline_result
