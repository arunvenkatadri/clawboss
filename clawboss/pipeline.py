"""Pipeline orchestration — supervised sequential and conditional tool chains.

Level 1: Sequential steps with output chaining.
Level 2: Conditional branching — if/else/threshold routing.

Usage:
    from clawboss import Pipeline, SessionManager, MemoryStore

    store = MemoryStore()
    mgr = SessionManager(store)

    # Sequential
    pipeline = Pipeline(mgr, "my-agent", policy_dict={...})
    pipeline.add_step("search", search_fn, query="quantum computing")
    pipeline.add_step("summarize", summarize_fn)
    results = await pipeline.run()

    # Conditional
    pipeline = Pipeline(mgr, "analyst", policy_dict={...})
    pipeline.add_step("check_metric", sql.query, sql="SELECT count(*) as cnt FROM alerts")
    pipeline.add_condition(
        lambda output: output["rows"][0]["cnt"] > 10,
        then_step=("escalate", escalate_fn),
        else_step=("log_ok", log_fn),
    )
    results = await pipeline.run()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, Union

from .session import SessionManager
from .supervisor import SupervisedResult


@dataclass
class Step:
    """A single pipeline step."""

    name: str
    tool_name: str
    fn: Callable[..., Coroutine]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    chain_input: bool = True
    input_key: str = "input"


@dataclass
class Condition:
    """A conditional branch in the pipeline."""

    name: str
    predicate: Callable[[Any], bool]
    then_step: Step
    else_step: Optional[Step] = None  # None = skip (do nothing)


# A pipeline node is either a Step or a Condition
PipelineNode = Union[Step, Condition]


@dataclass
class StepResult:
    """Result of a single pipeline step."""

    name: str
    tool_name: str
    result: SupervisedResult
    step_index: int = 0
    branch: str = ""  # "then", "else", or "" for non-conditional


@dataclass
class PipelineResult:
    """Result of a full pipeline run."""

    session_id: str
    steps: List[StepResult] = field(default_factory=list)
    completed: bool = False
    stopped_at: Optional[str] = None
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
    """Supervised pipeline with sequential steps and conditional branching.

    Every step and branch runs through the Supervisor with full policy enforcement
    (timeouts, budgets, circuit breakers, PII redaction, approvals).

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
        self._nodes: List[PipelineNode] = []
        self._reuse_session_id: Optional[str] = None

    def with_context(self, session_id: str) -> "Pipeline":
        """Attach this pipeline to an existing session.

        Instead of starting a new session on run(), the pipeline reuses
        the given session. The previous payload is loaded as context and
        the new payload is merged into it after each run.

        Use this for stateful agents that accumulate history — e.g.,
        a streaming agent that sees the last N messages plus the current one.

        Args:
            session_id: Existing session to reuse.

        Returns:
            self, for chaining.
        """
        self._reuse_session_id = session_id
        return self

    def add_step(
        self,
        tool_name: str,
        fn: Callable[..., Coroutine],
        name: Optional[str] = None,
        chain_input: bool = True,
        input_key: str = "input",
        **kwargs: Any,
    ) -> "Pipeline":
        """Add a sequential step to the pipeline.

        Args:
            tool_name: Name of the tool (for supervision and audit).
            fn: Async callable to execute.
            name: Human-readable step name (defaults to tool_name).
            chain_input: If True, pass previous step's output as a kwarg.
            input_key: Name of the kwarg for the chained input.
            **kwargs: Additional arguments passed to fn.

        Returns:
            self, for chaining.
        """
        self._nodes.append(
            Step(
                name=name or f"{len(self._nodes) + 1}_{tool_name}",
                tool_name=tool_name,
                fn=fn,
                kwargs=kwargs,
                chain_input=chain_input,
                input_key=input_key,
            )
        )
        return self

    def add_condition(
        self,
        predicate: Callable[[Any], bool],
        then_step: Tuple[str, Callable[..., Coroutine]],
        else_step: Optional[Tuple[str, Callable[..., Coroutine]]] = None,
        name: Optional[str] = None,
        then_kwargs: Optional[Dict[str, Any]] = None,
        else_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """Add a conditional branch.

        Evaluates ``predicate(previous_output)`` and routes to the
        appropriate step. Both branches are fully supervised.

        Args:
            predicate: Function that takes the previous step's output and
                      returns True (take then_step) or False (take else_step).
            then_step: Tuple of (tool_name, fn) to execute if predicate is True.
            else_step: Tuple of (tool_name, fn) if predicate is False.
                      If None, the condition is skipped when False.
            name: Human-readable name for the condition.
            then_kwargs: Extra kwargs for the then_step.
            else_kwargs: Extra kwargs for the else_step.

        Returns:
            self, for chaining.
        """
        then = Step(
            name=f"{then_step[0]}_then",
            tool_name=then_step[0],
            fn=then_step[1],
            kwargs=then_kwargs or {},
            chain_input=True,
        )
        else_s = None
        if else_step is not None:
            else_s = Step(
                name=f"{else_step[0]}_else",
                tool_name=else_step[0],
                fn=else_step[1],
                kwargs=else_kwargs or {},
                chain_input=True,
            )
        self._nodes.append(
            Condition(
                name=name or f"{len(self._nodes) + 1}_condition",
                predicate=predicate,
                then_step=then,
                else_step=else_s,
            )
        )
        return self

    def add_threshold(
        self,
        key: str,
        threshold: float,
        above_step: Tuple[str, Callable[..., Coroutine]],
        below_step: Optional[Tuple[str, Callable[..., Coroutine]]] = None,
        name: Optional[str] = None,
        above_kwargs: Optional[Dict[str, Any]] = None,
        below_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """Add a threshold-based branch — sugar for add_condition.

        Extracts a numeric value from the previous step's output and
        routes based on whether it's above or below the threshold.

        Args:
            key: Dot-notation key to extract from previous output.
                 e.g., "rows.0.cnt" extracts output["rows"][0]["cnt"]
            threshold: The threshold value.
            above_step: (tool_name, fn) if value >= threshold.
            below_step: (tool_name, fn) if value < threshold. None = skip.
            name: Name for the condition.
            above_kwargs: Extra kwargs for above_step.
            below_kwargs: Extra kwargs for below_step.

        Returns:
            self, for chaining.
        """

        def _extract(output: Any) -> bool:
            val = output
            for part in key.split("."):
                if isinstance(val, dict):
                    val = val.get(part)
                elif isinstance(val, (list, tuple)):
                    try:
                        val = val[int(part)]
                    except (IndexError, ValueError):
                        return False
                else:
                    return False
            try:
                return float(val) >= threshold  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return False

        return self.add_condition(
            predicate=_extract,
            then_step=above_step,
            else_step=below_step,
            name=name or f"threshold_{key}_{threshold}",
            then_kwargs=above_kwargs,
            else_kwargs=below_kwargs,
        )

    def add_llm_decision(
        self,
        llm: Callable[[str], Coroutine[Any, Any, str]],
        prompt_template: str,
        name: str = "llm_decision",
        output_schema: Optional[Dict[str, Any]] = None,
        include_context: bool = False,
    ) -> "Pipeline":
        """Add an LLM-backed decision step.

        The LLM reads the previous step's output (and optionally the
        session payload for context) and returns a structured decision
        as JSON. The pipeline then routes on the decision.

        Example:
            pipeline.add_step("fetch", fetch_fn)
            pipeline.add_llm_decision(
                my_llm,
                prompt_template=\"\"\"
                You are a fraud detector. Given this transaction:
                {input}

                Historical context: {context}

                Return JSON: {{"action": "block|approve|escalate", "reason": "..."}}
                \"\"\",
                include_context=True,
            )
            pipeline.add_condition(
                lambda out: out["action"] == "block",
                then_step=("block", block_fn),
                else_step=("approve", approve_fn),
            )

        Args:
            llm: Async callable — your LLM (same pattern as SkillBuilder).
            prompt_template: String with {input} and optionally {context} placeholders.
            name: Name for this step.
            output_schema: Optional JSON schema hint for the LLM.
            include_context: If True, pass the session payload as {context}.

        Returns:
            self, for chaining.
        """

        async def llm_step(input: Any = None, _context: Any = None) -> Dict[str, Any]:
            """Call the LLM with the templated prompt and parse JSON output."""
            import json as _json

            input_str = _json.dumps(input, default=str) if input is not None else ""
            context_str = _json.dumps(_context, default=str) if _context is not None else "{}"

            prompt = prompt_template.replace("{input}", input_str).replace("{context}", context_str)
            if output_schema:
                schema_json = _json.dumps(output_schema)
                prompt += f"\n\nRespond with ONLY JSON matching: {schema_json}"

            raw = await llm(prompt)

            # Strip markdown fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                if text.startswith("json\n"):
                    text = text[5:]

            try:
                result: Dict[str, Any] = _json.loads(text)
                return result
            except (ValueError, _json.JSONDecodeError):
                return {"raw": raw, "error": "Failed to parse JSON"}

        # Mark the step so Pipeline.run() knows to inject context
        step = Step(
            name=name,
            tool_name=name,
            fn=llm_step,
            kwargs={},
            chain_input=True,
            input_key="input",
        )
        step.__dict__["_include_context"] = include_context
        self._nodes.append(step)
        return self

    async def run(
        self,
        initial_input: Optional[Any] = None,
    ) -> PipelineResult:
        """Execute the pipeline — steps and conditions in sequence.

        Returns:
            PipelineResult with all step results and the final output.
        """
        # Reuse an existing session if with_context() was called
        if self._reuse_session_id:
            sid = self._reuse_session_id
            sv = self._manager.get_supervisor(sid)
            if sv is None:
                # Session not in memory — try to resume it
                try:
                    sv = self._manager.resume(sid)
                except Exception as e:
                    return PipelineResult(
                        session_id=sid, error=f"Failed to resume context session: {e}"
                    )
        else:
            sid = self._manager.start(
                self._agent_id,
                self._policy_dict,
                self._payload,
                stateless=self._stateless,
            )
            sv = self._manager.get_supervisor(sid)
            if sv is None:
                return PipelineResult(session_id=sid, error="Failed to create supervisor")

        pipeline_result = PipelineResult(session_id=sid)
        # Initial input for the first step (from stream payload or explicit param)
        previous_output: Any = initial_input

        # Load session context if any step needs it
        session_context = None
        needs_context = any(
            isinstance(n, Step) and n.__dict__.get("_include_context", False) for n in self._nodes
        )
        if needs_context:
            cp = self._manager.status(sid)
            session_context = cp.payload if cp else {}

        for i, node in enumerate(self._nodes):
            sv.record_iteration()

            if isinstance(node, Step):
                # Inject context for LLM decision steps
                if node.__dict__.get("_include_context", False):
                    node.kwargs = dict(node.kwargs)
                    node.kwargs["_context"] = session_context

                result, step_result = await self._run_step(sv, node, previous_output, i)
                pipeline_result.steps.append(step_result)

                if not result.succeeded:
                    pipeline_result.stopped_at = node.name
                    error_kind = result.error.kind if result.error else "unknown"
                    if error_kind == "approval_pending":
                        err = result.error
                        aid = err.details.get("approval_id", "") if err else ""
                        pipeline_result.error = f"Step '{node.name}' awaiting approval: {aid}"
                    else:
                        pipeline_result.error = f"Step '{node.name}' failed: {error_kind}"
                        self._manager.stop(sid)
                    return pipeline_result

                previous_output = result.output

            elif isinstance(node, Condition):
                # Evaluate the predicate
                try:
                    take_then = node.predicate(previous_output)
                except Exception as e:
                    pipeline_result.stopped_at = node.name
                    pipeline_result.error = f"Condition '{node.name}' predicate failed: {e}"
                    self._manager.stop(sid)
                    return pipeline_result

                chosen = node.then_step if take_then else node.else_step
                branch = "then" if take_then else "else"

                if chosen is None:
                    # else_step is None — skip, keep previous output
                    continue

                result, step_result = await self._run_step(
                    sv, chosen, previous_output, i, branch=branch
                )
                pipeline_result.steps.append(step_result)

                if not result.succeeded:
                    pipeline_result.stopped_at = chosen.name
                    error_kind = result.error.kind if result.error else "unknown"
                    pipeline_result.error = f"Step '{chosen.name}' ({branch}) failed: {error_kind}"
                    self._manager.stop(sid)
                    return pipeline_result

                previous_output = result.output

            # Update payload with progress (preserve existing payload)
            existing_cp = self._manager.status(sid)
            existing_payload = existing_cp.payload if existing_cp else {}
            merged = dict(existing_payload)
            merged["pipeline_step"] = i
            merged["pipeline_total_steps"] = len(self._nodes)
            self._manager.update_payload(sid, merged)

        pipeline_result.completed = True
        # If running with context, keep the session alive for future runs
        if not self._reuse_session_id:
            self._manager.stop(sid)
        else:
            # Accumulate the output into the session payload for next run
            cp = self._manager.status(sid)
            if cp:
                merged_payload = dict(cp.payload)
                history = merged_payload.get("history", [])
                if not isinstance(history, list):
                    history = []
                history.append(
                    {
                        "input": initial_input,
                        "output": pipeline_result.final_output,
                    }
                )
                # Keep only last 20 runs to bound payload size
                merged_payload["history"] = history[-20:]
                self._manager.update_payload(sid, merged_payload)
        return pipeline_result

    async def _run_step(
        self,
        sv: Any,
        step: Step,
        previous_output: Any,
        index: int,
        branch: str = "",
    ) -> Tuple[SupervisedResult, StepResult]:
        """Execute a single step through the supervisor."""
        call_kwargs = dict(step.kwargs)
        if step.chain_input and previous_output is not None:
            call_kwargs[step.input_key] = previous_output

        result = await sv.call(step.tool_name, step.fn, **call_kwargs)

        step_result = StepResult(
            name=step.name,
            tool_name=step.tool_name,
            result=result,
            step_index=index,
            branch=branch,
        )
        return result, step_result
