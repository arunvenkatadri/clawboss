"""Reflection loop — structured think → act → observe → reflect cycles.

Long-duration agents don't just loop `run()` forever. They think, act,
observe what happened, reflect on whether it worked, and adjust.

Each cycle is four LLM-backed phases:
1. **Think** — what should I do next, given the goal and what's happened?
2. **Act** — actually run a tool (fully supervised by Clawboss).
3. **Observe** — what came back, and how does it compare to what I expected?
4. **Reflect** — did this advance the goal? Should I change approach?

Every phase is a supervised call, every decision is recorded, and the
reflection output feeds into the next think step. This is what
distinguishes "agent runs 72 hours doing useful work" from "agent runs
in circles for 72 hours."

Bring-your-own-LLM — same pattern as SkillBuilder, PipelineBuilder.

Usage:
    from clawboss import ReflectionLoop, SessionManager, MemoryStore

    store = MemoryStore()
    mgr = SessionManager(store)

    loop = ReflectionLoop(
        manager=mgr,
        agent_id="research-agent",
        goal="Write a report on quantum computing in 2026",
        llm=my_llm,
        tools={"search": search_fn, "write": write_fn},
    )

    result = await loop.run(max_cycles=20)
    print(result.final_answer)
    print(result.cycles_used)
    print(result.reflections)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Coroutine, Dict, List, Optional

from .errors import ClawbossError
from .session import SessionManager

LLMFn = Callable[[str], Awaitable[str]]
ToolFn = Callable[..., Coroutine[Any, Any, Any]]


@dataclass
class ReflectionCycle:
    """One think → act → observe → reflect cycle."""

    cycle_number: int
    thought: str = ""
    tool_called: str = ""
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_output: Any = None
    tool_succeeded: bool = True
    observation: str = ""
    reflection: str = ""
    goal_progress: float = 0.0  # LLM-estimated, 0.0 to 1.0
    should_stop: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_number": self.cycle_number,
            "thought": self.thought,
            "tool_called": self.tool_called,
            "tool_args": self.tool_args,
            "tool_succeeded": self.tool_succeeded,
            "observation": self.observation,
            "reflection": self.reflection,
            "goal_progress": self.goal_progress,
            "should_stop": self.should_stop,
            "error": self.error,
        }


@dataclass
class ReflectionResult:
    """Final result of a reflection loop run."""

    session_id: str
    goal: str
    cycles: List[ReflectionCycle] = field(default_factory=list)
    completed: bool = False
    final_answer: str = ""
    stopped_reason: str = ""

    @property
    def cycles_used(self) -> int:
        return len(self.cycles)

    @property
    def reflections(self) -> List[str]:
        return [c.reflection for c in self.cycles if c.reflection]

    @property
    def total_tool_calls(self) -> int:
        return sum(1 for c in self.cycles if c.tool_called)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


_THINK_PROMPT = """\
You are an AI agent working toward a goal.

Goal: {goal}

Available tools: {tools}

History of what you've done so far (most recent last):
{history}

What should you do next to make progress? Respond with ONLY JSON:

{{
  "thought": "your reasoning in 1-2 sentences",
  "tool": "tool_name_or_null_if_done",
  "args": {{"...": "..."}},
  "done": false
}}

If you believe the goal is achieved, set "done": true and "tool": null. \
Include a "final_answer" field with your completed work.
"""


_OBSERVE_PROMPT = """\
You just called a tool. Observe the result.

Goal: {goal}
Tool: {tool}
Args: {args}
Result: {result}

What does this result tell you? Respond with ONLY a 1-2 sentence observation.
"""


_REFLECT_PROMPT = """\
You are reflecting on the last action you took.

Goal: {goal}
Last thought: {thought}
Last action: called "{tool}" with {args}
Result: {result}
Your observation: {observation}

Reflect on this cycle:
- Did the action advance the goal?
- Was it the right choice?
- What should change for the next cycle?
- Estimate overall progress toward the goal (0.0 to 1.0).
- Should we stop? (only stop if goal is achieved or clearly unreachable)

Respond with ONLY JSON:
{{
  "reflection": "1-3 sentences",
  "goal_progress": 0.0-1.0,
  "should_stop": false,
  "reason": "why you think you should stop, if applicable"
}}
"""


# ---------------------------------------------------------------------------
# ReflectionLoop
# ---------------------------------------------------------------------------


class ReflectionLoop:
    """Structured think → act → observe → reflect loop.

    Built on top of SessionManager + Supervisor — every cycle runs
    through full Clawboss supervision (budgets, policies, guardrails,
    audit, observability).

    Args:
        manager: SessionManager for the underlying session.
        agent_id: Agent identifier.
        goal: The goal the agent is working toward (plain English).
        llm: Async callable for all four phases (think, observe, reflect,
             and the final summarization). Same pattern as SkillBuilder.
        tools: Dict mapping tool name → async callable.
        policy_dict: Policy for the underlying session.
        max_context_history: How many past cycles to show in think prompts.
    """

    def __init__(
        self,
        manager: SessionManager,
        agent_id: str,
        goal: str,
        llm: LLMFn,
        tools: Dict[str, ToolFn],
        policy_dict: Optional[Dict[str, Any]] = None,
        max_context_history: int = 10,
    ):
        self._manager = manager
        self._agent_id = agent_id
        self._goal = goal
        self._llm = llm
        self._tools = tools
        self._policy_dict = policy_dict or {"max_iterations": 100, "tool_timeout": 60}
        self._max_history = max_context_history

    async def run(self, max_cycles: int = 20) -> ReflectionResult:
        """Run the reflection loop for up to max_cycles."""
        sid = self._manager.start(self._agent_id, self._policy_dict)
        sv = self._manager.get_supervisor(sid)
        if sv is None:
            return ReflectionResult(
                session_id=sid,
                goal=self._goal,
                stopped_reason="Failed to create supervisor",
            )

        result = ReflectionResult(session_id=sid, goal=self._goal)
        final_answer = ""

        for cycle_num in range(1, max_cycles + 1):
            cycle = ReflectionCycle(cycle_number=cycle_num)

            # -- 1. THINK --
            try:
                sv.record_iteration()
            except ClawbossError as e:
                cycle.error = f"Iteration limit reached: {e}"
                result.cycles.append(cycle)
                result.stopped_reason = "iteration_limit"
                break

            history_str = self._format_history(result.cycles)
            tools_str = ", ".join(self._tools.keys())
            think_prompt = (
                _THINK_PROMPT.replace("{goal}", self._goal)
                .replace("{tools}", tools_str)
                .replace("{history}", history_str)
            )
            try:
                raw = await self._llm(think_prompt)
                think = self._parse_json(raw)
            except Exception as e:
                cycle.error = f"Think phase failed: {e}"
                result.cycles.append(cycle)
                result.stopped_reason = "think_error"
                break

            cycle.thought = think.get("thought", "")
            tool_name = think.get("tool")

            # Check for completion
            if think.get("done") or tool_name is None:
                final_answer = think.get("final_answer", "")
                cycle.should_stop = True
                cycle.reflection = "Agent determined goal was achieved."
                cycle.goal_progress = 1.0
                result.cycles.append(cycle)
                result.completed = True
                result.stopped_reason = "goal_achieved"
                break

            if tool_name not in self._tools:
                cycle.error = f"Unknown tool: {tool_name}"
                result.cycles.append(cycle)
                result.stopped_reason = "unknown_tool"
                break

            # -- 2. ACT --
            tool_args = think.get("args", {}) or {}
            cycle.tool_called = tool_name
            cycle.tool_args = tool_args

            tool_result = await sv.call(tool_name, self._tools[tool_name], **tool_args)
            cycle.tool_succeeded = tool_result.succeeded
            cycle.tool_output = tool_result.output

            if not tool_result.succeeded:
                cycle.error = (
                    tool_result.error.user_message() if tool_result.error else "unknown error"
                )
                # Don't stop on tool error — let the agent reflect and try again
                cycle.observation = f"Tool call failed: {cycle.error}"
                cycle.reflection = "The tool call failed. The next cycle should adapt."
                result.cycles.append(cycle)
                continue

            # -- 3. OBSERVE --
            observe_prompt = (
                _OBSERVE_PROMPT.replace("{goal}", self._goal)
                .replace("{tool}", tool_name)
                .replace("{args}", json.dumps(tool_args, default=str)[:500])
                .replace("{result}", json.dumps(tool_result.output, default=str)[:2000])
            )
            try:
                cycle.observation = (await self._llm(observe_prompt)).strip()
            except Exception as e:
                cycle.observation = f"(observation failed: {e})"

            # -- 4. REFLECT --
            reflect_prompt = (
                _REFLECT_PROMPT.replace("{goal}", self._goal)
                .replace("{thought}", cycle.thought)
                .replace("{tool}", tool_name)
                .replace("{args}", json.dumps(tool_args, default=str)[:500])
                .replace("{result}", json.dumps(tool_result.output, default=str)[:1000])
                .replace("{observation}", cycle.observation)
            )
            try:
                raw = await self._llm(reflect_prompt)
                reflect = self._parse_json(raw)
                cycle.reflection = reflect.get("reflection", "")
                cycle.goal_progress = float(reflect.get("goal_progress", 0.0))
                cycle.should_stop = bool(reflect.get("should_stop", False))
                stop_reason = reflect.get("reason", "")
            except Exception as e:
                cycle.reflection = f"(reflection failed: {e})"
                stop_reason = ""

            result.cycles.append(cycle)

            if cycle.should_stop:
                result.stopped_reason = stop_reason or "agent_requested_stop"
                break
        else:
            # max_cycles exhausted
            result.stopped_reason = "max_cycles_reached"

        result.final_answer = final_answer
        self._manager.stop(sid)
        return result

    def _format_history(self, cycles: List[ReflectionCycle]) -> str:
        """Format recent cycles as a compact history string for the think prompt."""
        if not cycles:
            return "(no previous cycles)"
        recent = cycles[-self._max_history :]
        lines = []
        for c in recent:
            line = f'Cycle {c.cycle_number}: thought="{c.thought[:80]}"'
            if c.tool_called:
                line += f" tool={c.tool_called}"
            if c.reflection:
                line += f' reflection="{c.reflection[:80]}"'
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON from an LLM response, stripping markdown fences."""
        t = text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t[3:]
            if t.endswith("```"):
                t = t[:-3]
            t = t.strip()
            if t.startswith("json\n"):
                t = t[5:]
        try:
            result: Dict[str, Any] = json.loads(t)
            return result
        except (ValueError, json.JSONDecodeError):
            return {}
