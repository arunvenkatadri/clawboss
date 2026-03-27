"""Context compression — supervision-anchored context window management.

Supervised agents can compress more aggressively than unsupervised ones because
safety-critical state (policies, budgets, circuit breakers) is enforced by the
supervisor, not by the LLM's memory. This module reconstructs anchored state
fresh each turn and compresses older history safely.

Usage:
    from clawboss import Supervisor, Policy, ContextWindow

    supervisor = Supervisor(Policy(max_iterations=5, token_budget=10000))
    ctx = ContextWindow(supervisor, max_recent_turns=10)

    ctx.add_turn("user", "Search for quantum computing")
    ctx.add_turn("assistant", "I'll search for that.", tool_calls=[...])

    prompt = ctx.to_prompt()  # anchored state + compressed history + recent turns
"""

import asyncio
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .budget import BudgetSnapshot
from .supervisor import Supervisor


@dataclass
class Turn:
    """A single turn in the conversation history."""

    role: str  # "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    token_estimate: int = 0
    skill_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.token_estimate == 0:
            self.token_estimate = len(self.content) // 4
            if self.tool_calls:
                for tc in self.tool_calls:
                    self.token_estimate += len(str(tc)) // 4

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != 0}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Turn":
        return cls(
            role=d["role"],
            content=d.get("content", ""),
            tool_calls=d.get("tool_calls"),
            timestamp=d.get("timestamp", ""),
            token_estimate=d.get("token_estimate", 0),
            skill_name=d.get("skill_name"),
        )


@dataclass
class AnchoredState:
    """Safety-critical state reconstructed fresh from the supervisor every turn.

    Never compressed — always accurate because it's read from the supervisor's
    live state, not from LLM memory.
    """

    budget_snapshot: BudgetSnapshot
    circuit_states: Dict[str, str]
    policy_summary: Dict[str, Any]
    confirmed_tools: List[str]
    active_skill: Optional[str] = None

    def to_prompt(self) -> str:
        lines = ["[SUPERVISION STATE]"]
        b = self.budget_snapshot
        budget_str = f"{b.tokens_used}"
        if b.token_limit is not None:
            budget_str += f"/{b.token_limit}"
        else:
            budget_str += " (no limit)"
        lines.append(
            f"Budget: {budget_str} tokens used, {b.iterations}/{b.iteration_limit} iterations"
        )

        if self.circuit_states:
            states = ", ".join(f"{name}={state}" for name, state in self.circuit_states.items())
            lines.append(f"Circuit breakers: {states}")

        ps = self.policy_summary
        parts = []
        if "max_iterations" in ps:
            parts.append(f"max_iterations={ps['max_iterations']}")
        if "tool_timeout" in ps:
            parts.append(f"tool_timeout={ps['tool_timeout']}s")
        if "token_budget" in ps and ps["token_budget"] is not None:
            parts.append(f"token_budget={ps['token_budget']}")
        if parts:
            lines.append(f"Policy: {', '.join(parts)}")

        if self.confirmed_tools:
            lines.append(f"Confirmation required: {', '.join(self.confirmed_tools)}")

        if self.active_skill:
            lines.append(f"Active skill: {self.active_skill}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "budget_snapshot": asdict(self.budget_snapshot),
            "circuit_states": self.circuit_states,
            "policy_summary": self.policy_summary,
            "confirmed_tools": self.confirmed_tools,
        }
        if self.active_skill:
            d["active_skill"] = self.active_skill
        return d


@dataclass
class CompressedHistory:
    """Lossy but safe summary of older conversation turns."""

    summary: str
    turn_count: int
    tool_call_summaries: List[Dict[str, Any]]
    original_turn_range: str = ""

    def to_prompt(self) -> str:
        lines = [f"[CONVERSATION HISTORY (turns {self.original_turn_range}, compressed)]"]
        lines.append(self.summary)
        return "\n".join(lines)

    def token_estimate(self) -> int:
        return len(self.to_prompt()) // 4


@dataclass
class CompressedContext:
    """The full compressed context ready for injection into LLM prompts."""

    anchored: AnchoredState
    history: Optional[CompressedHistory]
    recent_turns: List[Turn]

    def to_prompt(self) -> str:
        parts = [self.anchored.to_prompt()]
        if self.history:
            parts.append(self.history.to_prompt())
        if self.recent_turns:
            lines = ["[RECENT CONVERSATION]"]
            for turn in self.recent_turns:
                lines.append(f"[{turn.role}]: {turn.content}")
                if turn.tool_calls:
                    for tc in turn.tool_calls:
                        name = tc.get("tool_name", "unknown")
                        params = tc.get("params", {})
                        result = tc.get("result_summary", "")
                        param_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
                        line = f"  -> {name}({param_str})"
                        if result:
                            line += f" = {result}"
                        lines.append(line)
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def token_estimate(self) -> int:
        return len(self.to_prompt()) // 4


class ContextWindow:
    """Supervision-anchored context window with automatic compression.

    Maintains a sliding window of conversation turns. Older turns are compressed
    using tool call summaries and optional LLM summarization. Safety-critical
    state is always reconstructed fresh from the supervisor.

    Usage:
        ctx = ContextWindow(supervisor, max_recent_turns=10)
        ctx.add_turn("user", "Search for quantum computing")
        prompt = ctx.to_prompt()
    """

    def __init__(
        self,
        supervisor: Supervisor,
        max_recent_turns: int = 10,
        max_summary_tokens: int = 500,
        summarizer: Optional[Callable[[str], Coroutine[Any, Any, str]]] = None,
        skill_name: Optional[str] = None,
    ):
        self._supervisor = supervisor
        self._max_recent_turns = max_recent_turns
        self._max_summary_tokens = max_summary_tokens
        self._summarizer = summarizer
        self._skill_name = skill_name
        self._turns: List[Turn] = []
        self._compressed: Optional[CompressedHistory] = None
        self._lock = threading.Lock()

    def add_turn(
        self,
        role: str,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        skill_name: Optional[str] = None,
    ) -> None:
        """Add a turn to the conversation history."""
        turn = Turn(
            role=role,
            content=content,
            tool_calls=tool_calls,
            skill_name=skill_name or self._skill_name,
        )
        with self._lock:
            self._turns.append(turn)

    def get_anchored_state(self) -> AnchoredState:
        """Reconstruct safety-critical state fresh from the supervisor."""
        budget = self._supervisor.budget()
        circuits = self._supervisor.circuit_breaker_states()
        policy = self._supervisor.policy
        return AnchoredState(
            budget_snapshot=budget,
            circuit_states=circuits,
            policy_summary={
                "max_iterations": policy.max_iterations,
                "tool_timeout": policy.tool_timeout,
                "token_budget": policy.token_budget,
                "require_confirm": list(policy.require_confirm),
            },
            confirmed_tools=list(policy.require_confirm),
            active_skill=self._skill_name,
        )

    def _compress_turns(self, turns: List[Turn]) -> CompressedHistory:
        """Compress turns using audit-based extraction (no LLM needed)."""
        groups: List[Dict[str, Any]] = []
        current_skill: Optional[str] = None
        current_group: List[Turn] = []

        for turn in turns:
            skill = turn.skill_name or "general"
            if skill != current_skill and current_group:
                groups.append({"skill": current_skill, "turns": current_group})
                current_group = []
            current_skill = skill
            current_group.append(turn)
        if current_group:
            groups.append({"skill": current_skill, "turns": current_group})

        summary_parts = []
        all_tool_summaries: List[Dict[str, Any]] = []

        for group in groups:
            skill = group["skill"]
            group_turns: List[Turn] = group["turns"]
            tool_counts: Dict[str, int] = {}
            tool_params: Dict[str, str] = {}
            user_snippets: List[str] = []

            for turn in group_turns:
                if turn.tool_calls:
                    for tc in turn.tool_calls:
                        name = tc.get("tool_name", "unknown")
                        tool_counts[name] = tool_counts.get(name, 0) + 1
                        if name not in tool_params:
                            params = tc.get("params", {})
                            tool_params[name] = str(params)[:80]
                        all_tool_summaries.append(tc)
                elif turn.role == "user":
                    user_snippets.append(turn.content[:100])

            parts = []
            if tool_counts:
                tool_strs = []
                for name, count in tool_counts.items():
                    s = f"{name}({tool_params.get(name, '')})"
                    if count > 1:
                        s += f" x{count}"
                    tool_strs.append(s)
                parts.append("called " + ", ".join(tool_strs))
            if user_snippets:
                parts.append("User: " + "; ".join(user_snippets))

            line = f"[{skill}]"
            if parts:
                line += " -- " + ". ".join(parts)
            summary_parts.append(line)

        summary = "\n".join(summary_parts)
        max_chars = self._max_summary_tokens * 4
        if len(summary) > max_chars:
            summary = summary[: max_chars - 3] + "..."

        start = 1
        if self._compressed:
            start = self._compressed.turn_count + 1
        end = start + len(turns) - 1

        return CompressedHistory(
            summary=summary,
            turn_count=(self._compressed.turn_count if self._compressed else 0) + len(turns),
            tool_call_summaries=(self._compressed.tool_call_summaries if self._compressed else [])
            + all_tool_summaries,
            original_turn_range=f"1-{end}",
        )

    async def compress(self) -> CompressedContext:
        """Compress older turns and return the full context.

        Keeps the last max_recent_turns at full fidelity. Older turns are
        compressed into a summary. If a summarizer is provided, it's used
        for a richer summary; otherwise, audit-based extraction is used.
        """
        with self._lock:
            if len(self._turns) <= self._max_recent_turns:
                return CompressedContext(
                    anchored=self.get_anchored_state(),
                    history=self._compressed,
                    recent_turns=list(self._turns),
                )
            to_compress = self._turns[: -self._max_recent_turns]
            recent = self._turns[-self._max_recent_turns :]
            self._turns = recent

        compressed = self._compress_turns(to_compress)

        if self._summarizer:
            try:
                llm_summary = await self._summarizer(compressed.summary)
                compressed.summary = llm_summary
            except Exception:
                pass  # fall back to audit-based summary

        with self._lock:
            if self._compressed:
                compressed.summary = self._compressed.summary + "\n" + compressed.summary
                max_chars = self._max_summary_tokens * 4
                if len(compressed.summary) > max_chars:
                    compressed.summary = compressed.summary[: max_chars - 3] + "..."
            self._compressed = compressed

        return CompressedContext(
            anchored=self.get_anchored_state(),
            history=self._compressed,
            recent_turns=list(recent),
        )

    def compress_sync(self) -> CompressedContext:
        """Synchronous version of compress(). Works when no summarizer is set."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.compress())
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(self.compress())
        except RuntimeError:
            return asyncio.run(self.compress())

    def to_prompt(self) -> str:
        """Render the current context as a prompt string.

        Does NOT trigger compression — just renders the current state.
        Call compress() first if you want to compress older turns.
        """
        anchored = self.get_anchored_state()
        with self._lock:
            compressed = self._compressed
            turns = list(self._turns)

        parts = [anchored.to_prompt()]
        if compressed:
            parts.append(compressed.to_prompt())
        if turns:
            lines = ["[RECENT CONVERSATION]"]
            for turn in turns:
                lines.append(f"[{turn.role}]: {turn.content}")
                if turn.tool_calls:
                    for tc in turn.tool_calls:
                        name = tc.get("tool_name", "unknown")
                        params = tc.get("params", {})
                        result = tc.get("result_summary", "")
                        param_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
                        line = f"  -> {name}({param_str})"
                        if result:
                            line += f" = {result}"
                        lines.append(line)
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def token_estimate(self) -> int:
        """Rough token estimate for the current context."""
        return len(self.to_prompt()) // 4

    @property
    def turn_count(self) -> int:
        """Total turns including compressed."""
        with self._lock:
            compressed_count = self._compressed.turn_count if self._compressed else 0
            return compressed_count + len(self._turns)
