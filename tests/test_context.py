"""Tests for clawboss.context — context compression module."""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from clawboss.budget import BudgetSnapshot
from clawboss.context import (
    AnchoredState,
    CompressedContext,
    CompressedHistory,
    ContextWindow,
    Turn,
)
from clawboss.policy import Policy
from clawboss.supervisor import Supervisor

# ---------------------------------------------------------------------------
# Helper tool functions
# ---------------------------------------------------------------------------


async def _ok_tool(**kwargs):
    return "ok"


async def _fail_tool(**kwargs):
    raise ValueError("boom")


def _make_supervisor(**overrides):
    defaults = {
        "max_iterations": 5,
        "tool_timeout": 30.0,
        "token_budget": 10000,
        "require_confirm": ["dangerous_tool"],
    }
    defaults.update(overrides)
    return Supervisor(Policy(**defaults))


# ---------------------------------------------------------------------------
# TestTurn
# ---------------------------------------------------------------------------


class TestTurn:
    def test_creation_with_defaults(self):
        t = Turn(role="user", content="hello world")
        assert t.role == "user"
        assert t.content == "hello world"
        assert t.tool_calls is None
        assert t.timestamp  # non-empty
        assert t.skill_name is None

    def test_auto_token_estimate(self):
        content = "x" * 100
        t = Turn(role="user", content=content)
        assert t.token_estimate == 25  # 100 // 4

    def test_auto_token_estimate_with_tool_calls(self):
        tc = {"tool_name": "search", "params": {"query": "test"}}
        t = Turn(role="assistant", content="calling tool", tool_calls=[tc])
        expected = len("calling tool") // 4 + len(str(tc)) // 4
        assert t.token_estimate == expected

    def test_explicit_token_estimate_not_overridden(self):
        """When token_estimate is nonzero, __post_init__ should leave it."""
        t = Turn(role="user", content="x" * 100, token_estimate=999)
        # token_estimate == 999 which is nonzero, so __post_init__ skips
        # Actually __post_init__ checks == 0, so 999 stays
        assert t.token_estimate == 999

    def test_to_dict_roundtrip(self):
        tc = [{"tool_name": "search", "params": {"q": "hi"}}]
        t = Turn(
            role="assistant",
            content="result",
            tool_calls=tc,
            skill_name="research",
        )
        d = t.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "result"
        assert d["tool_calls"] == tc
        assert d["skill_name"] == "research"

        t2 = Turn.from_dict(d)
        assert t2.role == t.role
        assert t2.content == t.content
        assert t2.tool_calls == t.tool_calls
        assert t2.skill_name == t.skill_name

    def test_to_dict_omits_none_and_zero(self):
        t = Turn(role="user", content="hi")
        d = t.to_dict()
        assert "tool_calls" not in d
        assert "skill_name" not in d
        # token_estimate of len("hi")//4 == 0, so omitted
        assert "token_estimate" not in d


# ---------------------------------------------------------------------------
# TestAnchoredState
# ---------------------------------------------------------------------------


class TestAnchoredState:
    def test_from_supervisor(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv)
        state = ctx.get_anchored_state()
        assert isinstance(state, AnchoredState)
        assert isinstance(state.budget_snapshot, BudgetSnapshot)
        assert state.budget_snapshot.tokens_used == 0
        assert state.confirmed_tools == ["dangerous_tool"]

    def test_to_prompt_includes_budget(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv)
        prompt = ctx.get_anchored_state().to_prompt()
        assert "[SUPERVISION STATE]" in prompt
        assert "Budget:" in prompt
        assert "0/10000 tokens used" in prompt

    def test_to_prompt_includes_circuit_states(self):
        state = AnchoredState(
            budget_snapshot=BudgetSnapshot(0, 10000, 0, 5),
            circuit_states={"web_search": "closed", "db_query": "open"},
            policy_summary={"max_iterations": 5},
            confirmed_tools=[],
        )
        prompt = state.to_prompt()
        assert "Circuit breakers:" in prompt
        assert "web_search=closed" in prompt
        assert "db_query=open" in prompt

    def test_to_prompt_includes_policy(self):
        sv = _make_supervisor()
        prompt = ContextWindow(sv).get_anchored_state().to_prompt()
        assert "Policy:" in prompt
        assert "max_iterations=5" in prompt
        assert "tool_timeout=30.0s" in prompt
        assert "token_budget=10000" in prompt

    def test_to_prompt_includes_confirmed_tools(self):
        sv = _make_supervisor()
        prompt = ContextWindow(sv).get_anchored_state().to_prompt()
        assert "Confirmation required: dangerous_tool" in prompt

    def test_to_prompt_includes_active_skill(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, skill_name="research")
        prompt = ctx.get_anchored_state().to_prompt()
        assert "Active skill: research" in prompt

    def test_to_prompt_no_active_skill_when_none(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv)
        prompt = ctx.get_anchored_state().to_prompt()
        assert "Active skill" not in prompt

    def test_to_dict_serializable(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, skill_name="demo")
        state = ctx.get_anchored_state()
        d = state.to_dict()
        # Should be JSON-serializable
        serialized = json.dumps(d)
        assert serialized
        parsed = json.loads(serialized)
        assert parsed["budget_snapshot"]["tokens_used"] == 0
        assert parsed["confirmed_tools"] == ["dangerous_tool"]
        assert parsed["active_skill"] == "demo"


# ---------------------------------------------------------------------------
# TestCompressedHistory
# ---------------------------------------------------------------------------


class TestCompressedHistory:
    def test_to_prompt_format(self):
        ch = CompressedHistory(
            summary="User asked about Python. called web_search.",
            turn_count=5,
            tool_call_summaries=[{"tool_name": "web_search"}],
            original_turn_range="1-5",
        )
        prompt = ch.to_prompt()
        assert "[CONVERSATION HISTORY (turns 1-5, compressed)]" in prompt
        assert "User asked about Python" in prompt

    def test_token_estimate(self):
        ch = CompressedHistory(
            summary="A" * 400,
            turn_count=10,
            tool_call_summaries=[],
            original_turn_range="1-10",
        )
        est = ch.token_estimate()
        assert est == len(ch.to_prompt()) // 4
        assert est > 0


# ---------------------------------------------------------------------------
# TestCompressedContext
# ---------------------------------------------------------------------------


class TestCompressedContext:
    def test_to_prompt_combines_all_zones(self):
        anchored = AnchoredState(
            budget_snapshot=BudgetSnapshot(100, 10000, 1, 5),
            circuit_states={},
            policy_summary={"max_iterations": 5},
            confirmed_tools=[],
        )
        history = CompressedHistory(
            summary="Previous conversation.",
            turn_count=5,
            tool_call_summaries=[],
            original_turn_range="1-5",
        )
        recent = [Turn(role="user", content="What next?")]
        cc = CompressedContext(anchored=anchored, history=history, recent_turns=recent)
        prompt = cc.to_prompt()
        assert "[SUPERVISION STATE]" in prompt
        assert "[CONVERSATION HISTORY" in prompt
        assert "[RECENT CONVERSATION]" in prompt
        assert "[user]: What next?" in prompt

    def test_to_prompt_with_tool_calls_in_recent(self):
        anchored = AnchoredState(
            budget_snapshot=BudgetSnapshot(0, None, 0, 5),
            circuit_states={},
            policy_summary={},
            confirmed_tools=[],
        )
        tc = {
            "tool_name": "search",
            "params": {"query": "test"},
            "result_summary": "found 3 results",
        }
        recent = [
            Turn(
                role="assistant",
                content="Searching",
                tool_calls=[tc],
            )
        ]
        cc = CompressedContext(anchored=anchored, history=None, recent_turns=recent)
        prompt = cc.to_prompt()
        assert "-> search(query='test')" in prompt
        assert "found 3 results" in prompt

    def test_token_estimate(self):
        anchored = AnchoredState(
            budget_snapshot=BudgetSnapshot(0, 10000, 0, 5),
            circuit_states={},
            policy_summary={"max_iterations": 5},
            confirmed_tools=[],
        )
        cc = CompressedContext(anchored=anchored, history=None, recent_turns=[])
        est = cc.token_estimate()
        assert est == len(cc.to_prompt()) // 4


# ---------------------------------------------------------------------------
# TestContextWindowAddTurn
# ---------------------------------------------------------------------------


class TestContextWindowAddTurn:
    def test_accumulates_turns(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv)
        ctx.add_turn("user", "hello")
        ctx.add_turn("assistant", "hi there")
        assert ctx.turn_count == 2

    def test_with_tool_calls(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv)
        tc = [{"tool_name": "search", "params": {"q": "test"}}]
        ctx.add_turn("assistant", "searching", tool_calls=tc)
        prompt = ctx.to_prompt()
        assert "search" in prompt

    def test_inherits_skill_name(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, skill_name="research")
        ctx.add_turn("user", "find info")
        # The turn should inherit the context-level skill_name
        with ctx._lock:
            assert ctx._turns[0].skill_name == "research"

    def test_overrides_skill_name(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, skill_name="research")
        ctx.add_turn("user", "find info", skill_name="coding")
        with ctx._lock:
            assert ctx._turns[0].skill_name == "coding"


# ---------------------------------------------------------------------------
# TestContextWindowCompression
# ---------------------------------------------------------------------------


class TestContextWindowCompression:
    @pytest.mark.asyncio
    async def test_no_compression_under_limit(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, max_recent_turns=10)
        for i in range(5):
            ctx.add_turn("user", f"message {i}")
        result = await ctx.compress()
        assert result.history is None
        assert len(result.recent_turns) == 5

    @pytest.mark.asyncio
    async def test_triggers_at_limit(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, max_recent_turns=3)
        for i in range(6):
            ctx.add_turn("user", f"message {i}")
        result = await ctx.compress()
        assert result.history is not None
        assert result.history.turn_count == 3  # 3 compressed
        assert len(result.recent_turns) == 3  # 3 kept recent

    @pytest.mark.asyncio
    async def test_preserves_tool_summaries_in_compressed_history(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, max_recent_turns=2)
        tc = [{"tool_name": "search", "params": {"q": "ai"}}]
        ctx.add_turn("assistant", "searching", tool_calls=tc)
        ctx.add_turn("user", "thanks")
        ctx.add_turn("user", "more please")
        ctx.add_turn("assistant", "here you go")
        result = await ctx.compress()
        assert result.history is not None
        assert len(result.history.tool_call_summaries) >= 1
        assert result.history.tool_call_summaries[0]["tool_name"] == "search"

    @pytest.mark.asyncio
    async def test_keeps_recent_turns_at_full_fidelity(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, max_recent_turns=2)
        ctx.add_turn("user", "old message")
        ctx.add_turn("assistant", "old reply")
        ctx.add_turn("user", "recent message")
        ctx.add_turn("assistant", "recent reply")
        result = await ctx.compress()
        assert len(result.recent_turns) == 2
        assert result.recent_turns[0].content == "recent message"
        assert result.recent_turns[1].content == "recent reply"

    @pytest.mark.asyncio
    async def test_incremental_compression(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, max_recent_turns=2)
        # First batch
        for i in range(4):
            ctx.add_turn("user", f"batch1 msg {i}")
        result1 = await ctx.compress()
        assert result1.history is not None
        first_count = result1.history.turn_count

        # Second batch
        for i in range(4):
            ctx.add_turn("user", f"batch2 msg {i}")
        result2 = await ctx.compress()
        assert result2.history is not None
        assert result2.history.turn_count > first_count


# ---------------------------------------------------------------------------
# TestAnchoredStateFreshness
# ---------------------------------------------------------------------------


class TestAnchoredStateFreshness:
    def test_reflects_budget_changes_after_recording_tokens(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv)
        state_before = ctx.get_anchored_state()
        assert state_before.budget_snapshot.tokens_used == 0

        sv.record_tokens(500)
        state_after = ctx.get_anchored_state()
        assert state_after.budget_snapshot.tokens_used == 500

    @pytest.mark.asyncio
    async def test_reflects_circuit_breaker_changes_after_failing(self):
        sv = _make_supervisor(circuit_breaker_threshold=2)
        ctx = ContextWindow(sv)
        state_before = ctx.get_anchored_state()
        assert state_before.circuit_states == {}

        # Cause failures to trip the circuit breaker
        await sv.call("flaky", _fail_tool)
        await sv.call("flaky", _fail_tool)

        state_after = ctx.get_anchored_state()
        assert "flaky" in state_after.circuit_states
        assert state_after.circuit_states["flaky"] == "open"


# ---------------------------------------------------------------------------
# TestSkillGrouping
# ---------------------------------------------------------------------------


class TestSkillGrouping:
    @pytest.mark.asyncio
    async def test_compression_groups_by_skill_name(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, max_recent_turns=2)
        ctx.add_turn("user", "research task", skill_name="research")
        ctx.add_turn("assistant", "researching", skill_name="research")
        ctx.add_turn("user", "code task", skill_name="coding")
        ctx.add_turn("assistant", "coding now", skill_name="coding")
        result = await ctx.compress()
        assert result.history is not None
        summary = result.history.summary
        assert "[research]" in summary
        # The coding turns may be in recent or compressed depending on
        # max_recent_turns; check that at least research is grouped
        assert "research task" in summary

    @pytest.mark.asyncio
    async def test_general_skill_for_no_skill_name(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, max_recent_turns=2)
        ctx.add_turn("user", "hello")
        ctx.add_turn("assistant", "hi")
        ctx.add_turn("user", "recent msg")
        ctx.add_turn("assistant", "recent reply")
        result = await ctx.compress()
        assert result.history is not None
        assert "[general]" in result.history.summary


# ---------------------------------------------------------------------------
# TestToPrompt
# ---------------------------------------------------------------------------


class TestToPrompt:
    def test_renders_all_three_zones(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, max_recent_turns=2)
        # Add enough turns to trigger compression on compress()
        for i in range(4):
            ctx.add_turn("user", f"msg {i}")
        asyncio.run(ctx.compress())
        prompt = ctx.to_prompt()
        assert "[SUPERVISION STATE]" in prompt
        assert "[CONVERSATION HISTORY" in prompt
        assert "[RECENT CONVERSATION]" in prompt

    def test_works_without_compression(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv)
        ctx.add_turn("user", "hello")
        prompt = ctx.to_prompt()
        assert "[SUPERVISION STATE]" in prompt
        assert "[CONVERSATION HISTORY" not in prompt
        assert "[RECENT CONVERSATION]" in prompt
        assert "[user]: hello" in prompt

    def test_works_with_compression(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, max_recent_turns=1)
        ctx.add_turn("user", "old")
        ctx.add_turn("user", "recent")
        asyncio.run(ctx.compress())
        prompt = ctx.to_prompt()
        assert "[SUPERVISION STATE]" in prompt
        assert "[CONVERSATION HISTORY" in prompt
        assert "[RECENT CONVERSATION]" in prompt
        assert "[user]: recent" in prompt


# ---------------------------------------------------------------------------
# TestWithSummarizer
# ---------------------------------------------------------------------------


class TestWithSummarizer:
    @pytest.mark.asyncio
    async def test_custom_summarizer_called(self):
        called_with = []

        async def mock_summarizer(text: str) -> str:
            called_with.append(text)
            return "LLM-generated summary"

        sv = _make_supervisor()
        ctx = ContextWindow(sv, max_recent_turns=2, summarizer=mock_summarizer)
        for i in range(4):
            ctx.add_turn("user", f"message {i}")
        result = await ctx.compress()
        assert len(called_with) == 1
        assert result.history is not None
        assert "LLM-generated summary" in result.history.summary

    @pytest.mark.asyncio
    async def test_summarizer_result_used(self):
        async def custom_summary(text: str) -> str:
            return "Custom: " + text[:20]

        sv = _make_supervisor()
        ctx = ContextWindow(sv, max_recent_turns=2, summarizer=custom_summary)
        for i in range(4):
            ctx.add_turn("user", f"msg {i}")
        result = await ctx.compress()
        assert result.history is not None
        assert result.history.summary.startswith("Custom:")

    @pytest.mark.asyncio
    async def test_summarizer_failure_falls_back_to_audit_based(self):
        async def failing_summarizer(text: str) -> str:
            raise RuntimeError("LLM unavailable")

        sv = _make_supervisor()
        ctx = ContextWindow(sv, max_recent_turns=2, summarizer=failing_summarizer)
        for i in range(4):
            ctx.add_turn("user", f"message {i}")
        result = await ctx.compress()
        # Should still have a history with audit-based summary
        assert result.history is not None
        assert len(result.history.summary) > 0
        # The audit-based summary should include [general] group tag
        assert "[general]" in result.history.summary


# ---------------------------------------------------------------------------
# TestTokenEstimate
# ---------------------------------------------------------------------------


class TestTokenEstimate:
    def test_increases_with_turns(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv)
        est_empty = ctx.token_estimate()
        ctx.add_turn("user", "A" * 400)
        est_one = ctx.token_estimate()
        assert est_one > est_empty
        ctx.add_turn("assistant", "B" * 400)
        est_two = ctx.token_estimate()
        assert est_two > est_one

    @pytest.mark.asyncio
    async def test_decreases_after_compression(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv, max_recent_turns=2)
        # Add many verbose turns
        for i in range(10):
            ctx.add_turn("user", f"{'x' * 200} message {i}")
        est_before = ctx.token_estimate()
        await ctx.compress()
        est_after = ctx.token_estimate()
        assert est_after < est_before


# ---------------------------------------------------------------------------
# TestThreadSafety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_add_turn_from_multiple_threads(self):
        sv = _make_supervisor()
        ctx = ContextWindow(sv)
        errors = []
        num_threads = 10
        turns_per_thread = 50

        def add_turns(thread_id: int):
            try:
                for i in range(turns_per_thread):
                    ctx.add_turn("user", f"t{thread_id}-m{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_turns, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert ctx.turn_count == num_threads * turns_per_thread
