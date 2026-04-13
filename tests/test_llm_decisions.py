"""Tests for LLM decision steps, with_context, and initial_input in pipelines."""

import pytest

from clawboss.pipeline import Pipeline
from clawboss.session import SessionManager
from clawboss.store import MemoryStore

POLICY = {"max_iterations": 10, "tool_timeout": 10}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def enrich(input=None):
    return {"user_id": 42, "amount": 15000, "country": "US"}


async def block(input=None):
    return "BLOCKED"


async def approve(input=None):
    return "APPROVED"


async def escalate(input=None):
    return "ESCALATED"


async def passthrough(input=None):
    return input


# ---------------------------------------------------------------------------
# LLM decision step
# ---------------------------------------------------------------------------


class TestLlmDecision:
    @pytest.mark.asyncio
    async def test_llm_decision_routes_to_block(self):
        store = MemoryStore()
        mgr = SessionManager(store)

        async def fake_llm(prompt: str) -> str:
            # Mock LLM that returns block for amount > 10000
            if "15000" in prompt:
                return '{"action": "block", "reason": "Amount too high"}'
            return '{"action": "approve"}'

        result = await (
            Pipeline(mgr, "fraud-detector", POLICY)
            .add_step("enrich", enrich)
            .add_llm_decision(
                fake_llm,
                prompt_template="Transaction: {input}\nDecide action.",
            )
            .add_condition(
                lambda out: out.get("action") == "block",
                then_step=("block", block),
                else_step=("approve", approve),
            )
            .run()
        )
        assert result.completed is True
        assert result.final_output == "BLOCKED"

    @pytest.mark.asyncio
    async def test_llm_decision_returns_parsed_json(self):
        store = MemoryStore()
        mgr = SessionManager(store)

        async def fake_llm(prompt: str) -> str:
            return '{"action": "escalate", "confidence": 0.95}'

        result = await (
            Pipeline(mgr, "test", POLICY)
            .add_step("enrich", enrich)
            .add_llm_decision(fake_llm, "Decide: {input}")
            .run()
        )
        assert result.completed is True
        assert result.final_output["action"] == "escalate"
        assert result.final_output["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_llm_decision_strips_markdown_fences(self):
        store = MemoryStore()
        mgr = SessionManager(store)

        async def fake_llm(prompt: str) -> str:
            return '```json\n{"action": "approve"}\n```'

        result = await (
            Pipeline(mgr, "test", POLICY)
            .add_step("enrich", enrich)
            .add_llm_decision(fake_llm, "Decide: {input}")
            .run()
        )
        assert result.completed is True
        assert result.final_output["action"] == "approve"

    @pytest.mark.asyncio
    async def test_llm_decision_malformed_json(self):
        store = MemoryStore()
        mgr = SessionManager(store)

        async def fake_llm(prompt: str) -> str:
            return "not json at all"

        result = await (
            Pipeline(mgr, "test", POLICY)
            .add_step("enrich", enrich)
            .add_llm_decision(fake_llm, "Decide: {input}")
            .run()
        )
        assert result.completed is True
        assert "error" in result.final_output
        assert result.final_output["raw"] == "not json at all"

    @pytest.mark.asyncio
    async def test_llm_sees_input(self):
        """LLM prompt should contain the previous step's output as {input}."""
        store = MemoryStore()
        mgr = SessionManager(store)
        received = []

        async def capture_llm(prompt: str) -> str:
            received.append(prompt)
            return '{"action": "approve"}'

        await (
            Pipeline(mgr, "test", POLICY)
            .add_step("enrich", enrich)
            .add_llm_decision(capture_llm, "Input was: {input}")
            .run()
        )
        assert len(received) == 1
        assert "15000" in received[0]
        assert "user_id" in received[0]


# ---------------------------------------------------------------------------
# with_context — session reuse and payload accumulation
# ---------------------------------------------------------------------------


class TestWithContext:
    @pytest.mark.asyncio
    async def test_initial_input_flows_into_pipeline(self):
        store = MemoryStore()
        mgr = SessionManager(store)

        result = await (
            Pipeline(mgr, "test", POLICY)
            .add_step("passthrough", passthrough)
            .run(initial_input={"event": "user_signup", "user_id": 1})
        )
        assert result.completed is True
        assert result.final_output["event"] == "user_signup"

    @pytest.mark.asyncio
    async def test_with_context_reuses_session(self):
        store = MemoryStore()
        mgr = SessionManager(store)

        # Start a session
        sid = mgr.start("stateful-agent", POLICY)

        # Run a pipeline against that session
        result1 = await (
            Pipeline(mgr, "stateful-agent", POLICY)
            .with_context(sid)
            .add_step("echo", passthrough)
            .run(initial_input={"msg": "first"})
        )
        assert result1.completed is True
        assert result1.session_id == sid

        # Run another pipeline against the same session
        result2 = await (
            Pipeline(mgr, "stateful-agent", POLICY)
            .with_context(sid)
            .add_step("echo", passthrough)
            .run(initial_input={"msg": "second"})
        )
        assert result2.completed is True
        assert result2.session_id == sid

    @pytest.mark.asyncio
    async def test_with_context_accumulates_history(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("stateful-agent", POLICY)

        for i in range(3):
            await (
                Pipeline(mgr, "stateful-agent", POLICY)
                .with_context(sid)
                .add_step("echo", passthrough)
                .run(initial_input={"msg": f"event-{i}"})
            )

        cp = mgr.status(sid)
        history = cp.payload.get("history", [])
        assert len(history) == 3
        assert history[0]["input"]["msg"] == "event-0"
        assert history[2]["input"]["msg"] == "event-2"

    @pytest.mark.asyncio
    async def test_llm_decision_sees_context(self):
        """LLM step with include_context=True receives the session payload."""
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("stateful", POLICY, payload={"user_risk_score": 75})

        received_prompts = []

        async def context_aware_llm(prompt: str) -> str:
            received_prompts.append(prompt)
            return '{"action": "approve"}'

        await (
            Pipeline(mgr, "stateful", POLICY)
            .with_context(sid)
            .add_step("enrich", enrich)
            .add_llm_decision(
                context_aware_llm,
                "Input: {input}\nContext: {context}",
                include_context=True,
            )
            .run()
        )

        assert len(received_prompts) == 1
        assert "user_risk_score" in received_prompts[0]
        assert "75" in received_prompts[0]


# ---------------------------------------------------------------------------
# Stream integration pattern
# ---------------------------------------------------------------------------


class TestStreamIntegration:
    @pytest.mark.asyncio
    async def test_stream_to_pipeline_pattern(self):
        """Simulate a stream message flowing through a pipeline."""
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("stream-agent", POLICY)

        async def classify(input=None):
            amount = input.get("amount", 0) if input else 0
            return {"amount": amount, "large": amount > 1000}

        async def handle_large(input=None):
            return f"LARGE: ${input['amount']}"

        async def handle_small(input=None):
            return f"small: ${input['amount']}"

        # Simulate receiving multiple stream messages
        messages = [
            {"user": "alice", "amount": 50},
            {"user": "bob", "amount": 5000},
            {"user": "carol", "amount": 100},
        ]

        outputs = []
        for msg in messages:
            result = await (
                Pipeline(mgr, "stream-agent", POLICY)
                .with_context(sid)
                .add_step("classify", classify)
                .add_condition(
                    lambda out: out["large"],
                    then_step=("large", handle_large),
                    else_step=("small", handle_small),
                )
                .run(initial_input=msg)
            )
            outputs.append(result.final_output)

        assert outputs[0] == "small: $50"
        assert outputs[1] == "LARGE: $5000"
        assert outputs[2] == "small: $100"

        # Session should still be alive with all 3 history entries
        cp = mgr.status(sid)
        assert len(cp.payload["history"]) == 3
