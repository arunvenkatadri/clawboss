"""Tests for cost attribution — PricingTable, input/output tokens, dashboard data."""

import pytest

from clawboss.observe import ModelPricing, Observer, PricingTable
from clawboss.session import SessionManager
from clawboss.store import MemoryStore

# ---------------------------------------------------------------------------
# PricingTable
# ---------------------------------------------------------------------------


class TestPricingTable:
    def test_default_table_has_common_models(self):
        t = PricingTable.default()
        assert "claude-sonnet-4-6" in t.models
        assert "gpt-4o" in t.models
        assert "gemini-1.5-flash" in t.models

    def test_cost_computation(self):
        t = PricingTable.default()
        # claude-sonnet-4-6: $3/M input, $15/M output
        # 1M input tokens + 1M output tokens = $3 + $15 = $18
        cost = t.cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000)
        assert abs(cost - 18.0) < 0.001

    def test_small_amounts(self):
        t = PricingTable.default()
        # 1000 input + 500 output on claude-sonnet-4-6
        # (1000/1M)*3 + (500/1M)*15 = 0.003 + 0.0075 = 0.0105
        cost = t.cost_usd("claude-sonnet-4-6", 1000, 500)
        assert abs(cost - 0.0105) < 0.00001

    def test_unknown_model_returns_zero(self):
        t = PricingTable.default()
        assert t.cost_usd("unknown-model", 1000, 1000) == 0.0

    def test_custom_pricing(self):
        t = PricingTable()
        t.set_model("my-model", input_per_million=1.0, output_per_million=2.0)
        cost = t.cost_usd("my-model", 1_000_000, 1_000_000)
        assert abs(cost - 3.0) < 0.001

    def test_to_dict(self):
        t = PricingTable.default()
        d = t.to_dict()
        assert "models" in d
        assert "claude-sonnet-4-6" in d["models"]
        assert d["models"]["claude-sonnet-4-6"]["input_per_million"] == 3.0


# ---------------------------------------------------------------------------
# Observer with pricing
# ---------------------------------------------------------------------------


class TestObserverWithPricing:
    def test_records_cost(self):
        obs = Observer(pricing=PricingTable.default())
        obs.record_tool_call(
            "llm_call",
            duration_ms=100,
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-6",
        )
        summary = obs.tool_summary("llm_call")
        assert summary["total_input_tokens"] == 1000
        assert summary["total_output_tokens"] == 500
        assert summary["total_cost_usd"] > 0

    def test_records_zero_cost_for_unknown_model(self):
        obs = Observer(pricing=PricingTable.default())
        obs.record_tool_call("tool", input_tokens=1000, output_tokens=500, model="unknown")
        summary = obs.tool_summary("tool")
        assert summary["total_cost_usd"] == 0.0
        assert summary["total_tokens"] == 1500

    def test_no_pricing_means_zero_cost(self):
        obs = Observer()  # no pricing
        obs.record_tool_call(
            "tool", input_tokens=1000, output_tokens=500, model="claude-sonnet-4-6"
        )
        summary = obs.tool_summary("tool")
        assert summary["total_cost_usd"] == 0.0
        assert summary["total_tokens"] == 1500

    def test_legacy_tokens_param_still_works(self):
        """Backward-compat: old `tokens=N` still works."""
        obs = Observer(pricing=PricingTable.default())
        obs.record_tool_call("tool", tokens=500)
        summary = obs.tool_summary("tool")
        assert summary["total_tokens"] == 500
        # Counts as output tokens in legacy mode
        assert summary["total_output_tokens"] == 500


# ---------------------------------------------------------------------------
# Cost summary endpoint (aggregations)
# ---------------------------------------------------------------------------


class TestCostSummary:
    def test_empty(self):
        obs = Observer(pricing=PricingTable.default())
        summary = obs.cost_summary()
        assert summary["total_cost_usd"] == 0.0
        assert summary["total_calls"] == 0
        assert summary["by_agent"] == []

    def test_by_agent(self):
        obs = Observer(pricing=PricingTable.default())
        obs.record_tool_call(
            "search",
            agent_id="researcher",
            input_tokens=1000,
            output_tokens=500,
            model="claude-sonnet-4-6",
        )
        obs.record_tool_call(
            "fetch",
            agent_id="researcher",
            input_tokens=500,
            output_tokens=200,
            model="claude-sonnet-4-6",
        )
        obs.record_tool_call(
            "write",
            agent_id="writer",
            input_tokens=2000,
            output_tokens=1000,
            model="claude-sonnet-4-6",
        )
        summary = obs.cost_summary()
        assert summary["total_calls"] == 3
        by_agent = {a["agent"]: a for a in summary["by_agent"]}
        assert by_agent["researcher"]["calls"] == 2
        assert by_agent["writer"]["calls"] == 1
        # Writer has more tokens → higher cost → should be first
        assert summary["by_agent"][0]["agent"] == "writer"

    def test_by_session(self):
        obs = Observer(pricing=PricingTable.default())
        obs.record_tool_call(
            "t",
            session_id="s1",
            input_tokens=100,
            output_tokens=50,
            model="claude-sonnet-4-6",
        )
        obs.record_tool_call(
            "t",
            session_id="s2",
            input_tokens=200,
            output_tokens=100,
            model="claude-sonnet-4-6",
        )
        summary = obs.cost_summary()
        sessions = {s["session_id"]: s for s in summary["by_session"]}
        assert "s1" in sessions
        assert "s2" in sessions

    def test_by_model(self):
        obs = Observer(pricing=PricingTable.default())
        obs.record_tool_call("t", input_tokens=1000, output_tokens=500, model="claude-sonnet-4-6")
        obs.record_tool_call("t", input_tokens=1000, output_tokens=500, model="gpt-4o")
        summary = obs.cost_summary()
        by_model = {m["model"]: m for m in summary["by_model"]}
        assert "claude-sonnet-4-6" in by_model
        assert "gpt-4o" in by_model
        # Same token counts, different models → different costs
        assert by_model["claude-sonnet-4-6"]["cost"] != by_model["gpt-4o"]["cost"]

    def test_totals_add_up(self):
        obs = Observer(pricing=PricingTable.default())
        for _ in range(5):
            obs.record_tool_call(
                "t",
                input_tokens=1000,
                output_tokens=500,
                model="claude-sonnet-4-6",
            )
        summary = obs.cost_summary()
        assert summary["total_input_tokens"] == 5000
        assert summary["total_output_tokens"] == 2500
        assert summary["total_calls"] == 5
        # 5 * (0.003 + 0.0075) = 5 * 0.0105 = 0.0525
        assert abs(summary["total_cost_usd"] - 0.0525) < 0.00001


# ---------------------------------------------------------------------------
# Session integration
# ---------------------------------------------------------------------------


class TestSessionManagerCosts:
    def test_default_pricing_enabled(self):
        """SessionManager defaults to the built-in pricing table."""
        store = MemoryStore()
        mgr = SessionManager(store)
        assert mgr.observer.pricing is not None
        assert "claude-sonnet-4-6" in mgr.observer.pricing.models

    def test_custom_pricing(self):
        store = MemoryStore()
        pricing = PricingTable()
        pricing.set_model("my-model", 100.0, 200.0)
        mgr = SessionManager(store, pricing=pricing)
        assert mgr.observer.pricing is pricing

    @pytest.mark.asyncio
    async def test_tool_output_with_input_output_tokens(self):
        """Tools that return {input_tokens, output_tokens, model} get cost attribution."""
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", {"max_iterations": 5, "token_budget": 100000})
        sv = mgr.get_supervisor(sid)

        async def llm_call():
            return {
                "result": "hello",
                "input_tokens": 1000,
                "output_tokens": 500,
                "model": "claude-sonnet-4-6",
            }

        result = await sv.call("llm_call", llm_call)
        assert result.succeeded is True

        summary = mgr.observer.session_summary(sid)
        assert summary["total_input_tokens"] == 1000
        assert summary["total_output_tokens"] == 500
        assert summary["total_cost_usd"] > 0

    @pytest.mark.asyncio
    async def test_legacy_tokens_used_still_works(self):
        """Old tool outputs with just tokens_used keep working."""
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", {"max_iterations": 5, "token_budget": 100000})
        sv = mgr.get_supervisor(sid)

        async def old_tool():
            return {"result": "hi", "tokens_used": 500}

        result = await sv.call("old_tool", old_tool)
        assert result.succeeded is True

        summary = mgr.observer.session_summary(sid)
        assert summary["total_output_tokens"] == 0
        # No model → no cost
        assert summary["total_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# ModelPricing math
# ---------------------------------------------------------------------------


class TestModelPricing:
    def test_zero_pricing(self):
        p = ModelPricing()
        assert p.cost_usd(1_000_000, 1_000_000) == 0.0

    def test_input_only(self):
        p = ModelPricing(input_per_million=5.0, output_per_million=0.0)
        assert p.cost_usd(1_000_000, 1_000_000) == 5.0

    def test_output_only(self):
        p = ModelPricing(input_per_million=0.0, output_per_million=10.0)
        assert p.cost_usd(1_000_000, 1_000_000) == 10.0
