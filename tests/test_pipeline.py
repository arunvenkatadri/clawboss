"""Tests for clawboss.pipeline — sequential supervised orchestration."""

from typing import Any

import pytest

from clawboss.pipeline import Pipeline
from clawboss.session import SessionManager
from clawboss.store import MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POLICY = {"max_iterations": 10, "tool_timeout": 10, "token_budget": 50000}


async def search(query: str = "test") -> str:
    return f"results for {query}"


async def summarize(input: str = "") -> str:
    return f"summary of: {input}"


async def write_report(input: str = "", title: str = "Report") -> str:
    return f"# {title}\n\n{input}"


async def failing_tool(input: str = "") -> str:
    raise RuntimeError("tool broke")


async def token_tool(input: str = "") -> dict:
    return {"text": f"processed {input}", "tokens_used": 500}


# ---------------------------------------------------------------------------
# Basic pipeline
# ---------------------------------------------------------------------------


class TestPipelineBasic:
    @pytest.mark.asyncio
    async def test_simple_chain(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        p = Pipeline(mgr, "test-agent", POLICY)
        p.add_step("search", search, query="quantum")
        p.add_step("summarize", summarize)

        result = await p.run()
        assert result.completed is True
        assert len(result.steps) == 2
        assert result.steps[0].result.succeeded is True
        assert result.steps[1].result.succeeded is True
        assert "summary of: results for quantum" in result.final_output

    @pytest.mark.asyncio
    async def test_three_step_chain(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        p = Pipeline(mgr, "test-agent", POLICY)
        p.add_step("search", search, query="AI")
        p.add_step("summarize", summarize)
        p.add_step("write_report", write_report, title="AI Report")

        result = await p.run()
        assert result.completed is True
        assert len(result.steps) == 3
        assert "# AI Report" in result.final_output
        assert "summary of:" in result.final_output

    @pytest.mark.asyncio
    async def test_chaining_passes_output(self):
        """Each step receives the previous step's output."""
        store = MemoryStore()
        mgr = SessionManager(store)
        p = Pipeline(mgr, "test-agent", POLICY)
        p.add_step("search", search, query="test")
        p.add_step("summarize", summarize)

        result = await p.run()
        # summarize should have received "results for test" as input
        assert "results for test" in result.final_output

    @pytest.mark.asyncio
    async def test_no_chain_input_on_first_step(self):
        """First step doesn't get chained input."""
        store = MemoryStore()
        mgr = SessionManager(store)
        p = Pipeline(mgr, "test-agent", POLICY)
        p.add_step("search", search, query="hello")

        result = await p.run()
        assert result.completed is True
        assert result.final_output == "results for hello"


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


class TestPipelineFailures:
    @pytest.mark.asyncio
    async def test_stops_on_failure(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        p = Pipeline(mgr, "test-agent", POLICY)
        p.add_step("search", search, query="test")
        p.add_step("broken", failing_tool)
        p.add_step("summarize", summarize)  # should never run

        result = await p.run()
        assert result.completed is False
        assert result.stopped_at == "2_broken"
        assert len(result.steps) == 2  # search + broken, not summarize
        assert "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_final_output_is_last_success(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        p = Pipeline(mgr, "test-agent", POLICY)
        p.add_step("search", search, query="data")
        p.add_step("broken", failing_tool)

        result = await p.run()
        assert result.final_output == "results for data"


# ---------------------------------------------------------------------------
# Session integration
# ---------------------------------------------------------------------------


class TestPipelineSession:
    @pytest.mark.asyncio
    async def test_creates_and_stops_session(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        p = Pipeline(mgr, "test-agent", POLICY)
        p.add_step("search", search, query="test")

        result = await p.run()
        # Session should be stopped after pipeline completes
        cp = mgr.status(result.session_id)
        assert cp.status.value == "stopped"

    @pytest.mark.asyncio
    async def test_updates_payload_with_progress(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        p = Pipeline(mgr, "test-agent", POLICY)
        p.add_step("search", search, query="test")
        p.add_step("summarize", summarize)

        result = await p.run()
        cp = mgr.status(result.session_id)
        assert cp.payload["pipeline_step"] == 1  # 0-indexed, last step
        assert cp.payload["pipeline_total_steps"] == 2

    @pytest.mark.asyncio
    async def test_stateless_pipeline(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        p = Pipeline(mgr, "test-agent", POLICY, stateless=True)
        p.add_step("search", search, query="fast")

        result = await p.run()
        assert result.completed is True

    @pytest.mark.asyncio
    async def test_observability_records_calls(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        p = Pipeline(mgr, "test-agent", POLICY)
        p.add_step("search", search, query="test")
        p.add_step("summarize", summarize)

        await p.run()
        summary = mgr.observer.all_tools_summary()
        assert "search" in summary
        assert "summarize" in summary
        assert summary["search"]["calls"] == 1
        assert summary["summarize"]["calls"] == 1

    @pytest.mark.asyncio
    async def test_total_duration(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        p = Pipeline(mgr, "test-agent", POLICY)
        p.add_step("search", search, query="test")
        p.add_step("summarize", summarize)

        result = await p.run()
        assert result.total_duration_ms >= 0


# ---------------------------------------------------------------------------
# Chaining API
# ---------------------------------------------------------------------------


class TestPipelineChainingApi:
    @pytest.mark.asyncio
    async def test_fluent_api(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        result = await (
            Pipeline(mgr, "test-agent", POLICY)
            .add_step("search", search, query="test")
            .add_step("summarize", summarize)
            .run()
        )
        assert result.completed is True

    @pytest.mark.asyncio
    async def test_custom_input_key(self):
        async def process(text: str = "") -> str:
            return f"processed: {text}"

        store = MemoryStore()
        mgr = SessionManager(store)
        result = await (
            Pipeline(mgr, "test-agent", POLICY)
            .add_step("search", search, query="data")
            .add_step("process", process, chain_input=True, input_key="text")
            .run()
        )
        assert result.completed is True
        assert "processed: results for data" in result.final_output

    @pytest.mark.asyncio
    async def test_no_chain_input(self):
        """Step with chain_input=False ignores previous output."""
        store = MemoryStore()
        mgr = SessionManager(store)
        result = await (
            Pipeline(mgr, "test-agent", POLICY)
            .add_step("search", search, query="first")
            .add_step("search", search, chain_input=False, query="independent")
            .run()
        )
        assert result.completed is True
        assert result.final_output == "results for independent"


# ---------------------------------------------------------------------------
# Approval handling
# ---------------------------------------------------------------------------


class TestPipelineApproval:
    @pytest.mark.asyncio
    async def test_stops_at_approval(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        policy = {**POLICY, "require_confirm": ["dangerous"]}

        async def dangerous_tool(input: str = "") -> str:
            return "deleted everything"

        result = await (
            Pipeline(mgr, "test-agent", policy)
            .add_step("search", search, query="test")
            .add_step("dangerous", dangerous_tool)
            .add_step("summarize", summarize)
            .run()
        )
        assert result.completed is False
        assert "approval" in result.error.lower()
        assert len(result.steps) == 2  # search + dangerous (pending)


# ---------------------------------------------------------------------------
# Conditional routing (Level 2)
# ---------------------------------------------------------------------------


class TestPipelineCondition:
    @pytest.mark.asyncio
    async def test_condition_takes_then_branch(self):
        async def high_action(input: str = "") -> str:
            return "took high path"

        async def low_action(input: str = "") -> str:
            return "took low path"

        store = MemoryStore()
        mgr = SessionManager(store)
        result = await (
            Pipeline(mgr, "test-agent", POLICY)
            .add_step("search", search, query="data")
            .add_condition(
                lambda output: "results" in output,
                then_step=("high", high_action),
                else_step=("low", low_action),
            )
            .run()
        )
        assert result.completed is True
        assert result.final_output == "took high path"
        assert result.steps[-1].branch == "then"

    @pytest.mark.asyncio
    async def test_condition_takes_else_branch(self):
        async def high_action(input: str = "") -> str:
            return "took high path"

        async def low_action(input: str = "") -> str:
            return "took low path"

        store = MemoryStore()
        mgr = SessionManager(store)
        result = await (
            Pipeline(mgr, "test-agent", POLICY)
            .add_step("search", search, query="data")
            .add_condition(
                lambda output: "NOTFOUND" in output,
                then_step=("high", high_action),
                else_step=("low", low_action),
            )
            .run()
        )
        assert result.completed is True
        assert result.final_output == "took low path"
        assert result.steps[-1].branch == "else"

    @pytest.mark.asyncio
    async def test_condition_else_none_skips(self):
        """If else_step is None and predicate is False, skip and keep previous output."""
        store = MemoryStore()
        mgr = SessionManager(store)

        async def action(input: str = "") -> str:
            return "should not run"

        result = await (
            Pipeline(mgr, "test-agent", POLICY)
            .add_step("search", search, query="data")
            .add_condition(
                lambda output: False,
                then_step=("action", action),
                else_step=None,
            )
            .run()
        )
        assert result.completed is True
        assert result.final_output == "results for data"  # unchanged

    @pytest.mark.asyncio
    async def test_condition_predicate_error(self):
        store = MemoryStore()
        mgr = SessionManager(store)

        async def dummy(input: str = "") -> str:
            return "x"

        result = await (
            Pipeline(mgr, "test-agent", POLICY)
            .add_step("search", search, query="test")
            .add_condition(
                lambda output: 1 / 0,  # will raise
                then_step=("dummy", dummy),
            )
            .run()
        )
        assert result.completed is False
        assert "predicate failed" in result.error.lower()


class TestPipelineThreshold:
    @pytest.mark.asyncio
    async def test_above_threshold(self):
        async def get_data() -> dict:
            return {"rows": [{"cnt": 15}]}

        async def escalate(input: Any = None) -> str:
            return "escalated"

        async def skip(input: Any = None) -> str:
            return "skipped"

        store = MemoryStore()
        mgr = SessionManager(store)
        result = await (
            Pipeline(mgr, "test-agent", POLICY)
            .add_step("check", get_data, chain_input=False)
            .add_threshold(
                key="rows.0.cnt",
                threshold=10,
                above_step=("escalate", escalate),
                below_step=("skip", skip),
            )
            .run()
        )
        assert result.completed is True
        assert result.final_output == "escalated"

    @pytest.mark.asyncio
    async def test_below_threshold(self):
        async def get_data() -> dict:
            return {"rows": [{"cnt": 3}]}

        async def escalate(input: Any = None) -> str:
            return "escalated"

        async def skip(input: Any = None) -> str:
            return "all clear"

        store = MemoryStore()
        mgr = SessionManager(store)
        result = await (
            Pipeline(mgr, "test-agent", POLICY)
            .add_step("check", get_data, chain_input=False)
            .add_threshold(
                key="rows.0.cnt",
                threshold=10,
                above_step=("escalate", escalate),
                below_step=("skip", skip),
            )
            .run()
        )
        assert result.completed is True
        assert result.final_output == "all clear"

    @pytest.mark.asyncio
    async def test_threshold_below_no_else(self):
        """Below threshold with no below_step — skip."""

        async def get_data() -> dict:
            return {"rows": [{"cnt": 3}]}

        async def escalate(input: Any = None) -> str:
            return "escalated"

        store = MemoryStore()
        mgr = SessionManager(store)
        result = await (
            Pipeline(mgr, "test-agent", POLICY)
            .add_step("check", get_data, chain_input=False)
            .add_threshold(
                key="rows.0.cnt",
                threshold=10,
                above_step=("escalate", escalate),
            )
            .run()
        )
        assert result.completed is True
        # Previous output preserved since threshold wasn't met
        assert result.final_output == {"rows": [{"cnt": 3}]}

    @pytest.mark.asyncio
    async def test_threshold_bad_key(self):
        """Invalid key path returns False (takes else branch)."""

        async def get_data() -> dict:
            return {"something": "else"}

        async def above(input: Any = None) -> str:
            return "above"

        async def below(input: Any = None) -> str:
            return "below"

        store = MemoryStore()
        mgr = SessionManager(store)
        result = await (
            Pipeline(mgr, "test-agent", POLICY)
            .add_step("check", get_data, chain_input=False)
            .add_threshold(
                key="rows.0.cnt",
                threshold=10,
                above_step=("above", above),
                below_step=("below", below),
            )
            .run()
        )
        assert result.completed is True
        assert result.final_output == "below"
