"""Tests for clawboss.pipeline_poml — POML parsing and PipelineBuilder."""

import pytest

from clawboss.pipeline_poml import PipelineBuilder, _make_predicate, parse_pipeline_poml
from clawboss.session import SessionManager
from clawboss.store import MemoryStore

POLICY = {"max_iterations": 10, "tool_timeout": 10}


# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


async def search(query: str = "", input: str = "") -> str:
    return f"results for {query or input}"


async def summarize(input: str = "") -> str:
    return f"summary of: {input}"


async def escalate(input: str = "", description: str = "") -> str:
    return "ESCALATED"


async def log_ok(input: str = "", description: str = "") -> str:
    return "all clear"


async def handle_error(input: str = "") -> str:
    return "error handled"


async def sql_query(sql: str = "", input: str = "") -> dict:
    """Execute a SQL query."""
    if "count" in sql.lower():
        return {"rows": [{"cnt": 15}], "row_count": 1, "columns": ["cnt"]}
    return {"rows": [{"id": 1, "name": "test"}], "row_count": 1, "columns": ["id", "name"]}


TOOLS = {
    "search": search,
    "summarize": summarize,
    "escalate": escalate,
    "log_ok": log_ok,
    "handle_error": handle_error,
    "sql.query": sql_query,
}


# ---------------------------------------------------------------------------
# POML parsing — steps
# ---------------------------------------------------------------------------


class TestParseSteps:
    @pytest.mark.asyncio
    async def test_single_step(self):
        poml = """
        <pipeline>
          <step tool="search">find quantum computing papers</step>
        </pipeline>
        """
        store = MemoryStore()
        mgr = SessionManager(store)
        pipeline = parse_pipeline_poml(poml, TOOLS, mgr, "test", POLICY)
        result = await pipeline.run()
        assert result.completed is True
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_multi_step_chain(self):
        poml = """
        <pipeline>
          <step tool="search">quantum computing</step>
          <step tool="summarize">summarize findings</step>
        </pipeline>
        """
        store = MemoryStore()
        mgr = SessionManager(store)
        pipeline = parse_pipeline_poml(poml, TOOLS, mgr, "test", POLICY)
        result = await pipeline.run()
        assert result.completed is True
        assert len(result.steps) == 2
        assert "summary of:" in result.final_output

    @pytest.mark.asyncio
    async def test_sql_step(self):
        poml = """
        <pipeline>
          <step tool="sql.query">SELECT count(*) as cnt FROM alerts</step>
        </pipeline>
        """
        store = MemoryStore()
        mgr = SessionManager(store)
        pipeline = parse_pipeline_poml(poml, TOOLS, mgr, "test", POLICY)
        result = await pipeline.run()
        assert result.completed is True
        assert result.final_output["rows"][0]["cnt"] == 15


# ---------------------------------------------------------------------------
# POML parsing — threshold
# ---------------------------------------------------------------------------


class TestParseThreshold:
    @pytest.mark.asyncio
    async def test_threshold_above(self):
        poml = """
        <pipeline>
          <step tool="sql.query">SELECT count(*) as cnt FROM alerts</step>
          <threshold key="rows.0.cnt" value="10">
            <above tool="escalate">Alert the team</above>
            <below tool="log_ok">Everything fine</below>
          </threshold>
        </pipeline>
        """
        store = MemoryStore()
        mgr = SessionManager(store)
        pipeline = parse_pipeline_poml(poml, TOOLS, mgr, "test", POLICY)
        result = await pipeline.run()
        assert result.completed is True
        assert result.final_output == "ESCALATED"

    @pytest.mark.asyncio
    async def test_threshold_no_below(self):
        """Threshold with no <below> — skip when below."""
        poml = """
        <pipeline>
          <step tool="sql.query">SELECT count(*) as cnt FROM alerts</step>
          <threshold key="rows.0.cnt" value="100">
            <above tool="escalate">Alert</above>
          </threshold>
        </pipeline>
        """
        store = MemoryStore()
        mgr = SessionManager(store)
        pipeline = parse_pipeline_poml(poml, TOOLS, mgr, "test", POLICY)
        result = await pipeline.run()
        assert result.completed is True
        # 15 < 100, no below step, previous output preserved
        assert result.final_output["rows"][0]["cnt"] == 15


# ---------------------------------------------------------------------------
# POML parsing — condition
# ---------------------------------------------------------------------------


class TestParseCondition:
    @pytest.mark.asyncio
    async def test_condition_true(self):
        poml = """
        <pipeline>
          <step tool="search">test query</step>
          <condition if="'results' in output">
            <then tool="summarize">summarize it</then>
            <else tool="handle_error">handle the error</else>
          </condition>
        </pipeline>
        """
        store = MemoryStore()
        mgr = SessionManager(store)
        pipeline = parse_pipeline_poml(poml, TOOLS, mgr, "test", POLICY)
        result = await pipeline.run()
        assert result.completed is True
        assert "summary of:" in result.final_output

    @pytest.mark.asyncio
    async def test_condition_false(self):
        poml = """
        <pipeline>
          <step tool="search">test</step>
          <condition if="'NOTFOUND' in output">
            <then tool="escalate">escalate</then>
            <else tool="log_ok">all good</else>
          </condition>
        </pipeline>
        """
        store = MemoryStore()
        mgr = SessionManager(store)
        pipeline = parse_pipeline_poml(poml, TOOLS, mgr, "test", POLICY)
        result = await pipeline.run()
        assert result.completed is True
        assert result.final_output == "all clear"


# ---------------------------------------------------------------------------
# POML parsing — errors
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_no_pipeline_block(self):
        with pytest.raises(ValueError, match="No <pipeline>"):
            store = MemoryStore()
            mgr = SessionManager(store)
            parse_pipeline_poml("<step tool='x'>y</step>", TOOLS, mgr, "t", POLICY)

    def test_missing_tool_attribute(self):
        with pytest.raises(ValueError, match="requires a 'tool'"):
            store = MemoryStore()
            mgr = SessionManager(store)
            parse_pipeline_poml(
                "<pipeline><step>no tool</step></pipeline>", TOOLS, mgr, "t", POLICY
            )

    def test_unknown_tool(self):
        with pytest.raises(ValueError, match="not found in registry"):
            store = MemoryStore()
            mgr = SessionManager(store)
            parse_pipeline_poml(
                '<pipeline><step tool="nonexistent">x</step></pipeline>',
                TOOLS,
                mgr,
                "t",
                POLICY,
            )

    def test_threshold_missing_key(self):
        with pytest.raises(ValueError, match="requires a 'key'"):
            store = MemoryStore()
            mgr = SessionManager(store)
            poml = (
                '<pipeline><threshold value="10">'
                '<above tool="escalate">x</above>'
                "</threshold></pipeline>"
            )
            parse_pipeline_poml(
                poml,
                TOOLS,
                mgr,
                "t",
                POLICY,
            )


# ---------------------------------------------------------------------------
# Full POML document with metadata
# ---------------------------------------------------------------------------


class TestFullPomlDocument:
    @pytest.mark.asyncio
    async def test_full_document_with_metadata(self):
        poml = """
        <!--
        metadata:
          name: alert-monitor
          triggers: check alerts, monitor
          version: 1.0
        -->

        <task>
        Check critical alerts. If more than 10, escalate. Otherwise log.
        </task>

        <pipeline>
          <step tool="sql.query">SELECT count(*) as cnt FROM alerts WHERE severity='critical'</step>
          <threshold key="rows.0.cnt" value="10">
            <above tool="escalate">Notify on-call team</above>
            <below tool="log_ok">Log all clear</below>
          </threshold>
        </pipeline>
        """
        store = MemoryStore()
        mgr = SessionManager(store)
        pipeline = parse_pipeline_poml(poml, TOOLS, mgr, "monitor", POLICY)
        result = await pipeline.run()
        assert result.completed is True
        assert result.final_output == "ESCALATED"


# ---------------------------------------------------------------------------
# PipelineBuilder — LLM integration
# ---------------------------------------------------------------------------


class TestPipelineBuilder:
    @pytest.mark.asyncio
    async def test_create_from_natural_language(self):
        """Simulate LLM returning POML from a description."""

        async def fake_llm(prompt: str) -> str:
            return """
            <!--
            metadata:
              name: search-pipeline
            -->
            <task>Search and summarize</task>
            <pipeline>
              <step tool="search">find information</step>
              <step tool="summarize">summarize findings</step>
            </pipeline>
            """

        store = MemoryStore()
        mgr = SessionManager(store)
        builder = PipelineBuilder(fake_llm, TOOLS, mgr)
        pipeline = await builder.create("Search for information and summarize it")
        result = await pipeline.run()
        assert result.completed is True
        assert "summary of:" in result.final_output

    @pytest.mark.asyncio
    async def test_create_poml_only(self):
        """create_poml returns POML text without parsing."""

        async def fake_llm(prompt: str) -> str:
            return """```xml
            <pipeline>
              <step tool="search">test</step>
            </pipeline>
            ```"""

        store = MemoryStore()
        mgr = SessionManager(store)
        builder = PipelineBuilder(fake_llm, TOOLS, mgr)
        poml = await builder.create_poml("Search for something")
        assert "<pipeline>" in poml
        assert "<step" in poml

    @pytest.mark.asyncio
    async def test_refine(self):
        """Refine an existing POML with feedback."""

        async def fake_llm(prompt: str) -> str:
            if "changes" in prompt or "feedback" in prompt:
                return """
                <pipeline>
                  <step tool="search">updated query</step>
                  <step tool="summarize">better summary</step>
                  <step tool="escalate">and escalate</step>
                </pipeline>
                """
            return "<pipeline><step tool='search'>original</step></pipeline>"

        store = MemoryStore()
        mgr = SessionManager(store)
        builder = PipelineBuilder(fake_llm, TOOLS, mgr)
        refined = await builder.refine(
            "<pipeline><step tool='search'>original</step></pipeline>",
            "Add a summarize step and then escalate",
        )
        assert "summarize" in refined
        assert "escalate" in refined

    @pytest.mark.asyncio
    async def test_create_with_threshold(self):
        """LLM generates a pipeline with threshold branching."""

        async def fake_llm(prompt: str) -> str:
            return """
            <pipeline>
              <step tool="sql.query">SELECT count(*) as cnt FROM alerts</step>
              <threshold key="rows.0.cnt" value="10">
                <above tool="escalate">Escalate</above>
                <below tool="log_ok">Log ok</below>
              </threshold>
            </pipeline>
            """

        store = MemoryStore()
        mgr = SessionManager(store)
        builder = PipelineBuilder(fake_llm, TOOLS, mgr)
        pipeline = await builder.create("Check alerts and escalate if more than 10")
        result = await pipeline.run()
        assert result.completed is True
        assert result.final_output == "ESCALATED"

    @pytest.mark.asyncio
    async def test_tools_description_in_prompt(self):
        """Verify the LLM receives the tool descriptions."""
        received_prompt = []

        async def capture_llm(prompt: str) -> str:
            received_prompt.append(prompt)
            return "<pipeline><step tool='search'>x</step></pipeline>"

        store = MemoryStore()
        mgr = SessionManager(store)
        builder = PipelineBuilder(capture_llm, TOOLS, mgr)
        await builder.create_poml("test")
        assert "sql.query" in received_prompt[0]
        assert "search" in received_prompt[0]

    @pytest.mark.asyncio
    async def test_create_poml_validates_output(self):
        """create_poml rejects LLM output without <pipeline> block."""

        async def bad_llm(prompt: str) -> str:
            return "Here is your pipeline: <step tool='search'>x</step>"

        store = MemoryStore()
        mgr = SessionManager(store)
        builder = PipelineBuilder(bad_llm, TOOLS, mgr)
        with pytest.raises(ValueError, match="missing <pipeline>"):
            await builder.create_poml("test")

    @pytest.mark.asyncio
    async def test_create_poml_validates_xml(self):
        """create_poml rejects LLM output with broken XML."""

        async def bad_xml_llm(prompt: str) -> str:
            return "<pipeline><step tool='search'>unclosed</pipeline>"

        store = MemoryStore()
        mgr = SessionManager(store)
        builder = PipelineBuilder(bad_xml_llm, TOOLS, mgr)
        with pytest.raises(ValueError, match="invalid XML"):
            await builder.create_poml("test")

    @pytest.mark.asyncio
    async def test_refine_validates_output(self):
        """refine rejects LLM output without <pipeline> block."""

        async def bad_llm(prompt: str) -> str:
            return "Sorry, I can't help with that."

        store = MemoryStore()
        mgr = SessionManager(store)
        builder = PipelineBuilder(bad_llm, TOOLS, mgr)
        with pytest.raises(ValueError, match="missing <pipeline>"):
            await builder.refine("<pipeline><step tool='search'>x</step></pipeline>", "change it")


# ---------------------------------------------------------------------------
# Additional parse error coverage
# ---------------------------------------------------------------------------


class TestParseErrorsExtended:
    def test_threshold_non_numeric_value(self):
        with pytest.raises(ValueError, match="must be numeric"):
            store = MemoryStore()
            mgr = SessionManager(store)
            poml = (
                '<pipeline><threshold key="x" value="abc">'
                '<above tool="escalate">x</above>'
                "</threshold></pipeline>"
            )
            parse_pipeline_poml(poml, TOOLS, mgr, "t", POLICY)

    def test_threshold_missing_above(self):
        with pytest.raises(ValueError, match="requires an <above> child"):
            store = MemoryStore()
            mgr = SessionManager(store)
            poml = (
                '<pipeline><threshold key="x" value="10">'
                '<below tool="log_ok">x</below>'
                "</threshold></pipeline>"
            )
            parse_pipeline_poml(poml, TOOLS, mgr, "t", POLICY)

    def test_condition_missing_then(self):
        with pytest.raises(ValueError, match="requires a <then> child"):
            store = MemoryStore()
            mgr = SessionManager(store)
            poml = (
                "<pipeline><condition if=\"'x' in output\">"
                '<else tool="log_ok">x</else>'
                "</condition></pipeline>"
            )
            parse_pipeline_poml(poml, TOOLS, mgr, "t", POLICY)

    def test_condition_missing_if(self):
        with pytest.raises(ValueError, match="requires an 'if'"):
            store = MemoryStore()
            mgr = SessionManager(store)
            poml = '<pipeline><condition><then tool="escalate">x</then></condition></pipeline>'
            parse_pipeline_poml(poml, TOOLS, mgr, "t", POLICY)

    def test_unknown_tag(self):
        with pytest.raises(ValueError, match="Unknown pipeline tag"):
            store = MemoryStore()
            mgr = SessionManager(store)
            poml = '<pipeline><loop tool="search">x</loop></pipeline>'
            parse_pipeline_poml(poml, TOOLS, mgr, "t", POLICY)

    def test_invalid_xml(self):
        with pytest.raises(ValueError, match="Invalid XML"):
            store = MemoryStore()
            mgr = SessionManager(store)
            poml = "<pipeline><step tool='search'>unclosed</pipeline>"
            parse_pipeline_poml(poml, TOOLS, mgr, "t", POLICY)

    def test_threshold_child_invalid_tool(self):
        with pytest.raises(ValueError, match="requires a valid 'tool'"):
            store = MemoryStore()
            mgr = SessionManager(store)
            poml = (
                '<pipeline><threshold key="x" value="10">'
                '<above tool="nonexistent">x</above>'
                "</threshold></pipeline>"
            )
            parse_pipeline_poml(poml, TOOLS, mgr, "t", POLICY)

    def test_condition_child_invalid_tool(self):
        with pytest.raises(ValueError, match="requires a valid 'tool'"):
            store = MemoryStore()
            mgr = SessionManager(store)
            poml = (
                "<pipeline><condition if=\"'x' in output\">"
                '<then tool="nonexistent">x</then>'
                "</condition></pipeline>"
            )
            parse_pipeline_poml(poml, TOOLS, mgr, "t", POLICY)


# ---------------------------------------------------------------------------
# Expression evaluator safety
# ---------------------------------------------------------------------------


class TestExpressionEvaluator:
    def test_containment_true(self):
        pred = _make_predicate("'error' in output")
        assert pred("some error here") is True

    def test_containment_false(self):
        pred = _make_predicate("'error' in output")
        assert pred("all good") is False

    def test_equality(self):
        pred = _make_predicate("output == 'done'")
        assert pred("done") is True
        assert pred("not done") is False

    def test_not_equal(self):
        pred = _make_predicate("output != 'done'")
        assert pred("other") is True
        assert pred("done") is False

    def test_numeric_comparison(self):
        pred = _make_predicate("output > 10")
        assert pred(15) is True
        assert pred(5) is False

    def test_dot_notation(self):
        pred = _make_predicate("'error' in output")
        assert pred("error found") is True

    def test_not_in(self):
        pred = _make_predicate("'fail' not in output")
        assert pred("success") is True
        assert pred("failure") is False

    def test_rejects_function_calls(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _make_predicate("len(output) > 5")

    def test_rejects_attribute_access_exploits(self):
        with pytest.raises(ValueError, match="Unsupported"):
            _make_predicate("output.__class__.__bases__[0].__subclasses__()")

    def test_rejects_import(self):
        with pytest.raises(ValueError, match="Unsupported|Invalid"):
            _make_predicate("__import__('os').system('echo pwned')")

    def test_rejects_lambda(self):
        with pytest.raises(ValueError, match="Unsupported|Invalid"):
            _make_predicate("(lambda: None)()")

    def test_rejects_comprehension(self):
        with pytest.raises(ValueError, match="Unsupported|Invalid"):
            _make_predicate("[x for x in output]")

    def test_invalid_syntax(self):
        with pytest.raises(ValueError, match="Invalid condition"):
            _make_predicate("if output then true")

    def test_predicate_returns_false_on_runtime_error(self):
        pred = _make_predicate("output > 10")
        assert pred("not a number") is False

    def test_bool_and(self):
        pred = _make_predicate("output > 5 and output < 15")
        assert pred(10) is True
        assert pred(20) is False

    def test_bool_or(self):
        pred = _make_predicate("output == 'a' or output == 'b'")
        assert pred("a") is True
        assert pred("c") is False

    def test_not_operator(self):
        pred = _make_predicate("not output == 'bad'")
        assert pred("good") is True
        assert pred("bad") is False
