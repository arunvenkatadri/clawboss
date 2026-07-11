"""Tests for agenthandler.sdk_adapters — SDK integration wrappers."""

import pytest

from agenthandler.errors import AgentHandlerError
from agenthandler.guardrails.types import GuardrailResult
from agenthandler.policy import Policy
from agenthandler.sdk_adapters import (
    AgentHandlerMiddleware,
    openai_guardrail_adapter,
    supervised_tool_registry,
    wrap_claude_tool,
    wrap_openai_tool,
)
from agenthandler.session import SessionManager
from agenthandler.store import MemoryStore
from agenthandler.supervisor import Supervisor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def async_search(query: str = "") -> dict:
    return {"results": [f"Result for '{query}'"], "count": 1}


def sync_search(query: str = "") -> dict:
    return {"results": [f"Sync result for '{query}'"], "count": 1}


async def slow_tool(delay: float = 100.0) -> str:
    import asyncio

    await asyncio.sleep(delay)
    return "done"


async def failing_tool() -> str:
    raise ValueError("something broke")


POLICY = {"max_iterations": 10, "tool_timeout": 5, "token_budget": 50000}


# ---------------------------------------------------------------------------
# wrap_openai_tool
# ---------------------------------------------------------------------------


class TestWrapOpenaiTool:
    @pytest.mark.asyncio
    async def test_wraps_async_fn(self):
        sv = Supervisor(Policy.from_dict(POLICY))
        wrapped = wrap_openai_tool(async_search, sv, tool_name="search")
        result = await wrapped(query="python")
        assert result["count"] == 1
        assert "python" in result["results"][0]

    @pytest.mark.asyncio
    async def test_wraps_sync_fn(self):
        sv = Supervisor(Policy.from_dict(POLICY))
        wrapped = wrap_openai_tool(sync_search, sv, tool_name="search")
        result = await wrapped(query="python")
        assert result["count"] == 1
        assert "Sync result" in result["results"][0]

    @pytest.mark.asyncio
    async def test_uses_fn_name_as_default(self):
        sv = Supervisor(Policy.from_dict(POLICY))
        wrapped = wrap_openai_tool(async_search, sv)
        result = await wrapped(query="test")
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        sv = Supervisor(Policy.from_dict({"tool_timeout": 0.01}))
        wrapped = wrap_openai_tool(slow_tool, sv, tool_name="slow")
        with pytest.raises(AgentHandlerError):
            await wrapped(delay=10.0)

    @pytest.mark.asyncio
    async def test_tool_error_raises(self):
        sv = Supervisor(Policy.from_dict(POLICY))
        wrapped = wrap_openai_tool(failing_tool, sv, tool_name="bad")
        with pytest.raises(AgentHandlerError):
            await wrapped()

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips(self):
        sv = Supervisor(Policy.from_dict({**POLICY, "circuit_breaker_threshold": 2}))
        wrapped = wrap_openai_tool(failing_tool, sv, tool_name="flaky")
        for _ in range(3):
            with pytest.raises(AgentHandlerError):
                await wrapped()
        with pytest.raises(AgentHandlerError) as exc_info:
            await wrapped()
        assert "circuit" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# wrap_claude_tool
# ---------------------------------------------------------------------------


class TestWrapClaudeTool:
    @pytest.mark.asyncio
    async def test_wraps_async_fn(self):
        sv = Supervisor(Policy.from_dict(POLICY))
        wrapped = wrap_claude_tool(async_search, sv, tool_name="search")
        result = await wrapped(query="claude")
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_wraps_sync_fn(self):
        sv = Supervisor(Policy.from_dict(POLICY))
        wrapped = wrap_claude_tool(sync_search, sv)
        result = await wrapped(query="test")
        assert "Sync result" in result["results"][0]


# ---------------------------------------------------------------------------
# openai_guardrail_adapter
# ---------------------------------------------------------------------------


class SimplePreGuardrail:
    name = "test-pre"

    def check(self, tool_name: str, kwargs: dict, context: dict) -> GuardrailResult:
        if kwargs.get("blocked"):
            return GuardrailResult.block("blocked by test", self.name)
        return GuardrailResult.allow(self.name)


class SimplePostGuardrail:
    name = "test-post"

    def check(self, tool_name: str, kwargs: dict, output: object, context: dict) -> GuardrailResult:
        if output == "bad":
            return GuardrailResult.block("bad output", self.name)
        return GuardrailResult.allow(self.name)


class TestOpenaiGuardrailAdapter:
    def test_converts_pre_guardrail(self):
        adapted = openai_guardrail_adapter([SimplePreGuardrail()])
        assert len(adapted) == 1
        entry = adapted[0]
        assert isinstance(entry, dict)
        assert entry["type"] == "input"
        assert entry["name"] == "test-pre"

    def test_converts_post_guardrail(self):
        adapted = openai_guardrail_adapter([SimplePostGuardrail()])
        assert len(adapted) == 1
        entry = adapted[0]
        assert isinstance(entry, dict)
        assert entry["type"] == "output"

    def test_converts_mixed(self):
        adapted = openai_guardrail_adapter([SimplePreGuardrail(), SimplePostGuardrail()])
        assert len(adapted) == 2
        types = {e["type"] for e in adapted}
        assert types == {"input", "output"}

    @pytest.mark.asyncio
    async def test_input_guardrail_allows(self):
        adapted = openai_guardrail_adapter([SimplePreGuardrail()])
        result = await adapted[0]["check"](None, None, {"query": "ok"})
        assert result["tripwire_triggered"] is False

    @pytest.mark.asyncio
    async def test_input_guardrail_blocks(self):
        adapted = openai_guardrail_adapter([SimplePreGuardrail()])
        result = await adapted[0]["check"](None, None, {"blocked": True})
        assert result["tripwire_triggered"] is True
        assert "blocked by test" in result["output"]

    @pytest.mark.asyncio
    async def test_output_guardrail_allows(self):
        adapted = openai_guardrail_adapter([SimplePostGuardrail()])
        result = await adapted[0]["check"](None, None, "good")
        assert result["tripwire_triggered"] is False

    @pytest.mark.asyncio
    async def test_output_guardrail_blocks(self):
        adapted = openai_guardrail_adapter([SimplePostGuardrail()])
        result = await adapted[0]["check"](None, None, "bad")
        assert result["tripwire_triggered"] is True

    def test_skips_non_guardrail(self):
        adapted = openai_guardrail_adapter([object()])
        assert len(adapted) == 0


# ---------------------------------------------------------------------------
# supervised_tool_registry
# ---------------------------------------------------------------------------


class TestSupervisedToolRegistry:
    def test_creates_supervised_versions(self):
        store = MemoryStore()
        manager = SessionManager(store)
        supervised = supervised_tool_registry(
            {"search": async_search, "sync_search": sync_search},
            manager,
            policy=POLICY,
        )
        assert "search" in supervised
        assert "sync_search" in supervised
        assert callable(supervised["search"])
        assert callable(supervised["sync_search"])

    @pytest.mark.asyncio
    async def test_supervised_tools_work(self):
        store = MemoryStore()
        manager = SessionManager(store)
        supervised = supervised_tool_registry(
            {"search": async_search},
            manager,
            policy=POLICY,
        )
        result = await supervised["search"](query="test")
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_supervised_sync_tool_works(self):
        store = MemoryStore()
        manager = SessionManager(store)
        supervised = supervised_tool_registry(
            {"calc": sync_search},
            manager,
            policy=POLICY,
        )
        result = await supervised["calc"](query="hello")
        assert "Sync result" in result["results"][0]

    def test_starts_session(self):
        store = MemoryStore()
        manager = SessionManager(store)
        supervised_tool_registry({"search": async_search}, manager, policy=POLICY)
        sessions = manager.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].agent_id == "sdk-agent"

    def test_custom_agent_id(self):
        store = MemoryStore()
        manager = SessionManager(store)
        supervised_tool_registry(
            {"search": async_search},
            manager,
            policy=POLICY,
            agent_id="custom-agent",
        )
        sessions = manager.list_sessions()
        assert sessions[0].agent_id == "custom-agent"

    def test_empty_tools(self):
        store = MemoryStore()
        manager = SessionManager(store)
        supervised = supervised_tool_registry({}, manager, policy=POLICY)
        assert supervised == {}


# ---------------------------------------------------------------------------
# AgentHandlerMiddleware
# ---------------------------------------------------------------------------


class TestAgentHandlerMiddleware:
    @pytest.mark.asyncio
    async def test_runs_agent_fn(self):
        store = MemoryStore()
        manager = SessionManager(store)
        middleware = AgentHandlerMiddleware(manager, policy=POLICY)

        async def agent_fn(tools: dict, task: str = "") -> str:
            result = await tools["search"](query=task)
            return f"found {result['count']}"

        result = await middleware.run(
            agent_fn,
            tools={"search": async_search},
            task="python",
        )
        assert result == "found 1"

    @pytest.mark.asyncio
    async def test_stops_session_on_success(self):
        store = MemoryStore()
        manager = SessionManager(store)
        middleware = AgentHandlerMiddleware(manager, policy=POLICY)

        async def agent_fn(tools: dict) -> str:
            return "done"

        await middleware.run(agent_fn, tools={})
        sid = middleware.session_id
        assert sid is not None
        cp = manager.status(sid)
        assert cp.status.value == "stopped"

    @pytest.mark.asyncio
    async def test_stops_session_on_error(self):
        store = MemoryStore()
        manager = SessionManager(store)
        middleware = AgentHandlerMiddleware(manager, policy=POLICY)

        async def agent_fn(tools: dict) -> str:
            raise RuntimeError("agent crashed")

        with pytest.raises(RuntimeError, match="agent crashed"):
            await middleware.run(agent_fn, tools={})

        sid = middleware.session_id
        cp = manager.status(sid)
        assert cp.status.value == "stopped"

    @pytest.mark.asyncio
    async def test_session_id_set(self):
        store = MemoryStore()
        manager = SessionManager(store)
        middleware = AgentHandlerMiddleware(manager, policy=POLICY)

        assert middleware.session_id is None

        async def agent_fn(tools: dict) -> str:
            return "ok"

        await middleware.run(agent_fn, tools={})
        assert middleware.session_id is not None

    @pytest.mark.asyncio
    async def test_custom_agent_id(self):
        store = MemoryStore()
        manager = SessionManager(store)
        middleware = AgentHandlerMiddleware(manager, policy=POLICY, agent_id="my-agent")

        async def agent_fn(tools: dict) -> str:
            return "ok"

        await middleware.run(agent_fn, tools={})
        cp = manager.status(middleware.session_id)
        assert cp.agent_id == "my-agent"

    @pytest.mark.asyncio
    async def test_no_tools(self):
        store = MemoryStore()
        manager = SessionManager(store)
        middleware = AgentHandlerMiddleware(manager, policy=POLICY)

        async def agent_fn(tools: dict) -> str:
            assert tools == {}
            return "no tools needed"

        result = await middleware.run(agent_fn)
        assert result == "no tools needed"

    @pytest.mark.asyncio
    async def test_tools_are_supervised(self):
        store = MemoryStore()
        manager = SessionManager(store)
        middleware = AgentHandlerMiddleware(manager, policy={"tool_timeout": 0.01})

        async def agent_fn(tools: dict) -> str:
            await tools["slow"](delay=10.0)
            return "should not reach"

        with pytest.raises(AgentHandlerError):
            await middleware.run(agent_fn, tools={"slow": slow_tool})
