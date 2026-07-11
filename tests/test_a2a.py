"""Tests for agenthandler.a2a — Agent-to-Agent protocol support."""

from __future__ import annotations

import pytest

from agenthandler.a2a import (
    A2AAgentCard,
    A2AAuthentication,
    A2AClient,
    A2ASkill,
    A2ASupervisedEndpoint,
    A2ATask,
    TaskState,
)
from agenthandler.policy import Policy
from agenthandler.session import SessionManager
from agenthandler.store import MemoryStore
from agenthandler.supervisor import Supervisor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POLICY = {"max_iterations": 10, "tool_timeout": 10, "token_budget": 50000}


async def echo_tool(input: str = "") -> dict:
    return {"echo": input}


# ---------------------------------------------------------------------------
# A2AAgentCard serialization
# ---------------------------------------------------------------------------


class TestA2AAgentCard:
    def test_roundtrip_minimal(self):
        card = A2AAgentCard(name="test-agent")
        d = card.to_dict()
        restored = A2AAgentCard.from_dict(d)
        assert restored.name == "test-agent"
        assert restored.url == ""
        assert restored.skills == []

    def test_roundtrip_full(self):
        card = A2AAgentCard(
            name="research-agent",
            description="Researches topics",
            url="https://agent.example.com",
            version="2.0",
            skills=[
                A2ASkill(
                    name="research",
                    description="Deep research",
                    input_schema={"type": "object", "properties": {"topic": {"type": "string"}}},
                    output_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
                ),
                A2ASkill(name="summarize"),
            ],
            authentication=A2AAuthentication(
                schemes=["bearer"], credentials="https://auth.example.com"
            ),
            metadata={"tier": "premium"},
        )
        d = card.to_dict()
        restored = A2AAgentCard.from_dict(d)

        assert restored.name == "research-agent"
        assert restored.description == "Researches topics"
        assert restored.url == "https://agent.example.com"
        assert restored.version == "2.0"
        assert len(restored.skills) == 2
        assert restored.skills[0].name == "research"
        assert restored.skills[0].input_schema is not None
        assert restored.skills[0].output_schema is not None
        assert restored.skills[1].name == "summarize"
        assert restored.authentication is not None
        assert restored.authentication.schemes == ["bearer"]
        assert restored.metadata == {"tier": "premium"}

    def test_from_supervisor(self):
        policy = Policy(max_iterations=20, tool_timeout=15, token_budget=25000)
        sv = Supervisor(policy)

        card = A2AAgentCard.from_supervisor(
            sv,
            name="coding-agent",
            url="https://code.example.com",
            description="Writes code",
            tool_names=["write_code", "review_code"],
        )

        assert card.name == "coding-agent"
        assert card.url == "https://code.example.com"
        assert len(card.skills) == 2
        assert card.skills[0].name == "write_code"
        assert card.skills[1].name == "review_code"
        assert card.metadata["supervision"]["max_iterations"] == 20
        assert card.metadata["supervision"]["tool_timeout"] == 15
        assert card.metadata["supervision"]["token_budget"] == 25000

        sv.finish()

    def test_from_supervisor_no_tools(self):
        sv = Supervisor(Policy())
        card = A2AAgentCard.from_supervisor(sv, name="bare-agent")
        assert len(card.skills) == 1
        assert card.skills[0].name == "bare-agent"
        sv.finish()


# ---------------------------------------------------------------------------
# A2ATask serialization
# ---------------------------------------------------------------------------


class TestA2ATask:
    def test_roundtrip(self):
        task = A2ATask(
            id="task-123",
            state=TaskState.COMPLETED,
            skill="research",
            input={"topic": "AI"},
            output={"summary": "..."},
            metadata={"source": "test"},
        )
        d = task.to_dict()
        restored = A2ATask.from_dict(d)
        assert restored.id == "task-123"
        assert restored.state == TaskState.COMPLETED
        assert restored.skill == "research"
        assert restored.input == {"topic": "AI"}
        assert restored.output == {"summary": "..."}

    def test_task_states(self):
        for state in TaskState:
            task = A2ATask(id="t", state=state)
            d = task.to_dict()
            assert d["state"] == state.value
            restored = A2ATask.from_dict(d)
            assert restored.state == state


# ---------------------------------------------------------------------------
# A2ASkill and A2AAuthentication
# ---------------------------------------------------------------------------


class TestA2ASkill:
    def test_roundtrip(self):
        skill = A2ASkill(
            name="search",
            description="Search the web",
            input_schema={"type": "object"},
        )
        d = skill.to_dict()
        restored = A2ASkill.from_dict(d)
        assert restored.name == "search"
        assert restored.description == "Search the web"
        assert restored.input_schema == {"type": "object"}


class TestA2AAuthentication:
    def test_roundtrip(self):
        auth = A2AAuthentication(
            schemes=["bearer", "apiKey"], credentials="https://auth.example.com"
        )
        d = auth.to_dict()
        restored = A2AAuthentication.from_dict(d)
        assert restored.schemes == ["bearer", "apiKey"]
        assert restored.credentials == "https://auth.example.com"


# ---------------------------------------------------------------------------
# A2AClient — calls go through supervisor
# ---------------------------------------------------------------------------


class TestA2AClient:
    def test_requires_httpx(self, monkeypatch):
        import agenthandler.a2a as a2a_mod

        original = a2a_mod.httpx
        monkeypatch.setattr(a2a_mod, "httpx", None)
        try:
            sv = Supervisor(Policy())
            with pytest.raises(ImportError, match="httpx"):
                A2AClient(supervisor=sv)
            sv.finish()
        finally:
            monkeypatch.setattr(a2a_mod, "httpx", original)

    @pytest.mark.asyncio
    async def test_send_task_goes_through_supervisor(self):
        """Verify that send_task routes through supervisor.call as 'a2a_call'."""
        policy = Policy(max_iterations=10, tool_timeout=10)
        sv = Supervisor(policy)

        calls_made: list = []
        original_call = sv.call

        async def tracking_call(tool_name, fn, **kwargs):
            calls_made.append(tool_name)
            return await original_call(tool_name, fn, **kwargs)

        sv.call = tracking_call  # type: ignore[method-assign]

        client = A2AClient(supervisor=sv, timeout=5)

        # The HTTP call will fail (no server), but we verify the supervisor
        # was invoked with "a2a_call"
        result = await client.send_task(
            agent_url="http://localhost:99999",
            task={"skill": "test", "input": "hello"},
        )

        assert "a2a_call" in calls_made
        assert result.tool_name == "a2a_call"
        sv.finish()

    @pytest.mark.asyncio
    async def test_send_task_with_circuit_breaker(self):
        """A2A calls should trip the circuit breaker after repeated failures."""
        policy = Policy(circuit_breaker_threshold=2, tool_timeout=2)
        sv = Supervisor(policy)
        client = A2AClient(supervisor=sv, timeout=1)

        for _ in range(3):
            await client.send_task(
                agent_url="http://localhost:99999",
                task={"skill": "test", "input": "fail"},
            )

        states = sv.circuit_breaker_states()
        assert states.get("a2a_call") == "open"
        sv.finish()


# ---------------------------------------------------------------------------
# A2ASupervisedEndpoint
# ---------------------------------------------------------------------------


class TestA2ASupervisedEndpoint:
    def test_generates_card_from_supervisor(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("test-agent", POLICY)

        endpoint = A2ASupervisedEndpoint(
            manager=mgr,
            session_id=sid,
            name="my-agent",
            description="Test agent",
            url="https://test.example.com",
        )

        card = endpoint.agent_card
        assert card.name == "my-agent"
        assert card.url == "https://test.example.com"
        assert card.description == "Test agent"
        assert len(card.skills) >= 1
        assert card.metadata["supervision"]["max_iterations"] == POLICY["max_iterations"]

        mgr.stop(sid)

    def test_uses_provided_card(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("test-agent", POLICY)

        custom_card = A2AAgentCard(
            name="custom",
            url="https://custom.example.com",
            skills=[A2ASkill(name="custom-skill")],
        )
        endpoint = A2ASupervisedEndpoint(
            manager=mgr,
            session_id=sid,
            agent_card=custom_card,
        )

        assert endpoint.agent_card.name == "custom"
        assert endpoint.agent_card.skills[0].name == "custom-skill"
        mgr.stop(sid)

    @pytest.mark.asyncio
    async def test_handle_send_with_tool_router(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("test-agent", POLICY)

        def route(skill: str):
            if skill == "echo":
                return echo_tool
            return None

        endpoint = A2ASupervisedEndpoint(
            manager=mgr,
            session_id=sid,
            tool_router=route,
            name="echo-agent",
        )

        response = await endpoint.handle_send(
            {
                "jsonrpc": "2.0",
                "id": "req-1",
                "method": "tasks/send",
                "params": {
                    "id": "task-1",
                    "skill": "echo",
                    "input": {"input": "hello world"},
                },
            }
        )

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "req-1"
        result = response["result"]
        assert result["state"] == "completed"
        assert result["output"] == {"echo": "hello world"}

        mgr.stop(sid)

    @pytest.mark.asyncio
    async def test_handle_send_without_router(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("test-agent", POLICY)

        endpoint = A2ASupervisedEndpoint(manager=mgr, session_id=sid, name="bare-agent")

        response = await endpoint.handle_send(
            {
                "jsonrpc": "2.0",
                "id": "req-1",
                "params": {"id": "task-1", "skill": "anything"},
            }
        )

        assert response["result"]["state"] == "completed"
        mgr.stop(sid)

    @pytest.mark.asyncio
    async def test_handle_get_known_task(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("test-agent", POLICY)

        endpoint = A2ASupervisedEndpoint(manager=mgr, session_id=sid, name="agent")

        await endpoint.handle_send(
            {
                "jsonrpc": "2.0",
                "id": "r1",
                "params": {"id": "task-42", "skill": "test"},
            }
        )

        response = await endpoint.handle_get(
            {
                "jsonrpc": "2.0",
                "id": "r2",
                "params": {"id": "task-42"},
            }
        )

        assert "result" in response
        assert response["result"]["id"] == "task-42"
        mgr.stop(sid)

    @pytest.mark.asyncio
    async def test_handle_get_unknown_task(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("test-agent", POLICY)

        endpoint = A2ASupervisedEndpoint(manager=mgr, session_id=sid, name="agent")

        response = await endpoint.handle_get(
            {
                "jsonrpc": "2.0",
                "id": "r1",
                "params": {"id": "nonexistent"},
            }
        )

        assert "error" in response
        assert response["error"]["code"] == -32001
        mgr.stop(sid)

    @pytest.mark.asyncio
    async def test_handle_cancel(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("test-agent", POLICY)

        endpoint = A2ASupervisedEndpoint(manager=mgr, session_id=sid, name="agent")

        await endpoint.handle_send(
            {
                "jsonrpc": "2.0",
                "id": "r1",
                "params": {"id": "task-99", "skill": "test"},
            }
        )

        response = await endpoint.handle_cancel(
            {
                "jsonrpc": "2.0",
                "id": "r2",
                "params": {"id": "task-99"},
            }
        )

        assert response["result"]["state"] == "canceled"

        get_response = await endpoint.handle_get(
            {
                "jsonrpc": "2.0",
                "id": "r3",
                "params": {"id": "task-99"},
            }
        )
        assert get_response["result"]["state"] == "canceled"
        mgr.stop(sid)

    @pytest.mark.asyncio
    async def test_handle_send_failed_tool(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("test-agent", POLICY)

        async def broken_tool(**kwargs):
            raise ValueError("tool is broken")

        endpoint = A2ASupervisedEndpoint(
            manager=mgr,
            session_id=sid,
            tool_router=lambda skill: broken_tool if skill == "broken" else None,
            name="agent",
        )

        response = await endpoint.handle_send(
            {
                "jsonrpc": "2.0",
                "id": "r1",
                "params": {"id": "task-fail", "skill": "broken", "input": {}},
            }
        )

        assert response["result"]["state"] == "failed"
        assert response["result"]["error"] is not None
        mgr.stop(sid)
