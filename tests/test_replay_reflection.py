"""Tests for session replay and reflection loop."""

import json

import pytest

from clawboss.reflection import ReflectionLoop
from clawboss.replay import SessionReplay
from clawboss.session import SessionManager
from clawboss.store import MemoryStore

POLICY = {"max_iterations": 50, "tool_timeout": 10, "token_budget": 100000}


# ---------------------------------------------------------------------------
# Session replay
# ---------------------------------------------------------------------------


class TestSessionReplay:
    @pytest.mark.asyncio
    async def test_replay_empty_session(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("test", POLICY)
        mgr.stop(sid)

        replay = SessionReplay(mgr, sid)
        assert replay.exists is True
        assert len(replay.frames()) >= 1  # at least request_start

    @pytest.mark.asyncio
    async def test_replay_nonexistent_session(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        replay = SessionReplay(mgr, "nonexistent")
        assert replay.exists is False

    @pytest.mark.asyncio
    async def test_replay_records_tool_calls(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("test", POLICY)
        sv = mgr.get_supervisor(sid)

        async def tool(x=""):
            return f"result: {x}"

        await sv.call("search", tool, x="python")
        await sv.call("search", tool, x="rust")
        mgr.stop(sid)

        replay = SessionReplay(mgr, sid)
        tool_call_frames = replay.filter(phase="tool_call", outcome="allowed")
        assert len(tool_call_frames) >= 2

    @pytest.mark.asyncio
    async def test_replay_summary(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("researcher", POLICY)
        sv = mgr.get_supervisor(sid)

        async def search(q=""):
            return f"results for {q}"

        async def fetch(url=""):
            return f"content from {url}"

        await sv.call("search", search, q="test")
        await sv.call("fetch", fetch, url="https://x.com")
        await sv.call("search", search, q="again")
        mgr.stop(sid)

        replay = SessionReplay(mgr, sid)
        summary = replay.summary()
        assert summary.agent_id == "researcher"
        assert summary.status == "stopped"
        assert summary.total_tool_calls >= 3
        assert "search" in summary.unique_tools
        assert "fetch" in summary.unique_tools

    @pytest.mark.asyncio
    async def test_replay_state_at_frame(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("test", POLICY)
        sv = mgr.get_supervisor(sid)

        async def tool():
            return "ok"

        sv.record_iteration()
        await sv.call("search", tool)
        sv.record_iteration()
        await sv.call("search", tool)
        mgr.stop(sid)

        replay = SessionReplay(mgr, sid)
        # State at end should have iterations counted
        last_state = replay.state_at(len(replay.frames()) - 1)
        assert last_state["iterations"] >= 2

    @pytest.mark.asyncio
    async def test_replay_to_timeline(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("test", POLICY)
        sv = mgr.get_supervisor(sid)

        async def tool():
            return "ok"

        await sv.call("search", tool)
        mgr.stop(sid)

        replay = SessionReplay(mgr, sid)
        timeline = replay.to_timeline()
        assert isinstance(timeline, list)
        assert len(timeline) > 0
        assert "phase" in timeline[0]
        assert "timestamp" in timeline[0]


# ---------------------------------------------------------------------------
# ReflectionLoop
# ---------------------------------------------------------------------------


class TestReflectionLoop:
    @pytest.mark.asyncio
    async def test_immediate_done(self):
        """LLM says done on the first think phase — loop exits cleanly."""
        store = MemoryStore()
        mgr = SessionManager(store)

        async def llm(prompt):
            return json.dumps(
                {
                    "thought": "already done",
                    "tool": None,
                    "done": True,
                    "final_answer": "42",
                }
            )

        loop = ReflectionLoop(
            manager=mgr,
            agent_id="test",
            goal="Find the answer",
            llm=llm,
            tools={},
            policy_dict=POLICY,
        )
        result = await loop.run(max_cycles=5)
        assert result.completed is True
        assert result.final_answer == "42"
        assert result.cycles_used == 1
        assert result.stopped_reason == "goal_achieved"

    @pytest.mark.asyncio
    async def test_single_cycle_with_tool(self):
        """One cycle: think → act → observe → reflect → done."""
        store = MemoryStore()
        mgr = SessionManager(store)
        call_count = [0]

        async def llm(prompt):
            call_count[0] += 1
            if "Reflect on this cycle" in prompt:
                return json.dumps(
                    {
                        "reflection": "progress made",
                        "goal_progress": 0.5,
                        "should_stop": False,
                    }
                )
            if "What should you do next" in prompt:
                if "Cycle 1" in prompt:
                    # Second think — done
                    return json.dumps(
                        {
                            "thought": "done",
                            "tool": None,
                            "done": True,
                            "final_answer": "result",
                        }
                    )
                return json.dumps(
                    {
                        "thought": "call the tool",
                        "tool": "search",
                        "args": {"q": "x"},
                        "done": False,
                    }
                )
            # Observe phase — returns plain text
            return "tool returned something"

        async def search(q=""):
            return {"results": [q]}

        loop = ReflectionLoop(
            manager=mgr,
            agent_id="test",
            goal="Test",
            llm=llm,
            tools={"search": search},
            policy_dict=POLICY,
        )
        result = await loop.run(max_cycles=5)
        assert result.completed is True
        assert result.cycles_used == 2
        # Cycle 1 called the tool
        assert result.cycles[0].tool_called == "search"
        assert result.cycles[0].tool_succeeded is True
        assert result.cycles[0].reflection == "progress made"

    @pytest.mark.asyncio
    async def test_tool_error_continues(self):
        """Tool failure doesn't stop the loop — agent reflects and continues."""
        store = MemoryStore()
        mgr = SessionManager(store)

        async def llm(prompt):
            if "What should you do next" in prompt:
                if "Cycle 1" in prompt or "Cycle 2" in prompt:
                    return json.dumps(
                        {
                            "thought": "done",
                            "tool": None,
                            "done": True,
                            "final_answer": "ok",
                        }
                    )
                return json.dumps(
                    {
                        "thought": "try",
                        "tool": "broken",
                        "args": {},
                        "done": False,
                    }
                )
            return "{}"

        async def broken():
            raise RuntimeError("boom")

        loop = ReflectionLoop(
            manager=mgr,
            agent_id="test",
            goal="Test",
            llm=llm,
            tools={"broken": broken},
            policy_dict=POLICY,
        )
        result = await loop.run(max_cycles=5)
        # First cycle fails, loop continues, second cycle stops
        assert len(result.cycles) >= 1
        assert result.cycles[0].tool_succeeded is False
        assert result.cycles[0].error is not None

    @pytest.mark.asyncio
    async def test_max_cycles_respected(self):
        """Loop stops at max_cycles even if LLM never signals done."""
        store = MemoryStore()
        mgr = SessionManager(store)

        async def llm(prompt):
            if "What should you do next" in prompt:
                return json.dumps(
                    {
                        "thought": "keep going",
                        "tool": "noop",
                        "args": {},
                        "done": False,
                    }
                )
            if "Reflect on this cycle" in prompt:
                return json.dumps(
                    {
                        "reflection": "more needed",
                        "goal_progress": 0.1,
                        "should_stop": False,
                    }
                )
            return "nothing"

        async def noop():
            return "ok"

        loop = ReflectionLoop(
            manager=mgr,
            agent_id="test",
            goal="Infinite",
            llm=llm,
            tools={"noop": noop},
            policy_dict={"max_iterations": 1000, "tool_timeout": 10},
        )
        result = await loop.run(max_cycles=3)
        assert result.cycles_used == 3
        assert result.stopped_reason == "max_cycles_reached"

    @pytest.mark.asyncio
    async def test_unknown_tool_stops_loop(self):
        store = MemoryStore()
        mgr = SessionManager(store)

        async def llm(prompt):
            return json.dumps(
                {
                    "thought": "call something that doesn't exist",
                    "tool": "nonexistent",
                    "args": {},
                    "done": False,
                }
            )

        loop = ReflectionLoop(
            manager=mgr,
            agent_id="test",
            goal="Test",
            llm=llm,
            tools={"real_tool": lambda: None},
            policy_dict=POLICY,
        )
        result = await loop.run(max_cycles=5)
        assert result.completed is False
        assert result.stopped_reason == "unknown_tool"

    @pytest.mark.asyncio
    async def test_reflection_history_in_prompt(self):
        """Later cycles should see earlier cycles in the history."""
        store = MemoryStore()
        mgr = SessionManager(store)
        received_prompts = []

        async def llm(prompt):
            received_prompts.append(prompt)
            if "What should you do next" in prompt:
                if len([p for p in received_prompts if "What should" in p]) >= 2:
                    return json.dumps(
                        {"thought": "done", "tool": None, "done": True, "final_answer": "done"}
                    )
                return json.dumps(
                    {
                        "thought": "first action",
                        "tool": "search",
                        "args": {"q": "a"},
                        "done": False,
                    }
                )
            if "Reflect on this cycle" in prompt:
                return json.dumps(
                    {"reflection": "good start", "goal_progress": 0.5, "should_stop": False}
                )
            return "observed"

        async def search(q=""):
            return q

        loop = ReflectionLoop(
            manager=mgr,
            agent_id="test",
            goal="Test",
            llm=llm,
            tools={"search": search},
            policy_dict=POLICY,
        )
        await loop.run(max_cycles=5)

        # The second think prompt should reference cycle 1
        think_prompts = [p for p in received_prompts if "What should" in p]
        assert len(think_prompts) >= 2
        assert "Cycle 1" in think_prompts[1] or "first action" in think_prompts[1]
