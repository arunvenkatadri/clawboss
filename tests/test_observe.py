"""Tests for clawboss.observe — agent observability and metrics."""

import pytest

from clawboss.observe import Observer
from clawboss.session import SessionManager
from clawboss.store import MemoryStore

# ---------------------------------------------------------------------------
# Observer unit tests
# ---------------------------------------------------------------------------


class TestObserver:
    def test_record_and_tool_summary(self):
        obs = Observer()
        obs.record_tool_call("search", duration_ms=100, succeeded=True, tokens=500)
        obs.record_tool_call("search", duration_ms=200, succeeded=True, tokens=300)
        obs.record_tool_call("search", duration_ms=150, succeeded=False, error_kind="timeout")
        summary = obs.tool_summary("search")
        assert summary["calls"] == 3
        assert summary["successes"] == 2
        assert summary["failures"] == 1
        assert summary["avg_latency_ms"] == 150.0
        assert summary["min_latency_ms"] == 100
        assert summary["max_latency_ms"] == 200
        assert summary["total_tokens"] == 800
        assert summary["error_counts"]["timeout"] == 1

    def test_session_summary(self):
        obs = Observer()
        obs.record_tool_call("search", duration_ms=100, tokens=500, session_id="s1")
        obs.record_tool_call("fetch", duration_ms=200, tokens=300, session_id="s1")
        obs.record_tool_call("search", duration_ms=50, tokens=100, session_id="s2")
        summary = obs.session_summary("s1")
        assert summary["total_calls"] == 2
        assert summary["total_tokens"] == 800
        assert "search" in summary["tools"]
        assert "fetch" in summary["tools"]

    def test_all_tools_summary(self):
        obs = Observer()
        obs.record_tool_call("a", duration_ms=10)
        obs.record_tool_call("b", duration_ms=20)
        obs.record_tool_call("a", duration_ms=30)
        result = obs.all_tools_summary()
        assert result["a"]["calls"] == 2
        assert result["b"]["calls"] == 1

    def test_recent_calls(self):
        obs = Observer()
        for i in range(10):
            obs.record_tool_call(f"tool_{i}", duration_ms=i * 10)
        recent = obs.recent_calls(limit=5)
        assert len(recent) == 5
        assert recent[0]["tool_name"] == "tool_9"  # most recent first

    def test_empty_summary(self):
        obs = Observer()
        summary = obs.tool_summary("nonexistent")
        assert summary["calls"] == 0
        assert summary["success_rate"] == 0.0

    def test_success_rate(self):
        obs = Observer()
        obs.record_tool_call("t", succeeded=True)
        obs.record_tool_call("t", succeeded=True)
        obs.record_tool_call("t", succeeded=False)
        summary = obs.tool_summary("t")
        assert abs(summary["success_rate"] - 0.6667) < 0.01


# ---------------------------------------------------------------------------
# Integration with SessionManager
# ---------------------------------------------------------------------------


class TestObserverIntegration:
    @pytest.mark.asyncio
    async def test_supervisor_records_to_observer(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", {"max_iterations": 5})
        sv = mgr.get_supervisor(sid)

        async def good_tool(query="test"):
            return f"result for {query}"

        await sv.call("search", good_tool, query="test")
        await sv.call("search", good_tool, query="test2")

        summary = mgr.observer.tool_summary("search")
        assert summary["calls"] == 2
        assert summary["successes"] == 2

    @pytest.mark.asyncio
    async def test_observer_records_failures(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", {"max_iterations": 5})
        sv = mgr.get_supervisor(sid)

        async def bad_tool():
            raise RuntimeError("boom")

        await sv.call("broken", bad_tool)

        summary = mgr.observer.tool_summary("broken")
        assert summary["calls"] == 1
        assert summary["failures"] == 1
        assert "tool_error" in summary["error_counts"]

    @pytest.mark.asyncio
    async def test_session_metrics_via_observer(self):
        store = MemoryStore()
        mgr = SessionManager(store)
        sid = mgr.start("agent-1", {"max_iterations": 5})
        sv = mgr.get_supervisor(sid)

        async def tool(x="1"):
            return x

        await sv.call("a", tool)
        await sv.call("b", tool)

        summary = mgr.observer.session_summary(sid)
        assert summary["total_calls"] == 2
        assert "a" in summary["tools"]
        assert "b" in summary["tools"]
