"""Cost attribution example — track token usage and dollar costs across agents.

Demonstrates creating an Observer with a PricingTable, recording tool calls
with token counts and model info, and inspecting cost breakdowns.

Run:
    python examples/cost_attribution.py
"""

import asyncio
import json

from agenthandler import MemoryStore, Observer, PricingTable, SessionManager


def print_json(label: str, data: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(json.dumps(data, indent=2))


async def fake_tool(query: str) -> str:
    return f"Result for: {query}"


async def main():
    # --- Standalone Observer usage ---
    pricing = PricingTable.default()
    obs = Observer(pricing=pricing)

    # Simulate a series of LLM-backed tool calls across two agents
    obs.record_tool_call(
        "web_search",
        duration_ms=250,
        succeeded=True,
        input_tokens=800,
        output_tokens=150,
        model="claude-sonnet-4-6",
        session_id="sess-1",
        agent_id="researcher",
    )
    obs.record_tool_call(
        "summarize",
        duration_ms=1200,
        succeeded=True,
        input_tokens=2000,
        output_tokens=500,
        model="claude-sonnet-4-6",
        session_id="sess-1",
        agent_id="researcher",
    )
    obs.record_tool_call(
        "generate_code",
        duration_ms=3000,
        succeeded=True,
        input_tokens=4000,
        output_tokens=2000,
        model="claude-opus-4-6",
        session_id="sess-1",
        agent_id="coder",
    )
    obs.record_tool_call(
        "web_search",
        duration_ms=180,
        succeeded=False,
        input_tokens=500,
        output_tokens=0,
        model="gpt-4o-mini",
        session_id="sess-2",
        agent_id="researcher",
        error_kind="timeout",
    )

    # --- tool_summary: metrics for a single tool ---
    print_json("tool_summary('web_search')", obs.tool_summary("web_search"))

    # --- cost_summary: full breakdown ---
    cost = obs.cost_summary()
    print_json(
        "cost_summary() - totals",
        {
            "total_cost_usd": cost["total_cost_usd"],
            "total_tokens": cost["total_tokens"],
            "total_calls": cost["total_calls"],
        },
    )
    print_json("cost_summary() - by_agent", {"by_agent": cost["by_agent"]})
    print_json("cost_summary() - by_model", {"by_model": cost["by_model"]})

    # --- SessionManager with default pricing ---
    print(f"\n{'=' * 60}")
    print("  SessionManager with built-in pricing")
    print(f"{'=' * 60}")

    store = MemoryStore()
    mgr = SessionManager(store)  # uses PricingTable.default() automatically

    sid = mgr.start(agent_id="demo-agent", policy_dict={"max_iterations": 5})
    sv = mgr.get_supervisor(sid)

    # Make a supervised call — the observer tracks it automatically
    await sv.call("fake_tool", fake_tool, query="hello")
    await sv.call("fake_tool", fake_tool, query="world")

    session_costs = mgr.observer.cost_summary()
    print(f"  Session calls recorded: {session_costs['total_calls']}")
    print(f"  Total cost: ${session_costs['total_cost_usd']:.6f}")
    print("  (Tool calls without a model field have $0 cost)")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
