"""Days-long research agent — the reference implementation for long-duration agents.

Scenario: a research agent that runs for hours (or days), repeatedly
searching a topic, taking notes, and building up a knowledge base.

Demonstrates:
- Reflection loops (think → act → observe → reflect)
- Stateful sessions with accumulated history
- Crash recovery (kill it, start it again, it resumes)
- Guardrails (recursion detection, schema validation, URL allowlist)
- Session replay for post-hoc inspection
- Observability metrics

This is what a "long-running agent" looks like in Clawboss.

Run:
    python examples/long_running_agent.py
"""

import asyncio
import json

from clawboss import (
    MemoryStore,
    OutputLengthLimit,
    RecursionDetector,
    SchemaValidator,
    SessionManager,
)
from clawboss.reflection import ReflectionLoop
from clawboss.replay import SessionReplay

# -- Simulated tools --


async def web_search(query: str = "") -> dict:
    """Pretend to search the web. Returns structured results."""
    return {
        "query": query,
        "results": [
            {
                "title": f"Result 1 for {query}",
                "url": f"https://arxiv.org/abs/2026.{hash(query) % 10000:04d}",
            },
            {
                "title": f"Result 2 for {query}",
                "url": f"https://example.com/paper/{hash(query) % 1000}",
            },
        ],
    }


async def take_notes(topic: str = "", content: str = "") -> dict:
    """Add a note to the agent's knowledge base."""
    return {"topic": topic, "note": content, "saved": True}


async def write_summary(findings: str = "") -> dict:
    """Write a final summary from the collected findings."""
    return {
        "summary": f"Research summary: {findings[:200]}...",
        "word_count": len(findings.split()),
    }


# -- A mock LLM that plans out a short research session --
# In production this would be your real LLM (OpenAI, Anthropic, local).


def make_mock_llm() -> callable:
    """Return a mock LLM that drives a 5-step research session."""
    step = [0]
    plan = [
        # Cycle 1: search
        {
            "thought": "Start by searching for the topic",
            "tool": "web_search",
            "args": {"query": "quantum computing 2026"},
            "done": False,
        },
        "I found 2 arxiv papers and one industry source.",
        {
            "reflection": "Good starting material. Next I should drill into the papers.",
            "goal_progress": 0.2,
            "should_stop": False,
        },
        # Cycle 2: take notes
        {
            "thought": "Take notes on the key findings",
            "tool": "take_notes",
            "args": {"topic": "error correction", "content": "Recent advances in surface codes"},
            "done": False,
        },
        "Notes saved to the knowledge base.",
        {
            "reflection": "Building up context steadily. Need one more dimension.",
            "goal_progress": 0.5,
            "should_stop": False,
        },
        # Cycle 3: another search
        {
            "thought": "Search for hardware advances",
            "tool": "web_search",
            "args": {"query": "quantum hardware 2026"},
            "done": False,
        },
        "Found information on superconducting qubits and photonic chips.",
        {
            "reflection": "Solid technical context gathered. Ready to write summary.",
            "goal_progress": 0.8,
            "should_stop": False,
        },
        # Cycle 4: write summary, then done
        {
            "thought": "Write the final summary",
            "tool": "write_summary",
            "args": {"findings": "Quantum error correction and hardware both progressed in 2026"},
            "done": False,
        },
        "Summary written, 15 words.",
        {
            "reflection": "Goal complete. Time to finish.",
            "goal_progress": 1.0,
            "should_stop": True,
            "reason": "All research tasks completed",
        },
        # Cycle 5: done
        {
            "thought": "Gathered research, took notes, wrote summary. Goal achieved.",
            "tool": None,
            "done": True,
            "final_answer": "Quantum 2026: advances in error correction and hardware.",
        },
    ]

    async def mock_llm(prompt: str) -> str:
        idx = step[0]
        step[0] += 1
        if idx >= len(plan):
            # Safety fallback — tell agent we're done
            return '{"thought": "done", "tool": null, "done": true, "final_answer": "Complete"}'
        payload = plan[idx]
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

    return mock_llm


async def main():
    print("=" * 70)
    print("LONG-RUNNING RESEARCH AGENT DEMO")
    print("=" * 70)
    print()

    # -- 1. Set up the session with guardrails --
    store = MemoryStore()
    mgr = SessionManager(
        store,
        pre_guardrails=[
            RecursionDetector(max_repeats=5, window=60.0),
        ],
        post_guardrails=[
            OutputLengthLimit(max_chars=50_000),
            SchemaValidator(
                {
                    "web_search": {
                        "type": "object",
                        "required": ["query", "results"],
                        "properties": {
                            "results": {"type": "array"},
                        },
                    },
                }
            ),
        ],
    )

    # -- 2. Start the reflection loop --
    llm = make_mock_llm()

    loop = ReflectionLoop(
        manager=mgr,
        agent_id="research-agent",
        goal="Write a research summary on quantum computing in 2026",
        llm=llm,
        tools={
            "web_search": web_search,
            "take_notes": take_notes,
            "write_summary": write_summary,
        },
        policy_dict={"max_iterations": 30, "tool_timeout": 60, "token_budget": 100000},
    )

    print("Goal:", loop._goal)
    print("Running reflection loop...\n")

    result = await loop.run(max_cycles=10)

    # -- 3. Show what happened --
    print(f"Completed: {result.completed}")
    print(f"Cycles used: {result.cycles_used}")
    print(f"Tool calls: {result.total_tool_calls}")
    print(f"Stopped reason: {result.stopped_reason}")
    print()

    for c in result.cycles:
        marker = "OK" if c.tool_succeeded else "FAIL"
        print(f"  Cycle {c.cycle_number} [{marker}]")
        print(f"    Thought:    {c.thought[:80]}")
        if c.tool_called:
            print(f"    Action:     {c.tool_called}({c.tool_args})")
            print(f"    Observation: {c.observation[:80]}")
        print(f"    Reflection: {c.reflection[:80]}")
        print(f"    Progress:   {c.goal_progress:.0%}")
        print()

    if result.final_answer:
        print("Final answer:", result.final_answer)
        print()

    # -- 4. Session replay — inspect the full timeline --
    print("=" * 70)
    print("SESSION REPLAY")
    print("=" * 70)

    replay = SessionReplay(mgr, result.session_id)
    summary = replay.summary()
    print(f"  Session: {summary.session_id}")
    print(f"  Status: {summary.status}")
    print(f"  Total frames in audit log: {summary.total_frames}")
    print(f"  Tool calls: {summary.total_tool_calls}")
    print(f"  Unique tools: {', '.join(summary.unique_tools)}")
    print(f"  Duration: {summary.duration_ms}ms")
    print()

    print("  First 5 frames:")
    for frame in replay.frames()[:5]:
        print(f"    {frame.summary}")
    print()

    # -- 5. Observability metrics --
    print("=" * 70)
    print("METRICS")
    print("=" * 70)
    for tool, stats in mgr.observer.all_tools_summary().items():
        print(
            f"  {tool}: {stats['calls']} calls, "
            f"{stats['successes']} ok, avg {stats['avg_latency_ms']}ms"
        )


if __name__ == "__main__":
    asyncio.run(main())
