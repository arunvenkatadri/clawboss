"""Context compression — supervision-anchored context window management.

Shows how ContextWindow compresses conversation history while keeping
safety-critical state (policies, budgets, circuit breakers) fresh from
the supervisor. The key insight: supervised agents can compress more
aggressively because the supervisor enforces safety, not the LLM's memory.
"""

import asyncio

from clawboss import Policy, Supervisor
from clawboss.context import ContextWindow


async def main():
    # Create a supervisor with limits
    policy = Policy(
        max_iterations=10,
        tool_timeout=15.0,
        token_budget=10000,
        require_confirm=["delete_file"],
    )
    supervisor = Supervisor(policy)

    # Create a context window that keeps last 5 turns at full fidelity
    ctx = ContextWindow(supervisor, max_recent_turns=5, skill_name="web-search")

    # Simulate a conversation with tool calls
    ctx.add_turn("user", "Search for quantum computing breakthroughs in 2026")
    ctx.add_turn(
        "assistant",
        "I'll search for that.",
        tool_calls=[
            {
                "tool_name": "web_search",
                "params": {"query": "quantum computing breakthroughs 2026"},
                "result_summary": "Found 5 results",
            }
        ],
    )
    ctx.add_turn("assistant", "Here are the top results on quantum computing...")
    ctx.add_turn("user", "Tell me more about IBM's processor")
    ctx.add_turn(
        "assistant",
        "Let me look that up.",
        tool_calls=[
            {
                "tool_name": "web_search",
                "params": {"query": "IBM quantum processor 2026"},
                "result_summary": "Found 3 results",
            },
            {
                "tool_name": "fetch_page",
                "params": {"url": "https://research.ibm.com/quantum"},
                "result_summary": "Page fetched",
            },
        ],
    )
    ctx.add_turn("assistant", "IBM announced their new 1000+ qubit processor...")
    ctx.add_turn("user", "How does this compare to Google's approach?")
    ctx.add_turn(
        "assistant",
        "Searching for Google's quantum work.",
        tool_calls=[
            {
                "tool_name": "web_search",
                "params": {"query": "Google quantum computing 2026"},
                "result_summary": "Found 4 results",
            }
        ],
    )
    ctx.add_turn("assistant", "Google has been focusing on error correction...")
    ctx.add_turn("user", "Can you summarize the key differences?")

    # Before compression
    print("=== BEFORE COMPRESSION ===")
    print(f"Total turns: {ctx.turn_count}")
    print(f"Token estimate: ~{ctx.token_estimate()}")
    print()

    # Compress — moves older turns to summary, keeps last 5 at full fidelity
    result = await ctx.compress()
    print("=== AFTER COMPRESSION ===")
    print(f"Total turns: {ctx.turn_count}")
    print(f"Token estimate: ~{ctx.token_estimate()}")
    print(f"Compressed turns: {result.history.turn_count if result.history else 0}")
    print(f"Recent turns: {len(result.recent_turns)}")
    print()

    # The full prompt — all three zones
    print("=== FULL CONTEXT PROMPT ===")
    print(result.to_prompt())
    print()

    # Now update the supervisor — the anchored state stays fresh
    supervisor.record_tokens(3500)
    print("=== AFTER RECORDING 3500 TOKENS ===")
    print("(notice budget updates in anchored state)")
    print()
    print(ctx.to_prompt())


if __name__ == "__main__":
    asyncio.run(main())
