"""Basic clawboss usage — supervise tool calls with timeouts and budgets."""

import asyncio

from clawboss import Policy, Supervisor


async def web_search(query: str) -> str:
    """Simulate a web search tool."""
    await asyncio.sleep(0.1)  # pretend this hits an API
    return f"Top result for '{query}': https://example.com"


async def slow_tool(query: str) -> str:
    """A tool that takes too long."""
    await asyncio.sleep(60)
    return "this never returns"


async def main():
    policy = Policy(
        max_iterations=3,
        tool_timeout=5.0,
        token_budget=10_000,
    )
    supervisor = Supervisor(policy)

    # Normal call — succeeds
    result = await supervisor.call("web_search", web_search, query="clawboss python")
    print(f"Search: {result.user_message()}")
    print(f"  took {result.duration_ms}ms, succeeded={result.succeeded}")

    # Track tokens from your LLM
    supervisor.record_tokens(2500)
    print(f"Budget: {supervisor.budget()}")

    # Record iteration
    supervisor.record_iteration()

    # Slow call — times out
    result = await supervisor.call("slow_tool", slow_tool, query="waiting...")
    print(f"Slow tool: {result.user_message()}")
    print(f"  succeeded={result.succeeded}, error={result.error.kind}")

    # Finish
    snapshot = supervisor.finish()
    print(f"\nDone: {snapshot.tokens_used} tokens, {snapshot.iterations} iterations")


if __name__ == "__main__":
    asyncio.run(main())
