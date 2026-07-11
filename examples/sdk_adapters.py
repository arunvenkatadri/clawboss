"""SDK adapters — integrate AgentHandler with OpenAI Agents SDK and Claude Agent SDK.

Shows how to wrap tools, convert guardrails, use the tool registry, and
run an agent through AgentHandlerMiddleware.

Usage:
    python examples/sdk_adapters.py
"""

import asyncio

from agenthandler import (
    MemoryStore,
    RecursionDetector,
    SessionManager,
    UrlGuard,
)
from agenthandler.sdk_adapters import (
    AgentHandlerMiddleware,
    openai_guardrail_adapter,
    supervised_tool_registry,
    wrap_claude_tool,
    wrap_openai_tool,
)
from agenthandler.supervisor import Supervisor

# ---------------------------------------------------------------------------
# Sample tools
# ---------------------------------------------------------------------------


async def web_search(query: str = "") -> dict:
    """Search the web."""
    return {"results": [f"Result for '{query}'"], "count": 1}


async def write_file(path: str = "", content: str = "") -> str:
    """Write content to a file."""
    return f"Wrote {len(content)} chars to {path}"


def calculator(expression: str = "") -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# 1. wrap_openai_tool — single tool wrapping for OpenAI Agents SDK
# ---------------------------------------------------------------------------


async def demo_wrap_openai_tool():
    policy_dict = {"max_iterations": 10, "tool_timeout": 15, "token_budget": 50000}
    sv = Supervisor(
        __import__("agenthandler").Policy.from_dict(policy_dict),
    )
    supervised_search = wrap_openai_tool(web_search, sv, tool_name="web_search")

    result = await supervised_search(query="python async")
    print(f"OpenAI tool result: {result}")


# ---------------------------------------------------------------------------
# 2. wrap_claude_tool — single tool wrapping for Claude Agent SDK
# ---------------------------------------------------------------------------


async def demo_wrap_claude_tool():
    from agenthandler import Policy

    sv = Supervisor(Policy(max_iterations=10, tool_timeout=15))
    supervised_calc = wrap_claude_tool(calculator, sv, tool_name="calculator")

    result = await supervised_calc(expression="2 + 2")
    print(f"Claude tool result: {result}")


# ---------------------------------------------------------------------------
# 3. openai_guardrail_adapter — convert guardrails
# ---------------------------------------------------------------------------


def demo_guardrail_adapter():
    guardrails = [
        UrlGuard(allowlist=["*.example.com", "*.github.com"]),
        RecursionDetector(max_repeats=3),
    ]
    oai_guardrails = openai_guardrail_adapter(guardrails)
    print(f"Converted {len(guardrails)} guardrails -> {len(oai_guardrails)} OpenAI guardrails")
    for g in oai_guardrails:
        if isinstance(g, dict):
            print(f"  {g['type']}: {g['name']}")
        else:
            print(f"  {type(g).__name__}: {g.name}")


# ---------------------------------------------------------------------------
# 4. supervised_tool_registry — bulk wrapping
# ---------------------------------------------------------------------------


async def demo_supervised_registry():
    store = MemoryStore()
    manager = SessionManager(store)

    supervised = supervised_tool_registry(
        {"web_search": web_search, "write_file": write_file, "calculator": calculator},
        manager,
        policy={"tool_timeout": 30, "token_budget": 50000},
    )

    result = await supervised["web_search"](query="agenthandler")
    print(f"Registry result: {result}")

    result = await supervised["calculator"](expression="10 * 5")
    print(f"Registry calc: {result}")


# ---------------------------------------------------------------------------
# 5. AgentHandlerMiddleware — full session lifecycle
# ---------------------------------------------------------------------------


async def demo_middleware():
    store = MemoryStore()
    manager = SessionManager(store)

    async def my_agent(tools: dict, task: str = "") -> str:
        """A simple agent loop that uses supervised tools."""
        search_result = await tools["web_search"](query=task)
        summary = f"Found {search_result['count']} results for '{task}'"
        await tools["write_file"](path="/tmp/output.txt", content=summary)
        return summary

    middleware = AgentHandlerMiddleware(
        manager,
        policy={"tool_timeout": 30, "token_budget": 50000},
        agent_id="demo-agent",
    )

    result = await middleware.run(
        my_agent,
        tools={"web_search": web_search, "write_file": write_file},
        task="python async patterns",
    )
    print(f"Middleware result: {result}")


# ---------------------------------------------------------------------------
# Run all demos
# ---------------------------------------------------------------------------


async def main():
    print("=== wrap_openai_tool ===")
    await demo_wrap_openai_tool()

    print("\n=== wrap_claude_tool ===")
    await demo_wrap_claude_tool()

    print("\n=== openai_guardrail_adapter ===")
    demo_guardrail_adapter()

    print("\n=== supervised_tool_registry ===")
    await demo_supervised_registry()

    print("\n=== AgentHandlerMiddleware ===")
    await demo_middleware()


if __name__ == "__main__":
    asyncio.run(main())
