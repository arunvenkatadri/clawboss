"""MCP server mode — expose supervised tools via Model Context Protocol.

Any MCP client (Claude Desktop, Claude Code, Cursor) gets governed tool access
with budgets, timeouts, circuit breakers, guardrails, and audit logging.

Usage:
    python examples/mcp_server.py                    # stdio (for Claude Desktop)
    python examples/mcp_server.py --http              # network mode

Requires: pip install agenthandler[mcp]
"""

import sys

from agenthandler import RecursionDetector, SupervisedMCPServer, UrlGuard

# --- Your tools (any async or sync callable) ---


async def web_search(query: str = "") -> dict:
    """Search the web for information."""
    return {"results": [f"Result 1 for '{query}'", f"Result 2 for '{query}'"], "count": 2}


async def write_file(path: str = "", content: str = "") -> str:
    """Write content to a file."""
    return f"Wrote {len(content)} chars to {path}"


async def send_email(to: str = "", subject: str = "", body: str = "") -> str:
    """Send an email notification."""
    return f"Email sent to {to}: {subject}"


def calculator(expression: str = "") -> str:
    """Evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# --- Build the supervised MCP server ---

server = SupervisedMCPServer(
    name="demo-supervised-tools",
    tools={
        "web_search": web_search,
        "write_file": write_file,
        "send_email": send_email,
        "calculator": calculator,
    },
    policy={
        "max_iterations": 100,
        "tool_timeout": 30,
        "token_budget": 100000,
        "circuit_breaker_threshold": 5,
        "require_confirm": ["send_email"],
    },
    pre_guardrails=[
        UrlGuard(allowlist=["*.example.com", "*.github.com"]),
        RecursionDetector(max_repeats=3),
    ],
)

if __name__ == "__main__":
    if "--http" in sys.argv:
        print("Starting supervised MCP server on http://0.0.0.0:8765/mcp")
        server.run(transport="streamable-http")
    else:
        server.run()  # stdio for Claude Desktop
