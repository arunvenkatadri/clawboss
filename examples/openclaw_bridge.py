"""OpenClaw bridge — expose clawboss-supervised tools to OpenClaw.

This example:
1. Defines tools with ToolDefinition
2. Registers them with an OpenClawBridge
3. Starts the HTTP bridge server

To use with OpenClaw:
1. Run this script: python examples/openclaw_bridge.py
2. Copy openclaw-plugin/ into your OpenClaw plugins directory
3. Configure the plugin's bridgeUrl to http://localhost:9229

Test with curl:
    curl http://localhost:9229/health
    curl http://localhost:9229/tools
    curl -X POST http://localhost:9229/execute/web_search \\
         -H 'Content-Type: application/json' \\
         -d '{"params": {"query": "python async"}}'
"""

import asyncio
import json

from clawboss import (
    OpenClawBridge,
    Skill,
    ToolDefinition,
    ToolParameter,
    to_openclaw_manifest,
    to_openclaw_tool_schema,
)

# ── Tool implementations ──────────────────────────────────────────


async def web_search(query: str, max_results: int = 5) -> str:
    """Simulate a web search."""
    await asyncio.sleep(0.1)
    return f"Found {max_results} results for '{query}'"


async def fetch_page(url: str) -> str:
    """Simulate fetching a web page."""
    await asyncio.sleep(0.1)
    return f"Content of {url}: <html>...</html>"


# ── Skill definition ──────────────────────────────────────────────

research_skill = Skill(
    name="web_research",
    description="Research topics using web search",
    tools=[
        ToolDefinition(
            name="web_search",
            description="Search the web for information",
            parameters=[
                ToolParameter(
                    name="query", type="string", description="Search query", required=True
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum results to return",
                    default=5,
                ),
            ],
        ),
        ToolDefinition(
            name="fetch_page",
            description="Fetch the content of a web page",
            parameters=[
                ToolParameter(name="url", type="string", description="URL to fetch", required=True),
            ],
        ),
    ],
    supervision={
        "max_iterations": 5,
        "tool_timeout": 15,
        "token_budget": 10000,
    },
)


async def main():
    # Show the OpenClaw schema conversion
    print("=== OpenClaw Tool Schemas ===")
    for tool in research_skill.tools:
        schema = to_openclaw_tool_schema(tool)
        print(json.dumps(schema, indent=2))
    print()

    # Show the plugin manifest
    print("=== Plugin Manifest ===")
    manifest = to_openclaw_manifest(research_skill)
    print(json.dumps(manifest, indent=2))
    print()

    # Start the bridge
    bridge = OpenClawBridge(port=9229)
    bridge.register_skill(
        research_skill,
        {
            "web_search": web_search,
            "fetch_page": fetch_page,
        },
    )

    print("=== Bridge Running ===")
    print("http://127.0.0.1:9229")
    print("  GET  /health          Health check")
    print("  GET  /tools           List tools (OpenClaw format)")
    print("  POST /execute/<name>  Execute a supervised tool")
    print()
    print("Press Ctrl+C to stop.")
    bridge.serve()


if __name__ == "__main__":
    asyncio.run(main())
