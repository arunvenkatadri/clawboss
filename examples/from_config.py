"""Load policy from a config dict — works with YAML, JSON, database, whatever."""

import asyncio
import json

from clawboss import Policy, Supervisor

# This could come from a YAML file, database, API, etc.
CONFIG = {
    "max_iterations": 10,
    "tool_timeout": 30,
    "token_budget": 50000,
    "on_timeout": "return_error",
    "on_budget_exceeded": {"action": "respond_with_best_effort", "retries": 1},
    "require_confirm": ["delete_file", "send_email"],
}


async def safe_tool(path: str) -> str:
    return f"Read {path}"


async def dangerous_tool(path: str) -> str:
    return f"Deleted {path}"


async def main():
    # Load from dict
    policy = Policy.from_dict(CONFIG)
    supervisor = Supervisor(policy)

    print(f"Policy: {json.dumps(CONFIG, indent=2)}\n")

    # Safe tool — allowed
    result = await supervisor.call("read_file", safe_tool, path="/tmp/data.txt")
    print(f"read_file: {result.user_message()}")

    # Dangerous tool — blocked (requires confirmation)
    result = await supervisor.call("delete_file", dangerous_tool, path="/tmp/data.txt")
    print(f"delete_file: {result.user_message()}")

    supervisor.finish()


if __name__ == "__main__":
    asyncio.run(main())
