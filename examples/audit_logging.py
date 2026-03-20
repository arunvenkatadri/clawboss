"""Audit logging — record everything your agent does to JSONL or stdout."""

import asyncio
from clawboss import Supervisor, Policy, AuditLog, JsonlAuditSink


async def calculator(expression: str) -> str:
    """A simple tool that evaluates math."""
    return str(eval(expression))  # noqa: S307 — demo only


async def main():
    policy = Policy(max_iterations=5, tool_timeout=10.0)

    # Log to stdout (or use JsonlAuditSink.file("audit.jsonl") for files)
    sink = JsonlAuditSink.stdout()
    audit = AuditLog("demo-request-1", [sink])
    supervisor = Supervisor(policy, audit)

    # Each call gets logged
    r1 = await supervisor.call("calculator", calculator, expression="2 + 2")
    print(f"\n=> {r1.output}\n")

    supervisor.record_iteration()

    r2 = await supervisor.call("calculator", calculator, expression="10 * 42")
    print(f"\n=> {r2.output}\n")

    supervisor.finish()


if __name__ == "__main__":
    asyncio.run(main())
