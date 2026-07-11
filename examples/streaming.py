"""Streaming connector example.

Demonstrates how to wire a Kafka stream into a supervised pipeline.
Each message fires the pipeline with the message payload as initial input.
Uses Pipeline.with_context() for stateful stream processing.

Run:
    python examples/streaming.py
"""

import asyncio

from agenthandler import MemoryStore, SessionManager
from agenthandler.pipeline import Pipeline
from agenthandler.streams import KafkaStreamConnector

# -- Simulated tools --


async def classify(input: str = "") -> str:
    """Classify incoming event severity."""
    if "error" in input.lower():
        return "critical"
    return "info"


async def route(input: str = "") -> str:
    """Route event to the appropriate handler."""
    return f"Routed [{input}] to alerts queue"


async def main() -> None:
    store = MemoryStore()
    mgr = SessionManager(store)

    # Create a persistent session for stateful processing.
    # The pipeline accumulates context across stream messages.
    session_id = mgr.start("stream-agent", policy_dict={"max_iterations": 100})

    # Build a pipeline that reuses the session across invocations
    pipeline = (
        Pipeline(mgr, agent_id="stream-agent", stateless=True)
        .with_context(session_id)
        .add_step("classify", classify)
        .add_step("route", route)
    )

    # Define the message handler that fires the pipeline per message
    async def on_message(payload: dict) -> None:
        msg = payload.get("value", {})
        result = await pipeline.run(initial_input=str(msg))
        print(f"  Processed: {result.final_output}")

    # Set up the Kafka connector (won't actually connect without a broker)
    _connector = KafkaStreamConnector(
        bootstrap_servers="localhost:9092",
        topic="agent-events",
        group_id="agenthandler-demo",
        on_message=on_message,
    )

    # Simulate what happens when messages arrive (no real broker needed)
    print("=== Simulating stream messages ===\n")
    fake_messages = [
        {"value": {"event": "deploy", "status": "ok"}},
        {"value": {"event": "healthcheck", "status": "error: timeout"}},
        {"value": {"event": "scale-up", "status": "ok"}},
    ]
    for msg in fake_messages:
        await on_message(msg)

    print("\n=== Connector setup (would connect to Kafka) ===")
    print("  Topic: agent-events")
    print("  Group: agenthandler-demo")
    print("  Servers: localhost:9092")
    print(f"  Session: {session_id} (reused across messages)")


if __name__ == "__main__":
    asyncio.run(main())
