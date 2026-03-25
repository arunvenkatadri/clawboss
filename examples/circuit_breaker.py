"""Circuit breaker — stop hammering tools that keep failing."""

import asyncio

from clawboss import Policy, Supervisor

call_count = 0


async def flaky_api(query: str) -> str:
    """A tool that fails the first 3 times, then works."""
    global call_count
    call_count += 1
    if call_count <= 3:
        raise ConnectionError(f"API unavailable (attempt {call_count})")
    return f"Success on attempt {call_count}"


async def main():
    policy = Policy(
        max_iterations=10,
        tool_timeout=5.0,
        circuit_breaker_threshold=3,  # open after 3 failures
        circuit_breaker_reset=2.0,  # try again after 2 seconds
    )
    supervisor = Supervisor(policy)

    # First 3 calls fail — circuit breaker opens
    for i in range(4):
        supervisor.record_iteration()
        result = await supervisor.call("flaky_api", flaky_api, query="test")
        print(f"Call {i + 1}: succeeded={result.succeeded}, error={result.error}")

    # Wait for circuit breaker reset
    print("\nWaiting for circuit breaker reset...")
    await asyncio.sleep(2.5)

    # This call should go through (half-open → test call succeeds)
    supervisor.record_iteration()
    result = await supervisor.call("flaky_api", flaky_api, query="test")
    print(f"Call 5: succeeded={result.succeeded}, output={result.output}")

    supervisor.finish()


if __name__ == "__main__":
    asyncio.run(main())
