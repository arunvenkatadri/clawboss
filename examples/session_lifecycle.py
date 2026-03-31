"""End-to-end session lifecycle example.

Demonstrates: start → use → pause → resume → stop, with crash recovery.

Run:
    python examples/session_lifecycle.py
"""

import asyncio

from clawboss import MemoryStore, SessionManager

# -- Simulated tools --


async def web_search(query: str) -> str:
    """Pretend to search the web."""
    return f"Results for: {query}"


async def summarize(text: str) -> dict:
    """Pretend to summarize with an LLM (reports token usage)."""
    return {"summary": f"Summary of: {text[:50]}...", "tokens_used": 200}


async def main():
    store = MemoryStore()  # swap with SqliteStore("sessions.db") for persistence
    mgr = SessionManager(store)

    # -- 1. Start a session --
    print("=== Starting session ===")
    sid = mgr.start(
        agent_id="researcher",
        policy_dict={
            "max_iterations": 10,
            "tool_timeout": 30,
            "token_budget": 5000,
            "require_confirm": ["delete_file"],
        },
        payload={"topic": "quantum computing", "step": 0},
    )
    print(f"Session ID: {sid}")

    # -- 2. Use the supervisor in an agent loop --
    sv = mgr.get_supervisor(sid)

    sv.record_iteration()
    result = await sv.call("web_search", web_search, query="quantum computing 2026")
    print(f"Search: {result.user_message()}")

    sv.record_iteration()
    result = await sv.call("summarize", summarize, text=result.output)
    print(f"Summary: {result.user_message()}")

    # Update payload with intermediate results
    mgr.update_payload(sid, {"topic": "quantum computing", "step": 2, "done_search": True})

    # Check budget
    snap = sv.budget()
    print(
        f"Budget: {snap.tokens_used}/{snap.token_limit} tokens, "
        f"{snap.iterations}/{snap.iteration_limit} iterations"
    )

    # -- 3. Pause --
    print("\n=== Pausing ===")
    mgr.pause(sid)
    cp = mgr.status(sid)
    print(f"Status: {cp.status.value}")

    # Trying to call a tool while paused → blocked
    result = await sv.call("web_search", web_search, query="should fail")
    print(f"Call while paused: succeeded={result.succeeded}, error={result.error.kind}")

    # -- 4. Simulate crash & recovery --
    print("\n=== Simulating crash ===")
    del mgr, sv  # everything in memory is gone

    mgr2 = SessionManager(store)
    print(f"Recovering session {sid}...")
    sv2 = mgr2.resume(sid)

    # Budget state is restored
    snap = sv2.budget()
    print(
        f"Restored budget: {snap.tokens_used}/{snap.token_limit} tokens, "
        f"{snap.iterations}/{snap.iteration_limit} iterations"
    )

    # Payload is preserved
    cp = mgr2.status(sid)
    print(f"Restored payload: {cp.payload}")

    # Policy is immutable — still has original limits
    print(f"Policy max_iterations: {sv2.policy.max_iterations}")
    print(f"Policy require_confirm: {sv2.policy.require_confirm}")

    # -- 5. Continue working --
    sv2.record_iteration()
    result = await sv2.call("web_search", web_search, query="quantum error correction")
    print(f"\nResumed search: {result.user_message()}")

    # -- 6. Stop --
    print("\n=== Stopping ===")
    mgr2.stop(sid)
    cp = mgr2.status(sid)
    print(f"Final status: {cp.status.value}")
    print(f"Final iterations: {cp.iterations}")
    print(f"Final tokens: {cp.tokens_used}")

    # Audit trail persisted
    entries = mgr2.get_audit_entries(sid)
    print(f"Audit entries: {len(entries)}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
