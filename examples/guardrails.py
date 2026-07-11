"""Guardrails example — deterministic and LLM-backed safety checks.

Demonstrates wiring pre-call and post-call guardrails into a SessionManager,
then running supervised tool calls that trigger blocking and allow decisions.

Run:
    python examples/guardrails.py
"""

import asyncio
import json

from agenthandler import (
    IntentDriftDetector,
    MemoryStore,
    OutputLengthLimit,
    PromptInjectionDetector,
    RecursionDetector,
    SchemaValidator,
    SessionManager,
    UrlGuard,
)

# -- Fake LLM for LLM-backed guardrails --


async def fake_llm(prompt: str) -> str:
    """Simulate an LLM that detects injection and drift."""
    if "prompt injection" in prompt.lower() or "injection" in prompt.lower():
        if "ignore all previous instructions" in prompt:
            return json.dumps(
                {
                    "injection_detected": True,
                    "confidence": 0.95,
                    "reason": "instruction override attempt",
                }
            )
        return json.dumps({"injection_detected": False, "confidence": 0.1, "reason": ""})

    if "staying on task" in prompt.lower():
        if "delete_everything" in prompt:
            return json.dumps(
                {
                    "on_task": False,
                    "confidence": 0.9,
                    "reason": "destructive action unrelated to search task",
                }
            )
        return json.dumps({"on_task": True, "confidence": 0.95, "reason": ""})

    return json.dumps({})


# -- Simulated tools --


async def web_search(url: str) -> str:
    return f"Fetched content from {url}"


async def summarize(text: str) -> dict:
    return {"summary": text[:50], "word_count": len(text.split())}


async def huge_output(query: str) -> str:
    return "x" * 200_000  # deliberately oversized


async def main():
    # --- Deterministic guardrails (pre-call) ---
    url_guard = UrlGuard(
        allowlist=["*.example.com", "api.trusted.io"],
        blocklist=["*.evil.com"],
    )
    recursion_detector = RecursionDetector(max_repeats=2, window=10.0)

    # --- Deterministic guardrails (post-call) ---
    output_limit = OutputLengthLimit(max_chars=50_000)
    schema_validator = SchemaValidator(
        schemas={
            "summarize": {
                "type": "object",
                "required": ["summary"],
                "properties": {
                    "summary": {"type": "string"},
                    "word_count": {"type": "integer", "minimum": 0},
                },
            }
        }
    )

    # --- LLM-backed guardrails ---
    injection_detector = PromptInjectionDetector(llm=fake_llm)
    drift_detector = IntentDriftDetector(llm=fake_llm, original_task_key="original_task")

    # --- Wire into SessionManager ---
    store = MemoryStore()
    mgr = SessionManager(
        store,
        pre_guardrails=[url_guard, recursion_detector, injection_detector, drift_detector],
        post_guardrails=[output_limit, schema_validator],
    )

    sid = mgr.start(
        agent_id="search-agent",
        policy_dict={"max_iterations": 20, "tool_timeout": 10},
    )
    sv = mgr.get_supervisor(sid)

    print("=== 1. Allowed URL ===")
    result = await sv.call("web_search", web_search, url="https://docs.example.com/api")
    print(f"  {result.user_message()}")
    print(f"  succeeded={result.succeeded}")

    print("\n=== 2. Blocked URL (not in allowlist) ===")
    result = await sv.call("web_search", web_search, url="https://shady.io/hack")
    print(f"  {result.user_message()}")
    print(f"  succeeded={result.succeeded}")

    print("\n=== 3. Schema-validated output (passes) ===")
    result = await sv.call("summarize", summarize, text="Hello world this is a test")
    print(f"  {result.user_message()}")

    print("\n=== 4. Output too large (blocked by OutputLengthLimit) ===")
    result = await sv.call("huge_output", huge_output, query="give me everything")
    print(f"  {result.user_message()}")
    print(f"  succeeded={result.succeeded}")

    print("\n=== 5. Recursion detection ===")
    for i in range(4):
        result = await sv.call("web_search", web_search, url="https://docs.example.com/same")
        status = "OK" if result.succeeded else "BLOCKED"
        print(f"  call {i + 1}: {status} - {result.user_message()[:60]}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
