"""Edge-reduce integration — supervised hybrid local/cloud LLM routing.

Uses edge-reduce-llm to route easy queries to a local model (Ollama)
and hard queries to a frontier model (Claude), with AgentHandler
supervising every call for budgets, timeouts, and cost tracking.

Requirements:
    pip install agenthandler
    pip install local-reduce  # or: pip install git+https://github.com/arunvenkatadri/edge-reduce-llm.git
    ollama pull llama3.2       # or any local model

Usage:
    python examples/edge_reduce.py
"""

import asyncio
import os

from agenthandler import (
    MemoryStore,
    Pipeline,
    Policy,
    PricingTable,
    SessionManager,
    Supervisor,
)
from agenthandler.edge_reduce import EdgeReduceLLM, make_supervised_llm


async def main() -> None:
    # --- Setup ---
    store = MemoryStore()
    mgr = SessionManager(store, pricing=PricingTable.default())
    policy = Policy(tool_timeout=60, token_budget=100000, max_iterations=50)
    supervisor = Supervisor(policy)

    # --- Create the hybrid LLM ---
    llm = EdgeReduceLLM(
        supervisor=supervisor,
        local_base_url="http://localhost:11434/v1",
        cloud_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        cloud_model="claude-sonnet-4-20250514",
        policy_path="routing_policy.yaml",  # optional — uses defaults if missing
    )

    # --- Use it anywhere AgentHandler expects an LLM ---

    # 1. Direct call
    print("=== Direct LLM call ===")
    try:
        response = await llm("What is 2 + 2?")
        print(f"Response: {response}")
        details = llm.last_result
        if details:
            print(f"Route: {details.route} (model: {details.model})")
            print(f"Cost: ${details.cost_usd:.6f}")
    except Exception as e:
        print(f"(Skipped — needs running Ollama/API key: {e})")

    # 2. In a pipeline with LLM decisions
    print("\n=== Pipeline with hybrid LLM ===")
    print("(Would use edge-reduce to route each LLM call)")

    async def fetch_data(input: str = "") -> dict:
        return {"transactions": [{"amount": 500, "merchant": "Unknown Corp"}]}

    try:
        result = await (
            Pipeline(mgr, "fraud-detector", policy_dict=policy.to_dict())
            .add_step("fetch", fetch_data)
            .add_llm_decision(
                llm,
                prompt_template="""
                Analyze this transaction for fraud risk:
                {input}

                Return JSON: {{"risk": "low|medium|high", "reason": "..."}}
                """,
            )
            .run()
        )
        print(f"Result: {result.final_output}")
    except Exception as e:
        print(f"(Skipped — needs running Ollama/API key: {e})")

    # 3. Shorthand with make_supervised_llm
    print("\n=== Shorthand ===")
    quick_llm = make_supervised_llm(
        supervisor,
        cloud_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )
    print(f"Created hybrid LLM: {type(quick_llm).__name__}")

    # Show budget status
    budget = supervisor.budget()
    print(f"\nBudget: {budget.tokens_used}/{budget.token_budget} tokens used")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
