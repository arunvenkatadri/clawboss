"""Agent-to-Agent protocol — supervised inter-agent communication.

Shows how to:
1. Set up an A2A client to call other agents with supervision
2. Send a task to another agent
3. Publish your own agent card

Requires: pip install agenthandler httpx

Usage:
    python examples/a2a_agents.py
"""

from agenthandler import MemoryStore, Policy, SessionManager, Supervisor
from agenthandler.a2a import A2AAgentCard, A2AClient, A2ASkill, A2ASupervisedEndpoint

# --- 1. Set up an A2A client with supervision ---


async def demo_client():
    """Send a supervised task to another agent."""
    policy = Policy(max_iterations=10, tool_timeout=30, token_budget=50000)
    sv = Supervisor(policy)

    client = A2AClient(supervisor=sv, timeout=15)

    # Send a task (would hit a real agent at this URL)
    result = await client.send_task(
        agent_url="https://research-agent.example.com",
        task={"skill": "research", "input": "quantum computing breakthroughs 2025"},
    )

    if result.succeeded:
        print(f"Task result: {result.output}")
    else:
        print(f"Task failed: {result.error}")

    sv.finish()


# --- 2. Build an agent card ---


def demo_agent_card():
    """Create and inspect an agent card."""
    card = A2AAgentCard(
        name="research-agent",
        description="Performs deep research on any topic",
        url="https://research-agent.example.com",
        skills=[
            A2ASkill(
                name="research",
                description="Deep research with citations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "depth": {"type": "string", "enum": ["shallow", "deep"]},
                    },
                    "required": ["topic"],
                },
            ),
            A2ASkill(name="summarize", description="Summarize a document"),
        ],
    )

    # Serialize and deserialize
    card_dict = card.to_dict()
    print(f"Agent card: {card_dict['name']} at {card_dict['url']}")
    print(f"Skills: {[s['name'] for s in card_dict['skills']]}")

    restored = A2AAgentCard.from_dict(card_dict)
    assert restored.name == card.name
    assert len(restored.skills) == len(card.skills)
    print("Roundtrip OK")


# --- 3. Set up a supervised A2A endpoint ---


def demo_endpoint():
    """Make your agent discoverable and callable via A2A."""
    store = MemoryStore()
    mgr = SessionManager(store)
    sid = mgr.start(
        "my-agent",
        {"max_iterations": 50, "tool_timeout": 30, "token_budget": 100000},
    )

    async def research_tool(input: str = "") -> dict:
        return {"summary": f"Research results for: {input}", "sources": 3}

    def route_skill(skill_name: str):
        tools = {"research": research_tool}
        return tools.get(skill_name)

    endpoint = A2ASupervisedEndpoint(
        manager=mgr,
        session_id=sid,
        tool_router=route_skill,
        name="my-research-agent",
        description="A supervised research agent",
        url="https://my-agent.example.com",
    )

    card = endpoint.agent_card
    print(f"Endpoint card: {card.name}")
    print(f"Skills: {[s.name for s in card.skills]}")

    # In a real app, you'd mount the router:
    #   from fastapi import FastAPI
    #   app = FastAPI()
    #   app.include_router(endpoint.router())

    mgr.stop(sid)


# --- 4. Generate card from supervisor ---


def demo_card_from_supervisor():
    """Generate an agent card from an existing Supervisor."""
    policy = Policy(max_iterations=20, tool_timeout=15, token_budget=25000)
    sv = Supervisor(policy)

    card = A2AAgentCard.from_supervisor(
        sv,
        name="coding-agent",
        url="https://code.example.com",
        description="Writes and reviews code",
        tool_names=["write_code", "review_code", "run_tests"],
    )

    print(f"\nGenerated card: {card.name}")
    print(f"Skills: {[s.name for s in card.skills]}")
    print(f"Supervision: {card.metadata.get('supervision', {})}")

    sv.finish()


if __name__ == "__main__":
    demo_agent_card()
    demo_endpoint()
    demo_card_from_supervisor()
    print("\nAll demos passed.")
