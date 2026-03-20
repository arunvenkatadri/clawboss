"""Skill Builder — create agent skills from natural language.

This example uses a mock LLM. Replace mock_llm() with your real LLM client
(OpenAI, Anthropic, local model, etc).
"""

import asyncio
import json
import tempfile
from clawboss import SkillBuilder, SkillStore, Skill


# ── Replace this with your real LLM ──────────────────────────────────
async def mock_llm(prompt: str) -> str:
    """Pretend to be an LLM that generates skill definitions."""
    return json.dumps({
        "name": "web_research",
        "description": "Research a topic using web search and summarize findings",
        "triggers": ["research", "look up", "find out", "search for"],
        "role": "You are a research assistant that finds and synthesizes information from the web.",
        "task": (
            "When the user asks you to research something:\n"
            "1. Search the web for relevant information\n"
            "2. Read the most relevant results\n"
            "3. Synthesize findings into a clear summary\n"
            "4. Cite your sources"
        ),
        "instructions": [
            "Always cite sources with URLs",
            "Prefer recent sources over older ones",
            "If the first search doesn't work, refine your query",
            "Summarize in plain language, not jargon",
            "If you can't find reliable info, say so honestly",
        ],
        "tools": [
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": [
                    {"name": "query", "type": "string", "description": "Search query", "required": True},
                    {"name": "max_results", "type": "integer", "description": "Max results", "required": False, "default": 5},
                ],
            },
            {
                "name": "fetch_page",
                "description": "Fetch text content from a URL",
                "parameters": [
                    {"name": "url", "type": "string", "description": "URL to fetch", "required": True},
                ],
            },
        ],
        "supervision": {
            "max_iterations": 5,
            "tool_timeout": 30,
            "token_budget": 15000,
            "on_timeout": "return_error",
            "on_budget_exceeded": "respond_with_best_effort",
        },
    })
# ─────────────────────────────────────────────────────────────────────


async def main():
    builder = SkillBuilder(mock_llm)

    # 1. Create a skill from natural language
    print("Creating skill from description...\n")
    skill = await builder.create(
        "A skill that researches topics on the web, limited to 5 tool calls, "
        "with a 30 second timeout per search"
    )

    print(f"Name: {skill.name}")
    print(f"Description: {skill.description}")
    print(f"Triggers: {skill.triggers}")
    print(f"Tools: {[t.name for t in skill.tools]}")
    print(f"Supervision: {skill.supervision}")
    print(f"Instructions ({len(skill.instructions)}):")
    for inst in skill.instructions:
        print(f"  - {inst}")

    # 2. Save to a skill store
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SkillStore(tmpdir)
        path = store.save(skill)
        print(f"\nSaved to: {path}")

        # 3. List skills
        print("\nAll skills:")
        for s in store.list():
            print(f"  {s['name']}: {s['description']}")

        # 4. Load it back
        loaded = store.get("web_research")
        print(f"\nLoaded: {loaded.name} (v{loaded.version})")

        # 5. Export as POML
        poml = store.export_poml("web_research")
        print(f"\n--- POML output ---\n{poml}")


if __name__ == "__main__":
    asyncio.run(main())
