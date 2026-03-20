"""SkillBuilder — generate skills from natural language descriptions.

Bring your own LLM. Pass any async callable that takes a prompt string
and returns a string. Works with OpenAI, Anthropic, local models, whatever.

Usage:
    from clawboss import SkillBuilder, SkillStore

    async def my_llm(prompt: str) -> str:
        return await openai.chat(messages=[{"role": "user", "content": prompt}])

    builder = SkillBuilder(my_llm)
    skill = await builder.create("A skill that researches topics on the web")

    store = SkillStore("~/.clawboss/skills")
    store.save(skill)
"""

import json
import re
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .skill import Skill

# The prompt that teaches the LLM how to produce a valid skill definition.
# Kept as a module constant so users can inspect/override it.
SKILL_GENERATION_PROMPT = '''\
You are a skill builder for AI agents. Given a natural language description of what \
a skill should do, produce a complete skill definition as JSON.

The JSON must have this structure (all fields shown — omit any that aren't relevant):

```json
{
  "name": "snake_case_name",
  "description": "One sentence describing what this skill does",
  "triggers": ["keyword1", "keyword2"],
  "version": "1.0",
  "role": "You are a ... (persona for the agent)",
  "task": "When the user asks ... (what the agent should do step by step)",
  "instructions": [
    "First instruction",
    "Second instruction"
  ],
  "examples": [
    {"user": "Example user message", "assistant": "Example response"}
  ],
  "output_format": "Optional template for response structure",
  "tools": [
    {
      "name": "tool_name",
      "description": "What the tool does",
      "parameters": [
        {"name": "param_name", "type": "string", "description": "...", "required": true}
      ]
    }
  ],
  "supervision": {
    "max_iterations": 5,
    "tool_timeout": 30,
    "token_budget": 10000,
    "on_timeout": "return_error",
    "on_budget_exceeded": "respond_with_best_effort",
    "require_confirm": ["dangerous_tool_name"]
  }
}
```

Rules:
- name must be snake_case, short, descriptive
- triggers are keywords/phrases that should activate this skill
- tools: only include tools the skill actually needs. If no tools needed, omit the field
- supervision: always include sensible limits. Think about what could go wrong:
  - How many iterations could this reasonably need?
  - How long should each tool call take?
  - What's a reasonable token budget?
  - Which tools are dangerous enough to need confirmation?
- instructions: be specific and actionable. Include safety rules
- on_timeout options: "return_error", "respond_with_best_effort", "kill"
- on_budget_exceeded options: same as above
- require_confirm: list tool names that should need human approval before running

Respond with ONLY the JSON object. No markdown fences, no explanation.'''

SKILL_REFINEMENT_PROMPT = '''\
You are refining an existing skill definition based on user feedback.

Current skill definition:
```json
{current_skill}
```

The user wants the following changes:
{feedback}

Apply the requested changes and return the complete updated skill definition as JSON.
Follow the same schema as the original. Respond with ONLY the JSON object.'''


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from LLM output, handling markdown fences."""
    # Strip markdown code fences if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return json.loads(text.strip())


class SkillBuilder:
    """Build skills from natural language using any LLM.

    Args:
        llm: An async callable that takes a prompt string and returns a string.
             This is your LLM — OpenAI, Anthropic, local model, whatever.
        system_prompt: Override the default generation prompt if you want.
    """

    def __init__(
        self,
        llm: Callable[[str], Coroutine[Any, Any, str]],
        system_prompt: Optional[str] = None,
    ):
        self._llm = llm
        self._system_prompt = system_prompt or SKILL_GENERATION_PROMPT

    async def create(self, description: str) -> Skill:
        """Generate a skill from a natural language description.

        Args:
            description: What you want the skill to do, in plain English.

        Returns:
            A Skill object ready to save, inspect, or export.

        Raises:
            ValueError: If the LLM output can't be parsed as a valid skill.
        """
        prompt = f"{self._system_prompt}\n\nUser's description:\n{description}"
        raw = await self._llm(prompt)

        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"LLM returned invalid JSON. Raw output:\n{raw[:500]}"
            ) from e

        if "name" not in data:
            raise ValueError("LLM output missing required 'name' field")

        return Skill.from_dict(data)

    async def refine(self, skill: Skill, feedback: str) -> Skill:
        """Refine an existing skill based on natural language feedback.

        Args:
            skill: The current skill to modify.
            feedback: What to change, in plain English.

        Returns:
            An updated Skill object.
        """
        prompt = SKILL_REFINEMENT_PROMPT.format(
            current_skill=skill.to_json(),
            feedback=feedback,
        )
        raw = await self._llm(prompt)

        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"LLM returned invalid JSON during refinement. Raw output:\n{raw[:500]}"
            ) from e

        if "name" not in data:
            raise ValueError("LLM output missing required 'name' field")

        return Skill.from_dict(data)
