"""Model router — cost-optimized LLM routing with per-rule model selection.

Route queries to different cloud models based on configurable rules.
Cheap models handle simple queries; expensive models handle complex ones.
Every call goes through AgentHandler supervision.

Usage:
    from agenthandler.model_router import ModelRouter, RoutingRule

    router = ModelRouter(
        api_key="sk-ant-...",
        rules=[
            RoutingRule(name="simple", match_default=True, model="claude-haiku-4-5"),
            RoutingRule(name="complex", keywords=["analyze", "compare"], model="claude-sonnet-4-6"),
            RoutingRule(name="critical", keywords=["audit", "legal"], model="claude-opus-4-6"),
        ],
        supervisor=sv,
    )

    response = await router("Summarize this document")
    # Routes to haiku (cheap) — simple query, no keywords matched

    response = await router("Analyze the security architecture")
    # Routes to sonnet (mid) — matched "analyze" keyword
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .supervisor import Supervisor


@dataclass
class RoutingRule:
    """A rule that maps query patterns to a specific model.

    Rules are evaluated top-down (first match wins).
    """

    name: str
    model: str
    keywords: List[str] = field(default_factory=list)
    regex: Optional[str] = None
    match_default: bool = False
    escalation: str = ""

    def matches(self, text: str) -> bool:
        if self.match_default:
            return True
        lower = text.lower()
        if self.keywords and any(kw.lower() in lower for kw in self.keywords):
            return True
        if self.regex and re.search(self.regex, text, re.IGNORECASE):
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "model": self.model}
        if self.keywords:
            d["keywords"] = self.keywords
        if self.regex:
            d["regex"] = self.regex
        if self.match_default:
            d["match_default"] = True
        if self.escalation:
            d["escalation"] = self.escalation
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoutingRule":
        return cls(
            name=d["name"],
            model=d["model"],
            keywords=d.get("keywords", []),
            regex=d.get("regex"),
            match_default=d.get("match_default", False),
            escalation=d.get("escalation", ""),
        )


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    text: str
    model: str
    rule_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class ModelRouter:
    """Cost-optimized LLM router with per-rule model selection.

    Evaluates rules top-down (first match wins) and calls the matched
    model. Every call goes through AgentHandler supervision.

    The callable interface matches ``async def llm(prompt: str) -> str``,
    so it plugs into PipelineBuilder, SkillBuilder, ReflectionLoop, etc.

    Args:
        llm_caller: Async function that takes (prompt, model) and returns
                    a dict with {text, input_tokens, output_tokens, model}.
                    If None, uses a built-in Anthropic caller (requires
                    anthropic SDK).
        api_key: API key for the built-in Anthropic caller.
        rules: List of RoutingRule (evaluated top-down, first match wins).
        default_model: Fallback model if no rule matches.
        supervisor: Supervisor for policy enforcement.
        tool_name: Name in audit log.
    """

    def __init__(
        self,
        rules: Optional[List[RoutingRule]] = None,
        default_model: str = "claude-haiku-4-5",
        supervisor: Optional[Supervisor] = None,
        llm_caller: Optional[Callable[..., Coroutine[Any, Any, Dict[str, Any]]]] = None,
        api_key: Optional[str] = None,
        tool_name: str = "llm_call",
    ) -> None:
        self._rules = rules or [
            RoutingRule(name="default", model=default_model, match_default=True),
        ]
        self._default_model = default_model
        self._supervisor = supervisor
        self._llm_caller = llm_caller
        self._api_key = api_key
        self._tool_name = tool_name
        self._last_decision: Optional[RoutingDecision] = None

    def add_rule(
        self,
        name: str,
        model: str,
        keywords: Optional[List[str]] = None,
        regex: Optional[str] = None,
        match_default: bool = False,
        escalation: str = "",
    ) -> "ModelRouter":
        """Add a routing rule. Inserts before the last rule (to keep default last)."""
        rule = RoutingRule(
            name=name,
            model=model,
            keywords=keywords or [],
            regex=regex,
            match_default=match_default,
            escalation=escalation,
        )
        if len(self._rules) > 0:
            self._rules.insert(len(self._rules) - 1, rule)
        else:
            self._rules.append(rule)
        return self

    def route(self, prompt: str) -> RoutingRule:
        """Determine which rule matches a prompt (first match wins)."""
        for rule in self._rules:
            if rule.matches(prompt):
                return rule
        return RoutingRule(name="fallback", model=self._default_model, match_default=True)

    async def _default_caller(self, prompt: str, model: str) -> Dict[str, Any]:
        """Built-in Anthropic API caller."""
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic SDK required for built-in caller: pip install anthropic"
            ) from e

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""
        return {
            "text": text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "model": model,
        }

    async def __call__(self, prompt: str) -> str:
        """Route and execute an LLM call. Returns the response text."""
        matched_rule = self.route(prompt)
        caller = self._llm_caller or self._default_caller

        async def _execute(prompt: str = "", model: str = "") -> Dict[str, Any]:
            return await caller(prompt, model)

        if self._supervisor:
            result = await self._supervisor.call(
                self._tool_name, _execute, prompt=prompt, model=matched_rule.model
            )
            if not result.succeeded:
                msg = result.error.user_message() if result.error else "LLM call failed"
                raise RuntimeError(f"[AgentHandler] {msg}")
            output = result.output
        else:
            output = await caller(prompt, matched_rule.model)

        if isinstance(output, dict):
            self._last_decision = RoutingDecision(
                text=str(output.get("text", "")),
                model=matched_rule.model,
                rule_name=matched_rule.name,
                input_tokens=output.get("input_tokens", 0),
                output_tokens=output.get("output_tokens", 0),
                cost_usd=output.get("cost_usd", 0.0),
            )
            return str(output.get("text", ""))

        self._last_decision = RoutingDecision(
            text=str(output), model=matched_rule.model, rule_name=matched_rule.name
        )
        return str(output)

    @property
    def last_decision(self) -> Optional[RoutingDecision]:
        """Routing details from the most recent call."""
        return self._last_decision

    @property
    def rules(self) -> List[RoutingRule]:
        """Current routing rules."""
        return list(self._rules)

    def to_dict(self) -> Dict[str, Any]:
        """Export routing config as a dict."""
        return {
            "default_model": self._default_model,
            "rules": [r.to_dict() for r in self._rules],
        }

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        supervisor: Optional[Supervisor] = None,
        llm_caller: Optional[Callable[..., Coroutine[Any, Any, Dict[str, Any]]]] = None,
        api_key: Optional[str] = None,
    ) -> "ModelRouter":
        """Load from a dict (e.g., parsed from YAML/JSON config)."""
        rules = [RoutingRule.from_dict(r) for r in d.get("rules", [])]
        return cls(
            rules=rules,
            default_model=d.get("default_model", "claude-haiku-4-5"),
            supervisor=supervisor,
            llm_caller=llm_caller,
            api_key=api_key,
        )
