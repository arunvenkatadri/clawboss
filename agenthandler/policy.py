"""Supervision policy — what limits to enforce and what to do when they're hit."""

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Action(Enum):
    """What to do when a limit is hit."""

    RETURN_ERROR = "return_error"
    RESPOND_WITH_BEST_EFFORT = "respond_with_best_effort"
    KILL = "kill"

    @classmethod
    def from_str(cls, s: str) -> "Action":
        mapping = {
            "return_error": cls.RETURN_ERROR,
            "respond_with_best_effort": cls.RESPOND_WITH_BEST_EFFORT,
            "kill": cls.KILL,
        }
        return mapping.get(s, cls.RETURN_ERROR)


@dataclass
class OnFailure:
    """What to do on failure + how many retries."""

    action: Action = Action.RETURN_ERROR
    retries: int = 0


@dataclass
class ScopeRule:
    """A single scope constraint on a tool parameter."""

    param: str  # parameter name to constrain
    constraint: str  # "allow", "block", or "match"
    values: List[str]  # patterns (glob-style for allow/block, regex for match)

    def check(self, param_value: Any) -> bool:
        """Returns True if the value is allowed by this rule."""
        str_value = str(param_value)
        if self.constraint == "allow":
            return any(fnmatch.fnmatch(str_value, p) for p in self.values)
        if self.constraint == "block":
            return not any(fnmatch.fnmatch(str_value, p) for p in self.values)
        if self.constraint == "match":
            return any(re.search(p, str_value) for p in self.values)
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "param": self.param,
            "constraint": self.constraint,
            "values": list(self.values),
        }


@dataclass
class ToolScope:
    """Scoped permissions for a specific tool."""

    tool_name: str
    rules: List[ScopeRule]
    max_calls_per_minute: Optional[int] = None  # rate limit

    def check_args(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """Check if kwargs satisfy all rules.

        Returns None if OK, error message if violated.
        """
        for rule in self.rules:
            if rule.param in kwargs:
                if not rule.check(kwargs[rule.param]):
                    return (
                        f"Parameter '{rule.param}' value "
                        f"'{kwargs[rule.param]}' blocked by scope rule"
                    )
        return None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "tool_name": self.tool_name,
            "rules": [r.to_dict() for r in self.rules],
        }
        if self.max_calls_per_minute is not None:
            d["max_calls_per_minute"] = self.max_calls_per_minute
        return d


@dataclass
class Policy:
    """Supervision policy for a skill or agent.

    Configure limits, failure handlers, and confirmation gates.
    """

    # Tool execution limits
    tool_timeout: float = 30.0  # seconds per tool call
    max_iterations: int = 5  # max tool call rounds
    token_budget: Optional[int] = None  # max tokens for the whole skill

    # Request-level limits
    request_timeout: float = 300.0  # seconds for the entire request
    silence_timeout: Optional[float] = None  # dead man's switch (seconds)

    # Circuit breaker
    circuit_breaker_threshold: int = 5  # failures before opening
    circuit_breaker_reset: float = 60.0  # seconds before half-open

    # Failure handlers
    on_timeout: OnFailure = field(default_factory=lambda: OnFailure(Action.RETURN_ERROR))
    on_budget_exceeded: OnFailure = field(
        default_factory=lambda: OnFailure(Action.RESPOND_WITH_BEST_EFFORT)
    )
    on_max_iterations: OnFailure = field(default_factory=lambda: OnFailure(Action.RETURN_ERROR))
    on_circuit_open: OnFailure = field(default_factory=lambda: OnFailure(Action.RETURN_ERROR))
    on_silence: OnFailure = field(default_factory=lambda: OnFailure(Action.RETURN_ERROR))

    # Crash loop protection
    max_resumes: int = 3  # max times a session can be resumed before marked failed

    # Audit
    audit_enabled: bool = True

    # Confirmation gates (tool names that require user confirm)
    require_confirm: List[str] = field(default_factory=list)

    # Tool scopes (parameter-level permission rules)
    tool_scopes: List[ToolScope] = field(default_factory=list)

    # Privacy shielding — PII redaction categories
    # Options: "email", "phone", "ssn", "credit_card", "api_key", "ip_address"
    # Empty list = no redaction. None = redact all categories.
    redact: Optional[List[str]] = field(default_factory=list)
    redact_direction: str = "both"  # "inbound", "outbound", or "both"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary (round-trips with from_dict)."""
        d: Dict[str, Any] = {
            "tool_timeout": self.tool_timeout,
            "max_iterations": self.max_iterations,
            "request_timeout": self.request_timeout,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_reset": self.circuit_breaker_reset,
            "audit_enabled": self.audit_enabled,
        }
        if self.token_budget is not None:
            d["token_budget"] = self.token_budget
        if self.silence_timeout is not None:
            d["silence_timeout"] = self.silence_timeout
        for key in (
            "on_timeout",
            "on_budget_exceeded",
            "on_max_iterations",
            "on_circuit_open",
            "on_silence",
        ):
            val: OnFailure = getattr(self, key)
            d[key] = {"action": val.action.value, "retries": val.retries}
        d["max_resumes"] = self.max_resumes
        if self.require_confirm:
            d["require_confirm"] = list(self.require_confirm)
        if self.tool_scopes:
            d["tool_scopes"] = [s.to_dict() for s in self.tool_scopes]
        if self.redact is not None and self.redact:
            d["redact"] = list(self.redact)
        elif self.redact is None:
            d["redact"] = None
        d["redact_direction"] = self.redact_direction
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Policy":
        """Create a Policy from a plain dictionary.

        Recognised keys (all optional — missing keys use defaults):
            tool_timeout, max_iterations, token_budget,
            request_timeout, silence_timeout,
            circuit_breaker_threshold, circuit_breaker_reset,
            max_resumes,
            on_timeout, on_budget_exceeded, on_max_iterations,
            on_circuit_open, on_silence, audit_enabled, require_confirm
        """
        kwargs: Dict[str, Any] = {}
        # Simple scalar fields
        for key in (
            "tool_timeout",
            "max_iterations",
            "token_budget",
            "request_timeout",
            "silence_timeout",
            "circuit_breaker_threshold",
            "circuit_breaker_reset",
            "max_resumes",
            "audit_enabled",
        ):
            if key in d:
                kwargs[key] = d[key]

        # OnFailure fields (accept string or dict with action + retries)
        for key in (
            "on_timeout",
            "on_budget_exceeded",
            "on_max_iterations",
            "on_circuit_open",
            "on_silence",
        ):
            if key in d:
                val = d[key]
                if isinstance(val, str):
                    kwargs[key] = OnFailure(Action.from_str(val))
                elif isinstance(val, dict):
                    kwargs[key] = OnFailure(
                        Action.from_str(val.get("action", "return_error")),
                        retries=val.get("retries", 0),
                    )

        if "require_confirm" in d:
            kwargs["require_confirm"] = list(d["require_confirm"])

        if "tool_scopes" in d:
            scopes = []
            for scope_dict in d["tool_scopes"]:
                rules = []
                for r in scope_dict.get("rules", []):
                    rules.append(
                        ScopeRule(
                            param=r["param"],
                            constraint=r.get("constraint", "allow"),
                            values=r.get("values", []),
                        )
                    )
                scopes.append(
                    ToolScope(
                        tool_name=scope_dict["tool_name"],
                        rules=rules,
                        max_calls_per_minute=scope_dict.get("max_calls_per_minute"),
                    )
                )
            kwargs["tool_scopes"] = scopes

        if "redact" in d:
            kwargs["redact"] = d["redact"]  # None or list of strings
        if "redact_direction" in d:
            kwargs["redact_direction"] = d["redact_direction"]

        return cls(**kwargs)
