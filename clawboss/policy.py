"""Supervision policy — what limits to enforce and what to do when they're hit."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, List


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
class Policy:
    """Supervision policy for a skill or agent.

    Configure limits, failure handlers, and confirmation gates.
    """
    # Tool execution limits
    tool_timeout: float = 30.0       # seconds per tool call
    max_iterations: int = 5          # max tool call rounds
    token_budget: Optional[int] = None  # max tokens for the whole skill

    # Request-level limits
    request_timeout: float = 300.0   # seconds for the entire request
    silence_timeout: Optional[float] = None  # dead man's switch (seconds)

    # Circuit breaker
    circuit_breaker_threshold: int = 5   # failures before opening
    circuit_breaker_reset: float = 60.0  # seconds before half-open

    # Failure handlers
    on_timeout: OnFailure = field(default_factory=lambda: OnFailure(Action.RETURN_ERROR))
    on_budget_exceeded: OnFailure = field(default_factory=lambda: OnFailure(Action.RESPOND_WITH_BEST_EFFORT))
    on_max_iterations: OnFailure = field(default_factory=lambda: OnFailure(Action.RETURN_ERROR))
    on_circuit_open: OnFailure = field(default_factory=lambda: OnFailure(Action.RETURN_ERROR))
    on_silence: OnFailure = field(default_factory=lambda: OnFailure(Action.RETURN_ERROR))

    # Audit
    audit_enabled: bool = True

    # Confirmation gates (tool names that require user confirm)
    require_confirm: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Policy":
        """Create a Policy from a plain dictionary.

        Recognised keys (all optional — missing keys use defaults):
            tool_timeout, max_iterations, token_budget,
            request_timeout, silence_timeout,
            circuit_breaker_threshold, circuit_breaker_reset,
            on_timeout, on_budget_exceeded, on_max_iterations,
            on_circuit_open, on_silence, audit_enabled, require_confirm
        """
        kwargs: Dict[str, Any] = {}
        # Simple scalar fields
        for key in (
            "tool_timeout", "max_iterations", "token_budget",
            "request_timeout", "silence_timeout",
            "circuit_breaker_threshold", "circuit_breaker_reset",
            "audit_enabled",
        ):
            if key in d:
                kwargs[key] = d[key]

        # OnFailure fields (accept string or dict with action + retries)
        for key in (
            "on_timeout", "on_budget_exceeded", "on_max_iterations",
            "on_circuit_open", "on_silence",
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

        return cls(**kwargs)
