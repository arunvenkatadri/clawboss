"""Common types for guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, runtime_checkable


@dataclass
class GuardrailResult:
    """Result of a guardrail check.

    allowed=True means the tool call (or output) passes the check.
    allowed=False blocks it with the given reason.
    replacement_output, if set, is substituted for the original output
    (used for e.g. PII redaction where we modify rather than block).
    """

    allowed: bool
    reason: str = ""
    guardrail_name: str = ""
    replacement_output: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def allow(cls, guardrail_name: str = "") -> "GuardrailResult":
        return cls(allowed=True, guardrail_name=guardrail_name)

    @classmethod
    def block(cls, reason: str, guardrail_name: str = "") -> "GuardrailResult":
        return cls(allowed=False, reason=reason, guardrail_name=guardrail_name)

    @classmethod
    def replace(
        cls, new_output: Any, reason: str = "", guardrail_name: str = ""
    ) -> "GuardrailResult":
        return cls(
            allowed=True,
            reason=reason,
            guardrail_name=guardrail_name,
            replacement_output=new_output,
        )


@runtime_checkable
class PreCallGuardrail(Protocol):
    """A guardrail that runs BEFORE a tool call.

    Receives the tool name and kwargs. Returns a GuardrailResult.
    Can be sync or async.
    """

    name: str

    def check(
        self, tool_name: str, kwargs: Dict[str, Any], context: Dict[str, Any]
    ) -> GuardrailResult: ...


@runtime_checkable
class PostCallGuardrail(Protocol):
    """A guardrail that runs AFTER a tool call.

    Receives the tool name, kwargs, and output. Returns a GuardrailResult.
    Can block the output, replace it, or allow it.
    """

    name: str

    def check(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        output: Any,
        context: Dict[str, Any],
    ) -> GuardrailResult: ...


# Async versions for LLM-backed guardrails
AsyncPreCallCheck = Callable[[str, Dict[str, Any], Dict[str, Any]], Awaitable[GuardrailResult]]
AsyncPostCallCheck = Callable[
    [str, Dict[str, Any], Any, Dict[str, Any]], Awaitable[GuardrailResult]
]
