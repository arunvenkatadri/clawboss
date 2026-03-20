"""Clawboss error types — every error has a user_message()."""

from enum import Enum
from typing import Optional


class ClawbossError(Exception):
    """Base error for all clawboss supervision failures."""

    def __init__(self, kind: str, message: str, **details):
        self.kind = kind
        self.details = details
        super().__init__(message)

    def user_message(self) -> str:
        return str(self)

    @staticmethod
    def timeout(timeout_ms: int) -> "ClawbossError":
        return ClawbossError(
            "timeout",
            f"Tool call timed out after {timeout_ms}ms",
            timeout_ms=timeout_ms,
        )

    @staticmethod
    def budget_exceeded(used: int, limit: int) -> "ClawbossError":
        return ClawbossError(
            "budget_exceeded",
            f"Token budget exceeded: {used} / {limit}",
            used=used, limit=limit,
        )

    @staticmethod
    def max_iterations(iterations: int, max_iter: int) -> "ClawbossError":
        return ClawbossError(
            "max_iterations",
            f"Maximum iterations reached: {iterations} / {max_iter}",
            iterations=iterations, max=max_iter,
        )

    @staticmethod
    def circuit_open(tool: str, consecutive_failures: int) -> "ClawbossError":
        return ClawbossError(
            "circuit_open",
            f"Circuit breaker open for '{tool}': {consecutive_failures} consecutive failures",
            tool=tool, consecutive_failures=consecutive_failures,
        )

    @staticmethod
    def tool_error(message: str) -> "ClawbossError":
        return ClawbossError("tool_error", message)

    @staticmethod
    def dead_man_switch(silence_ms: int) -> "ClawbossError":
        return ClawbossError(
            "dead_man_switch",
            f"Dead man's switch: no activity for {silence_ms}ms",
            silence_ms=silence_ms,
        )

    @staticmethod
    def policy_denied(reason: str) -> "ClawbossError":
        return ClawbossError("policy_denied", f"Policy denied: {reason}", reason=reason)
