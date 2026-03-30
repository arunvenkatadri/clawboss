"""Clawboss — supervision for AI agent tool execution.

Timeouts, budgets, circuit breakers, and audit for tool calls.
Works with any agent framework — just wrap your tool calls.

Usage:
    from clawboss import Supervisor, Policy

    policy = Policy(max_iterations=3, tool_timeout=15, token_budget=10000)
    supervisor = Supervisor(policy)

    # Supervise a tool call
    result = await supervisor.call("web_search", my_tool_fn, query="python async")
    if result.succeeded:
        print(result.output)
    else:
        print(result.error)
"""

__version__ = "0.1.0"

from .audit import AuditEntry, AuditLog, AuditOutcome, AuditPhase, AuditSink, JsonlAuditSink
from .budget import BudgetSnapshot, BudgetTracker
from .builder import SkillBuilder
from .circuit_breaker import CircuitBreaker, CircuitState
from .errors import ClawbossError
from .openclaw import OpenClawBridge, to_openclaw_manifest, to_openclaw_tool_schema
from .policy import Action, OnFailure, Policy
from .session import SessionManager
from .skill import Skill, SkillStore, ToolDefinition, ToolParameter
from .store import (
    Checkpoint,
    MemoryStore,
    SessionStatus,
    SqliteStore,
    StateStore,
    validate_payload,
)
from .supervisor import SupervisedResult, Supervisor

__all__ = [
    "Policy",
    "OnFailure",
    "Action",
    "Supervisor",
    "SupervisedResult",
    "BudgetTracker",
    "BudgetSnapshot",
    "CircuitBreaker",
    "CircuitState",
    "AuditLog",
    "AuditEntry",
    "AuditPhase",
    "AuditOutcome",
    "AuditSink",
    "JsonlAuditSink",
    "ClawbossError",
    "Skill",
    "SkillStore",
    "ToolDefinition",
    "ToolParameter",
    "SkillBuilder",
    "OpenClawBridge",
    "to_openclaw_tool_schema",
    "to_openclaw_manifest",
    "StateStore",
    "MemoryStore",
    "SqliteStore",
    "Checkpoint",
    "SessionStatus",
    "SessionManager",
    "validate_payload",
]
