"""AgentHandler — supervision for AI agent tool execution.

Timeouts, budgets, circuit breakers, and audit for tool calls.
Works with any agent framework — just wrap your tool calls.

Usage:
    from agenthandler import Supervisor, Policy

    policy = Policy(max_iterations=3, tool_timeout=15, token_budget=10000)
    supervisor = Supervisor(policy)

    # Supervise a tool call
    result = await supervisor.call("web_search", my_tool_fn, query="python async")
    if result.succeeded:
        print(result.output)
    else:
        print(result.error)
"""

__version__ = "0.92.0"

from .approval import ApprovalQueue, ApprovalRequest, ApprovalStatus
from .audit import AuditEntry, AuditLog, AuditOutcome, AuditPhase, AuditSink, JsonlAuditSink
from .budget import BudgetSnapshot, BudgetTracker
from .builder import SkillBuilder
from .circuit_breaker import CircuitBreaker, CircuitState
from .connectors import MongoConnector, SqlConnector
from .context import AnchoredState, CompressedContext, CompressedHistory, ContextWindow, Turn
from .errors import AgentHandlerError
from .guardrails import (
    ActiveHours,
    AnomalyScorer,
    CategoryRateLimit,
    EnsembleDecision,
    ExplanationRequired,
    GoalVerifier,
    GuardrailResult,
    IdempotencyGuard,
    IntentDriftDetector,
    OutputLengthLimit,
    PromptInjectionDetector,
    RecursionDetector,
    ResourceQuota,
    SafetyClassifier,
    SchemaValidator,
    SemanticPiiRedactor,
    UrlGuard,
)
from .observe import ModelPricing, Observer, PricingTable, ToolCallRecord, ToolMetrics
from .openclaw import OpenClawBridge, to_openclaw_manifest, to_openclaw_tool_schema
from .pipeline import Pipeline, PipelineResult
from .pipeline_poml import PipelineBuilder, parse_pipeline_poml
from .policy import Action, OnFailure, Policy, ScopeRule, ToolScope
from .redact import Redactor
from .reflection import ReflectionCycle, ReflectionLoop, ReflectionResult
from .replay import ReplayFrame, ReplaySummary, SessionReplay
from .sdk_adapters import (
    AgentHandlerMiddleware,
    openai_guardrail_adapter,
    supervised_tool_registry,
    wrap_claude_tool,
    wrap_openai_tool,
)
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
from .streams import (
    KafkaStreamConnector,
    KinesisStreamConnector,
    RedisStreamConnector,
    StreamConnector,
)
from .supervisor import SupervisedResult, Supervisor
from .triggers import DbWatchEntry, Scheduler, WebhookTrigger

try:
    from .mcp_server import SupervisedMCPServer
except ImportError:
    pass

try:
    from .a2a import A2AAgentCard, A2AClient, A2ASkill, A2ASupervisedEndpoint, A2ATask, TaskState
except ImportError:
    pass

__all__ = [
    "Policy",
    "OnFailure",
    "Action",
    "ScopeRule",
    "ToolScope",
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
    "AgentHandlerError",
    "Skill",
    "SkillStore",
    "ToolDefinition",
    "ToolParameter",
    "SkillBuilder",
    "ContextWindow",
    "CompressedContext",
    "CompressedHistory",
    "AnchoredState",
    "Turn",
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
    "ApprovalQueue",
    "ApprovalRequest",
    "ApprovalStatus",
    "Redactor",
    "Observer",
    "ToolCallRecord",
    "ToolMetrics",
    "PricingTable",
    "ModelPricing",
    "Pipeline",
    "PipelineResult",
    "PipelineBuilder",
    "parse_pipeline_poml",
    "SqlConnector",
    "MongoConnector",
    "Scheduler",
    "WebhookTrigger",
    "DbWatchEntry",
    "StreamConnector",
    "KafkaStreamConnector",
    "KinesisStreamConnector",
    "RedisStreamConnector",
    # Guardrails — deterministic
    "GuardrailResult",
    "SchemaValidator",
    "CategoryRateLimit",
    "RecursionDetector",
    "IdempotencyGuard",
    "ResourceQuota",
    "OutputLengthLimit",
    "UrlGuard",
    "ActiveHours",
    # Guardrails — LLM-based
    "PromptInjectionDetector",
    "SafetyClassifier",
    "IntentDriftDetector",
    "SemanticPiiRedactor",
    "AnomalyScorer",
    "GoalVerifier",
    "ExplanationRequired",
    "EnsembleDecision",
    # Long-duration features
    "ReflectionLoop",
    "ReflectionCycle",
    "ReflectionResult",
    "SessionReplay",
    "ReplayFrame",
    "ReplaySummary",
    # MCP
    "SupervisedMCPServer",
    # A2A
    "A2AAgentCard",
    "A2AClient",
    "A2ASkill",
    "A2ASupervisedEndpoint",
    "A2ATask",
    "TaskState",
    # SDK adapters
    "wrap_openai_tool",
    "wrap_claude_tool",
    "openai_guardrail_adapter",
    "supervised_tool_registry",
    "AgentHandlerMiddleware",
]
