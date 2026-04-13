"""Clawboss guardrails — deterministic and LLM-backed safety checks.

Guardrails hook into Supervisor.call() as pre-call and post-call checks.
Pre-call checks can block a tool from executing. Post-call checks can
block the result from reaching the agent.

Two flavors:
- Deterministic: rule-based, fast, zero-overhead when disabled
- Non-deterministic: LLM-backed, bring-your-own-LLM, opt-in

Deterministic:
- SchemaValidator — enforce JSON schema on tool outputs
- CategoryRateLimit — rate limit across tool categories
- RecursionDetector — detect tool call loops
- IdempotencyGuard — dedupe retry calls
- ResourceQuota — CPU/memory/time caps
- OutputLengthLimit — block oversized outputs
- UrlGuard — allowlist/blocklist URLs
- ActiveHours — time-of-day restrictions

Non-deterministic (bring-your-own-LLM):
- PromptInjectionDetector — catch injection attempts
- SafetyClassifier — toxic content / bias
- IntentDriftDetector — off-task detection
- SemanticPiiRedactor — NLP-based PII
- AnomalyScorer — LLM anomaly scoring
- GoalVerifier — "is this on-task?"
- ExplanationRequired — force justification
- EnsembleDecision — multi-LLM agreement

All guardrails return a GuardrailResult — (allowed, reason) — so the
Supervisor can uniformly block or allow.
"""

from .deterministic import (
    ActiveHours,
    CategoryRateLimit,
    IdempotencyGuard,
    OutputLengthLimit,
    RecursionDetector,
    ResourceQuota,
    SchemaValidator,
    UrlGuard,
)
from .llm_based import (
    AnomalyScorer,
    EnsembleDecision,
    ExplanationRequired,
    GoalVerifier,
    IntentDriftDetector,
    PromptInjectionDetector,
    SafetyClassifier,
    SemanticPiiRedactor,
)
from .types import GuardrailResult, PostCallGuardrail, PreCallGuardrail

__all__ = [
    "GuardrailResult",
    "PreCallGuardrail",
    "PostCallGuardrail",
    # Deterministic
    "SchemaValidator",
    "CategoryRateLimit",
    "RecursionDetector",
    "IdempotencyGuard",
    "ResourceQuota",
    "OutputLengthLimit",
    "UrlGuard",
    "ActiveHours",
    # LLM-based
    "PromptInjectionDetector",
    "SafetyClassifier",
    "IntentDriftDetector",
    "SemanticPiiRedactor",
    "AnomalyScorer",
    "GoalVerifier",
    "ExplanationRequired",
    "EnsembleDecision",
]
