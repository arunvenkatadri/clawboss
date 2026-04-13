"""Agent observability — structured telemetry for tool calls and sessions.

Collects per-tool and per-session metrics: latency, success/failure rates,
token counts, dollar cost. Optional OpenTelemetry export for Datadog, Grafana,
Honeycomb, etc.

Zero overhead when disabled. No required dependencies.

Usage:
    from clawboss.observe import Observer, PricingTable

    # Optional: configure pricing for real dollar cost attribution
    pricing = PricingTable.default()  # or PricingTable(models={...})

    obs = Observer(pricing=pricing)
    obs.record_tool_call(
        "llm_call",
        duration_ms=120,
        succeeded=True,
        input_tokens=500,
        output_tokens=200,
        model="claude-sonnet-4-6",
    )

    summary = obs.tool_summary("llm_call")
    # {"calls": 1, "total_input_tokens": 500, "total_cost_usd": 0.0035, ...}

OpenTelemetry (optional):
    obs = Observer(otlp_endpoint="http://localhost:4317")
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------


@dataclass
class ModelPricing:
    """Pricing for one model, in USD per million tokens."""

    input_per_million: float = 0.0
    output_per_million: float = 0.0

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * self.input_per_million + (
            output_tokens / 1_000_000
        ) * self.output_per_million


@dataclass
class PricingTable:
    """Model → pricing map, with a default fallback.

    Use ``PricingTable.default()`` for a built-in table covering common models,
    or construct your own with ``PricingTable(models={...})``.
    """

    models: Dict[str, ModelPricing] = field(default_factory=dict)
    default_model: str = "claude-sonnet-4-6"

    @classmethod
    def default(cls) -> "PricingTable":
        """Built-in pricing for common models (USD per million tokens).

        Approximate pricing as of 2026. Update as needed for your use case.
        """
        return cls(
            models={
                # Anthropic
                "claude-opus-4-6": ModelPricing(input_per_million=15.0, output_per_million=75.0),
                "claude-sonnet-4-6": ModelPricing(input_per_million=3.0, output_per_million=15.0),
                "claude-haiku-4-5": ModelPricing(input_per_million=0.25, output_per_million=1.25),
                # OpenAI
                "gpt-4o": ModelPricing(input_per_million=2.5, output_per_million=10.0),
                "gpt-4o-mini": ModelPricing(input_per_million=0.15, output_per_million=0.6),
                "gpt-4-turbo": ModelPricing(input_per_million=10.0, output_per_million=30.0),
                # Google
                "gemini-1.5-pro": ModelPricing(input_per_million=1.25, output_per_million=5.0),
                "gemini-1.5-flash": ModelPricing(input_per_million=0.075, output_per_million=0.3),
            }
        )

    def cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Compute dollar cost for a tool call. Returns 0 if model not in table."""
        pricing = self.models.get(model)
        if pricing is None:
            return 0.0
        return pricing.cost_usd(input_tokens, output_tokens)

    def set_model(self, model: str, input_per_million: float, output_per_million: float) -> None:
        """Add or update a model's pricing."""
        self.models[model] = ModelPricing(input_per_million, output_per_million)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "models": {
                name: {
                    "input_per_million": p.input_per_million,
                    "output_per_million": p.output_per_million,
                }
                for name, p in self.models.items()
            },
            "default_model": self.default_model,
        }


# ---------------------------------------------------------------------------
# Records and metrics
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    """A single observed tool call."""

    tool_name: str
    session_id: str = ""
    agent_id: str = ""
    duration_ms: int = 0
    succeeded: bool = True
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    timestamp: float = 0.0
    error_kind: str = ""

    @property
    def tokens(self) -> int:
        """Backward-compat: total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class ToolMetrics:
    """Aggregated metrics for a single tool."""

    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    min_latency_ms: int = 0
    max_latency_ms: int = 0
    error_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls > 0 else 0.0

    @property
    def avg_cost_usd(self) -> float:
        return self.total_cost_usd / self.calls if self.calls > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_cost_usd": round(self.avg_cost_usd, 6),
            "error_counts": dict(self.error_counts),
        }


@dataclass
class SessionMetrics:
    """Aggregated metrics for a session."""

    session_id: str
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    tool_metrics: Dict[str, ToolMetrics] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_duration_ms": self.total_duration_ms,
            "tools": {k: v.to_dict() for k, v in self.tool_metrics.items()},
        }


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------


class Observer:
    """Collects and aggregates tool call telemetry.

    Thread-safe. Attach to a Supervisor or use standalone.

    Args:
        otlp_endpoint: If set, export spans/metrics via OpenTelemetry.
                       Requires opentelemetry-sdk (optional dep).
                       Example: "http://localhost:4317"
        pricing: Optional PricingTable for dollar cost attribution.
                 Without it, costs are always 0.0.
    """

    def __init__(
        self,
        otlp_endpoint: Optional[str] = None,
        pricing: Optional[PricingTable] = None,
    ):
        self._records: List[ToolCallRecord] = []
        self._lock = threading.Lock()
        self._otel_tracer = None
        self._otel_meter = None
        self._pricing = pricing

        if otlp_endpoint:
            self._init_otel(otlp_endpoint)

    @property
    def pricing(self) -> Optional[PricingTable]:
        return self._pricing

    def set_pricing(self, pricing: PricingTable) -> None:
        self._pricing = pricing

    def _init_otel(self, endpoint: str) -> None:
        """Initialize OpenTelemetry tracer and meter. Fails silently if not installed."""
        try:
            from opentelemetry import metrics, trace  # type: ignore[import-not-found]
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (  # type: ignore[import-not-found]
                OTLPMetricExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider  # type: ignore[import-not-found]
            from opentelemetry.sdk.metrics.export import (  # type: ignore[import-not-found]
                PeriodicExportingMetricReader,
            )
            from opentelemetry.sdk.resources import Resource  # type: ignore[import-not-found]
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-not-found]
            from opentelemetry.sdk.trace.export import (  # type: ignore[import-not-found]
                BatchSpanProcessor,
            )

            resource = Resource.create({"service.name": "clawboss"})

            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
            trace.set_tracer_provider(tracer_provider)
            self._otel_tracer = trace.get_tracer("clawboss")

            reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint))
            meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(meter_provider)
            self._otel_meter = metrics.get_meter("clawboss")
        except ImportError:
            pass

    def record_tool_call(
        self,
        tool_name: str,
        duration_ms: int = 0,
        succeeded: bool = True,
        tokens: int = 0,
        session_id: str = "",
        agent_id: str = "",
        error_kind: str = "",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        model: str = "",
    ) -> None:
        """Record a tool call observation.

        Backward-compatible: if input_tokens/output_tokens aren't provided,
        the legacy ``tokens`` value is treated as output tokens.
        """
        if input_tokens is None and output_tokens is None:
            # Legacy path — single tokens value counts as output
            in_tok = 0
            out_tok = tokens
        else:
            in_tok = input_tokens or 0
            out_tok = output_tokens or 0

        cost = 0.0
        if self._pricing and model:
            cost = self._pricing.cost_usd(model, in_tok, out_tok)

        record = ToolCallRecord(
            tool_name=tool_name,
            session_id=session_id,
            agent_id=agent_id,
            duration_ms=duration_ms,
            succeeded=succeeded,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            model=model,
            error_kind=error_kind,
        )
        with self._lock:
            self._records.append(record)

        if self._otel_tracer:
            with self._otel_tracer.start_as_current_span(f"tool_call.{tool_name}") as span:
                span.set_attribute("tool.name", tool_name)
                span.set_attribute("tool.duration_ms", duration_ms)
                span.set_attribute("tool.succeeded", succeeded)
                span.set_attribute("tool.input_tokens", in_tok)
                span.set_attribute("tool.output_tokens", out_tok)
                span.set_attribute("tool.cost_usd", cost)
                if model:
                    span.set_attribute("tool.model", model)
                if session_id:
                    span.set_attribute("session.id", session_id)
                if error_kind:
                    span.set_attribute("error.kind", error_kind)

    def tool_summary(self, tool_name: str) -> Dict[str, Any]:
        """Get aggregated metrics for a specific tool."""
        with self._lock:
            records = [r for r in self._records if r.tool_name == tool_name]
        return self._aggregate(records).to_dict()

    def session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get aggregated metrics for a specific session."""
        with self._lock:
            records = [r for r in self._records if r.session_id == session_id]

        metrics = SessionMetrics(session_id=session_id)
        by_tool: Dict[str, List[ToolCallRecord]] = defaultdict(list)
        for r in records:
            by_tool[r.tool_name].append(r)
            metrics.total_calls += 1
            metrics.total_input_tokens += r.input_tokens
            metrics.total_output_tokens += r.output_tokens
            metrics.total_cost_usd += r.cost_usd
            metrics.total_duration_ms += r.duration_ms

        for name, tool_records in by_tool.items():
            metrics.tool_metrics[name] = self._aggregate(tool_records)

        return metrics.to_dict()

    def all_tools_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get aggregated metrics for all tools."""
        with self._lock:
            records = list(self._records)

        by_tool: Dict[str, List[ToolCallRecord]] = defaultdict(list)
        for r in records:
            by_tool[r.tool_name].append(r)

        return {name: self._aggregate(recs).to_dict() for name, recs in by_tool.items()}

    def cost_summary(self) -> Dict[str, Any]:
        """Total cost breakdown by agent, session, and tool.

        Returns a dict suitable for the dashboard Costs tab.
        """
        with self._lock:
            records = list(self._records)

        total_cost = sum(r.cost_usd for r in records)
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)

        # By agent
        by_agent: Dict[str, Dict[str, Any]] = {}
        for r in records:
            key = r.agent_id or "(unknown)"
            if key not in by_agent:
                by_agent[key] = {"agent": key, "calls": 0, "tokens": 0, "cost": 0.0}
            by_agent[key]["calls"] += 1
            by_agent[key]["tokens"] += r.input_tokens + r.output_tokens
            by_agent[key]["cost"] += r.cost_usd

        # By session
        by_session: Dict[str, Dict[str, Any]] = {}
        for r in records:
            key = r.session_id or "(unknown)"
            if key not in by_session:
                by_session[key] = {"session_id": key, "calls": 0, "tokens": 0, "cost": 0.0}
            by_session[key]["calls"] += 1
            by_session[key]["tokens"] += r.input_tokens + r.output_tokens
            by_session[key]["cost"] += r.cost_usd

        # By tool
        by_tool: Dict[str, Dict[str, Any]] = {}
        for r in records:
            key = r.tool_name
            if key not in by_tool:
                by_tool[key] = {"tool": key, "calls": 0, "tokens": 0, "cost": 0.0}
            by_tool[key]["calls"] += 1
            by_tool[key]["tokens"] += r.input_tokens + r.output_tokens
            by_tool[key]["cost"] += r.cost_usd

        # By model
        by_model: Dict[str, Dict[str, Any]] = {}
        for r in records:
            if not r.model:
                continue
            key = r.model
            if key not in by_model:
                by_model[key] = {
                    "model": key,
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                }
            by_model[key]["calls"] += 1
            by_model[key]["input_tokens"] += r.input_tokens
            by_model[key]["output_tokens"] += r.output_tokens
            by_model[key]["cost"] += r.cost_usd

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_calls": len(records),
            "by_agent": [
                {**v, "cost": round(v["cost"], 6)}
                for v in sorted(by_agent.values(), key=lambda x: -x["cost"])
            ],
            "by_session": [
                {**v, "cost": round(v["cost"], 6)}
                for v in sorted(by_session.values(), key=lambda x: -x["cost"])
            ],
            "by_tool": [
                {**v, "cost": round(v["cost"], 6)}
                for v in sorted(by_tool.values(), key=lambda x: -x["cost"])
            ],
            "by_model": [
                {**v, "cost": round(v["cost"], 6)}
                for v in sorted(by_model.values(), key=lambda x: -x["cost"])
            ],
        }

    def recent_calls(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the most recent tool call records."""
        with self._lock:
            records = self._records[-limit:]
        return [
            {
                "tool_name": r.tool_name,
                "session_id": r.session_id,
                "agent_id": r.agent_id,
                "duration_ms": r.duration_ms,
                "succeeded": r.succeeded,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "tokens": r.tokens,
                "cost_usd": round(r.cost_usd, 6),
                "model": r.model,
                "error_kind": r.error_kind,
                "timestamp": r.timestamp,
            }
            for r in reversed(records)
        ]

    @staticmethod
    def _aggregate(records: List[ToolCallRecord]) -> ToolMetrics:
        if not records:
            return ToolMetrics()
        m = ToolMetrics()
        m.min_latency_ms = records[0].duration_ms
        for r in records:
            m.calls += 1
            m.total_latency_ms += r.duration_ms
            m.total_input_tokens += r.input_tokens
            m.total_output_tokens += r.output_tokens
            m.total_cost_usd += r.cost_usd
            if r.succeeded:
                m.successes += 1
            else:
                m.failures += 1
                if r.error_kind:
                    m.error_counts[r.error_kind] = m.error_counts.get(r.error_kind, 0) + 1
            if r.duration_ms < m.min_latency_ms:
                m.min_latency_ms = r.duration_ms
            if r.duration_ms > m.max_latency_ms:
                m.max_latency_ms = r.duration_ms
        return m
