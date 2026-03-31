"""Agent observability — structured telemetry for tool calls and sessions.

Collects per-tool and per-session metrics: latency, success/failure rates,
token cost, call counts. Optional OpenTelemetry export for Datadog, Grafana,
Honeycomb, etc.

Zero overhead when disabled. No required dependencies.

Usage:
    from clawboss.observe import Observer

    obs = Observer()
    obs.record_tool_call("web_search", duration_ms=120, succeeded=True, tokens=500)
    obs.record_tool_call("web_search", duration_ms=95, succeeded=True, tokens=300)

    summary = obs.tool_summary("web_search")
    # {"calls": 2, "successes": 2, "failures": 0, "avg_latency_ms": 107.5, ...}

    all_metrics = obs.session_summary("session-123")

OpenTelemetry (optional):
    from clawboss.observe import Observer
    obs = Observer(otlp_endpoint="http://localhost:4317")
    # Spans and metrics exported automatically
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallRecord:
    """A single observed tool call."""

    tool_name: str
    session_id: str = ""
    duration_ms: int = 0
    succeeded: bool = True
    tokens: int = 0
    timestamp: float = 0.0
    error_kind: str = ""

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
    total_tokens: int = 0
    min_latency_ms: int = 0
    max_latency_ms: int = 0
    error_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls > 0 else 0.0

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
            "error_counts": dict(self.error_counts),
        }


@dataclass
class SessionMetrics:
    """Aggregated metrics for a session."""

    session_id: str
    total_calls: int = 0
    total_tokens: int = 0
    total_duration_ms: int = 0
    tool_metrics: Dict[str, ToolMetrics] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "tools": {k: v.to_dict() for k, v in self.tool_metrics.items()},
        }


class Observer:
    """Collects and aggregates tool call telemetry.

    Thread-safe. Attach to a Supervisor or use standalone.

    Args:
        otlp_endpoint: If set, export spans/metrics via OpenTelemetry.
                       Requires opentelemetry-sdk (optional dep).
                       Example: "http://localhost:4317"
    """

    def __init__(self, otlp_endpoint: Optional[str] = None):
        self._records: List[ToolCallRecord] = []
        self._lock = threading.Lock()
        self._otel_tracer = None
        self._otel_meter = None

        if otlp_endpoint:
            self._init_otel(otlp_endpoint)

    def _init_otel(self, endpoint: str) -> None:
        """Initialize OpenTelemetry tracer and meter. Fails silently if not installed."""
        try:
            from opentelemetry import metrics, trace
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            resource = Resource.create({"service.name": "clawboss"})

            # Traces
            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
            trace.set_tracer_provider(tracer_provider)
            self._otel_tracer = trace.get_tracer("clawboss")

            # Metrics
            reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint))
            meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(meter_provider)
            self._otel_meter = metrics.get_meter("clawboss")
        except ImportError:
            pass  # OpenTelemetry not installed — that's fine

    def record_tool_call(
        self,
        tool_name: str,
        duration_ms: int = 0,
        succeeded: bool = True,
        tokens: int = 0,
        session_id: str = "",
        error_kind: str = "",
    ) -> None:
        """Record a tool call observation."""
        record = ToolCallRecord(
            tool_name=tool_name,
            session_id=session_id,
            duration_ms=duration_ms,
            succeeded=succeeded,
            tokens=tokens,
            error_kind=error_kind,
        )
        with self._lock:
            self._records.append(record)

        # Export to OpenTelemetry if configured
        if self._otel_tracer:
            with self._otel_tracer.start_as_current_span(f"tool_call.{tool_name}") as span:
                span.set_attribute("tool.name", tool_name)
                span.set_attribute("tool.duration_ms", duration_ms)
                span.set_attribute("tool.succeeded", succeeded)
                span.set_attribute("tool.tokens", tokens)
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
            metrics.total_tokens += r.tokens
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

    def recent_calls(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the most recent tool call records."""
        with self._lock:
            records = self._records[-limit:]
        return [
            {
                "tool_name": r.tool_name,
                "session_id": r.session_id,
                "duration_ms": r.duration_ms,
                "succeeded": r.succeeded,
                "tokens": r.tokens,
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
            m.total_tokens += r.tokens
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
