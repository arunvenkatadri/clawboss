"""Deterministic guardrails — rule-based safety checks.

All eight run synchronously with zero overhead when not configured.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from .types import GuardrailResult

# ---------------------------------------------------------------------------
# 1. Output schema validation
# ---------------------------------------------------------------------------


class SchemaValidator:
    """Validate tool outputs against a JSON schema.

    Blocks outputs that don't match the expected structure. Catches
    broken tools, corrupted data, or LLM hallucinations.

    Supports a minimal JSON Schema subset: type, required, properties,
    items, minLength, maxLength, minimum, maximum, enum, pattern.

    Usage:
        validator = SchemaValidator({
            "search_results": {
                "type": "object",
                "required": ["rows"],
                "properties": {
                    "rows": {"type": "array"},
                    "count": {"type": "integer", "minimum": 0},
                },
            },
        })
    """

    def __init__(self, schemas: Optional[Dict[str, Dict[str, Any]]] = None):
        self.name = "schema_validator"
        self._schemas = schemas or {}

    def check(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        output: Any,
        context: Dict[str, Any],
    ) -> GuardrailResult:
        schema = self._schemas.get(tool_name)
        if schema is None:
            return GuardrailResult.allow(self.name)

        error = _validate_json_schema(output, schema)
        if error:
            return GuardrailResult.block(f"Output schema validation failed: {error}", self.name)
        return GuardrailResult.allow(self.name)


def _validate_json_schema(value: Any, schema: Dict[str, Any], path: str = "$") -> str:
    """Minimal JSON Schema validator. Returns empty string if valid, error if not."""
    stype = schema.get("type")
    if stype:
        type_map: Dict[str, Any] = {
            "object": dict,
            "array": list,
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "null": type(None),
        }
        expected = type_map.get(stype)
        if expected is not None and not isinstance(value, expected):
            return f"{path}: expected {stype}, got {type(value).__name__}"

    if stype == "object" and isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                return f"{path}: missing required field '{req}'"
        for key, subschema in schema.get("properties", {}).items():
            if key in value:
                err = _validate_json_schema(value[key], subschema, f"{path}.{key}")
                if err:
                    return err

    if stype == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                err = _validate_json_schema(item, item_schema, f"{path}[{i}]")
                if err:
                    return err

    if isinstance(value, str):
        min_len = schema.get("minLength")
        max_len = schema.get("maxLength")
        if min_len is not None and len(value) < min_len:
            return f"{path}: string too short (< {min_len})"
        if max_len is not None and len(value) > max_len:
            return f"{path}: string too long (> {max_len})"
        pattern = schema.get("pattern")
        if pattern and not re.search(pattern, value):
            return f"{path}: does not match pattern {pattern}"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            return f"{path}: value {value} < minimum {minimum}"
        if maximum is not None and value > maximum:
            return f"{path}: value {value} > maximum {maximum}"

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        return f"{path}: value {value} not in enum {enum}"

    return ""


# ---------------------------------------------------------------------------
# 2. Category rate limiting
# ---------------------------------------------------------------------------


@dataclass
class CategoryRateLimit:
    """Rate limit across tool categories.

    Unlike per-tool rate limits, this tracks calls across related tools.
    For example, limit all "network" tools combined to 100/minute.

    Usage:
        CategoryRateLimit(
            categories={
                "network": ["web_search", "fetch_url", "http_post"],
                "writes": ["write_file", "delete_file", "send_email"],
            },
            limits={"network": 100, "writes": 10},  # per minute
        )
    """

    categories: Dict[str, List[str]] = field(default_factory=dict)
    limits: Dict[str, int] = field(default_factory=dict)
    _call_times: Dict[str, List[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    name: str = "category_rate_limit"

    def check(
        self, tool_name: str, kwargs: Dict[str, Any], context: Dict[str, Any]
    ) -> GuardrailResult:
        now = time.monotonic()
        # Find which categories this tool belongs to
        for category, tools in self.categories.items():
            if tool_name in tools:
                limit = self.limits.get(category)
                if limit is None:
                    continue
                with self._lock:
                    times = self._call_times.setdefault(category, [])
                    cutoff = now - 60.0
                    recent = [t for t in times if t > cutoff]
                    if len(recent) >= limit:
                        return GuardrailResult.block(
                            f"Category '{category}' rate limit exceeded: "
                            f"{len(recent)}/{limit} per minute",
                            self.name,
                        )
                    recent.append(now)
                    self._call_times[category] = recent
        return GuardrailResult.allow(self.name)


# ---------------------------------------------------------------------------
# 3. Recursion / loop detection
# ---------------------------------------------------------------------------


@dataclass
class RecursionDetector:
    """Detect tool call loops.

    Tracks recent tool call sequences. Blocks if the same (tool, args_hash)
    appears more than ``max_repeats`` times within ``window`` seconds.

    Catches agents that get stuck calling the same thing over and over.
    """

    max_repeats: int = 3
    window: float = 30.0
    _history: List[Tuple[float, str, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    name: str = "recursion_detector"

    def check(
        self, tool_name: str, kwargs: Dict[str, Any], context: Dict[str, Any]
    ) -> GuardrailResult:
        now = time.monotonic()
        sig = hashlib.sha256(
            (tool_name + json.dumps(kwargs, sort_keys=True, default=str)).encode()
        ).hexdigest()[:16]

        with self._lock:
            cutoff = now - self.window
            self._history = [(t, n, s) for t, n, s in self._history if t > cutoff]
            repeats = sum(1 for _, n, s in self._history if n == tool_name and s == sig)
            if repeats >= self.max_repeats:
                return GuardrailResult.block(
                    f"Recursion detected: {tool_name} called {repeats + 1} times "
                    f"with identical args in {self.window}s",
                    self.name,
                )
            self._history.append((now, tool_name, sig))
        return GuardrailResult.allow(self.name)


# ---------------------------------------------------------------------------
# 4. Idempotency keys
# ---------------------------------------------------------------------------


@dataclass
class IdempotencyGuard:
    """Cache results of mutating tool calls to dedupe retries.

    If the same (tool, args) is called within the TTL window, returns
    the cached result instead of executing again. Useful for tools that
    are expensive or have side effects (send_email, charge_card).

    Only applies to tools in ``tracked_tools``.
    """

    tracked_tools: List[str] = field(default_factory=list)
    ttl_seconds: float = 300.0
    _cache: Dict[str, Tuple[float, Any]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    name: str = "idempotency_guard"

    def _key(self, tool_name: str, kwargs: Dict[str, Any]) -> str:
        return hashlib.sha256(
            (tool_name + json.dumps(kwargs, sort_keys=True, default=str)).encode()
        ).hexdigest()

    def check_pre(
        self, tool_name: str, kwargs: Dict[str, Any], context: Dict[str, Any]
    ) -> GuardrailResult:
        """Pre-call: return cached result if this is a retry."""
        if tool_name not in self.tracked_tools:
            return GuardrailResult.allow(self.name)
        now = time.monotonic()
        key = self._key(tool_name, kwargs)
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] < self.ttl_seconds:
                return GuardrailResult.replace(
                    cached[1],
                    reason=f"Returned cached result for {tool_name} (idempotency)",
                    guardrail_name=self.name,
                )
        return GuardrailResult.allow(self.name)

    def check(
        self, tool_name: str, kwargs: Dict[str, Any], context: Dict[str, Any]
    ) -> GuardrailResult:
        return self.check_pre(tool_name, kwargs, context)

    def record(self, tool_name: str, kwargs: Dict[str, Any], output: Any) -> None:
        """Called after a successful tool call to cache the result."""
        if tool_name not in self.tracked_tools:
            return
        key = self._key(tool_name, kwargs)
        with self._lock:
            self._cache[key] = (time.monotonic(), output)


# ---------------------------------------------------------------------------
# 5. Resource quotas
# ---------------------------------------------------------------------------


@dataclass
class ResourceQuota:
    """Per-session CPU time and memory caps.

    Tracks cumulative CPU time and peak memory across all tool calls
    in a session. Blocks new calls when the cap is reached.
    """

    max_cpu_seconds: Optional[float] = None
    max_memory_mb: Optional[float] = None
    _cpu_used: Dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    name: str = "resource_quota"

    def check(
        self, tool_name: str, kwargs: Dict[str, Any], context: Dict[str, Any]
    ) -> GuardrailResult:
        session_id = context.get("session_id", "default")
        with self._lock:
            used = self._cpu_used.get(session_id, 0.0)
        if self.max_cpu_seconds is not None and used >= self.max_cpu_seconds:
            return GuardrailResult.block(
                f"CPU quota exceeded: {used:.2f}s / {self.max_cpu_seconds}s",
                self.name,
            )

        if self.max_memory_mb is not None:
            try:
                import resource

                rusage = resource.getrusage(resource.RUSAGE_SELF)
                # ru_maxrss is in KB on Linux, bytes on macOS
                import sys

                mem_mb = rusage.ru_maxrss / (1024 if sys.platform == "linux" else 1024 * 1024)
                if mem_mb > self.max_memory_mb:
                    return GuardrailResult.block(
                        f"Memory quota exceeded: {mem_mb:.1f}MB / {self.max_memory_mb}MB",
                        self.name,
                    )
            except ImportError:
                pass  # resource module not available (Windows)
        return GuardrailResult.allow(self.name)

    def record_cpu(self, session_id: str, seconds: float) -> None:
        with self._lock:
            self._cpu_used[session_id] = self._cpu_used.get(session_id, 0.0) + seconds


# ---------------------------------------------------------------------------
# 6. Output length limits
# ---------------------------------------------------------------------------


@dataclass
class OutputLengthLimit:
    """Block tool outputs that exceed a size limit.

    Prevents context blowup attacks where a tool returns an enormous
    payload to poison the agent's context window.
    """

    max_chars: int = 100_000
    max_tokens: Optional[int] = None  # rough estimate: chars / 4
    name: str = "output_length_limit"

    def check(
        self,
        tool_name: str,
        kwargs: Dict[str, Any],
        output: Any,
        context: Dict[str, Any],
    ) -> GuardrailResult:
        text = json.dumps(output, default=str) if not isinstance(output, str) else output
        if len(text) > self.max_chars:
            return GuardrailResult.block(
                f"Output exceeds {self.max_chars} chars ({len(text)} chars)",
                self.name,
            )
        if self.max_tokens is not None:
            est_tokens = len(text) // 4
            if est_tokens > self.max_tokens:
                return GuardrailResult.block(
                    f"Output exceeds ~{self.max_tokens} tokens (~{est_tokens})",
                    self.name,
                )
        return GuardrailResult.allow(self.name)


# ---------------------------------------------------------------------------
# 7. URL allowlist / blocklist
# ---------------------------------------------------------------------------


@dataclass
class UrlGuard:
    """Allowlist or blocklist URLs/domains for tool arguments.

    Scans string kwargs for URLs and checks each against the lists.
    Patterns support wildcards (``*.example.com``).

    Usage:
        UrlGuard(
            allowlist=["*.mycompany.com", "api.trusted.com"],
            blocklist=["*.internal.com"],
        )
    """

    allowlist: List[str] = field(default_factory=list)
    blocklist: List[str] = field(default_factory=list)
    name: str = "url_guard"

    def _extract_urls(self, text: str) -> List[str]:
        return re.findall(r"https?://[^\s\"'<>]+", text)

    def _domain_matches(self, domain: str, patterns: List[str]) -> bool:
        for pat in patterns:
            if fnmatch.fnmatch(domain, pat):
                return True
        return False

    def check(
        self, tool_name: str, kwargs: Dict[str, Any], context: Dict[str, Any]
    ) -> GuardrailResult:
        urls: Set[str] = set()
        for v in kwargs.values():
            if isinstance(v, str):
                urls.update(self._extract_urls(v))
                # Also handle bare URLs
                if v.startswith("http://") or v.startswith("https://"):
                    urls.add(v)

        for url in urls:
            try:
                domain = urlparse(url).netloc.lower()
            except Exception:
                continue
            if self.blocklist and self._domain_matches(domain, self.blocklist):
                return GuardrailResult.block(f"URL blocked: {domain}", self.name)
            if self.allowlist and not self._domain_matches(domain, self.allowlist):
                return GuardrailResult.block(f"URL not in allowlist: {domain}", self.name)
        return GuardrailResult.allow(self.name)


# ---------------------------------------------------------------------------
# 8. Active hours
# ---------------------------------------------------------------------------


@dataclass
class ActiveHours:
    """Restrict when an agent can run (compliance use case).

    Blocks all tool calls outside of the configured hours.
    Useful for agents that should only run during business hours.

    Args:
        start_hour: 0-23, inclusive (UTC).
        end_hour: 0-23, exclusive (UTC).
        days_of_week: list of 0-6 (Mon=0). None = all days.
    """

    start_hour: int = 0
    end_hour: int = 24
    days_of_week: Optional[List[int]] = None
    name: str = "active_hours"

    def check(
        self, tool_name: str, kwargs: Dict[str, Any], context: Dict[str, Any]
    ) -> GuardrailResult:
        now = datetime.now(timezone.utc)
        if self.days_of_week is not None and now.weekday() not in self.days_of_week:
            return GuardrailResult.block(
                f"Outside active days: {now.strftime('%A')} not allowed",
                self.name,
            )
        if not (self.start_hour <= now.hour < self.end_hour):
            return GuardrailResult.block(
                f"Outside active hours: {now.hour}:00 not in "
                f"{self.start_hour}:00-{self.end_hour}:00 UTC",
                self.name,
            )
        return GuardrailResult.allow(self.name)
