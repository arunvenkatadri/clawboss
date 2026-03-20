"""Circuit breaker — stops calling tools that keep failing."""

import time
import threading
from enum import Enum
from typing import Optional

from .errors import ClawbossError


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Blocking calls (too many failures)
    HALF_OPEN = "half_open" # Allowing one test call


class CircuitBreaker:
    """Per-tool circuit breaker. Thread-safe.

    States:
        CLOSED  → failures < threshold, calls pass through
        OPEN    → failures >= threshold, calls blocked until reset_after
        HALF_OPEN → reset_after elapsed, allow one test call

    Usage:
        cb = CircuitBreaker(threshold=3, reset_after=60.0)
        cb.check("web_search")      # raises if circuit open
        try:
            result = call_tool()
            cb.record_success()
        except:
            cb.record_failure()
    """

    def __init__(self, threshold: int = 5, reset_after: float = 60.0):
        self._threshold = threshold
        self._reset_after = reset_after
        self._consecutive_failures = 0
        self._state = CircuitState.CLOSED
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    def check(self, tool_name: str) -> None:
        """Check if the circuit allows a call. Raises ClawbossError if open."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return
            if self._state == CircuitState.OPEN:
                # Check if enough time has passed to try again
                if self._opened_at and (time.monotonic() - self._opened_at) >= self._reset_after:
                    self._state = CircuitState.HALF_OPEN
                    return
                raise ClawbossError.circuit_open(tool_name, self._consecutive_failures)
            # HALF_OPEN: allow one test call
            return

    def record_success(self) -> None:
        """Record a successful call. Resets the breaker."""
        with self._lock:
            self._consecutive_failures = 0
            self._state = CircuitState.CLOSED
            self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed call. May trip the breaker."""
        with self._lock:
            self._consecutive_failures += 1
            if self._state == CircuitState.HALF_OPEN:
                # Test call failed, reopen
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
            elif self._consecutive_failures >= self._threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            # Check for auto-transition to half-open
            if self._state == CircuitState.OPEN and self._opened_at:
                if (time.monotonic() - self._opened_at) >= self._reset_after:
                    self._state = CircuitState.HALF_OPEN
            return self._state

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures
