"""Token and iteration budget tracking — thread-safe."""

import threading
from dataclasses import dataclass
from typing import Optional

from .errors import ClawbossError


@dataclass
class BudgetSnapshot:
    """Point-in-time snapshot of budget usage."""

    tokens_used: int
    token_limit: Optional[int]
    iterations: int
    iteration_limit: int

    @property
    def tokens_remaining(self) -> Optional[int]:
        if self.token_limit is None:
            return None
        return max(0, self.token_limit - self.tokens_used)

    @property
    def iterations_remaining(self) -> int:
        return max(0, self.iteration_limit - self.iterations)

    @property
    def is_over_token_budget(self) -> bool:
        if self.token_limit is None:
            return False
        return self.tokens_used >= self.token_limit

    @property
    def is_over_iteration_limit(self) -> bool:
        return self.iterations >= self.iteration_limit


class BudgetTracker:
    """Thread-safe token and iteration budget tracker.

    Usage:
        tracker = BudgetTracker(token_limit=10000, iteration_limit=5)
        tracker.record_tokens(500)    # raises ClawbossError if over
        tracker.record_iteration()    # raises ClawbossError if over
        snap = tracker.snapshot()     # read current state
    """

    def __init__(self, token_limit: Optional[int] = None, iteration_limit: int = 5):
        self._token_limit = token_limit
        self._iteration_limit = iteration_limit
        self._tokens_used = 0
        self._iterations = 0
        self._lock = threading.Lock()

    @classmethod
    def from_policy(cls, policy) -> "BudgetTracker":
        return cls(
            token_limit=policy.token_budget,
            iteration_limit=policy.max_iterations,
        )

    def record_tokens(self, tokens: int) -> int:
        """Record token usage. Returns new total. Raises if over budget."""
        with self._lock:
            self._tokens_used += tokens
            if self._token_limit is not None and self._tokens_used > self._token_limit:
                raise ClawbossError.budget_exceeded(self._tokens_used, self._token_limit)
            return self._tokens_used

    def record_iteration(self) -> int:
        """Record an iteration. Returns new count. Raises if over limit."""
        with self._lock:
            self._iterations += 1
            if self._iterations > self._iteration_limit:
                raise ClawbossError.max_iterations(self._iterations, self._iteration_limit)
            return self._iterations

    def snapshot(self) -> BudgetSnapshot:
        """Get a point-in-time snapshot."""
        with self._lock:
            return BudgetSnapshot(
                tokens_used=self._tokens_used,
                token_limit=self._token_limit,
                iterations=self._iterations,
                iteration_limit=self._iteration_limit,
            )

    def reset(self):
        """Reset all counters."""
        with self._lock:
            self._tokens_used = 0
            self._iterations = 0
