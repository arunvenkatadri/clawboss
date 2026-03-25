"""Tests for clawboss.circuit_breaker — CircuitBreaker state machine."""

import time
from unittest.mock import patch

import pytest

from clawboss.circuit_breaker import CircuitBreaker, CircuitState
from clawboss.errors import ClawbossError


class TestCircuitBreakerInitialState:
    def test_starts_closed(self):
        cb = CircuitBreaker(threshold=3, reset_after=10.0)
        assert cb.state is CircuitState.CLOSED

    def test_initial_failure_count_is_zero(self):
        cb = CircuitBreaker(threshold=3, reset_after=10.0)
        assert cb.consecutive_failures == 0


class TestCircuitBreakerClosed:
    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(threshold=3, reset_after=10.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state is CircuitState.CLOSED
        assert cb.consecutive_failures == 2

    def test_check_passes_when_closed(self):
        cb = CircuitBreaker(threshold=3, reset_after=10.0)
        # Should not raise
        cb.check("my_tool")

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(threshold=3, reset_after=10.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state is CircuitState.OPEN

    def test_opens_exactly_at_threshold(self):
        cb = CircuitBreaker(threshold=1, reset_after=10.0)
        cb.record_failure()
        assert cb.state is CircuitState.OPEN


class TestCircuitBreakerOpen:
    def test_blocks_calls(self):
        cb = CircuitBreaker(threshold=2, reset_after=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state is CircuitState.OPEN
        with pytest.raises(ClawbossError) as exc_info:
            cb.check("broken_tool")
        assert exc_info.value.kind == "circuit_open"
        assert "broken_tool" in str(exc_info.value)

    def test_failure_count_preserved_when_open(self):
        cb = CircuitBreaker(threshold=2, reset_after=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.consecutive_failures == 2


class TestCircuitBreakerHalfOpen:
    def test_transitions_to_half_open_after_reset_period(self):
        cb = CircuitBreaker(threshold=2, reset_after=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.state is CircuitState.OPEN
        time.sleep(0.1)
        assert cb.state is CircuitState.HALF_OPEN

    def test_check_passes_when_half_open(self):
        cb = CircuitBreaker(threshold=2, reset_after=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)
        # Should not raise — half-open allows a test call
        cb.check("my_tool")

    def test_half_open_success_closes_circuit(self):
        cb = CircuitBreaker(threshold=2, reset_after=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)
        # Should be half-open now
        cb.check("my_tool")
        cb.record_success()
        assert cb.state is CircuitState.CLOSED
        assert cb.consecutive_failures == 0

    def test_half_open_failure_reopens_circuit(self):
        cb = CircuitBreaker(threshold=2, reset_after=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)
        cb.check("my_tool")
        cb.record_failure()
        assert cb.state is CircuitState.OPEN

    def test_check_triggers_half_open_transition(self):
        """Calling check() on an OPEN circuit past its reset time transitions to HALF_OPEN."""
        cb = CircuitBreaker(threshold=2, reset_after=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.1)
        # check() should internally transition and not raise
        cb.check("my_tool")


class TestCircuitBreakerRecordSuccess:
    def test_resets_failure_count(self):
        cb = CircuitBreaker(threshold=5, reset_after=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.consecutive_failures == 3
        cb.record_success()
        assert cb.consecutive_failures == 0
        assert cb.state is CircuitState.CLOSED

    def test_success_after_initial_state(self):
        cb = CircuitBreaker(threshold=5, reset_after=60.0)
        cb.record_success()  # no-op effectively
        assert cb.consecutive_failures == 0
        assert cb.state is CircuitState.CLOSED

    def test_interleaved_success_and_failure(self):
        cb = CircuitBreaker(threshold=3, reset_after=60.0)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # resets
        cb.record_failure()
        assert cb.consecutive_failures == 1
        assert cb.state is CircuitState.CLOSED
