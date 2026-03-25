"""Tests for clawboss.errors — ClawbossError static factories and properties."""

import pytest

from clawboss.errors import ClawbossError


class TestClawbossErrorTimeout:
    def test_kind(self):
        err = ClawbossError.timeout(5000)
        assert err.kind == "timeout"

    def test_message_contains_ms(self):
        err = ClawbossError.timeout(5000)
        assert "5000" in str(err)
        assert "timed out" in str(err).lower()

    def test_user_message_is_str(self):
        err = ClawbossError.timeout(5000)
        assert isinstance(err.user_message(), str)
        assert len(err.user_message()) > 0

    def test_details_has_timeout_ms(self):
        err = ClawbossError.timeout(5000)
        assert err.details["timeout_ms"] == 5000


class TestClawbossErrorBudgetExceeded:
    def test_kind(self):
        err = ClawbossError.budget_exceeded(12000, 10000)
        assert err.kind == "budget_exceeded"

    def test_message_contains_values(self):
        err = ClawbossError.budget_exceeded(12000, 10000)
        msg = str(err)
        assert "12000" in msg
        assert "10000" in msg

    def test_details(self):
        err = ClawbossError.budget_exceeded(12000, 10000)
        assert err.details["used"] == 12000
        assert err.details["limit"] == 10000


class TestClawbossErrorMaxIterations:
    def test_kind(self):
        err = ClawbossError.max_iterations(6, 5)
        assert err.kind == "max_iterations"

    def test_message(self):
        err = ClawbossError.max_iterations(6, 5)
        assert "6" in str(err)
        assert "5" in str(err)

    def test_details(self):
        err = ClawbossError.max_iterations(6, 5)
        assert err.details["iterations"] == 6
        assert err.details["max"] == 5


class TestClawbossErrorCircuitOpen:
    def test_kind(self):
        err = ClawbossError.circuit_open("web_search", 5)
        assert err.kind == "circuit_open"

    def test_message_contains_tool(self):
        err = ClawbossError.circuit_open("web_search", 5)
        assert "web_search" in str(err)

    def test_details(self):
        err = ClawbossError.circuit_open("web_search", 5)
        assert err.details["tool"] == "web_search"
        assert err.details["consecutive_failures"] == 5


class TestClawbossErrorToolError:
    def test_kind(self):
        err = ClawbossError.tool_error("connection refused")
        assert err.kind == "tool_error"

    def test_message(self):
        err = ClawbossError.tool_error("connection refused")
        assert "connection refused" in str(err)

    def test_user_message(self):
        err = ClawbossError.tool_error("connection refused")
        assert err.user_message() == "connection refused"


class TestClawbossErrorDeadManSwitch:
    def test_kind(self):
        err = ClawbossError.dead_man_switch(60000)
        assert err.kind == "dead_man_switch"

    def test_message_contains_ms(self):
        err = ClawbossError.dead_man_switch(60000)
        assert "60000" in str(err)

    def test_details(self):
        err = ClawbossError.dead_man_switch(60000)
        assert err.details["silence_ms"] == 60000


class TestClawbossErrorPolicyDenied:
    def test_kind(self):
        err = ClawbossError.policy_denied("requires confirmation")
        assert err.kind == "policy_denied"

    def test_message_contains_reason(self):
        err = ClawbossError.policy_denied("requires confirmation")
        assert "requires confirmation" in str(err)

    def test_details(self):
        err = ClawbossError.policy_denied("requires confirmation")
        assert err.details["reason"] == "requires confirmation"


class TestClawbossErrorIsException:
    def test_can_be_raised_and_caught(self):
        with pytest.raises(ClawbossError) as exc_info:
            raise ClawbossError.timeout(1000)
        assert exc_info.value.kind == "timeout"

    def test_inherits_from_exception(self):
        err = ClawbossError.timeout(1000)
        assert isinstance(err, Exception)
