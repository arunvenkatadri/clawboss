"""Tests for agenthandler.errors — AgentHandlerError static factories and properties."""

import pytest

from agenthandler.errors import AgentHandlerError


class TestAgentHandlerErrorTimeout:
    def test_kind(self):
        err = AgentHandlerError.timeout(5000)
        assert err.kind == "timeout"

    def test_message_contains_ms(self):
        err = AgentHandlerError.timeout(5000)
        assert "5000" in str(err)
        assert "timed out" in str(err).lower()

    def test_user_message_is_str(self):
        err = AgentHandlerError.timeout(5000)
        assert isinstance(err.user_message(), str)
        assert len(err.user_message()) > 0

    def test_details_has_timeout_ms(self):
        err = AgentHandlerError.timeout(5000)
        assert err.details["timeout_ms"] == 5000


class TestAgentHandlerErrorBudgetExceeded:
    def test_kind(self):
        err = AgentHandlerError.budget_exceeded(12000, 10000)
        assert err.kind == "budget_exceeded"

    def test_message_contains_values(self):
        err = AgentHandlerError.budget_exceeded(12000, 10000)
        msg = str(err)
        assert "12000" in msg
        assert "10000" in msg

    def test_details(self):
        err = AgentHandlerError.budget_exceeded(12000, 10000)
        assert err.details["used"] == 12000
        assert err.details["limit"] == 10000


class TestAgentHandlerErrorMaxIterations:
    def test_kind(self):
        err = AgentHandlerError.max_iterations(6, 5)
        assert err.kind == "max_iterations"

    def test_message(self):
        err = AgentHandlerError.max_iterations(6, 5)
        assert "6" in str(err)
        assert "5" in str(err)

    def test_details(self):
        err = AgentHandlerError.max_iterations(6, 5)
        assert err.details["iterations"] == 6
        assert err.details["max"] == 5


class TestAgentHandlerErrorCircuitOpen:
    def test_kind(self):
        err = AgentHandlerError.circuit_open("web_search", 5)
        assert err.kind == "circuit_open"

    def test_message_contains_tool(self):
        err = AgentHandlerError.circuit_open("web_search", 5)
        assert "web_search" in str(err)

    def test_details(self):
        err = AgentHandlerError.circuit_open("web_search", 5)
        assert err.details["tool"] == "web_search"
        assert err.details["consecutive_failures"] == 5


class TestAgentHandlerErrorToolError:
    def test_kind(self):
        err = AgentHandlerError.tool_error("connection refused")
        assert err.kind == "tool_error"

    def test_message(self):
        err = AgentHandlerError.tool_error("connection refused")
        assert "connection refused" in str(err)

    def test_user_message(self):
        err = AgentHandlerError.tool_error("connection refused")
        assert err.user_message() == "connection refused"


class TestAgentHandlerErrorDeadManSwitch:
    def test_kind(self):
        err = AgentHandlerError.dead_man_switch(60000)
        assert err.kind == "dead_man_switch"

    def test_message_contains_ms(self):
        err = AgentHandlerError.dead_man_switch(60000)
        assert "60000" in str(err)

    def test_details(self):
        err = AgentHandlerError.dead_man_switch(60000)
        assert err.details["silence_ms"] == 60000


class TestAgentHandlerErrorPolicyDenied:
    def test_kind(self):
        err = AgentHandlerError.policy_denied("requires confirmation")
        assert err.kind == "policy_denied"

    def test_message_contains_reason(self):
        err = AgentHandlerError.policy_denied("requires confirmation")
        assert "requires confirmation" in str(err)

    def test_details(self):
        err = AgentHandlerError.policy_denied("requires confirmation")
        assert err.details["reason"] == "requires confirmation"


class TestAgentHandlerErrorIsException:
    def test_can_be_raised_and_caught(self):
        with pytest.raises(AgentHandlerError) as exc_info:
            raise AgentHandlerError.timeout(1000)
        assert exc_info.value.kind == "timeout"

    def test_inherits_from_exception(self):
        err = AgentHandlerError.timeout(1000)
        assert isinstance(err, Exception)
