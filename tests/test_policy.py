"""Tests for clawboss.policy — Policy dataclass, OnFailure, Action."""

from clawboss.policy import Action, Policy

# ---------------------------------------------------------------------------
# Action.from_str
# ---------------------------------------------------------------------------


class TestActionFromStr:
    def test_valid_return_error(self):
        assert Action.from_str("return_error") is Action.RETURN_ERROR

    def test_valid_kill(self):
        assert Action.from_str("kill") is Action.KILL

    def test_valid_respond_with_best_effort(self):
        assert Action.from_str("respond_with_best_effort") is Action.RESPOND_WITH_BEST_EFFORT

    def test_invalid_string_defaults_to_return_error(self):
        assert Action.from_str("nonexistent") is Action.RETURN_ERROR

    def test_empty_string_defaults_to_return_error(self):
        assert Action.from_str("") is Action.RETURN_ERROR


# ---------------------------------------------------------------------------
# Policy defaults
# ---------------------------------------------------------------------------


class TestPolicyDefaults:
    def test_default_tool_timeout(self):
        p = Policy()
        assert p.tool_timeout == 30.0

    def test_default_max_iterations(self):
        p = Policy()
        assert p.max_iterations == 5

    def test_default_token_budget_is_none(self):
        p = Policy()
        assert p.token_budget is None

    def test_default_request_timeout(self):
        p = Policy()
        assert p.request_timeout == 300.0

    def test_default_silence_timeout_is_none(self):
        p = Policy()
        assert p.silence_timeout is None

    def test_default_circuit_breaker_threshold(self):
        p = Policy()
        assert p.circuit_breaker_threshold == 5

    def test_default_circuit_breaker_reset(self):
        p = Policy()
        assert p.circuit_breaker_reset == 60.0

    def test_default_on_timeout_action(self):
        p = Policy()
        assert p.on_timeout.action is Action.RETURN_ERROR
        assert p.on_timeout.retries == 0

    def test_default_on_budget_exceeded_action(self):
        p = Policy()
        assert p.on_budget_exceeded.action is Action.RESPOND_WITH_BEST_EFFORT

    def test_default_audit_enabled(self):
        p = Policy()
        assert p.audit_enabled is True

    def test_default_require_confirm_empty(self):
        p = Policy()
        assert p.require_confirm == []


# ---------------------------------------------------------------------------
# Policy.from_dict
# ---------------------------------------------------------------------------


class TestPolicyFromDict:
    def test_empty_dict_returns_defaults(self):
        p = Policy.from_dict({})
        default = Policy()
        assert p.tool_timeout == default.tool_timeout
        assert p.max_iterations == default.max_iterations
        assert p.token_budget == default.token_budget

    def test_all_scalar_fields(self):
        d = {
            "tool_timeout": 10.0,
            "max_iterations": 3,
            "token_budget": 5000,
            "request_timeout": 120.0,
            "silence_timeout": 30.0,
            "circuit_breaker_threshold": 2,
            "circuit_breaker_reset": 20.0,
            "audit_enabled": False,
        }
        p = Policy.from_dict(d)
        assert p.tool_timeout == 10.0
        assert p.max_iterations == 3
        assert p.token_budget == 5000
        assert p.request_timeout == 120.0
        assert p.silence_timeout == 30.0
        assert p.circuit_breaker_threshold == 2
        assert p.circuit_breaker_reset == 20.0
        assert p.audit_enabled is False

    def test_on_failure_as_string(self):
        p = Policy.from_dict({"on_timeout": "return_error"})
        assert p.on_timeout.action is Action.RETURN_ERROR
        assert p.on_timeout.retries == 0

    def test_on_failure_as_string_kill(self):
        p = Policy.from_dict({"on_timeout": "kill"})
        assert p.on_timeout.action is Action.KILL

    def test_on_failure_as_dict(self):
        p = Policy.from_dict({"on_timeout": {"action": "kill", "retries": 2}})
        assert p.on_timeout.action is Action.KILL
        assert p.on_timeout.retries == 2

    def test_on_failure_as_dict_defaults(self):
        p = Policy.from_dict({"on_budget_exceeded": {"retries": 1}})
        assert p.on_budget_exceeded.action is Action.RETURN_ERROR
        assert p.on_budget_exceeded.retries == 1

    def test_require_confirm_list(self):
        p = Policy.from_dict({"require_confirm": ["delete_file", "send_email"]})
        assert p.require_confirm == ["delete_file", "send_email"]

    def test_require_confirm_preserves_order(self):
        tools = ["z_tool", "a_tool", "m_tool"]
        p = Policy.from_dict({"require_confirm": tools})
        assert p.require_confirm == tools

    def test_partial_dict_leaves_other_defaults(self):
        p = Policy.from_dict({"tool_timeout": 7.0})
        assert p.tool_timeout == 7.0
        assert p.max_iterations == 5  # default
        assert p.token_budget is None  # default

    def test_all_on_failure_fields(self):
        d = {
            "on_timeout": "kill",
            "on_budget_exceeded": "respond_with_best_effort",
            "on_max_iterations": "return_error",
            "on_circuit_open": {"action": "kill", "retries": 3},
            "on_silence": {"action": "return_error", "retries": 1},
        }
        p = Policy.from_dict(d)
        assert p.on_timeout.action is Action.KILL
        assert p.on_budget_exceeded.action is Action.RESPOND_WITH_BEST_EFFORT
        assert p.on_max_iterations.action is Action.RETURN_ERROR
        assert p.on_circuit_open.action is Action.KILL
        assert p.on_circuit_open.retries == 3
        assert p.on_silence.retries == 1
