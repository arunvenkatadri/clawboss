"""Tests for tool scoping — ScopeRule, ToolScope, and Supervisor integration."""

from __future__ import annotations

import pytest

from clawboss.audit import AuditLog, MemoryAuditSink
from clawboss.policy import Policy, ScopeRule, ToolScope
from clawboss.supervisor import Supervisor

# ---------------------------------------------------------------------------
# Helper tool functions
# ---------------------------------------------------------------------------


async def good_tool(**kwargs) -> str:
    return f"ok: {kwargs}"


async def read_file(path: str = "/tmp/test.txt") -> str:
    return f"contents of {path}"


async def send_email(to: str = "", subject: str = "") -> str:
    return f"sent to {to}"


# ---------------------------------------------------------------------------
# TestScopeRule
# ---------------------------------------------------------------------------


class TestScopeRule:
    def test_allow_matching_glob_passes(self):
        rule = ScopeRule(param="path", constraint="allow", values=["/tmp/*"])
        assert rule.check("/tmp/foo.txt") is True

    def test_allow_non_matching_glob_fails(self):
        rule = ScopeRule(param="path", constraint="allow", values=["/tmp/*"])
        assert rule.check("/etc/passwd") is False

    def test_allow_multiple_patterns(self):
        rule = ScopeRule(
            param="path",
            constraint="allow",
            values=["/tmp/*", "/home/*"],
        )
        assert rule.check("/tmp/a.txt") is True
        assert rule.check("/home/user/b.txt") is True
        assert rule.check("/etc/shadow") is False

    def test_block_matching_pattern_fails(self):
        rule = ScopeRule(
            param="path",
            constraint="block",
            values=["/etc/*"],
        )
        assert rule.check("/etc/passwd") is False

    def test_block_non_matching_pattern_passes(self):
        rule = ScopeRule(
            param="path",
            constraint="block",
            values=["/etc/*"],
        )
        assert rule.check("/tmp/safe.txt") is True

    def test_block_multiple_patterns(self):
        rule = ScopeRule(
            param="path",
            constraint="block",
            values=["/etc/*", "/var/log/*"],
        )
        assert rule.check("/etc/passwd") is False
        assert rule.check("/var/log/syslog") is False
        assert rule.check("/tmp/ok.txt") is True

    def test_match_matching_regex_passes(self):
        rule = ScopeRule(
            param="url",
            constraint="match",
            values=[r"^https://api\.example\.com/"],
        )
        assert rule.check("https://api.example.com/v1/data") is True

    def test_match_non_matching_regex_fails(self):
        rule = ScopeRule(
            param="url",
            constraint="match",
            values=[r"^https://api\.example\.com/"],
        )
        assert rule.check("https://evil.com/data") is False

    def test_match_multiple_regex_patterns(self):
        rule = ScopeRule(
            param="url",
            constraint="match",
            values=[r"^https://api\.example\.com/", r"^https://cdn\.example\.com/"],
        )
        assert rule.check("https://api.example.com/v1") is True
        assert rule.check("https://cdn.example.com/img") is True
        assert rule.check("https://evil.com/") is False

    def test_glob_question_mark_wildcard(self):
        rule = ScopeRule(
            param="name",
            constraint="allow",
            values=["file?.txt"],
        )
        assert rule.check("file1.txt") is True
        assert rule.check("fileAB.txt") is False

    def test_glob_star_wildcard(self):
        rule = ScopeRule(
            param="name",
            constraint="allow",
            values=["*.py"],
        )
        assert rule.check("main.py") is True
        assert rule.check("main.js") is False

    def test_glob_path_wildcard(self):
        rule = ScopeRule(
            param="path",
            constraint="allow",
            values=["/tmp/*"],
        )
        # fnmatch * matches path separators on most platforms
        assert rule.check("/tmp/subdir/file.txt") is True
        assert rule.check("/tmp/file.txt") is True
        assert rule.check("/var/file.txt") is False

    def test_string_conversion_of_values(self):
        """Non-string param values are converted to str for matching."""
        rule = ScopeRule(
            param="count",
            constraint="allow",
            values=["42"],
        )
        assert rule.check(42) is True
        assert rule.check(99) is False

    def test_unknown_constraint_passes(self):
        """Unknown constraint type defaults to allowing."""
        rule = ScopeRule(
            param="x",
            constraint="unknown",
            values=["anything"],
        )
        assert rule.check("whatever") is True

    def test_to_dict(self):
        rule = ScopeRule(
            param="path",
            constraint="allow",
            values=["/tmp/*"],
        )
        d = rule.to_dict()
        assert d == {
            "param": "path",
            "constraint": "allow",
            "values": ["/tmp/*"],
        }


# ---------------------------------------------------------------------------
# TestToolScope
# ---------------------------------------------------------------------------


class TestToolScope:
    def test_check_args_returns_none_when_all_pass(self):
        scope = ToolScope(
            tool_name="read_file",
            rules=[
                ScopeRule(
                    param="path",
                    constraint="allow",
                    values=["/tmp/*"],
                ),
            ],
        )
        result = scope.check_args({"path": "/tmp/test.txt"})
        assert result is None

    def test_check_args_returns_error_when_rule_fails(self):
        scope = ToolScope(
            tool_name="read_file",
            rules=[
                ScopeRule(
                    param="path",
                    constraint="allow",
                    values=["/tmp/*"],
                ),
            ],
        )
        result = scope.check_args({"path": "/etc/passwd"})
        assert result is not None
        assert "path" in result
        assert "/etc/passwd" in result
        assert "blocked by scope rule" in result

    def test_multiple_rules_all_must_pass(self):
        scope = ToolScope(
            tool_name="send_email",
            rules=[
                ScopeRule(
                    param="to",
                    constraint="allow",
                    values=["*@mycompany.com"],
                ),
                ScopeRule(
                    param="subject",
                    constraint="block",
                    values=["*CONFIDENTIAL*"],
                ),
            ],
        )
        # Both pass
        assert (
            scope.check_args(
                {
                    "to": "bob@mycompany.com",
                    "subject": "Hello",
                }
            )
            is None
        )
        # First fails
        result = scope.check_args(
            {
                "to": "bob@evil.com",
                "subject": "Hello",
            }
        )
        assert result is not None
        assert "to" in result
        # Second fails
        result = scope.check_args(
            {
                "to": "bob@mycompany.com",
                "subject": "TOP CONFIDENTIAL MEMO",
            }
        )
        assert result is not None
        assert "subject" in result

    def test_missing_param_in_kwargs_is_ignored(self):
        scope = ToolScope(
            tool_name="read_file",
            rules=[
                ScopeRule(
                    param="path",
                    constraint="allow",
                    values=["/tmp/*"],
                ),
            ],
        )
        # No 'path' key — should pass (not enforced)
        assert scope.check_args({"other_param": "value"}) is None

    def test_to_dict_without_rate_limit(self):
        scope = ToolScope(
            tool_name="read_file",
            rules=[
                ScopeRule(
                    param="path",
                    constraint="allow",
                    values=["/tmp/*"],
                ),
            ],
        )
        d = scope.to_dict()
        assert d["tool_name"] == "read_file"
        assert len(d["rules"]) == 1
        assert "max_calls_per_minute" not in d

    def test_to_dict_with_rate_limit(self):
        scope = ToolScope(
            tool_name="search",
            rules=[],
            max_calls_per_minute=10,
        )
        d = scope.to_dict()
        assert d["max_calls_per_minute"] == 10


# ---------------------------------------------------------------------------
# TestPolicyWithScopes
# ---------------------------------------------------------------------------


class TestPolicyWithScopes:
    def test_from_dict_parses_tool_scopes(self):
        d = {
            "tool_scopes": [
                {
                    "tool_name": "read_file",
                    "rules": [
                        {
                            "param": "path",
                            "constraint": "allow",
                            "values": ["/tmp/*"],
                        },
                    ],
                    "max_calls_per_minute": 30,
                },
            ],
        }
        policy = Policy.from_dict(d)
        assert len(policy.tool_scopes) == 1
        scope = policy.tool_scopes[0]
        assert scope.tool_name == "read_file"
        assert len(scope.rules) == 1
        assert scope.rules[0].param == "path"
        assert scope.rules[0].constraint == "allow"
        assert scope.rules[0].values == ["/tmp/*"]
        assert scope.max_calls_per_minute == 30

    def test_from_dict_defaults_constraint_to_allow(self):
        d = {
            "tool_scopes": [
                {
                    "tool_name": "tool",
                    "rules": [{"param": "x", "values": ["a"]}],
                },
            ],
        }
        policy = Policy.from_dict(d)
        assert policy.tool_scopes[0].rules[0].constraint == "allow"

    def test_from_dict_defaults_values_to_empty(self):
        d = {
            "tool_scopes": [
                {
                    "tool_name": "tool",
                    "rules": [{"param": "x", "constraint": "allow"}],
                },
            ],
        }
        policy = Policy.from_dict(d)
        assert policy.tool_scopes[0].rules[0].values == []

    def test_policy_with_empty_tool_scopes(self):
        policy = Policy.from_dict({"tool_scopes": []})
        assert policy.tool_scopes == []

    def test_policy_default_tool_scopes_empty(self):
        policy = Policy()
        assert policy.tool_scopes == []

    def test_round_trip(self):
        """Create a policy with scopes, verify fields survive."""
        scope = ToolScope(
            tool_name="search",
            rules=[
                ScopeRule(
                    param="query",
                    constraint="block",
                    values=["*password*"],
                ),
            ],
            max_calls_per_minute=5,
        )
        policy = Policy(tool_scopes=[scope])
        assert policy.tool_scopes[0].tool_name == "search"
        assert policy.tool_scopes[0].rules[0].param == "query"
        assert policy.tool_scopes[0].max_calls_per_minute == 5

    def test_from_dict_multiple_scopes(self):
        d = {
            "tool_scopes": [
                {
                    "tool_name": "read_file",
                    "rules": [
                        {
                            "param": "path",
                            "constraint": "allow",
                            "values": ["/tmp/*"],
                        },
                    ],
                },
                {
                    "tool_name": "send_email",
                    "rules": [
                        {
                            "param": "to",
                            "constraint": "allow",
                            "values": ["*@mycompany.com"],
                        },
                    ],
                },
            ],
        }
        policy = Policy.from_dict(d)
        assert len(policy.tool_scopes) == 2
        assert policy.tool_scopes[0].tool_name == "read_file"
        assert policy.tool_scopes[1].tool_name == "send_email"


# ---------------------------------------------------------------------------
# TestSupervisorScopes
# ---------------------------------------------------------------------------


class TestSupervisorScopes:
    @pytest.mark.asyncio
    async def test_tool_call_allowed_when_scope_passes(self):
        policy = Policy(
            tool_scopes=[
                ToolScope(
                    tool_name="read_file",
                    rules=[
                        ScopeRule(
                            param="path",
                            constraint="allow",
                            values=["/tmp/*"],
                        ),
                    ],
                ),
            ],
        )
        sv = Supervisor(policy)
        result = await sv.call(
            "read_file",
            read_file,
            path="/tmp/test.txt",
        )
        assert result.succeeded is True
        assert "contents of /tmp/test.txt" in result.output

    @pytest.mark.asyncio
    async def test_tool_call_denied_when_scope_fails(self):
        policy = Policy(
            tool_scopes=[
                ToolScope(
                    tool_name="read_file",
                    rules=[
                        ScopeRule(
                            param="path",
                            constraint="allow",
                            values=["/tmp/*"],
                        ),
                    ],
                ),
            ],
        )
        sv = Supervisor(policy)
        result = await sv.call(
            "read_file",
            read_file,
            path="/etc/passwd",
        )
        assert result.succeeded is False
        assert result.error is not None
        assert result.error.kind == "scope_denied"

    @pytest.mark.asyncio
    async def test_scope_violation_is_audited(self):
        sink = MemoryAuditSink()
        audit = AuditLog("req-scope", sinks=[sink])
        policy = Policy(
            tool_scopes=[
                ToolScope(
                    tool_name="read_file",
                    rules=[
                        ScopeRule(
                            param="path",
                            constraint="allow",
                            values=["/tmp/*"],
                        ),
                    ],
                ),
            ],
        )
        sv = Supervisor(policy, audit=audit)
        await sv.call("read_file", read_file, path="/etc/shadow")
        phases = [e.phase for e in sink.entries]
        assert "scope_check" in phases
        outcomes = [e.outcome for e in sink.entries if e.phase == "scope_check"]
        assert "denied" in outcomes

    @pytest.mark.asyncio
    async def test_tool_with_no_scope_is_allowed(self):
        """Scopes only apply to tools that have them defined."""
        policy = Policy(
            tool_scopes=[
                ToolScope(
                    tool_name="read_file",
                    rules=[
                        ScopeRule(
                            param="path",
                            constraint="allow",
                            values=["/tmp/*"],
                        ),
                    ],
                ),
            ],
        )
        sv = Supervisor(policy)
        result = await sv.call("other_tool", good_tool, x="hello")
        assert result.succeeded is True

    @pytest.mark.asyncio
    async def test_rate_limit_allows_under_limit(self):
        policy = Policy(
            tool_scopes=[
                ToolScope(
                    tool_name="search",
                    rules=[],
                    max_calls_per_minute=3,
                ),
            ],
        )
        sv = Supervisor(policy)
        for _ in range(3):
            result = await sv.call("search", good_tool, q="test")
            assert result.succeeded is True

    @pytest.mark.asyncio
    async def test_rate_limit_denies_over_limit(self):
        policy = Policy(
            tool_scopes=[
                ToolScope(
                    tool_name="search",
                    rules=[],
                    max_calls_per_minute=2,
                ),
            ],
        )
        sv = Supervisor(policy)
        # First two should succeed
        r1 = await sv.call("search", good_tool, q="a")
        r2 = await sv.call("search", good_tool, q="b")
        assert r1.succeeded is True
        assert r2.succeeded is True
        # Third should be rate limited
        r3 = await sv.call("search", good_tool, q="c")
        assert r3.succeeded is False
        assert r3.error is not None
        assert r3.error.kind == "rate_limited"

    @pytest.mark.asyncio
    async def test_multiple_scopes_independent(self):
        policy = Policy(
            tool_scopes=[
                ToolScope(
                    tool_name="read_file",
                    rules=[
                        ScopeRule(
                            param="path",
                            constraint="allow",
                            values=["/tmp/*"],
                        ),
                    ],
                ),
                ToolScope(
                    tool_name="send_email",
                    rules=[
                        ScopeRule(
                            param="to",
                            constraint="allow",
                            values=["*@mycompany.com"],
                        ),
                    ],
                ),
            ],
        )
        sv = Supervisor(policy)
        # read_file allowed
        r1 = await sv.call(
            "read_file",
            read_file,
            path="/tmp/ok.txt",
        )
        assert r1.succeeded is True
        # send_email allowed
        r2 = await sv.call(
            "send_email",
            send_email,
            to="bob@mycompany.com",
        )
        assert r2.succeeded is True
        # read_file denied
        r3 = await sv.call(
            "read_file",
            read_file,
            path="/etc/passwd",
        )
        assert r3.succeeded is False
        assert r3.error.kind == "scope_denied"
        # send_email denied
        r4 = await sv.call(
            "send_email",
            send_email,
            to="bob@competitor.com",
        )
        assert r4.succeeded is False
        assert r4.error.kind == "scope_denied"

    @pytest.mark.asyncio
    async def test_scope_with_block_constraint(self):
        policy = Policy(
            tool_scopes=[
                ToolScope(
                    tool_name="read_file",
                    rules=[
                        ScopeRule(
                            param="path",
                            constraint="block",
                            values=["/etc/*", "/var/log/*"],
                        ),
                    ],
                ),
            ],
        )
        sv = Supervisor(policy)
        # Allowed — not blocked
        r1 = await sv.call(
            "read_file",
            read_file,
            path="/tmp/safe.txt",
        )
        assert r1.succeeded is True
        # Blocked
        r2 = await sv.call(
            "read_file",
            read_file,
            path="/etc/shadow",
        )
        assert r2.succeeded is False
        assert r2.error.kind == "scope_denied"


# ---------------------------------------------------------------------------
# TestScopePatterns — practical examples
# ---------------------------------------------------------------------------


class TestScopePatterns:
    def test_file_path_scoping_allow_tmp_block_etc(self):
        """Allow /tmp/* but block /etc/*."""
        allow_rule = ScopeRule(
            param="path",
            constraint="allow",
            values=["/tmp/*"],
        )
        block_rule = ScopeRule(
            param="path",
            constraint="block",
            values=["/etc/*"],
        )
        # Allow rule
        assert allow_rule.check("/tmp/data.csv") is True
        assert allow_rule.check("/etc/passwd") is False
        # Block rule
        assert block_rule.check("/tmp/data.csv") is True
        assert block_rule.check("/etc/passwd") is False

    def test_email_domain_scoping(self):
        """Allow *@mycompany.com but block *@competitor.com."""
        allow_rule = ScopeRule(
            param="to",
            constraint="allow",
            values=["*@mycompany.com"],
        )
        block_rule = ScopeRule(
            param="to",
            constraint="block",
            values=["*@competitor.com"],
        )
        assert allow_rule.check("alice@mycompany.com") is True
        assert allow_rule.check("bob@competitor.com") is False
        assert block_rule.check("alice@mycompany.com") is True
        assert block_rule.check("bob@competitor.com") is False

    def test_url_domain_scoping(self):
        """Allow only https://api.example.com/* URLs."""
        rule = ScopeRule(
            param="url",
            constraint="allow",
            values=["https://api.example.com/*"],
        )
        assert rule.check("https://api.example.com/v1/users") is True
        assert rule.check("https://evil.com/steal") is False

    def test_url_regex_scoping(self):
        """Use regex match for more complex URL validation."""
        rule = ScopeRule(
            param="url",
            constraint="match",
            values=[r"^https://(api|cdn)\.example\.com/"],
        )
        assert rule.check("https://api.example.com/v1") is True
        assert rule.check("https://cdn.example.com/img.png") is True
        assert rule.check("https://evil.example.com/") is False
        assert rule.check("http://api.example.com/v1") is False

    def test_combined_allow_and_block_on_same_tool(self):
        """A tool scope with both allow and block rules."""
        scope = ToolScope(
            tool_name="fetch_url",
            rules=[
                ScopeRule(
                    param="url",
                    constraint="match",
                    values=[r"^https://"],
                ),
                ScopeRule(
                    param="url",
                    constraint="block",
                    values=["*evil.com*"],
                ),
            ],
        )
        # https + not evil -> OK
        assert scope.check_args({"url": "https://api.example.com/data"}) is None
        # http -> fails match rule
        result = scope.check_args({"url": "http://api.example.com/data"})
        assert result is not None
        # https + evil -> fails block rule
        result = scope.check_args({"url": "https://evil.com/steal"})
        assert result is not None
