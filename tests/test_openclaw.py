"""Tests for clawboss.openclaw — OpenClaw integration, bridge, schema conversion."""
from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import urllib.request
import urllib.error

import pytest

from clawboss.openclaw import (
    OpenClawBridge,
    to_openclaw_manifest,
    to_openclaw_tool_schema,
)
from clawboss.policy import Policy
from clawboss.skill import Skill, ToolDefinition, ToolParameter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_tool_def(
    name: str = "web_search",
    description: str = "Search the web",
    params: list | None = None,
) -> ToolDefinition:
    if params is None:
        params = [
            ToolParameter(name="query", type="string", description="Search query", required=True),
            ToolParameter(name="limit", type="integer", description="Max results", default=10),
        ]
    return ToolDefinition(name=name, description=description, parameters=params)


async def _echo_tool(query: str = "default", **kwargs) -> str:
    return f"echo: {query}"


async def _failing_tool(**kwargs) -> str:
    raise RuntimeError("tool exploded")


def _make_skill_with_tools() -> Skill:
    return Skill(
        name="test_skill",
        description="A test skill",
        tools=[
            ToolDefinition(
                name="echo",
                description="Echoes input",
                parameters=[
                    ToolParameter(name="query", type="string", required=True),
                ],
            ),
            ToolDefinition(
                name="fail",
                description="Always fails",
                parameters=[],
            ),
        ],
        supervision={"max_iterations": 3, "tool_timeout": 10},
    )


# ---------------------------------------------------------------------------
# to_openclaw_tool_schema
# ---------------------------------------------------------------------------


class TestToOpenclawToolSchema:
    def test_basic_conversion(self):
        tool = _make_tool_def()
        schema = to_openclaw_tool_schema(tool)
        assert schema["name"] == "web_search"
        assert schema["description"] == "Search the web"
        assert "parameters" in schema

    def test_required_params(self):
        tool = _make_tool_def()
        schema = to_openclaw_tool_schema(tool)
        assert "required" in schema["parameters"]
        assert "query" in schema["parameters"]["required"]

    def test_optional_params_not_in_required(self):
        tool = _make_tool_def()
        schema = to_openclaw_tool_schema(tool)
        required = schema["parameters"].get("required", [])
        assert "limit" not in required

    def test_default_values_included(self):
        tool = _make_tool_def()
        schema = to_openclaw_tool_schema(tool)
        assert schema["parameters"]["properties"]["limit"]["default"] == 10

    def test_no_required_params(self):
        tool = ToolDefinition(
            name="noop",
            description="No params",
            parameters=[
                ToolParameter(name="x", type="string"),
            ],
        )
        schema = to_openclaw_tool_schema(tool)
        assert "required" not in schema["parameters"]

    def test_empty_parameters(self):
        tool = ToolDefinition(name="noop", description="Nothing")
        schema = to_openclaw_tool_schema(tool)
        assert schema["parameters"]["properties"] == {}

    def test_parameter_descriptions(self):
        tool = _make_tool_def()
        schema = to_openclaw_tool_schema(tool)
        assert schema["parameters"]["properties"]["query"]["description"] == "Search query"

    def test_parameter_types(self):
        tool = _make_tool_def()
        schema = to_openclaw_tool_schema(tool)
        assert schema["parameters"]["properties"]["query"]["type"] == "string"
        assert schema["parameters"]["properties"]["limit"]["type"] == "integer"


# ---------------------------------------------------------------------------
# to_openclaw_manifest
# ---------------------------------------------------------------------------


class TestToOpenclawManifest:
    def test_manifest_structure(self):
        skill = Skill(
            name="web_research",
            description="Research topics",
            version="2.0",
            tools=[_make_tool_def()],
        )
        manifest = to_openclaw_manifest(skill)
        assert manifest["id"] == "clawboss-web-research"
        assert manifest["name"] == "Clawboss: web_research"
        assert manifest["version"] == "2.0"
        assert manifest["description"] == "Research topics"
        assert "configSchema" in manifest
        assert len(manifest["tools"]) == 1

    def test_custom_plugin_id(self):
        skill = Skill(name="test", description="test")
        manifest = to_openclaw_manifest(skill, plugin_id="my-plugin")
        assert manifest["id"] == "my-plugin"

    def test_custom_bridge_port(self):
        skill = Skill(name="test", description="test")
        manifest = to_openclaw_manifest(skill, bridge_port=8888)
        config_props = manifest["configSchema"]["properties"]
        assert "8888" in config_props["bridgeUrl"]["default"]

    def test_tools_converted_to_openclaw_format(self):
        skill = Skill(
            name="test",
            description="test",
            tools=[
                ToolDefinition(
                    name="search",
                    description="Search",
                    parameters=[
                        ToolParameter(name="q", type="string", required=True),
                    ],
                ),
            ],
        )
        manifest = to_openclaw_manifest(skill)
        assert manifest["tools"][0]["name"] == "search"
        assert "parameters" in manifest["tools"][0]


# ---------------------------------------------------------------------------
# OpenClawBridge — register_tool / register_skill
# ---------------------------------------------------------------------------


class TestOpenClawBridgeRegistration:
    def test_register_tool(self):
        bridge = OpenClawBridge()
        tool = _make_tool_def(name="echo")
        bridge.register_tool(tool, _echo_tool)
        assert "echo" in bridge._registry

    def test_register_skill(self):
        bridge = OpenClawBridge()
        skill = _make_skill_with_tools()
        bridge.register_skill(skill, {
            "echo": _echo_tool,
            "fail": _failing_tool,
        })
        assert "echo" in bridge._registry
        assert "fail" in bridge._registry

    def test_register_skill_missing_impl_raises(self):
        bridge = OpenClawBridge()
        skill = _make_skill_with_tools()
        with pytest.raises(ValueError, match="Missing implementation"):
            bridge.register_skill(skill, {"echo": _echo_tool})
            # "fail" is missing

    def test_register_skill_applies_supervision_policy(self):
        bridge = OpenClawBridge()
        skill = _make_skill_with_tools()
        bridge.register_skill(skill, {
            "echo": _echo_tool,
            "fail": _failing_tool,
        })
        assert bridge._policy.max_iterations == 3
        assert bridge._policy.tool_timeout == 10


# ---------------------------------------------------------------------------
# OpenClawBridge — HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def bridge_server():
    """Start a bridge server on a random port and yield (bridge, base_url).

    Shuts down the server after the test.
    """
    port = _find_free_port()
    bridge = OpenClawBridge(
        policy=Policy(tool_timeout=5.0),
        host="127.0.0.1",
        port=port,
    )

    tool = ToolDefinition(
        name="echo",
        description="Echoes input",
        parameters=[
            ToolParameter(name="query", type="string", required=True),
        ],
    )
    bridge.register_tool(tool, _echo_tool)

    fail_tool = ToolDefinition(
        name="fail",
        description="Always fails",
        parameters=[],
    )
    bridge.register_tool(fail_tool, _failing_tool)

    thread = bridge.serve_background()
    # Give server a moment to start
    time.sleep(0.2)

    base_url = f"http://127.0.0.1:{port}"
    yield bridge, base_url

    bridge.shutdown()


def _http_get(url: str) -> tuple[int, dict]:
    """Simple HTTP GET, returns (status_code, json_body)."""
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def _http_post(url: str, body: dict | str | None = None) -> tuple[int, dict]:
    """Simple HTTP POST, returns (status_code, json_body)."""
    if body is None:
        data = None
    elif isinstance(body, str):
        data = body.encode("utf-8")
    else:
        data = json.dumps(body).encode("utf-8")
    try:
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


class TestBridgeHealthEndpoint:
    def test_health_returns_ok(self, bridge_server):
        _, base_url = bridge_server
        status, body = _http_get(f"{base_url}/health")
        assert status == 200
        assert body["status"] == "ok"

    def test_health_returns_tool_count(self, bridge_server):
        _, base_url = bridge_server
        status, body = _http_get(f"{base_url}/health")
        assert body["tools"] == 2  # echo + fail


class TestBridgeToolsEndpoint:
    def test_tools_returns_schemas(self, bridge_server):
        _, base_url = bridge_server
        status, body = _http_get(f"{base_url}/tools")
        assert status == 200
        assert "tools" in body
        names = {t["name"] for t in body["tools"]}
        assert "echo" in names
        assert "fail" in names

    def test_tools_in_openclaw_format(self, bridge_server):
        _, base_url = bridge_server
        _, body = _http_get(f"{base_url}/tools")
        echo_tool = next(t for t in body["tools"] if t["name"] == "echo")
        assert "parameters" in echo_tool
        assert echo_tool["parameters"]["type"] == "object"


class TestBridgeExecuteEndpoint:
    def test_execute_tool_returns_result(self, bridge_server):
        _, base_url = bridge_server
        status, body = _http_post(
            f"{base_url}/execute/echo",
            {"params": {"query": "hello"}},
        )
        assert status == 200
        assert body["success"] is True
        assert body["result"] == "echo: hello"

    def test_execute_failing_tool_returns_error(self, bridge_server):
        _, base_url = bridge_server
        status, body = _http_post(
            f"{base_url}/execute/fail",
            {"params": {}},
        )
        assert status == 200
        assert body["success"] is False
        assert body["error"] is not None
        assert body["error"]["kind"] == "tool_error"

    def test_execute_unknown_tool_returns_404(self, bridge_server):
        _, base_url = bridge_server
        status, body = _http_post(
            f"{base_url}/execute/unknown",
            {"params": {}},
        )
        assert status == 404
        assert "error" in body

    def test_execute_with_invalid_json_returns_400(self, bridge_server):
        _, base_url = bridge_server
        status, body = _http_post(
            f"{base_url}/execute/echo",
            "not valid json{{{",
        )
        assert status == 400
        assert "error" in body

    def test_execute_without_params_returns_400(self, bridge_server):
        _, base_url = bridge_server
        status, body = _http_post(
            f"{base_url}/execute/echo",
            {"no_params_here": True},
        )
        assert status == 400
        assert "error" in body

    def test_execute_result_has_metadata(self, bridge_server):
        _, base_url = bridge_server
        _, body = _http_post(
            f"{base_url}/execute/echo",
            {"params": {"query": "meta"}},
        )
        assert "metadata" in body
        assert "duration_ms" in body["metadata"]
        assert body["metadata"]["tool_name"] == "echo"

    def test_execute_result_has_budget_metadata(self, bridge_server):
        _, base_url = bridge_server
        _, body = _http_post(
            f"{base_url}/execute/echo",
            {"params": {"query": "budget"}},
        )
        assert "budget" in body["metadata"]
        budget = body["metadata"]["budget"]
        assert "tokens_used" in budget
        assert "token_limit" in budget


class TestBridgeShutdown:
    def test_shutdown_cleans_up(self):
        port = _find_free_port()
        bridge = OpenClawBridge(host="127.0.0.1", port=port)
        tool = ToolDefinition(name="echo", description="Echo")
        bridge.register_tool(tool, _echo_tool)
        bridge.serve_background()
        time.sleep(0.2)
        bridge.shutdown()
        assert bridge._server is None

    def test_shutdown_idempotent(self):
        bridge = OpenClawBridge()
        # Calling shutdown without serve should not crash
        bridge.shutdown()
        bridge.shutdown()
