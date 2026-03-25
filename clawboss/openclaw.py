"""OpenClaw integration — bridge clawboss-supervised tools to OpenClaw.

Converts clawboss Skill/ToolDefinition to OpenClaw's JSON Schema format
and serves them over HTTP for the OpenClaw TypeScript plugin.

Usage:
    from clawboss import OpenClawBridge, Policy, ToolDefinition, ToolParameter

    bridge = OpenClawBridge(policy=Policy(tool_timeout=10))
    bridge.register_tool(tool_def, my_async_fn)
    bridge.serve()  # http://127.0.0.1:9229

Or convert schemas without running a server:
    from clawboss import to_openclaw_tool_schema
    schema = to_openclaw_tool_schema(tool_def)
"""

import asyncio
import json
import re
import threading
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from .audit import AuditLog
from .policy import Policy
from .skill import Skill, ToolDefinition
from .supervisor import Supervisor, SupervisedResult


def _slugify(name: str) -> str:
    """Convert a name to a safe identifier."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def to_openclaw_tool_schema(tool: ToolDefinition) -> dict:
    """Convert a clawboss ToolDefinition to OpenClaw's registerTool format.

    Returns a dict with name, description, and JSON Schema parameters
    compatible with OpenClaw's api.registerTool().
    """
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param in tool.parameters:
        prop: Dict[str, Any] = {"type": param.type}
        if param.description:
            prop["description"] = param.description
        if param.default is not None:
            prop["default"] = param.default
        properties[param.name] = prop
        if param.required:
            required.append(param.name)

    schema: Dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "parameters": {
            "type": "object",
            "properties": properties,
        },
    }
    if required:
        schema["parameters"]["required"] = required

    return schema


def to_openclaw_manifest(
    skill: Skill,
    plugin_id: Optional[str] = None,
    bridge_port: int = 9229,
) -> dict:
    """Generate an openclaw.plugin.json manifest for a skill.

    Args:
        skill: The skill to generate a manifest for.
        plugin_id: Plugin identifier. Defaults to "clawboss-{skill.name}".
        bridge_port: Port the bridge server will run on.
    """
    pid = plugin_id or f"clawboss-{_slugify(skill.name)}"
    return {
        "id": pid,
        "name": f"Clawboss: {skill.name}",
        "version": skill.version,
        "description": skill.description,
        "entry": "dist/index.js",
        "configSchema": {
            "type": "object",
            "properties": {
                "bridgeUrl": {
                    "type": "string",
                    "default": f"http://localhost:{bridge_port}",
                    "description": "URL of the clawboss bridge server",
                },
            },
        },
        "tools": [to_openclaw_tool_schema(t) for t in skill.tools],
    }


def _supervised_result_to_dict(result: SupervisedResult) -> dict:
    """Convert a SupervisedResult to a JSON-serializable response dict."""
    response: Dict[str, Any] = {
        "success": result.succeeded,
        "result": None,
        "error": None,
        "metadata": {
            "duration_ms": result.duration_ms,
            "tool_name": result.tool_name,
        },
    }

    if result.succeeded and result.output is not None:
        try:
            json.dumps(result.output)
            response["result"] = result.output
        except (TypeError, ValueError):
            response["result"] = str(result.output)

    if result.error:
        response["error"] = {
            "kind": result.error.kind,
            "message": str(result.error),
        }

    if result.budget:
        response["metadata"]["budget"] = {
            "tokens_used": result.budget.tokens_used,
            "token_limit": result.budget.token_limit,
            "iterations": result.budget.iterations,
            "iteration_limit": result.budget.iteration_limit,
        }

    return response


class OpenClawBridge:
    """HTTP bridge that exposes clawboss-supervised tools to OpenClaw.

    Registers tool definitions with their async callables, then serves
    them over HTTP for the OpenClaw TypeScript plugin to discover and invoke.

    Usage:
        bridge = OpenClawBridge(policy=Policy(tool_timeout=10))
        bridge.register_tool(tool_def, my_search_fn)
        bridge.serve()  # blocking, Ctrl+C to stop
    """

    def __init__(
        self,
        policy: Optional[Policy] = None,
        audit: Optional[AuditLog] = None,
        host: str = "127.0.0.1",
        port: int = 9229,
    ):
        self._policy = policy or Policy()
        self._audit = audit
        self._host = host
        self._port = port
        self._registry: Dict[str, Tuple[ToolDefinition, Callable[..., Coroutine]]] = {}
        self._lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None

    def register_tool(
        self,
        tool: ToolDefinition,
        fn: Callable[..., Coroutine],
    ) -> None:
        """Register a tool with its async implementation."""
        with self._lock:
            self._registry[tool.name] = (tool, fn)

    def register_skill(
        self,
        skill: Skill,
        tool_impls: Dict[str, Callable[..., Coroutine]],
    ) -> None:
        """Register all tools from a skill.

        Args:
            skill: Skill containing tool definitions.
            tool_impls: Dict mapping tool name to its async callable.
        """
        if skill.supervision:
            self._policy = Policy.from_dict(skill.supervision)

        for tool in skill.tools:
            if tool.name not in tool_impls:
                raise ValueError(
                    f"Missing implementation for tool '{tool.name}'. "
                    f"Provide it in tool_impls."
                )
            self.register_tool(tool, tool_impls[tool.name])

    def _start_event_loop(self) -> None:
        """Start a background event loop for async tool execution."""
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="clawboss-bridge-loop",
        )
        self._loop_thread.start()

    def _stop_event_loop(self) -> None:
        """Stop the background event loop."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread:
            self._loop_thread.join(timeout=5)

    def _execute_tool(self, tool_name: str, params: dict) -> dict:
        """Execute a tool through the Supervisor. Called from HTTP handler thread."""
        if not self._loop:
            return {"success": False, "error": {"kind": "internal", "message": "Event loop not running"}}

        supervisor = Supervisor(self._policy, audit=self._audit or AuditLog.noop())

        tool_def, fn = self._registry[tool_name]

        future = asyncio.run_coroutine_threadsafe(
            supervisor.call(tool_name, fn, **params),
            self._loop,
        )
        result = future.result(timeout=self._policy.request_timeout)
        return _supervised_result_to_dict(result)

    def _make_handler(self) -> type:
        """Create an HTTP request handler class bound to this bridge."""
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # suppress default stderr logging

            def _send_json(self, status: int, data: dict) -> None:
                body = json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path.strip("/")

                if path == "health":
                    self._send_json(200, {
                        "status": "ok",
                        "tools": len(bridge._registry),
                    })
                elif path == "tools":
                    tools = []
                    with bridge._lock:
                        for tool_def, _ in bridge._registry.values():
                            tools.append(to_openclaw_tool_schema(tool_def))
                    self._send_json(200, {"tools": tools})
                else:
                    self._send_json(404, {"error": "Not found"})

            def do_POST(self):
                parsed = urllib.parse.urlparse(self.path)
                segments = parsed.path.strip("/").split("/")

                if len(segments) != 2 or segments[0] != "execute":
                    self._send_json(404, {"error": "Not found"})
                    return

                tool_name = segments[1]

                with bridge._lock:
                    if tool_name not in bridge._registry:
                        self._send_json(404, {
                            "error": f"Tool '{tool_name}' not registered",
                        })
                        return

                # Parse request body
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length == 0:
                    self._send_json(400, {"error": "Request body required"})
                    return

                try:
                    body = json.loads(self.rfile.read(content_length))
                except (json.JSONDecodeError, ValueError):
                    self._send_json(400, {"error": "Invalid JSON in request body"})
                    return

                if not isinstance(body.get("params"), dict):
                    self._send_json(400, {
                        "error": "Request body must contain 'params' object",
                    })
                    return

                # Execute through supervisor
                try:
                    response = bridge._execute_tool(tool_name, body["params"])
                    self._send_json(200, response)
                except Exception as e:
                    self._send_json(500, {
                        "error": f"Internal server error: {e}",
                    })

        return Handler

    def serve(self) -> None:
        """Start the bridge server (blocking). Ctrl+C to stop."""
        self._start_event_loop()
        handler = self._make_handler()
        self._server = ThreadingHTTPServer((self._host, self._port), handler)
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def serve_background(self) -> threading.Thread:
        """Start the bridge in a background thread. Returns the thread."""
        self._start_event_loop()
        handler = self._make_handler()
        self._server = ThreadingHTTPServer((self._host, self._port), handler)
        thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="clawboss-bridge-http",
        )
        thread.start()
        return thread

    def shutdown(self) -> None:
        """Stop the server and clean up."""
        if self._server:
            self._server.shutdown()
            self._server = None
        self._stop_event_loop()
