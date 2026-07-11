"""MCP server mode — expose supervised tools via Model Context Protocol.

AgentHandler becomes an MCP proxy: wrap any set of tools with supervision
(budgets, timeouts, circuit breakers, guardrails, audit, cost tracking)
and expose them as MCP tools. Any MCP client gets governed tool access.

Usage:
    from agenthandler.mcp_server import SupervisedMCPServer

    server = SupervisedMCPServer(
        name="my-supervised-tools",
        tools={"search": search_fn, "write": write_fn},
        policy={"max_iterations": 100, "tool_timeout": 30, "token_budget": 50000},
    )
    server.run()  # stdio for Claude Desktop
    server.run(transport="streamable-http")  # network mode

Or wrap an existing MCP server's tools with supervision:
    server = SupervisedMCPServer.wrap_mcp(
        "npx -y @modelcontextprotocol/server-filesystem /tmp",
        policy={"tool_timeout": 10},
    )
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .observe import PricingTable
from .policy import Policy
from .session import SessionManager
from .store import MemoryStore

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP = None  # type: ignore[assignment, misc]


@dataclass
class ToolSpec:
    """Metadata for a supervised tool."""

    name: str
    fn: Callable[..., Any]
    description: str = ""
    parameter_schema: Optional[Dict[str, Any]] = None


@dataclass
class MCPServerConfig:
    """Configuration for the supervised MCP server."""

    name: str = "agenthandler"
    policy: Optional[Dict[str, Any]] = None
    agent_id: str = "mcp-agent"
    instructions: str = (
        "Tools supervised by AgentHandler — budgets, timeouts, and guardrails enforced."
    )
    host: str = "0.0.0.0"
    port: int = 8765
    stateless: bool = True
    pricing: Optional[PricingTable] = None


class SupervisedMCPServer:
    """MCP server that wraps tools with AgentHandler supervision.

    Every tool call goes through the Supervisor with full policy enforcement:
    timeouts, budgets, circuit breakers, PII redaction, guardrails, approvals,
    audit logging, and cost attribution.

    Args:
        name: Server name shown to MCP clients.
        tools: Dict mapping tool names to async/sync callables.
        policy: Policy dict for supervision.
        agent_id: Agent identifier for the session.
        instructions: Server description for MCP clients.
        host: Host for HTTP transport.
        port: Port for HTTP transport.
        pricing: PricingTable for cost attribution (defaults to built-in).
    """

    def __init__(
        self,
        name: str = "agenthandler",
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        policy: Optional[Dict[str, Any]] = None,
        agent_id: str = "mcp-agent",
        instructions: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8765,
        pricing: Optional[PricingTable] = None,
        pre_guardrails: Optional[List[Any]] = None,
        post_guardrails: Optional[List[Any]] = None,
    ) -> None:
        if FastMCP is None:
            raise ImportError(
                "MCP SDK required: pip install agenthandler[mcp]\nOr: pip install mcp"
            )

        self._config = MCPServerConfig(
            name=name,
            policy=policy or {"max_iterations": 1000, "tool_timeout": 30},
            agent_id=agent_id,
            instructions=instructions
            or "Tools supervised by AgentHandler — budgets, timeouts, and guardrails enforced.",
            host=host,
            port=port,
            pricing=pricing,
        )
        self._tool_specs: Dict[str, ToolSpec] = {}
        self._pre_guardrails = pre_guardrails or []
        self._post_guardrails = post_guardrails or []
        self._manager: Optional[SessionManager] = None
        self._session_id: Optional[str] = None
        self._mcp: Any = None

        if tools:
            for tool_name, fn in tools.items():
                self.add_tool(tool_name, fn)

    def add_tool(
        self,
        name: str,
        fn: Callable[..., Any],
        description: Optional[str] = None,
        parameter_schema: Optional[Dict[str, Any]] = None,
    ) -> "SupervisedMCPServer":
        """Register a tool to be exposed via MCP with supervision."""
        doc = description or (fn.__doc__ or "").strip().split("\n")[0] or name
        self._tool_specs[name] = ToolSpec(
            name=name,
            fn=fn,
            description=doc,
            parameter_schema=parameter_schema,
        )
        return self

    def _build_mcp(self) -> Any:
        """Build the FastMCP server with supervised tool wrappers."""

        @asynccontextmanager
        async def lifespan(server: Any) -> AsyncIterator[Dict[str, Any]]:
            store = MemoryStore()
            pricing = self._config.pricing or PricingTable.default()
            mgr = SessionManager(
                store,
                pricing=pricing,
                pre_guardrails=self._pre_guardrails,
                post_guardrails=self._post_guardrails,
            )
            sid = mgr.start(
                self._config.agent_id,
                self._config.policy,
                stateless=self._config.stateless,
            )
            self._manager = mgr
            self._session_id = sid
            try:
                yield {"manager": mgr, "session_id": sid}
            finally:
                try:
                    mgr.stop(sid)
                except Exception:
                    pass

        mcp = FastMCP(
            self._config.name,
            instructions=self._config.instructions,
            lifespan=lifespan,
            host=self._config.host,
            port=self._config.port,
        )

        for spec in self._tool_specs.values():
            self._register_tool(mcp, spec)

        self._register_manifest_resource(mcp)

        return mcp

    def _register_tool(self, mcp: Any, spec: ToolSpec) -> None:
        """Register a single supervised tool on the MCP server."""
        original_fn = spec.fn
        tool_name = spec.name

        if asyncio.iscoroutinefunction(original_fn):
            supervised_fn = original_fn
        else:

            async def _async_wrap(**kwargs: Any) -> Any:
                return original_fn(**kwargs)

            supervised_fn = _async_wrap

        captured_fn = supervised_fn

        async def _supervised_call(ctx: Any = None, **kwargs: Any) -> Any:
            if self._manager is None or self._session_id is None:
                raise RuntimeError("Server not initialized")
            sv = self._manager.get_supervisor(self._session_id)
            if sv is None:
                raise RuntimeError("Session not found")

            async def _execute(**kw: Any) -> Any:
                return await captured_fn(**kw)

            result = await sv.call(tool_name, _execute, **kwargs)
            if result.succeeded:
                return result.output
            msg = result.error.user_message() if result.error else "Unknown error"
            raise Exception(f"[AgentHandler] {msg}")

        _supervised_call.__name__ = tool_name
        _supervised_call.__doc__ = spec.description

        sig_params = []
        if spec.parameter_schema and "properties" in spec.parameter_schema:
            for param_name, param_info in spec.parameter_schema["properties"].items():
                default = param_info.get("default", inspect.Parameter.empty)
                sig_params.append(
                    inspect.Parameter(
                        param_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        default=default,
                        annotation=str,
                    )
                )
        else:
            try:
                orig_sig = inspect.signature(original_fn)
                for pname, param in orig_sig.parameters.items():
                    if pname in ("self", "cls", "ctx"):
                        continue
                    sig_params.append(
                        inspect.Parameter(
                            pname,
                            inspect.Parameter.KEYWORD_ONLY,
                            default=param.default,
                            annotation=param.annotation
                            if param.annotation != inspect.Parameter.empty
                            else str,
                        )
                    )
            except (ValueError, TypeError):
                pass

        if sig_params:
            _supervised_call.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
                parameters=sig_params,
                return_annotation=Any,
            )

        mcp.tool(name=tool_name, description=spec.description)(_supervised_call)

    def _register_manifest_resource(self, mcp: Any) -> None:
        """Register a capability manifest resource."""

        @mcp.resource("agenthandler://manifest")
        def get_manifest() -> str:
            """AgentHandler capability manifest."""
            policy = Policy.from_dict(self._config.policy or {})
            manifest = {
                "name": self._config.name,
                "version": "0.92.0",
                "supervision": {
                    "tool_timeout": policy.tool_timeout,
                    "max_iterations": policy.max_iterations,
                    "token_budget": policy.token_budget,
                    "circuit_breaker": policy.circuit_breaker_threshold > 0,
                    "pii_redaction": bool(policy.redact),
                    "require_confirm": list(policy.require_confirm),
                },
                "guardrails": {
                    "pre": len(self._pre_guardrails),
                    "post": len(self._post_guardrails),
                },
                "tools": list(self._tool_specs.keys()),
                "cost_attribution": True,
            }
            return json.dumps(manifest, indent=2)

    def run(self, transport: str = "stdio") -> None:
        """Start the MCP server.

        Args:
            transport: "stdio" for Claude Desktop/CLI, "streamable-http" for network.
        """
        mcp = self._build_mcp()
        self._mcp = mcp
        mcp.run(transport=transport)

    def get_mcp_app(self) -> Any:
        """Get the ASGI app for mounting in an existing server."""
        mcp = self._build_mcp()
        self._mcp = mcp
        return mcp.streamable_http_app()

    def status(self) -> Optional[Dict[str, Any]]:
        """Get current supervision status."""
        if self._manager is None or self._session_id is None:
            return None
        cp = self._manager.status(self._session_id)
        if cp is None:
            return None
        return {
            "session_id": self._session_id,
            "iterations": cp.iterations,
            "tokens_used": cp.tokens_used,
            "status": cp.status,
        }
