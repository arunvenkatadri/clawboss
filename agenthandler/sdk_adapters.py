"""SDK adapters — integrate AgentHandler supervision with agent SDKs.

Thin wrappers that make any function tool supervised by AgentHandler,
then expose it in the format expected by the OpenAI Agents SDK or
Claude Agent SDK. Works even without the target SDKs installed.

Usage:
    from agenthandler.sdk_adapters import wrap_openai_tool, supervised_tool_registry

    supervised_search = wrap_openai_tool(search_fn, supervisor, tool_name="search")
    supervised = supervised_tool_registry(
        {"search": search_fn, "write": write_fn},
        manager,
        policy={"tool_timeout": 30, "token_budget": 50000},
    )
"""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .errors import AgentHandlerError
from .session import SessionManager
from .supervisor import Supervisor

try:
    from openai_agents import Agent as OpenAIAgent  # noqa: F401
    from openai_agents import GuardrailFunctionOutput, InputGuardrail, OutputGuardrail

    _HAS_OPENAI_AGENTS = True
except ImportError:
    _HAS_OPENAI_AGENTS = False

try:
    from claude_agent_sdk import Tool as ClaudeTool  # noqa: F401

    _HAS_CLAUDE_SDK = True
except ImportError:
    _HAS_CLAUDE_SDK = False


def _ensure_async(fn: Callable[..., Any]) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Wrap a sync callable into an async one if needed."""
    if asyncio.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    async def _wrapper(**kwargs: Any) -> Any:
        return fn(**kwargs)

    return _wrapper


def wrap_openai_tool(
    fn: Callable[..., Any],
    supervisor: Supervisor,
    tool_name: Optional[str] = None,
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Wrap a function with AgentHandler supervision for use as an OpenAI tool.

    The returned async callable goes through the Supervisor's full policy
    enforcement (timeouts, budgets, circuit breakers, guardrails) before
    executing the original function.

    Args:
        fn: The tool function (sync or async).
        supervisor: Supervisor instance to enforce policy.
        tool_name: Name for audit/tracking. Defaults to fn.__name__.

    Returns:
        An async callable with the same signature, supervised by AgentHandler.
    """
    name = tool_name or getattr(fn, "__name__", "tool")
    async_fn = _ensure_async(fn)

    @functools.wraps(fn)
    async def _supervised(**kwargs: Any) -> Any:
        result = await supervisor.call(name, async_fn, **kwargs)
        if result.succeeded:
            return result.output
        raise AgentHandlerError.tool_error(
            result.error.user_message() if result.error else "Tool call failed"
        )

    return _supervised


def wrap_claude_tool(
    fn: Callable[..., Any],
    supervisor: Supervisor,
    tool_name: Optional[str] = None,
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Wrap a function with AgentHandler supervision for use as a Claude tool.

    Same pattern as wrap_openai_tool — the returned callable goes through
    the Supervisor before executing the original function.

    Args:
        fn: The tool function (sync or async).
        supervisor: Supervisor instance to enforce policy.
        tool_name: Name for audit/tracking. Defaults to fn.__name__.

    Returns:
        An async callable supervised by AgentHandler.
    """
    name = tool_name or getattr(fn, "__name__", "tool")
    async_fn = _ensure_async(fn)

    @functools.wraps(fn)
    async def _supervised(**kwargs: Any) -> Any:
        result = await supervisor.call(name, async_fn, **kwargs)
        if result.succeeded:
            return result.output
        raise AgentHandlerError.tool_error(
            result.error.user_message() if result.error else "Tool call failed"
        )

    return _supervised


def openai_guardrail_adapter(guardrails: List[Any]) -> List[Any]:
    """Convert AgentHandler guardrails to OpenAI Agents SDK guardrail format.

    Takes a list of AgentHandler guardrails (PreCallGuardrail / PostCallGuardrail)
    and returns a list of OpenAI InputGuardrail / OutputGuardrail objects.

    Falls back to plain wrapper dicts if the OpenAI Agents SDK is not installed.

    Args:
        guardrails: List of AgentHandler guardrail instances.

    Returns:
        List of OpenAI-compatible guardrail objects (or wrapper dicts if SDK
        is not installed).
    """
    adapted: List[Any] = []
    for gr in guardrails:
        gr_name = getattr(gr, "name", "guardrail")
        has_check = hasattr(gr, "check")
        has_check_async = hasattr(gr, "check_async")

        if has_check:
            sig = inspect.signature(gr.check)
            is_post = "output" in sig.parameters
        elif has_check_async:
            sig = inspect.signature(gr.check_async)
            is_post = "output" in sig.parameters
        else:
            continue

        if is_post:
            adapted.append(_adapt_output_guardrail(gr, gr_name, has_check_async))
        else:
            adapted.append(_adapt_input_guardrail(gr, gr_name, has_check_async))

    return adapted


def _adapt_input_guardrail(gr: Any, name: str, is_async: bool) -> Any:
    """Convert an AgentHandler pre-call guardrail to OpenAI InputGuardrail."""
    context: Dict[str, Any] = {}

    async def _check(ctx: Any, agent: Any, input_data: Any) -> Any:
        kwargs: Dict[str, Any] = {}
        if isinstance(input_data, dict):
            kwargs = input_data
        if is_async:
            result = await gr.check_async("", kwargs, context)
        else:
            result = gr.check("", kwargs, context)
        if not result.allowed:
            if _HAS_OPENAI_AGENTS:
                return GuardrailFunctionOutput(
                    output=result.reason,
                    tripwire_triggered=True,
                )
            return {"output": result.reason, "tripwire_triggered": True}
        if _HAS_OPENAI_AGENTS:
            return GuardrailFunctionOutput(output="", tripwire_triggered=False)
        return {"output": "", "tripwire_triggered": False}

    if _HAS_OPENAI_AGENTS:
        return InputGuardrail(guardrail_function=_check, name=name)
    return {"type": "input", "name": name, "check": _check}


def _adapt_output_guardrail(gr: Any, name: str, is_async: bool) -> Any:
    """Convert an AgentHandler post-call guardrail to OpenAI OutputGuardrail."""
    context: Dict[str, Any] = {}

    async def _check(ctx: Any, agent: Any, output_data: Any) -> Any:
        kwargs: Dict[str, Any] = {}
        if is_async:
            result = await gr.check_async("", kwargs, output_data, context)
        else:
            result = gr.check("", kwargs, output_data, context)
        if not result.allowed:
            if _HAS_OPENAI_AGENTS:
                return GuardrailFunctionOutput(
                    output=result.reason,
                    tripwire_triggered=True,
                )
            return {"output": result.reason, "tripwire_triggered": True}
        if _HAS_OPENAI_AGENTS:
            return GuardrailFunctionOutput(output="", tripwire_triggered=False)
        return {"output": "", "tripwire_triggered": False}

    if _HAS_OPENAI_AGENTS:
        return OutputGuardrail(guardrail_function=_check, name=name)
    return {"type": "output", "name": name, "check": _check}


def supervised_tool_registry(
    tools: Dict[str, Callable[..., Any]],
    manager: SessionManager,
    policy: Optional[Dict[str, Any]] = None,
    agent_id: str = "sdk-agent",
) -> Dict[str, Callable[..., Coroutine[Any, Any, Any]]]:
    """Wrap every tool in a dict with AgentHandler supervision.

    Creates a session and returns a new dict where each tool is supervised.
    Framework-agnostic — works with any SDK that accepts async callables.

    Args:
        tools: Dict mapping tool names to callables.
        manager: SessionManager for session lifecycle.
        policy: Policy dict for supervision.
        agent_id: Agent identifier for the session.

    Returns:
        Dict mapping tool names to supervised async callables.
    """
    sid = manager.start(agent_id, policy or {})
    sv = manager.get_supervisor(sid)

    supervised: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
    for name, fn in tools.items():
        supervised[name] = wrap_openai_tool(fn, sv, tool_name=name)

    return supervised


class AgentHandlerMiddleware:
    """Wraps an async agent callable with session management.

    Handles session lifecycle (start, stop) and injects a supervised
    tool set into the agent function.

    Usage:
        middleware = AgentHandlerMiddleware(manager, policy={"token_budget": 50000})
        result = await middleware.run(my_agent_fn, tools={"search": search_fn})
    """

    def __init__(
        self,
        manager: SessionManager,
        policy: Optional[Dict[str, Any]] = None,
        agent_id: str = "middleware-agent",
    ) -> None:
        self._manager = manager
        self._policy = policy or {}
        self._agent_id = agent_id
        self._session_id: Optional[str] = None

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    async def run(
        self,
        agent_fn: Callable[..., Coroutine[Any, Any, Any]],
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run an agent function with supervised tools and session management.

        Args:
            agent_fn: The agent's main loop (async callable).
            tools: Dict of tool name -> callable to supervise.
            **kwargs: Extra arguments passed to agent_fn.

        Returns:
            Whatever agent_fn returns.
        """
        sid = self._manager.start(self._agent_id, self._policy)
        self._session_id = sid
        sv = self._manager.get_supervisor(sid)

        supervised_tools: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        if tools:
            for name, fn in tools.items():
                supervised_tools[name] = wrap_openai_tool(fn, sv, tool_name=name)

        try:
            return await agent_fn(tools=supervised_tools, **kwargs)
        finally:
            try:
                self._manager.stop(sid)
            except Exception:
                pass
