"""Agent-to-Agent protocol support — supervised inter-agent communication.

Implements Google's Agent2Agent (A2A) protocol for AgentHandler. Every
outbound call to another agent goes through the Supervisor with full
policy enforcement (budgets, timeouts, circuit breakers, audit).

Usage:
    client = A2AClient(supervisor=sv, timeout=30)
    result = await client.send_task(
        agent_url="https://other-agent.example.com",
        task={"skill": "research", "input": "quantum computing"},
    )

To make your agent discoverable:
    endpoint = A2ASupervisedEndpoint(manager=mgr, session_id=sid)
    app.include_router(endpoint.router())
"""

from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .errors import AgentHandlerError
from .supervisor import SupervisedResult, Supervisor

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Task lifecycle
# ---------------------------------------------------------------------------


class TaskState(Enum):
    """A2A task lifecycle states (per spec)."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


@dataclass
class A2ATask:
    """Represents an A2A task with its current state."""

    id: str
    state: TaskState = TaskState.SUBMITTED
    skill: str = ""
    input: Any = None
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "state": self.state.value,
        }
        if self.skill:
            d["skill"] = self.skill
        if self.input is not None:
            d["input"] = self.input
        if self.output is not None:
            d["output"] = self.output
        if self.error is not None:
            d["error"] = self.error
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> A2ATask:
        return cls(
            id=d.get("id", str(uuid.uuid4())),
            state=TaskState(d["state"]) if "state" in d else TaskState.SUBMITTED,
            skill=d.get("skill", ""),
            input=d.get("input"),
            output=d.get("output"),
            error=d.get("error"),
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Agent card
# ---------------------------------------------------------------------------


@dataclass
class A2ASkill:
    """A skill advertised in an agent card."""

    name: str
    description: str = ""
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name}
        if self.description:
            d["description"] = self.description
        if self.input_schema is not None:
            d["inputSchema"] = self.input_schema
        if self.output_schema is not None:
            d["outputSchema"] = self.output_schema
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> A2ASkill:
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            input_schema=d.get("inputSchema"),
            output_schema=d.get("outputSchema"),
        )


@dataclass
class A2AAuthentication:
    """Authentication requirements for an agent endpoint."""

    schemes: List[str] = field(default_factory=list)
    credentials: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.schemes:
            d["schemes"] = list(self.schemes)
        if self.credentials is not None:
            d["credentials"] = self.credentials
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> A2AAuthentication:
        return cls(
            schemes=d.get("schemes", []),
            credentials=d.get("credentials"),
        )


@dataclass
class A2AAgentCard:
    """An agent's capability card (per A2A spec).

    Describes what an agent can do and how to reach it.
    Served at ``/.well-known/agent.json``.
    """

    name: str
    description: str = ""
    url: str = ""
    version: str = "1.0"
    skills: List[A2ASkill] = field(default_factory=list)
    authentication: Optional[A2AAuthentication] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "url": self.url,
            "version": self.version,
            "skills": [s.to_dict() for s in self.skills],
        }
        if self.description:
            d["description"] = self.description
        if self.authentication is not None:
            d["authentication"] = self.authentication.to_dict()
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> A2AAgentCard:
        skills = [A2ASkill.from_dict(s) for s in d.get("skills", [])]
        auth = None
        if "authentication" in d:
            auth = A2AAuthentication.from_dict(d["authentication"])
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            url=d.get("url", ""),
            version=d.get("version", "1.0"),
            skills=skills,
            authentication=auth,
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_supervisor(
        cls,
        supervisor: Supervisor,
        name: str,
        url: str = "",
        description: str = "",
        tool_names: Optional[List[str]] = None,
    ) -> A2AAgentCard:
        """Generate an agent card from a Supervisor's tools and policy.

        Args:
            supervisor: The Supervisor whose policy informs the card.
            name: Agent name for the card.
            url: Base URL where this agent is reachable.
            description: Human-readable description.
            tool_names: Tool names to advertise as skills. If None,
                        generates a single skill from the agent name.
        """
        skills: List[A2ASkill] = []
        if tool_names:
            for tool_name in tool_names:
                skills.append(A2ASkill(name=tool_name))
        else:
            skills.append(A2ASkill(name=name, description=description))

        meta: Dict[str, Any] = {
            "supervision": {
                "max_iterations": supervisor.policy.max_iterations,
                "tool_timeout": supervisor.policy.tool_timeout,
                "token_budget": supervisor.policy.token_budget,
            },
        }

        return cls(
            name=name,
            description=description,
            url=url,
            skills=skills,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# A2A client — supervised outbound calls
# ---------------------------------------------------------------------------


class A2AClient:
    """Supervised client for outbound A2A calls.

    Every HTTP call to another agent goes through the Supervisor, so
    budgets, timeouts, circuit breakers, and audit all apply.

    Args:
        supervisor: The Supervisor to route calls through.
        timeout: HTTP request timeout in seconds (separate from the
                 Supervisor's tool_timeout, which still applies).
        headers: Extra HTTP headers sent with every request.
    """

    def __init__(
        self,
        supervisor: Supervisor,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if httpx is None:
            raise ImportError(
                "httpx required for A2A client: pip install agenthandler[a2a]\n"
                "Or: pip install httpx"
            )
        self._supervisor = supervisor
        self._timeout = timeout
        self._headers = headers or {}

    async def send_task(
        self,
        agent_url: str,
        task: Dict[str, Any],
        *,
        task_id: Optional[str] = None,
    ) -> SupervisedResult:
        """Send a task to another agent via A2A.

        Args:
            agent_url: Base URL of the target agent.
            task: Task payload (should contain ``skill`` and ``input``).
            task_id: Optional task ID. Generated if not provided.

        Returns:
            SupervisedResult wrapping the A2A response.
        """
        tid = task_id or str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "id": tid,
            "params": {
                "id": tid,
                **task,
            },
        }

        async def _do_send(url: str = "", body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{url.rstrip('/')}/a2a/tasks/send",
                    json=body,
                    headers=self._headers,
                )
                resp.raise_for_status()
                return dict(resp.json())

        return await self._supervisor.call(
            "a2a_call",
            _do_send,
            url=agent_url,
            body=payload,
        )

    async def get_task(
        self,
        agent_url: str,
        task_id: str,
    ) -> SupervisedResult:
        """Get the status of a previously submitted task.

        Args:
            agent_url: Base URL of the target agent.
            task_id: The task ID to query.

        Returns:
            SupervisedResult wrapping the task status response.
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "id": task_id,
            "params": {"id": task_id},
        }

        async def _do_get(url: str = "", body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{url.rstrip('/')}/a2a/tasks/get",
                    json=body,
                    headers=self._headers,
                )
                resp.raise_for_status()
                return dict(resp.json())

        return await self._supervisor.call(
            "a2a_call",
            _do_get,
            url=agent_url,
            body=payload,
        )

    async def cancel_task(
        self,
        agent_url: str,
        task_id: str,
    ) -> SupervisedResult:
        """Cancel a previously submitted task.

        Args:
            agent_url: Base URL of the target agent.
            task_id: The task ID to cancel.

        Returns:
            SupervisedResult wrapping the cancellation response.
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "id": task_id,
            "params": {"id": task_id},
        }

        async def _do_cancel(
            url: str = "", body: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{url.rstrip('/')}/a2a/tasks/cancel",
                    json=body,
                    headers=self._headers,
                )
                resp.raise_for_status()
                return dict(resp.json())

        return await self._supervisor.call(
            "a2a_call",
            _do_cancel,
            url=agent_url,
            body=payload,
        )

    async def get_agent_card(
        self,
        agent_url: str,
    ) -> SupervisedResult:
        """Fetch another agent's capability card.

        Args:
            agent_url: Base URL of the target agent.

        Returns:
            SupervisedResult wrapping the parsed A2AAgentCard as a dict.
        """

        async def _do_fetch(url: str = "") -> Dict[str, Any]:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{url.rstrip('/')}/.well-known/agent.json",
                    headers=self._headers,
                )
                resp.raise_for_status()
                return dict(resp.json())

        return await self._supervisor.call(
            "a2a_call",
            _do_fetch,
            url=agent_url,
        )


# ---------------------------------------------------------------------------
# A2A supervised endpoint — makes your agent callable via A2A
# ---------------------------------------------------------------------------


class A2ASupervisedEndpoint:
    """Makes an AgentHandler-supervised agent discoverable and callable via A2A.

    Provides:
    - ``GET /.well-known/agent.json`` — agent card
    - ``POST /a2a/tasks/send`` — accept a task
    - ``POST /a2a/tasks/get`` — query task status
    - ``POST /a2a/tasks/cancel`` — cancel a task

    Incoming tasks are routed through a Supervisor so all policy
    enforcement applies.

    Args:
        manager: SessionManager that owns the session.
        session_id: The session to route incoming tasks through.
        agent_card: Pre-built agent card. If None, one is generated
                    from the Supervisor's policy.
        tool_router: Callable that maps a skill name to an async tool
                     function. If None, incoming tasks are stored but
                     not executed.
    """

    def __init__(
        self,
        manager: Any,
        session_id: str,
        agent_card: Optional[A2AAgentCard] = None,
        tool_router: Optional[
            Callable[[str], Optional[Callable[..., Coroutine[Any, Any, Any]]]]
        ] = None,
        name: str = "agenthandler",
        description: str = "",
        url: str = "",
        auth_token: Optional[str] = None,
    ) -> None:
        self._manager = manager
        self._session_id = session_id
        self._tool_router = tool_router
        self._tasks: Dict[str, A2ATask] = {}
        self._auth_token = auth_token

        sv = manager.get_supervisor(session_id)
        if agent_card is not None:
            self._card = agent_card
        elif sv is not None:
            self._card = A2AAgentCard.from_supervisor(
                sv, name=name, url=url, description=description
            )
        else:
            self._card = A2AAgentCard(name=name, url=url, description=description)

    @property
    def agent_card(self) -> A2AAgentCard:
        return self._card

    def _get_supervisor(self) -> Any:
        sv = self._manager.get_supervisor(self._session_id)
        if sv is None:
            raise AgentHandlerError.session_not_found(self._session_id)
        return sv

    async def handle_send(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a ``tasks/send`` JSON-RPC request.

        Routes the task through the Supervisor, records it, and returns
        the JSON-RPC response.
        """
        params = request_body.get("params", {})
        rpc_id = request_body.get("id", str(uuid.uuid4()))
        task_id = params.get("id", rpc_id)
        skill = params.get("skill", "")
        task_input = params.get("input")

        task = A2ATask(
            id=task_id,
            state=TaskState.WORKING,
            skill=skill,
            input=task_input,
            metadata=params.get("metadata", {}),
        )
        self._tasks[task_id] = task

        sv = self._get_supervisor()
        tool_fn = None
        if self._tool_router is not None:
            tool_fn = self._tool_router(skill)

        if tool_fn is not None:
            result = await sv.call(
                skill,
                tool_fn,
                **(task_input if isinstance(task_input, dict) else {"input": task_input}),
            )
            if result.succeeded:
                task.state = TaskState.COMPLETED
                task.output = result.output
            else:
                task.state = TaskState.FAILED
                task.error = result.error.user_message() if result.error else "Unknown error"
        else:
            task.state = TaskState.COMPLETED
            task.output = None

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": task.to_dict(),
        }

    async def handle_get(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a ``tasks/get`` JSON-RPC request."""
        params = request_body.get("params", {})
        rpc_id = request_body.get("id", "")
        task_id = params.get("id", "")

        task = self._tasks.get(task_id)
        if task is None:
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {
                    "code": -32001,
                    "message": f"Task not found: {task_id}",
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": task.to_dict(),
        }

    async def handle_cancel(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a ``tasks/cancel`` JSON-RPC request."""
        params = request_body.get("params", {})
        rpc_id = request_body.get("id", "")
        task_id = params.get("id", "")

        task = self._tasks.get(task_id)
        if task is None:
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {
                    "code": -32001,
                    "message": f"Task not found: {task_id}",
                },
            }

        task.state = TaskState.CANCELED
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": task.to_dict(),
        }

    def _check_auth(self, request: Any) -> None:
        """Validate the auth token on an incoming request if configured.

        Raises HTTPException(401) when a token is configured but the
        request does not supply a matching ``Authorization: Bearer <token>``.
        """
        if self._auth_token is None:
            return
        from fastapi import HTTPException

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing authentication")
        provided = auth_header[len("Bearer ") :]
        if not secrets.compare_digest(provided, self._auth_token):
            raise HTTPException(status_code=401, detail="Invalid authentication token")

    def router(self) -> Any:
        """Build a FastAPI APIRouter with A2A endpoints.

        Returns:
            A ``fastapi.APIRouter`` you can include in your app.

        Raises:
            ImportError: If ``fastapi`` is not installed.
        """
        try:
            from fastapi import APIRouter, Request
            from fastapi.responses import JSONResponse
        except ImportError:
            raise ImportError("FastAPI required for A2A endpoints: pip install fastapi")

        api = APIRouter()
        card_dict = self._card.to_dict()

        @api.get("/.well-known/agent.json")
        async def get_agent_card() -> JSONResponse:
            return JSONResponse(content=card_dict)

        @api.post("/a2a/tasks/send")
        async def send_task(request: Request) -> JSONResponse:
            self._check_auth(request)
            body = await request.json()
            result = await self.handle_send(body)
            return JSONResponse(content=result)

        @api.post("/a2a/tasks/get")
        async def get_task(request: Request) -> JSONResponse:
            self._check_auth(request)
            body = await request.json()
            result = await self.handle_get(body)
            return JSONResponse(content=result)

        @api.post("/a2a/tasks/cancel")
        async def cancel_task(request: Request) -> JSONResponse:
            self._check_auth(request)
            body = await request.json()
            result = await self.handle_cancel(body)
            return JSONResponse(content=result)

        return api
