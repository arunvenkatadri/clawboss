"""REST control plane — FastAPI app exposing the SessionManager.

Optional dependency: install with ``pip install clawboss[server]``.

Run with::

    uvicorn clawboss.server:app

Or with API key auth::

    CLAWBOSS_API_KEY=my-secret-key uvicorn clawboss.server:app

Or use ``create_app()`` to build an app with a custom store.

By default, CORS is restricted to localhost origins only.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from typing import Any, Dict, List, Optional

try:
    from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from pydantic import BaseModel
except ImportError as e:
    raise ImportError(
        "clawboss[server] extras required: pip install clawboss[server]"
    ) from e

from .session import SessionManager
from .store import SqliteStore

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    agent_id: str
    policy: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None
    stateless: bool = False


class SessionSummary(BaseModel):
    session_id: str
    agent_id: str
    status: str
    iterations: int = 0
    tokens_used: int = 0
    token_limit: Optional[int] = None
    iteration_limit: int = 5
    timestamp: str = ""
    payload: Optional[Dict[str, Any]] = None
    stateless: bool = False


class SessionDetail(SessionSummary):
    policy_dict: Optional[Dict[str, Any]] = None
    circuit_breaker_states: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


def _make_auth_dependency(api_key: Optional[str]):
    """Build a FastAPI dependency that enforces Bearer token auth.

    If api_key is None, auth is disabled (all requests pass).
    """

    async def _check_auth(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    ):
        if api_key is None:
            return  # auth disabled
        if credentials is None or not secrets.compare_digest(
            credentials.credentials, api_key
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return _check_auth


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


_LOCALHOST_ORIGINS = [
    "http://localhost",
    "http://localhost:*",
    "http://127.0.0.1",
    "http://127.0.0.1:*",
]


def create_app(
    manager: Optional[SessionManager] = None,
    allowed_origins: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> FastAPI:
    """Build a FastAPI app wired to the given SessionManager.

    If no manager is supplied, one is created with a SqliteStore.

    Args:
        manager: SessionManager to use. Created with SqliteStore if None.
        allowed_origins: CORS allowed origins. Defaults to localhost only.
                        Pass ["*"] to allow all origins (NOT recommended
                        for production without auth).
        api_key: If set, all HTTP endpoints require ``Authorization: Bearer <key>``.
                 If None, checks the CLAWBOSS_API_KEY environment variable.
                 If neither is set, auth is disabled (open access).
    """
    if manager is None:
        store = SqliteStore()
        manager = SessionManager(store)

    # Resolve API key: explicit param > env var > disabled
    resolved_key = api_key if api_key is not None else os.environ.get("CLAWBOSS_API_KEY")

    auth = _make_auth_dependency(resolved_key)

    app = FastAPI(
        title="Clawboss Control Plane",
        version="0.77.0",
        description=(
            "REST API for managing agent sessions. "
            + (
                "Authentication: Bearer token required."
                if resolved_key
                else "WARNING: No authentication configured. "
                "Set CLAWBOSS_API_KEY or pass api_key to create_app()."
            )
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or _LOCALHOST_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Stash on app state for tests
    app.state.manager = manager
    app.state.auth_enabled = resolved_key is not None

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.post("/sessions", response_model=SessionSummary, status_code=201)
    def create_session(req: CreateSessionRequest, _=Depends(auth)):
        sid = manager.start(req.agent_id, req.policy, req.payload, stateless=req.stateless)
        cp = manager.status(sid)
        return _checkpoint_to_summary(cp)

    @app.get("/sessions", response_model=List[SessionSummary])
    def list_sessions(_=Depends(auth)):
        return [_checkpoint_to_summary(cp) for cp in manager.list_sessions()]

    @app.get("/sessions/{session_id}", response_model=SessionDetail)
    def get_session(session_id: str, _=Depends(auth)):
        cp = manager.status(session_id)
        if cp is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return _checkpoint_to_detail(cp)

    @app.post("/sessions/{session_id}/pause", response_model=SessionSummary)
    def pause_session(session_id: str, _=Depends(auth)):
        try:
            manager.pause(session_id)
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail="Session not found")
            raise
        cp = manager.status(session_id)
        return _checkpoint_to_summary(cp)

    @app.post("/sessions/{session_id}/resume", response_model=SessionSummary)
    def resume_session(session_id: str, _=Depends(auth)):
        try:
            manager.resume(session_id)
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail="Session not found")
            raise
        cp = manager.status(session_id)
        return _checkpoint_to_summary(cp)

    @app.post("/sessions/{session_id}/stop", response_model=SessionSummary)
    def stop_session(session_id: str, _=Depends(auth)):
        try:
            manager.stop(session_id)
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail="Session not found")
            raise
        cp = manager.status(session_id)
        return _checkpoint_to_summary(cp)

    @app.get("/sessions/{session_id}/audit")
    def get_audit(session_id: str, _=Depends(auth)):
        cp = manager.status(session_id)
        if cp is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return manager.get_audit_entries(session_id)

    @app.websocket("/sessions/{session_id}/events")
    async def session_events(websocket: WebSocket, session_id: str):
        """Stream audit events and status changes for a session.

        WebSocket auth: pass the API key as a query param ?token=<key>
        or in the first message after connect.
        """
        # WebSocket auth via query param
        if resolved_key is not None:
            token = websocket.query_params.get("token", "")
            if not secrets.compare_digest(token, resolved_key):
                await websocket.close(code=4001)
                return

        cp = manager.status(session_id)
        if cp is None:
            await websocket.close(code=4004)
            return

        await websocket.accept()
        last_count = 0

        try:
            while True:
                entries = manager.get_audit_entries(session_id)
                if len(entries) > last_count:
                    for entry in entries[last_count:]:
                        await websocket.send_json({"type": "audit", "data": entry})
                    last_count = len(entries)

                cp = manager.status(session_id)
                if cp is not None:
                    await websocket.send_json({
                        "type": "status",
                        "data": {
                            "session_id": cp.session_id,
                            "status": cp.status.value,
                            "iterations": cp.iterations,
                            "tokens_used": cp.tokens_used,
                        },
                    })

                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _checkpoint_to_summary(cp) -> dict:
    return {
        "session_id": cp.session_id,
        "agent_id": cp.agent_id,
        "status": cp.status.value,
        "iterations": cp.iterations,
        "tokens_used": cp.tokens_used,
        "token_limit": cp.token_limit,
        "iteration_limit": cp.iteration_limit,
        "timestamp": cp.timestamp,
        "payload": cp.payload,
        "stateless": cp.stateless,
    }


def _checkpoint_to_detail(cp) -> dict:
    d = _checkpoint_to_summary(cp)
    d["policy_dict"] = cp.policy_dict
    d["circuit_breaker_states"] = cp.circuit_breaker_states
    return d


# ---------------------------------------------------------------------------
# Default app instance for ``uvicorn clawboss.server:app``
# ---------------------------------------------------------------------------

app = create_app()
