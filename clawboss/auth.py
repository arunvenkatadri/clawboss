"""Authentication providers for the REST control plane.

Supports:
- API key (Bearer token) — simple, self-issued
- OAuth2 (GitHub, Google) — for production deployments

Usage:
    # API key only
    app = create_app(api_key="my-secret")

    # OAuth2 via env vars:
    #   CLAWBOSS_OAUTH_PROVIDER=github
    #   CLAWBOSS_OAUTH_CLIENT_ID=...
    #   CLAWBOSS_OAUTH_CLIENT_SECRET=...
    app = create_app()
"""

from __future__ import annotations

import secrets
import time
from typing import Any, Dict, Optional

try:
    from fastapi import Depends, HTTPException, Request  # type: ignore[import-not-found]
    from fastapi.security import (  # type: ignore[import-not-found]
        HTTPAuthorizationCredentials,
        HTTPBearer,
    )
except ImportError:
    pass


_bearer_scheme = HTTPBearer(auto_error=False)

# In-memory token store for OAuth sessions (maps token -> user info + expiry)
_oauth_tokens: Dict[str, Dict[str, Any]] = {}


def make_auth_dependency(
    api_key: Optional[str] = None,
    oauth_enabled: bool = False,
):
    """Build a FastAPI dependency that enforces auth.

    Checks in order:
    1. Bearer token matches api_key (if set)
    2. Bearer token is a valid OAuth session token (if oauth enabled)
    3. Reject

    If api_key is None and oauth is disabled, all requests pass.
    """

    async def _check_auth(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    ) -> Optional[Dict[str, Any]]:
        # No auth configured — open access
        if api_key is None and not oauth_enabled:
            return None

        if credentials is None:
            raise HTTPException(status_code=401, detail="Missing authentication")

        token = credentials.credentials

        # Check API key
        if api_key is not None and secrets.compare_digest(token, api_key):
            return {"auth_type": "api_key"}

        # Check OAuth session token
        if oauth_enabled and token in _oauth_tokens:
            session = _oauth_tokens[token]
            if session.get("expires_at", 0) > time.time():
                return session
            else:
                del _oauth_tokens[token]  # expired

        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return _check_auth


def register_oauth_routes(app: Any, provider: str, client_id: str, client_secret: str) -> None:
    """Add OAuth2 login/callback routes to the FastAPI app.

    Supports: "github", "google"
    """
    providers = {
        "github": {
            "authorize_url": "https://github.com/login/oauth/authorize",
            "token_url": "https://github.com/login/oauth/access_token",
            "user_url": "https://api.github.com/user",
            "scope": "read:user",
        },
        "google": {
            "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "user_url": "https://www.googleapis.com/oauth2/v2/userinfo",
            "scope": "openid email profile",
        },
    }

    if provider not in providers:
        raise ValueError(f"Unsupported OAuth provider: {provider}. Use: {list(providers.keys())}")

    config = providers[provider]

    @app.get("/auth/login")
    def oauth_login(request: Request) -> dict:
        """Redirect URL for OAuth2 login."""
        callback_url = str(request.base_url) + "auth/callback"
        state = secrets.token_hex(16)
        url = (
            f"{config['authorize_url']}?"
            f"client_id={client_id}&"
            f"redirect_uri={callback_url}&"
            f"scope={config['scope']}&"
            f"state={state}"
        )
        return {"login_url": url, "state": state}

    @app.get("/auth/callback")
    async def oauth_callback(code: str, state: str = "") -> dict:
        """OAuth2 callback — exchange code for token, create session."""
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="httpx required for OAuth. pip install httpx",
            )

        # Exchange code for access token
        async with httpx.AsyncClient() as client:
            token_resp = await client.post(
                config["token_url"],
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                },
                headers={"Accept": "application/json"},
            )
            if token_resp.status_code != 200:
                raise HTTPException(status_code=401, detail="OAuth token exchange failed")
            token_data = token_resp.json()
            access_token = token_data.get("access_token")
            if not access_token:
                raise HTTPException(status_code=401, detail="No access token in response")

            # Get user info
            user_resp = await client.get(
                config["user_url"],
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if user_resp.status_code != 200:
                raise HTTPException(status_code=401, detail="Failed to fetch user info")
            user_info = user_resp.json()

        # Create a session token
        session_token = secrets.token_hex(32)
        _oauth_tokens[session_token] = {
            "auth_type": "oauth",
            "provider": provider,
            "user": user_info.get("login") or user_info.get("email", "unknown"),
            "user_info": user_info,
            "expires_at": time.time() + 86400,  # 24 hours
        }

        return {
            "token": session_token,
            "user": _oauth_tokens[session_token]["user"],
            "expires_in": 86400,
        }

    @app.get("/auth/me")
    async def auth_me(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    ) -> dict:
        """Get info about the current authenticated user."""
        if credentials is None:
            raise HTTPException(status_code=401, detail="Not authenticated")
        session = _oauth_tokens.get(credentials.credentials)
        if session and session.get("expires_at", 0) > time.time():
            return {"user": session["user"], "provider": session.get("provider")}
        raise HTTPException(status_code=401, detail="Invalid or expired token")
