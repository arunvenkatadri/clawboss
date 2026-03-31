"""Tests for clawboss.server — REST control plane endpoints, auth, and security."""

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip("fastapi not installed", allow_module_level=True)

from clawboss.server import create_app
from clawboss.session import SessionManager
from clawboss.store import MemoryStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    store = MemoryStore()
    manager = SessionManager(store)
    app = create_app(manager)
    return TestClient(app)


@pytest.fixture
def authed_client():
    """Client with API key auth enabled."""
    store = MemoryStore()
    manager = SessionManager(store)
    app = create_app(manager, api_key="test-secret-key")
    return TestClient(app)


@pytest.fixture
def client_with_session(client):
    """Client + a pre-created session."""
    resp = client.post(
        "/sessions",
        json={
            "agent_id": "test-agent",
            "policy": {"max_iterations": 5, "tool_timeout": 10, "token_budget": 10000},
            "payload": {"task": "research"},
        },
    )
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]
    return client, session_id


# ---------------------------------------------------------------------------
# POST /sessions
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_create_session(self, client):
        resp = client.post(
            "/sessions",
            json={"agent_id": "my-agent", "policy": {"max_iterations": 3}},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["agent_id"] == "my-agent"
        assert data["status"] == "running"
        assert "session_id" in data

    def test_create_session_minimal(self, client):
        resp = client.post("/sessions", json={"agent_id": "simple"})
        assert resp.status_code == 201

    def test_create_session_with_payload(self, client):
        resp = client.post(
            "/sessions",
            json={"agent_id": "a", "payload": {"key": "value"}},
        )
        assert resp.status_code == 201
        assert resp.json()["payload"] == {"key": "value"}


# ---------------------------------------------------------------------------
# GET /sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_list_empty(self, client):
        resp = client.get("/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_multiple(self, client):
        client.post("/sessions", json={"agent_id": "a"})
        client.post("/sessions", json={"agent_id": "b"})
        resp = client.get("/sessions")
        assert resp.status_code == 200
        assert len(resp.json()) == 2


# ---------------------------------------------------------------------------
# GET /sessions/{id}
# ---------------------------------------------------------------------------


class TestGetSession:
    def test_get_existing(self, client_with_session):
        client, sid = client_with_session
        resp = client.get(f"/sessions/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid
        assert data["agent_id"] == "test-agent"
        assert "policy_dict" in data
        assert "circuit_breaker_states" in data

    def test_get_missing(self, client):
        resp = client.get("/sessions/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /sessions/{id}/pause
# ---------------------------------------------------------------------------


class TestPauseSession:
    def test_pause(self, client_with_session):
        client, sid = client_with_session
        resp = client.post(f"/sessions/{sid}/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

    def test_pause_missing(self, client):
        resp = client.post("/sessions/nonexistent/pause")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /sessions/{id}/resume
# ---------------------------------------------------------------------------


class TestResumeSession:
    def test_resume_after_pause(self, client_with_session):
        client, sid = client_with_session
        client.post(f"/sessions/{sid}/pause")
        resp = client.post(f"/sessions/{sid}/resume")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_resume_missing(self, client):
        resp = client.post("/sessions/nonexistent/resume")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /sessions/{id}/stop
# ---------------------------------------------------------------------------


class TestStopSession:
    def test_stop(self, client_with_session):
        client, sid = client_with_session
        resp = client.post(f"/sessions/{sid}/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_stop_missing(self, client):
        resp = client.post("/sessions/nonexistent/stop")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /sessions/{id}/audit
# ---------------------------------------------------------------------------


class TestAuditEndpoint:
    def test_get_audit(self, client_with_session):
        client, sid = client_with_session
        resp = client.get(f"/sessions/{sid}/audit")
        assert resp.status_code == 200
        entries = resp.json()
        assert isinstance(entries, list)
        assert len(entries) >= 1

    def test_audit_missing(self, client):
        resp = client.get("/sessions/nonexistent/audit")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Full lifecycle via REST
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_create_pause_resume_stop(self, client):
        resp = client.post(
            "/sessions",
            json={
                "agent_id": "lifecycle-agent",
                "policy": {"max_iterations": 10},
                "payload": {"step": 1},
            },
        )
        assert resp.status_code == 201
        sid = resp.json()["session_id"]

        resp = client.get("/sessions")
        ids = [s["session_id"] for s in resp.json()]
        assert sid in ids

        resp = client.post(f"/sessions/{sid}/pause")
        assert resp.json()["status"] == "paused"

        resp = client.post(f"/sessions/{sid}/resume")
        assert resp.json()["status"] == "running"

        resp = client.post(f"/sessions/{sid}/stop")
        assert resp.json()["status"] == "stopped"

        resp = client.get(f"/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"


# ---------------------------------------------------------------------------
# API key auth
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    def test_no_key_returns_401(self, authed_client):
        resp = authed_client.get("/sessions")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, authed_client):
        resp = authed_client.get("/sessions", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401

    def test_correct_key_passes(self, authed_client):
        resp = authed_client.get("/sessions", headers={"Authorization": "Bearer test-secret-key"})
        assert resp.status_code == 200

    def test_auth_on_create(self, authed_client):
        resp = authed_client.post(
            "/sessions",
            json={"agent_id": "test"},
            headers={"Authorization": "Bearer test-secret-key"},
        )
        assert resp.status_code == 201

    def test_auth_on_create_rejected(self, authed_client):
        resp = authed_client.post("/sessions", json={"agent_id": "test"})
        assert resp.status_code == 401

    def test_auth_on_pause(self, authed_client):
        headers = {"Authorization": "Bearer test-secret-key"}
        resp = authed_client.post("/sessions", json={"agent_id": "test"}, headers=headers)
        sid = resp.json()["session_id"]

        # Without auth
        resp = authed_client.post(f"/sessions/{sid}/pause")
        assert resp.status_code == 401

        # With auth
        resp = authed_client.post(f"/sessions/{sid}/pause", headers=headers)
        assert resp.status_code == 200

    def test_no_auth_configured_allows_all(self, client):
        """When no API key is set, all requests pass without auth."""
        resp = client.get("/sessions")
        assert resp.status_code == 200
        resp = client.post("/sessions", json={"agent_id": "test"})
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Security: CORS defaults
# ---------------------------------------------------------------------------


class TestCORSDefaults:
    def test_cors_rejects_foreign_origin(self, client):
        resp = client.options(
            "/sessions",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        allow_origin = resp.headers.get("access-control-allow-origin", "")
        assert "evil.com" not in allow_origin

    def test_cors_allows_localhost(self, client):
        resp = client.options(
            "/sessions",
            headers={
                "Origin": "http://localhost",
                "Access-Control-Request-Method": "GET",
            },
        )
        allow_origin = resp.headers.get("access-control-allow-origin", "")
        assert "localhost" in allow_origin or allow_origin == "*"


# ---------------------------------------------------------------------------
# Security: Session ID via REST
# ---------------------------------------------------------------------------


class TestSessionIdSecurity:
    def test_session_id_high_entropy(self, client):
        resp = client.post("/sessions", json={"agent_id": "test"})
        sid = resp.json()["session_id"]
        assert len(sid) == 32
        assert all(c in "0123456789abcdef" for c in sid)
