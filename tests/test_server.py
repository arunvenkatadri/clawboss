"""Tests for clawboss.server — REST control plane endpoints and security."""

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
        # Should have at least the request_start entry
        assert len(entries) >= 1

    def test_audit_missing(self, client):
        resp = client.get("/sessions/nonexistent/audit")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Full lifecycle via REST
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    def test_create_pause_resume_stop(self, client):
        # Create
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

        # Verify listed
        resp = client.get("/sessions")
        ids = [s["session_id"] for s in resp.json()]
        assert sid in ids

        # Pause
        resp = client.post(f"/sessions/{sid}/pause")
        assert resp.json()["status"] == "paused"

        # Resume
        resp = client.post(f"/sessions/{sid}/resume")
        assert resp.json()["status"] == "running"

        # Stop
        resp = client.post(f"/sessions/{sid}/stop")
        assert resp.json()["status"] == "stopped"

        # Still accessible via GET
        resp = client.get(f"/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"


# ---------------------------------------------------------------------------
# Security: CORS defaults
# ---------------------------------------------------------------------------


class TestCORSDefaults:
    def test_cors_rejects_foreign_origin(self, client):
        """Non-localhost origins should be rejected by default CORS policy."""
        resp = client.options(
            "/sessions",
            headers={
                "Origin": "https://evil.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # The response should NOT include the evil origin in allow-origin
        allow_origin = resp.headers.get("access-control-allow-origin", "")
        assert "evil.com" not in allow_origin

    def test_cors_allows_localhost(self, client):
        """Localhost origins should be allowed."""
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
        """Session IDs returned by the API should be 32 hex chars (128 bits)."""
        resp = client.post("/sessions", json={"agent_id": "test"})
        sid = resp.json()["session_id"]
        assert len(sid) == 32
        assert all(c in "0123456789abcdef" for c in sid)
