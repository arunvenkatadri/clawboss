"""Tests for clawboss.connectors — SQL and NoSQL database tools."""

import os
import tempfile
from typing import Any

import pytest

from clawboss.connectors import SqlConnector
from clawboss.pipeline import Pipeline
from clawboss.session import SessionManager
from clawboss.store import MemoryStore

POLICY = {"max_iterations": 10, "tool_timeout": 10}


# ---------------------------------------------------------------------------
# SqlConnector — SQLite
# ---------------------------------------------------------------------------


class TestSqlConnectorSqlite:
    def _make_db(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        import sqlite3

        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT, email TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@test.com')")
        conn.execute("INSERT INTO users VALUES (2, 'Bob', 'bob@test.com')")
        conn.execute("INSERT INTO users VALUES (3, 'Charlie', 'charlie@test.com')")
        conn.execute("CREATE TABLE metrics (name TEXT, value REAL)")
        conn.execute("INSERT INTO metrics VALUES ('cpu', 85.5)")
        conn.execute("INSERT INTO metrics VALUES ('memory', 42.1)")
        conn.commit()
        conn.close()
        return path

    @pytest.mark.asyncio
    async def test_select_query(self):
        path = self._make_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}")
            result = await sql.query(sql="SELECT * FROM users")
            assert result["row_count"] == 3
            assert result["columns"] == ["id", "name", "email"]
            assert result["rows"][0]["name"] == "Alice"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_parameterized_query(self):
        path = self._make_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}")
            result = await sql.query(
                sql="SELECT * FROM users WHERE name = :name",
                params={"name": "Bob"},
            )
            assert result["row_count"] == 1
            assert result["rows"][0]["name"] == "Bob"
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_write_blocked_by_default(self):
        path = self._make_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}")
            with pytest.raises(PermissionError, match="blocked"):
                await sql.query(sql="INSERT INTO users VALUES (4, 'Dave', 'dave@test.com')")
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_write_allowed_when_enabled(self):
        path = self._make_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}", allow_write=True)
            result = await sql.execute(sql="INSERT INTO users VALUES (4, 'Dave', 'dave@test.com')")
            assert result["affected_rows"] == 1
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_drop_blocked(self):
        path = self._make_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}")
            with pytest.raises(PermissionError, match="DROP"):
                await sql.query(sql="DROP TABLE users")
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_empty_query(self):
        path = self._make_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}")
            result = await sql.query()
            assert result["rows"] == []
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_max_rows(self):
        path = self._make_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}", max_rows=2)
            result = await sql.query(sql="SELECT * FROM users")
            assert result["row_count"] == 2  # capped at 2
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_input_key_for_chaining(self):
        """The `input` param works as an alternative to `sql` for pipeline chaining."""
        path = self._make_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}")
            result = await sql.query(input="SELECT * FROM metrics")
            assert result["row_count"] == 2
        finally:
            os.unlink(path)

    def test_close(self):
        path = self._make_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}")
            sql._connect()
            sql.close()
            assert sql._conn is None
        finally:
            os.unlink(path)

    def test_invalid_connection_string(self):
        with pytest.raises(ValueError, match="Unsupported"):
            SqlConnector("oracle://localhost/db")._connect()


# ---------------------------------------------------------------------------
# SqlConnector in Pipeline
# ---------------------------------------------------------------------------


class TestSqlInPipeline:
    @pytest.mark.asyncio
    async def test_query_then_act(self):
        """Flow 1: query then perform actions based on the query."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        import sqlite3

        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE orders (product TEXT, amount REAL)")
        conn.execute("INSERT INTO orders VALUES ('widget', 99.99)")
        conn.execute("INSERT INTO orders VALUES ('gadget', 149.99)")
        conn.commit()
        conn.close()

        try:
            sql = SqlConnector(f"sqlite:///{path}")

            async def analyze(input: Any = None) -> str:
                total = sum(r["amount"] for r in input["rows"])
                return f"Total revenue: ${total:.2f}"

            store = MemoryStore()
            mgr = SessionManager(store)
            result = await (
                Pipeline(mgr, "analyst", POLICY)
                .add_step("query_orders", sql.query, sql="SELECT * FROM orders")
                .add_step("analyze", analyze)
                .run()
            )
            assert result.completed is True
            assert "249.98" in result.final_output
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_query_with_threshold(self):
        """Flow 3: query, check threshold, act or skip."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        import sqlite3

        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE alerts (severity TEXT)")
        for _ in range(15):
            conn.execute("INSERT INTO alerts VALUES ('critical')")
        conn.commit()
        conn.close()

        try:
            sql = SqlConnector(f"sqlite:///{path}")

            async def escalate(input: Any = None) -> str:
                return "ESCALATED: too many critical alerts"

            async def log_ok(input: Any = None) -> str:
                return "All clear"

            store = MemoryStore()
            mgr = SessionManager(store)
            result = await (
                Pipeline(mgr, "monitor", POLICY)
                .add_step(
                    "check_alerts",
                    sql.query,
                    sql="SELECT count(*) as cnt FROM alerts WHERE severity='critical'",
                )
                .add_threshold(
                    key="rows.0.cnt",
                    threshold=10,
                    above_step=("escalate", escalate),
                    below_step=("log_ok", log_ok),
                )
                .run()
            )
            assert result.completed is True
            assert "ESCALATED" in result.final_output
        finally:
            os.unlink(path)
