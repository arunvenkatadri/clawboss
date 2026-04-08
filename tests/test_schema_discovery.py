"""Tests for database schema discovery."""

import os
import sqlite3
import tempfile

import pytest

from clawboss.connectors import SqlConnector
from clawboss.pipeline_poml import PipelineBuilder
from clawboss.session import SessionManager
from clawboss.store import MemoryStore


def _make_test_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL)")
    conn.execute("INSERT INTO users VALUES (1, 'Alice', 'alice@test.com')")
    conn.execute("INSERT INTO orders VALUES (1, 1, 99.99)")
    conn.commit()
    conn.close()
    return path


class TestSqlSchemaDiscovery:
    @pytest.mark.asyncio
    async def test_discover_tables(self):
        path = _make_test_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}")
            schema = await sql.discover_schema()
            assert "tables" in schema
            table_names = [t["name"] for t in schema["tables"]]
            assert "users" in table_names
            assert "orders" in table_names
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_discover_columns(self):
        path = _make_test_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}")
            schema = await sql.discover_schema()
            users = next(t for t in schema["tables"] if t["name"] == "users")
            col_names = [c["name"] for c in users["columns"]]
            assert "id" in col_names
            assert "name" in col_names
            assert "email" in col_names
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_discover_column_types(self):
        path = _make_test_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}")
            schema = await sql.discover_schema()
            users = next(t for t in schema["tables"] if t["name"] == "users")
            id_col = next(c for c in users["columns"] if c["name"] == "id")
            assert id_col["type"] == "INTEGER"
            assert id_col["pk"] is True
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_discover_row_count(self):
        path = _make_test_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}")
            schema = await sql.discover_schema()
            users = next(t for t in schema["tables"] if t["name"] == "users")
            assert users["row_count"] == 1
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_schema_to_text(self):
        path = _make_test_db()
        try:
            sql = SqlConnector(f"sqlite:///{path}")
            schema = await sql.discover_schema()
            text = sql.schema_to_text(schema)
            assert "Table: users" in text
            assert "id INTEGER" in text
            assert "PK" in text
            assert "Table: orders" in text
        finally:
            os.unlink(path)


class TestSchemaInPipelineBuilder:
    @pytest.mark.asyncio
    async def test_schema_included_in_prompt(self):
        """The LLM should receive the database schema in its prompt."""
        path = _make_test_db()
        received_prompts = []

        async def capture_llm(prompt: str) -> str:
            received_prompts.append(prompt)
            return "<pipeline><step tool='search'>test</step></pipeline>"

        async def search(input: str = "") -> str:
            return "ok"

        try:
            sql = SqlConnector(f"sqlite:///{path}")
            schema = await sql.discover_schema()

            store = MemoryStore()
            mgr = SessionManager(store)
            builder = PipelineBuilder(
                capture_llm,
                {"search": search, "sql.query": sql.query},
                mgr,
                db_schema=schema,
            )
            await builder.create_poml("Show me all users")

            assert len(received_prompts) == 1
            prompt = received_prompts[0]
            assert "Table: users" in prompt
            assert "Table: orders" in prompt
            assert "id INTEGER" in prompt
            assert "email" in prompt
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_no_schema_no_crash(self):
        """PipelineBuilder works fine without a schema."""
        received_prompts = []

        async def capture_llm(prompt: str) -> str:
            received_prompts.append(prompt)
            return "<pipeline><step tool='search'>test</step></pipeline>"

        async def search(input: str = "") -> str:
            return "ok"

        store = MemoryStore()
        mgr = SessionManager(store)
        builder = PipelineBuilder(capture_llm, {"search": search}, mgr)
        await builder.create_poml("Do a search")

        prompt = received_prompts[0]
        assert "Database schema" not in prompt

    @pytest.mark.asyncio
    async def test_schema_helps_llm_write_sql(self):
        """Simulate an LLM that uses the schema to write correct SQL."""
        path = _make_test_db()

        async def smart_llm(prompt: str) -> str:
            # LLM "reads" the schema and writes SQL using actual column names
            if "Table: users" in prompt and "email" in prompt:
                return """
                <pipeline>
                  <step tool="sql.query">SELECT name, email FROM users</step>
                </pipeline>
                """
            return "<pipeline><step tool='search'>fallback</step></pipeline>"

        async def search(input: str = "") -> str:
            return "fallback"

        try:
            sql = SqlConnector(f"sqlite:///{path}")
            schema = await sql.discover_schema()

            store = MemoryStore()
            mgr = SessionManager(store)
            builder = PipelineBuilder(
                smart_llm,
                {"search": search, "sql.query": sql.query},
                mgr,
                db_schema=schema,
            )
            pipeline = await builder.create(
                "Show me all user names and emails",
                policy_dict={"max_iterations": 5},
            )
            result = await pipeline.run()
            assert result.completed is True
            assert result.final_output["rows"][0]["name"] == "Alice"
            assert result.final_output["rows"][0]["email"] == "alice@test.com"
        finally:
            os.unlink(path)
