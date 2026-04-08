"""Database connectors — supervised query tools for SQL and NoSQL.

Give agents the ability to query databases as regular supervised tool calls.
Results flow through the full Clawboss pipeline: PII redaction, audit logging,
tool scoping, budgets, and observability.

Connectors return async callables that work with ``supervisor.call()`` and
``Pipeline.add_step()``.

SQL (PostgreSQL, MySQL, SQLite) — uses DB-API 2.0 (stdlib sqlite3 or
any PEP 249 driver). No required dependencies beyond stdlib.

MongoDB — optional, requires ``pymongo``.

Usage:
    from clawboss.connectors import SqlConnector

    sql = SqlConnector("sqlite:///data.db")
    result = await supervisor.call("query", sql.query, sql="SELECT * FROM users LIMIT 10")

    # In a pipeline
    pipeline.add_step("get_metrics", sql.query, sql="SELECT avg(price) FROM orders")
    pipeline.add_step("analyze", my_analyze_fn)
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List, Optional


class SqlConnector:
    """Supervised SQL query tool.

    Connects to SQLite (stdlib), PostgreSQL, or MySQL via DB-API 2.0.
    Returns an async callable for use with Supervisor.call().

    Args:
        connection_string: Database connection string.
            - SQLite: "sqlite:///path/to/db.sqlite" or "sqlite://:memory:"
            - PostgreSQL: "postgresql://user:pass@host/db" (requires psycopg2)
            - MySQL: "mysql://user:pass@host/db" (requires mysql-connector-python)
        allow_write: If False (default), only SELECT statements are allowed.
                    INSERT, UPDATE, DELETE, DROP, ALTER, etc. are blocked.
        max_rows: Maximum rows to return per query (default: 1000).
    """

    def __init__(
        self,
        connection_string: str,
        allow_write: bool = False,
        max_rows: int = 1000,
    ):
        self._conn_str = connection_string
        self._allow_write = allow_write
        self._max_rows = max_rows
        self._conn: Any = None

    def _connect(self) -> Any:
        """Lazily connect to the database."""
        if self._conn is not None:
            return self._conn

        if self._conn_str.startswith("sqlite:///"):
            path = self._conn_str[len("sqlite:///") :]
            self._conn = sqlite3.connect(path)
            self._conn.row_factory = sqlite3.Row
        elif self._conn_str == "sqlite://:memory:":
            self._conn = sqlite3.connect(":memory:")
            self._conn.row_factory = sqlite3.Row
        elif self._conn_str.startswith("postgresql://"):
            try:
                import psycopg2  # type: ignore[import-not-found,import-untyped]
                import psycopg2.extras  # type: ignore[import-not-found,import-untyped]

                self._conn = psycopg2.connect(self._conn_str)
            except ImportError as e:
                raise ImportError(
                    "psycopg2 required for PostgreSQL: pip install psycopg2-binary"
                ) from e
        elif self._conn_str.startswith("mysql://"):
            try:
                import mysql.connector  # type: ignore[import-not-found]

                # Parse mysql://user:pass@host/db
                parsed = re.match(r"mysql://([^:]+):([^@]+)@([^/]+)/(.+)", self._conn_str)
                if not parsed:
                    raise ValueError(f"Invalid MySQL connection string: {self._conn_str}")
                self._conn = mysql.connector.connect(
                    user=parsed.group(1),
                    password=parsed.group(2),
                    host=parsed.group(3),
                    database=parsed.group(4),
                )
            except ImportError as e:
                raise ImportError(
                    "mysql-connector-python required for MySQL: pip install mysql-connector-python"
                ) from e
        else:
            raise ValueError(
                f"Unsupported connection string: {self._conn_str}. "
                "Use sqlite:///path, postgresql://..., or mysql://..."
            )
        return self._conn

    def _check_write(self, sql: str) -> None:
        """Block write operations if allow_write is False."""
        if self._allow_write:
            return
        normalized = sql.strip().upper()
        write_ops = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "REPLACE")
        for op in write_ops:
            if normalized.startswith(op):
                raise PermissionError(
                    f"Write operation '{op}' blocked. Set allow_write=True to enable."
                )

    async def query(
        self,
        sql: str = "",
        params: Optional[Dict[str, Any]] = None,
        input: str = "",
    ) -> Dict[str, Any]:
        """Execute a SQL query and return results.

        Args:
            sql: The SQL query to execute. If empty, uses ``input`` (for pipeline chaining).
            params: Query parameters (for parameterized queries).
            input: Alternative to ``sql`` — used when chained from a previous pipeline step.

        Returns:
            Dict with "rows" (list of dicts), "row_count", and "columns".
        """
        query_str = sql or input
        if not query_str:
            return {"rows": [], "row_count": 0, "columns": []}

        self._check_write(query_str)
        conn = self._connect()
        cursor = conn.cursor()

        try:
            if params:
                cursor.execute(query_str, params)
            else:
                cursor.execute(query_str)

            # For SELECT queries, fetch results
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows_raw = cursor.fetchmany(self._max_rows)
                rows = [dict(zip(columns, row)) for row in rows_raw]
                return {
                    "rows": rows,
                    "row_count": len(rows),
                    "columns": columns,
                }
            else:
                # Write operation (INSERT/UPDATE/DELETE)
                conn.commit()
                return {
                    "rows": [],
                    "row_count": cursor.rowcount,
                    "columns": [],
                    "affected_rows": cursor.rowcount,
                }
        finally:
            cursor.close()

    async def execute(
        self,
        sql: str = "",
        params: Optional[Dict[str, Any]] = None,
        input: str = "",
    ) -> Dict[str, Any]:
        """Execute a write SQL statement. Alias for query() with write intent.

        Raises PermissionError if allow_write is False.
        """
        return await self.query(sql=sql, params=params, input=input)

    async def discover_schema(self) -> Dict[str, Any]:
        """Discover the database schema — tables, columns, types.

        Returns a dict describing every table and its columns, suitable
        for passing to an LLM so it can write correct SQL.

        Returns:
            {"tables": [{"name": "...", "columns": [{"name": "...", "type": "..."}]}]}
        """
        conn = self._connect()
        tables: List[Dict[str, Any]] = []

        if self._conn_str.startswith("sqlite"):
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            table_names = [row[0] for row in cursor.fetchall()]
            for tname in table_names:
                if tname.startswith("sqlite_"):
                    continue
                cursor.execute(f"PRAGMA table_info({tname})")  # noqa: S608
                cols = [
                    {"name": row[1], "type": row[2], "nullable": not row[3], "pk": bool(row[5])}
                    for row in cursor.fetchall()
                ]
                # Sample a few rows for context
                cursor.execute(f"SELECT COUNT(*) FROM {tname}")  # noqa: S608
                row_count = cursor.fetchone()[0]
                tables.append(
                    {
                        "name": tname,
                        "columns": cols,
                        "row_count": row_count,
                    }
                )
            cursor.close()

        elif self._conn_str.startswith("postgresql"):
            cursor = conn.cursor()
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            )
            table_names = [row[0] for row in cursor.fetchall()]
            for tname in table_names:
                cursor.execute(
                    "SELECT column_name, data_type, is_nullable, "
                    "column_default FROM information_schema.columns "
                    "WHERE table_name = %s ORDER BY ordinal_position",
                    (tname,),
                )
                cols = [
                    {
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2] == "YES",
                        "default": row[3],
                    }
                    for row in cursor.fetchall()
                ]
                tables.append({"name": tname, "columns": cols})
            cursor.close()

        elif self._conn_str.startswith("mysql"):
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            table_names = [row[0] for row in cursor.fetchall()]
            for tname in table_names:
                cursor.execute(f"DESCRIBE {tname}")  # noqa: S608
                cols = [
                    {
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2] == "YES",
                        "pk": row[3] == "PRI",
                    }
                    for row in cursor.fetchall()
                ]
                tables.append({"name": tname, "columns": cols})
            cursor.close()

        return {"tables": tables}

    def schema_to_text(self, schema: Dict[str, Any]) -> str:
        """Format a schema dict as human-readable text for LLM prompts.

        Example output:
            Table: orders (1500 rows)
              - id INTEGER (PK)
              - product TEXT
              - amount REAL
        """
        lines = []
        for table in schema.get("tables", []):
            header = f"Table: {table['name']}"
            if "row_count" in table:
                header += f" ({table['row_count']} rows)"
            lines.append(header)
            for col in table.get("columns", []):
                parts = [f"  - {col['name']} {col.get('type', '')}"]
                if col.get("pk"):
                    parts.append("(PK)")
                if col.get("nullable") is False:
                    parts.append("NOT NULL")
                lines.append(" ".join(parts))
            lines.append("")
        return "\n".join(lines)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class MongoConnector:
    """Supervised MongoDB query tool.

    Requires ``pymongo``: pip install pymongo

    Args:
        uri: MongoDB connection URI (e.g., "mongodb://localhost:27017").
        database: Database name.
        allow_write: If False (default), only find operations are allowed.
        max_docs: Maximum documents to return per query (default: 1000).
    """

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        database: str = "default",
        allow_write: bool = False,
        max_docs: int = 1000,
    ):
        self._uri = uri
        self._database = database
        self._allow_write = allow_write
        self._max_docs = max_docs
        self._client: Any = None
        self._db: Any = None

    def _connect(self) -> Any:
        """Lazily connect to MongoDB."""
        if self._db is not None:
            return self._db
        try:
            import pymongo  # type: ignore[import-not-found]

            self._client = pymongo.MongoClient(self._uri)
            self._db = self._client[self._database]
        except ImportError as e:
            raise ImportError("pymongo required: pip install pymongo") from e
        return self._db

    async def find(
        self,
        collection: str = "",
        filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
        sort: Optional[List] = None,
        limit: Optional[int] = None,
        input: str = "",
    ) -> Dict[str, Any]:
        """Query a MongoDB collection.

        Args:
            collection: Collection name.
            filter: MongoDB query filter.
            projection: Fields to include/exclude.
            sort: Sort specification.
            limit: Max documents (capped at max_docs).
            input: Unused — for pipeline chaining compatibility.

        Returns:
            Dict with "documents" (list of dicts) and "count".
        """
        db = self._connect()
        coll = db[collection]
        cursor = coll.find(
            filter=filter or {},
            projection=projection,
        )
        if sort:
            cursor = cursor.sort(sort)
        cursor = cursor.limit(min(limit or self._max_docs, self._max_docs))

        docs = []
        for doc in cursor:
            # Convert ObjectId to string for JSON serialization
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
            docs.append(doc)

        return {"documents": docs, "count": len(docs)}

    async def insert(
        self,
        collection: str = "",
        documents: Optional[List[Dict[str, Any]]] = None,
        input: str = "",
    ) -> Dict[str, Any]:
        """Insert documents into a MongoDB collection.

        Raises PermissionError if allow_write is False.
        """
        if not self._allow_write:
            raise PermissionError("Write operation blocked. Set allow_write=True to enable.")
        db = self._connect()
        coll = db[collection]
        docs = documents or []
        if not docs:
            return {"inserted_count": 0}
        result = coll.insert_many(docs)
        return {"inserted_count": len(result.inserted_ids)}

    async def discover_schema(self) -> Dict[str, Any]:
        """Discover MongoDB collections and sample field structures.

        Samples a few documents from each collection to infer the schema.
        """
        db = self._connect()
        collections = []
        for name in db.list_collection_names():
            sample = list(db[name].find().limit(5))
            fields: Dict[str, str] = {}
            for doc in sample:
                for k, v in doc.items():
                    if k not in fields:
                        fields[k] = type(v).__name__
            collections.append(
                {
                    "name": name,
                    "document_count": db[name].estimated_document_count(),
                    "fields": [{"name": k, "type": v} for k, v in fields.items()],
                }
            )
        return {"collections": collections}

    def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
