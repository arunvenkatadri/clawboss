"""Data-driven agent flows with database connectors and conditional pipelines.

Demonstrates all three data flows:
1. Query then act — fetch data, analyze, report
2. Act then query to decide — do work, check status, decide next step
3. Query and threshold — check a metric, escalate if above threshold

Run:
    python examples/data_flows.py
"""

import asyncio
import os
import sqlite3
import tempfile

from clawboss import MemoryStore, SessionManager
from clawboss.connectors import SqlConnector
from clawboss.pipeline import Pipeline

# -- Setup: create a test database --


def create_test_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE orders (product TEXT, amount REAL, region TEXT)")
    conn.execute("INSERT INTO orders VALUES ('Widget A', 99.99, 'US')")
    conn.execute("INSERT INTO orders VALUES ('Widget B', 149.99, 'EU')")
    conn.execute("INSERT INTO orders VALUES ('Gadget X', 299.99, 'US')")
    conn.execute("INSERT INTO orders VALUES ('Gadget Y', 49.99, 'EU')")
    conn.execute("CREATE TABLE alerts (severity TEXT, message TEXT)")
    for i in range(12):
        conn.execute(f"INSERT INTO alerts VALUES ('critical', 'Alert #{i}')")
    for i in range(5):
        conn.execute(f"INSERT INTO alerts VALUES ('warning', 'Warning #{i}')")
    conn.commit()
    conn.close()
    return path


# -- Simulated action tools --


async def analyze_revenue(input=None):
    """Analyze query results."""
    total = sum(r["amount"] for r in input["rows"])
    by_region = {}
    for r in input["rows"]:
        by_region[r["region"]] = by_region.get(r["region"], 0) + r["amount"]
    return {"total": total, "by_region": by_region}


async def write_report(input=None, title="Report"):
    """Generate a report from analysis."""
    return f"# {title}\nTotal: ${input['total']:.2f}\nBy region: {input['by_region']}"


async def escalate(input=None):
    """Escalate — too many critical alerts."""
    return f"ESCALATED: {input['rows'][0]['cnt']} critical alerts detected"


async def log_ok(input=None):
    """All clear — below threshold."""
    return f"OK: only {input['rows'][0]['cnt']} critical alerts"


async def process_task(input=None):
    """Simulate doing some work."""
    return {"status": "completed", "items_processed": 42}


async def check_and_report(input=None):
    """Report on completed work."""
    return f"Processed {input['rows'][0]['total']} orders after task completion"


async def main():
    db_path = create_test_db()
    sql = SqlConnector(f"sqlite:///{db_path}")
    store = MemoryStore()
    mgr = SessionManager(store)
    policy = {"max_iterations": 10, "tool_timeout": 30}

    # -- Flow 1: Query then act --
    print("=" * 60)
    print("FLOW 1: Query then act")
    print("=" * 60)
    result = await (
        Pipeline(mgr, "revenue-analyst", policy)
        .add_step("get_orders", sql.query, sql="SELECT * FROM orders")
        .add_step("analyze", analyze_revenue)
        .add_step("report", write_report, title="Revenue Report")
        .run()
    )
    print(f"Completed: {result.completed}")
    print(result.final_output)
    print()

    # -- Flow 2: Act then query to decide --
    print("=" * 60)
    print("FLOW 2: Act then query to decide next step")
    print("=" * 60)
    result = await (
        Pipeline(mgr, "task-runner", policy)
        .add_step("process", process_task, chain_input=False)
        .add_step(
            "check_status",
            sql.query,
            chain_input=False,
            sql="SELECT count(*) as total FROM orders",
        )
        .add_step("report", check_and_report)
        .run()
    )
    print(f"Completed: {result.completed}")
    print(result.final_output)
    print()

    # -- Flow 3: Query and threshold --
    print("=" * 60)
    print("FLOW 3: Query and threshold check")
    print("=" * 60)
    result = await (
        Pipeline(mgr, "alert-monitor", policy)
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
    print(f"Completed: {result.completed}")
    print(result.final_output)
    print()

    # -- Metrics --
    print("=" * 60)
    print("METRICS")
    print("=" * 60)
    for tool_name, metrics in mgr.observer.all_tools_summary().items():
        print(f"  {tool_name}: {metrics['calls']} calls, {metrics['successes']} ok")

    # Cleanup
    sql.close()
    os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(main())
