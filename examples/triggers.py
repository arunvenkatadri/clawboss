"""Scheduling and triggers example.

Demonstrates interval triggers, cron triggers, webhook triggers, and
database watch triggers using the Scheduler.

Run:
    python examples/triggers.py
"""

import asyncio
import time

from clawboss import MemoryStore, SessionManager
from clawboss.connectors import SqlConnector
from clawboss.pipeline import Pipeline
from clawboss.triggers import Scheduler, WebhookTrigger

# -- Simulated tools --


async def check_alerts() -> str:
    """Check for new alerts."""
    return "Found 3 unread alerts"


async def send_report() -> str:
    """Generate and send a daily report."""
    return "Daily report sent to #ops"


async def handle_deploy() -> str:
    """Handle a deploy webhook event."""
    return "Deploy pipeline triggered"


async def main() -> None:
    store = MemoryStore()
    mgr = SessionManager(store)

    # Build pipelines for each trigger
    alert_pipeline = Pipeline(mgr, agent_id="alerter", stateless=True).add_step(
        "check_alerts", check_alerts
    )
    report_pipeline = Pipeline(mgr, agent_id="reporter", stateless=True).add_step(
        "send_report", send_report
    )

    # --- Scheduler with interval and cron triggers ---
    scheduler = Scheduler()

    scheduler.add_interval("check-alerts", alert_pipeline.run, minutes=15)
    scheduler.add_cron("morning-report", report_pipeline.run, cron="0 9 * * *")

    # --- Database watch trigger ---
    # Uses an in-memory SQLite database for demonstration
    db = SqlConnector("sqlite://:memory:", allow_write=True)
    await db.query(sql="CREATE TABLE alerts (id INTEGER PRIMARY KEY, severity TEXT, seen BOOLEAN)")
    await db.query(sql="INSERT INTO alerts VALUES (1, 'critical', 0)")
    await db.query(sql="INSERT INTO alerts VALUES (2, 'warning', 0)")

    scheduler.add_db_watch(
        "unseen-alerts",
        alert_pipeline.run,
        connector=db,
        query="SELECT count(*) as cnt FROM alerts WHERE seen = 0",
        condition=lambda result: result["rows"][0]["cnt"] > 0,
        poll_seconds=60,
    )

    # --- Webhook trigger ---
    webhook = WebhookTrigger("deploy-notify", handle_deploy)

    # Show what's registered
    print("=== Registered triggers ===\n")
    for t in scheduler.list_triggers():
        print(f"  {t['name']} ({t['type']})", end="")
        if "seconds" in t:
            print(f" - every {t['seconds']}s")
        elif "cron" in t:
            print(f" - {t['cron']}")
        elif "query" in t:
            print(f" - polls every {t['poll_seconds']}s")
        else:
            print()

    # Start the scheduler (runs in a background thread)
    scheduler.start()
    print("\nScheduler started. Letting it tick for 2 seconds...")
    time.sleep(2)

    # Fire the webhook manually
    print("\n=== Firing webhook ===\n")
    record = await webhook.fire(payload={"sha": "abc123"})
    print(f"  Webhook '{record.trigger_name}' fired at {record.fired_at}")

    # Check history
    print("\n=== Trigger history ===\n")
    for entry in scheduler.history():
        print(f"  {entry['trigger_name']} ({entry['trigger_type']}) at {entry['fired_at']}")

    # Clean up
    scheduler.stop()
    print("\nScheduler stopped.")


if __name__ == "__main__":
    asyncio.run(main())
