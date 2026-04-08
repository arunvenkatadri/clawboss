"""Triggers and scheduling — run pipelines on schedule, webhook, or data event.

Three trigger types:
- Schedule: cron expression or fixed interval
- Webhook: fires when an HTTP endpoint is called
- Database watch: polls a SQL query and fires when the result changes or
  crosses a threshold

Usage:
    from clawboss.triggers import Scheduler, WebhookTrigger, DbWatchTrigger

    scheduler = Scheduler()

    # Run every 15 minutes
    scheduler.add_interval("check-alerts", pipeline_fn, minutes=15)

    # Run on cron schedule
    scheduler.add_cron("morning-report", pipeline_fn, cron="0 9 * * *")

    # Run when webhook is called
    webhook = WebhookTrigger("deploy-notify", pipeline_fn)

    # Run when query result crosses threshold
    db_watch = DbWatchTrigger(
        "alert-monitor", pipeline_fn, sql_connector,
        query="SELECT count(*) as cnt FROM alerts WHERE seen=false",
        condition=lambda result: result["rows"][0]["cnt"] > 0,
        poll_seconds=60,
    )

    scheduler.start()  # background thread
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional


@dataclass
class TriggerRecord:
    """Record of a trigger firing."""

    trigger_name: str
    trigger_type: str  # "interval", "cron", "webhook", "db_watch"
    fired_at: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self):
        if not self.fired_at:
            self.fired_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "trigger_name": self.trigger_name,
            "trigger_type": self.trigger_type,
            "fired_at": self.fired_at,
        }
        if self.error:
            d["error"] = self.error
        if self.result:
            d["result"] = self.result
        return d


# ---------------------------------------------------------------------------
# Cron parser (minimal — handles standard 5-field cron expressions)
# ---------------------------------------------------------------------------


def _cron_matches(cron_expr: str, dt: datetime) -> bool:
    """Check if a datetime matches a cron expression (minute hour dom month dow)."""
    parts = cron_expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression: {cron_expr} (need 5 fields)")

    fields = [dt.minute, dt.hour, dt.day, dt.month, dt.weekday()]
    ranges = [
        (0, 59),  # minute
        (0, 23),  # hour
        (1, 31),  # day of month
        (1, 12),  # month
        (0, 6),  # day of week (0=Monday)
    ]

    for part, val, (lo, hi) in zip(parts, fields, ranges):
        if not _cron_field_matches(part, val, lo, hi):
            return False
    return True


def _cron_field_matches(field_expr: str, value: int, lo: int, hi: int) -> bool:
    """Check if a single cron field matches a value."""
    if field_expr == "*":
        return True

    for item in field_expr.split(","):
        # Handle */N
        if item.startswith("*/"):
            step = int(item[2:])
            if value % step == 0:
                return True
        # Handle N-M
        elif "-" in item:
            start, end = item.split("-", 1)
            if int(start) <= value <= int(end):
                return True
        # Handle plain N
        else:
            if int(item) == value:
                return True

    return False


# ---------------------------------------------------------------------------
# Trigger entries
# ---------------------------------------------------------------------------


@dataclass
class IntervalEntry:
    name: str
    fn: Callable[[], Coroutine[Any, Any, Any]]
    seconds: float
    last_run: float = 0.0
    enabled: bool = True


@dataclass
class CronEntry:
    name: str
    fn: Callable[[], Coroutine[Any, Any, Any]]
    cron: str
    last_run_minute: int = -1  # track minute to avoid double-firing
    enabled: bool = True


@dataclass
class DbWatchEntry:
    name: str
    fn: Callable[[], Coroutine[Any, Any, Any]]
    connector: Any  # SqlConnector
    query: str
    condition: Callable[[Dict[str, Any]], bool]
    poll_seconds: float = 60.0
    last_poll: float = 0.0
    last_result: Optional[Dict[str, Any]] = None
    enabled: bool = True


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class Scheduler:
    """Background scheduler for interval and cron triggers.

    Runs in a daemon thread. Non-blocking.
    """

    def __init__(self):
        self._intervals: Dict[str, IntervalEntry] = {}
        self._crons: Dict[str, CronEntry] = {}
        self._db_watches: Dict[str, DbWatchEntry] = {}
        self._history: List[TriggerRecord] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def add_interval(
        self,
        name: str,
        fn: Callable[[], Coroutine[Any, Any, Any]],
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
    ) -> None:
        """Add an interval trigger.

        Args:
            name: Unique trigger name.
            fn: Async callable to run (e.g., pipeline.run).
            seconds/minutes/hours: Interval (combined).
        """
        total = seconds + minutes * 60 + hours * 3600
        if total <= 0:
            raise ValueError("Interval must be > 0")
        with self._lock:
            self._intervals[name] = IntervalEntry(name=name, fn=fn, seconds=total)

    def add_cron(
        self,
        name: str,
        fn: Callable[[], Coroutine[Any, Any, Any]],
        cron: str,
    ) -> None:
        """Add a cron trigger.

        Args:
            name: Unique trigger name.
            fn: Async callable to run.
            cron: 5-field cron expression (minute hour dom month dow).
        """
        # Validate
        _cron_matches(cron, datetime.now(timezone.utc))
        with self._lock:
            self._crons[name] = CronEntry(name=name, fn=fn, cron=cron)

    def add_db_watch(
        self,
        name: str,
        fn: Callable[[], Coroutine[Any, Any, Any]],
        connector: Any,
        query: str,
        condition: Callable[[Dict[str, Any]], bool],
        poll_seconds: float = 60.0,
    ) -> None:
        """Add a database watch trigger.

        Polls the query at poll_seconds interval. Fires fn when
        condition(query_result) returns True.

        Args:
            name: Unique trigger name.
            fn: Async callable to run when condition fires.
            connector: SqlConnector instance.
            query: SQL query to poll.
            condition: Function that takes query result dict and returns bool.
            poll_seconds: How often to poll.
        """
        with self._lock:
            self._db_watches[name] = DbWatchEntry(
                name=name,
                fn=fn,
                connector=connector,
                query=query,
                condition=condition,
                poll_seconds=poll_seconds,
            )

    def remove(self, name: str) -> bool:
        """Remove a trigger by name."""
        with self._lock:
            found = False
            if name in self._intervals:
                del self._intervals[name]
                found = True
            if name in self._crons:
                del self._crons[name]
                found = True
            if name in self._db_watches:
                del self._db_watches[name]
                found = True
            return found

    def enable(self, name: str) -> None:
        """Enable a trigger."""
        with self._lock:
            for registry in (self._intervals, self._crons, self._db_watches):
                if name in registry:
                    registry[name].enabled = True

    def disable(self, name: str) -> None:
        """Disable a trigger without removing it."""
        with self._lock:
            for registry in (self._intervals, self._crons, self._db_watches):
                if name in registry:
                    registry[name].enabled = False

    def list_triggers(self) -> List[Dict[str, Any]]:
        """List all registered triggers."""
        with self._lock:
            result: List[Dict[str, Any]] = []
            for ie in self._intervals.values():
                result.append(
                    {
                        "name": ie.name,
                        "type": "interval",
                        "seconds": ie.seconds,
                        "enabled": ie.enabled,
                    }
                )
            for ce in self._crons.values():
                result.append(
                    {
                        "name": ce.name,
                        "type": "cron",
                        "cron": ce.cron,
                        "enabled": ce.enabled,
                    }
                )
            for de in self._db_watches.values():
                result.append(
                    {
                        "name": de.name,
                        "type": "db_watch",
                        "query": de.query,
                        "poll_seconds": de.poll_seconds,
                        "enabled": de.enabled,
                    }
                )
            return result

    def history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trigger firing history."""
        with self._lock:
            return [r.to_dict() for r in self._history[-limit:]]

    def start(self) -> None:
        """Start the scheduler in a background thread."""
        if self._running:
            return
        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="clawboss-scheduler"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        self._loop = None

    def _run_loop(self) -> None:
        """Main scheduler loop."""
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        while self._running:
            self._loop.run_until_complete(self._tick())
            time.sleep(1)  # 1-second resolution

    async def _tick(self) -> None:
        """Check all triggers and fire any that are due."""
        now = time.monotonic()
        now_dt = datetime.now(timezone.utc)

        with self._lock:
            intervals = list(self._intervals.values())
            crons = list(self._crons.values())
            db_watches = list(self._db_watches.values())

        # Interval triggers
        for ie in intervals:
            if not ie.enabled:
                continue
            if now - ie.last_run >= ie.seconds:
                ie.last_run = now
                await self._fire(ie.name, "interval", ie.fn)

        # Cron triggers
        current_minute = now_dt.hour * 60 + now_dt.minute
        for ce in crons:
            if not ce.enabled:
                continue
            if _cron_matches(ce.cron, now_dt) and ce.last_run_minute != current_minute:
                ce.last_run_minute = current_minute
                await self._fire(ce.name, "cron", ce.fn)

        # DB watch triggers
        for de in db_watches:
            if not de.enabled:
                continue
            if now - de.last_poll >= de.poll_seconds:
                de.last_poll = now
                try:
                    result = await de.connector.query(sql=de.query)
                    if de.condition(result):
                        de.last_result = result
                        await self._fire(de.name, "db_watch", de.fn)
                except Exception as e:
                    self._record(de.name, "db_watch", error=str(e))

    async def _fire(self, name: str, trigger_type: str, fn: Callable) -> None:
        """Fire a trigger and record the result."""
        try:
            result = await fn()
            result_dict = None
            if hasattr(result, "completed"):
                # PipelineResult
                result_dict = {
                    "completed": result.completed,
                    "steps": len(result.steps) if hasattr(result, "steps") else 0,
                }
            self._record(name, trigger_type, result=result_dict)
        except Exception as e:
            self._record(name, trigger_type, error=str(e))

    def _record(
        self,
        name: str,
        trigger_type: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record a trigger firing in history."""
        record = TriggerRecord(
            trigger_name=name,
            trigger_type=trigger_type,
            result=result,
            error=error,
        )
        with self._lock:
            self._history.append(record)
            if len(self._history) > 1000:
                self._history = self._history[-500:]


# ---------------------------------------------------------------------------
# WebhookTrigger — fires when an HTTP endpoint is called
# ---------------------------------------------------------------------------


class WebhookTrigger:
    """A trigger that fires when its HTTP endpoint is called.

    Register with the REST server via Scheduler or directly.

    Usage:
        trigger = WebhookTrigger("deploy", pipeline.run)
        # POST /triggers/deploy → fires pipeline.run()
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[], Coroutine[Any, Any, Any]],
    ):
        self.name = name
        self.fn = fn
        self._history: List[TriggerRecord] = []

    async def fire(self, payload: Optional[Dict[str, Any]] = None) -> TriggerRecord:
        """Fire the trigger. Called by the REST endpoint handler."""
        try:
            result = await self.fn()
            result_dict = None
            if hasattr(result, "completed"):
                result_dict = {
                    "completed": result.completed,
                    "steps": len(result.steps) if hasattr(result, "steps") else 0,
                }
            record = TriggerRecord(
                trigger_name=self.name,
                trigger_type="webhook",
                result=result_dict,
            )
        except Exception as e:
            record = TriggerRecord(
                trigger_name=self.name,
                trigger_type="webhook",
                error=str(e),
            )
        self._history.append(record)
        return record

    def history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._history[-limit:]]
