"""Tests for clawboss.triggers — scheduling, webhooks, and DB watch."""

import asyncio
import time

import pytest

from clawboss.triggers import (
    Scheduler,
    WebhookTrigger,
    _cron_matches,
)

# ---------------------------------------------------------------------------
# Cron parser
# ---------------------------------------------------------------------------


class TestCronParser:
    def test_wildcard_matches_everything(self):
        from datetime import datetime, timezone

        dt = datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc)
        assert _cron_matches("* * * * *", dt) is True

    def test_exact_minute(self):
        from datetime import datetime, timezone

        dt = datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc)
        assert _cron_matches("30 * * * *", dt) is True
        assert _cron_matches("31 * * * *", dt) is False

    def test_exact_hour(self):
        from datetime import datetime, timezone

        dt = datetime(2026, 3, 15, 9, 0, tzinfo=timezone.utc)
        assert _cron_matches("0 9 * * *", dt) is True
        assert _cron_matches("0 10 * * *", dt) is False

    def test_step(self):
        from datetime import datetime, timezone

        dt = datetime(2026, 3, 15, 14, 0, tzinfo=timezone.utc)
        assert _cron_matches("*/15 * * * *", dt) is True  # 0 matches */15
        dt2 = datetime(2026, 3, 15, 14, 7, tzinfo=timezone.utc)
        assert _cron_matches("*/15 * * * *", dt2) is False

    def test_range(self):
        from datetime import datetime, timezone

        dt = datetime(2026, 3, 15, 10, 0, tzinfo=timezone.utc)
        assert _cron_matches("0 9-17 * * *", dt) is True
        dt2 = datetime(2026, 3, 15, 20, 0, tzinfo=timezone.utc)
        assert _cron_matches("0 9-17 * * *", dt2) is False

    def test_invalid_cron(self):
        from datetime import datetime, timezone

        dt = datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="5 fields"):
            _cron_matches("* * *", dt)


# ---------------------------------------------------------------------------
# Scheduler — interval
# ---------------------------------------------------------------------------


class TestSchedulerInterval:
    def test_add_interval(self):
        s = Scheduler()

        async def noop():
            pass

        s.add_interval("test", noop, minutes=15)
        triggers = s.list_triggers()
        assert len(triggers) == 1
        assert triggers[0]["name"] == "test"
        assert triggers[0]["type"] == "interval"
        assert triggers[0]["seconds"] == 900

    def test_add_interval_zero_raises(self):
        s = Scheduler()

        async def noop():
            pass

        with pytest.raises(ValueError, match="> 0"):
            s.add_interval("test", noop, seconds=0)

    def test_remove(self):
        s = Scheduler()

        async def noop():
            pass

        s.add_interval("test", noop, seconds=10)
        assert s.remove("test") is True
        assert s.remove("test") is False
        assert len(s.list_triggers()) == 0


# ---------------------------------------------------------------------------
# Scheduler — cron
# ---------------------------------------------------------------------------


class TestSchedulerCron:
    def test_add_cron(self):
        s = Scheduler()

        async def noop():
            pass

        s.add_cron("morning", noop, cron="0 9 * * *")
        triggers = s.list_triggers()
        assert len(triggers) == 1
        assert triggers[0]["cron"] == "0 9 * * *"

    def test_enable_disable(self):
        s = Scheduler()

        async def noop():
            pass

        s.add_cron("test", noop, cron="* * * * *")
        s.disable("test")
        assert s.list_triggers()[0]["enabled"] is False
        s.enable("test")
        assert s.list_triggers()[0]["enabled"] is True


# ---------------------------------------------------------------------------
# WebhookTrigger
# ---------------------------------------------------------------------------


class TestWebhookTrigger:
    @pytest.mark.asyncio
    async def test_fire(self):
        fired = []

        async def on_fire():
            fired.append(True)
            return "done"

        wh = WebhookTrigger("test-hook", on_fire)
        record = await wh.fire()
        assert len(fired) == 1
        assert record.trigger_type == "webhook"
        assert record.error is None

    @pytest.mark.asyncio
    async def test_fire_with_error(self):
        async def bad_fn():
            raise RuntimeError("boom")

        wh = WebhookTrigger("bad-hook", bad_fn)
        record = await wh.fire()
        assert record.error == "boom"

    @pytest.mark.asyncio
    async def test_history(self):
        call_count = 0

        async def counter():
            nonlocal call_count
            call_count += 1

        wh = WebhookTrigger("test", counter)
        await wh.fire()
        await wh.fire()
        assert len(wh.history()) == 2


# ---------------------------------------------------------------------------
# Scheduler — DB watch
# ---------------------------------------------------------------------------


class TestDbWatch:
    def test_add_db_watch(self):
        s = Scheduler()

        async def noop():
            pass

        class FakeConnector:
            pass

        s.add_db_watch(
            "watch-alerts",
            noop,
            connector=FakeConnector(),
            query="SELECT count(*) FROM alerts",
            condition=lambda r: True,
            poll_seconds=30,
        )
        triggers = s.list_triggers()
        assert len(triggers) == 1
        assert triggers[0]["type"] == "db_watch"
        assert triggers[0]["poll_seconds"] == 30


# ---------------------------------------------------------------------------
# Integration — scheduler start/stop
# ---------------------------------------------------------------------------


class TestSchedulerLifecycle:
    def test_start_stop(self):
        s = Scheduler()
        s.start()
        assert s._running is True
        s.stop()
        assert s._running is False

    def test_start_idempotent(self):
        s = Scheduler()
        s.start()
        s.start()  # should not crash
        s.stop()

    @pytest.mark.asyncio
    async def test_interval_fires(self):
        """Interval trigger fires within expected time."""
        results = []

        async def record():
            results.append(time.time())

        s = Scheduler()
        s.add_interval("fast", record, seconds=1)
        s.start()
        await asyncio.sleep(2.5)
        s.stop()
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_history_recorded(self):
        async def noop():
            pass

        s = Scheduler()
        s.add_interval("test", noop, seconds=1)
        s.start()
        await asyncio.sleep(1.5)
        s.stop()
        assert len(s.history()) >= 1
        assert s.history()[0]["trigger_name"] == "test"
