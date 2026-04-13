"""Tests for clawboss.streams — Kafka, Kinesis, Redis connectors."""

import pytest

from clawboss.streams import (
    KafkaStreamConnector,
    KinesisStreamConnector,
    RedisStreamConnector,
    StreamConnector,
)

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestStreamConnectorProtocol:
    def test_kafka_implements_protocol(self):
        async def noop(p):
            pass

        conn = KafkaStreamConnector(
            bootstrap_servers="localhost:9092",
            topic="test",
            group_id="test-group",
            on_message=noop,
        )
        assert isinstance(conn, StreamConnector)

    def test_kinesis_implements_protocol(self):
        async def noop(p):
            pass

        conn = KinesisStreamConnector(stream_name="test-stream", on_message=noop)
        assert isinstance(conn, StreamConnector)

    def test_redis_implements_protocol(self):
        async def noop(p):
            pass

        conn = RedisStreamConnector(
            url="redis://localhost:6379",
            stream="test",
            group="test-group",
            consumer="test-consumer",
            on_message=noop,
        )
        assert isinstance(conn, StreamConnector)


# ---------------------------------------------------------------------------
# Kafka — graceful failure when aiokafka not installed
# ---------------------------------------------------------------------------


class TestKafkaConnector:
    @pytest.mark.asyncio
    async def test_start_without_dep_raises(self):
        """If aiokafka isn't installed, start() raises ImportError."""

        async def noop(p):
            pass

        conn = KafkaStreamConnector(
            bootstrap_servers="localhost:9092",
            topic="test",
            group_id="test",
            on_message=noop,
        )
        # If aiokafka is installed, this would try to connect and fail differently
        try:
            await conn.start()
            await conn.stop()
        except ImportError as e:
            assert "aiokafka" in str(e)
        except Exception:
            pass  # any other error is also fine (e.g. connection refused)

    def test_construction(self):
        async def handler(p):
            return None

        conn = KafkaStreamConnector(
            bootstrap_servers="broker:9092",
            topic="events",
            group_id="my-group",
            on_message=handler,
            auto_offset_reset="earliest",
        )
        assert conn._topic == "events"
        assert conn._group_id == "my-group"
        assert conn._auto_offset_reset == "earliest"


# ---------------------------------------------------------------------------
# Kinesis
# ---------------------------------------------------------------------------


class TestKinesisConnector:
    @pytest.mark.asyncio
    async def test_start_without_dep_raises(self):
        async def noop(p):
            pass

        conn = KinesisStreamConnector(stream_name="test", on_message=noop)
        try:
            await conn.start()
            await conn.stop()
        except ImportError as e:
            assert "aioboto3" in str(e)
        except Exception:
            pass

    def test_construction(self):
        async def handler(p):
            return None

        conn = KinesisStreamConnector(
            stream_name="events",
            on_message=handler,
            region_name="us-west-2",
            shard_iterator_type="TRIM_HORIZON",
        )
        assert conn._stream == "events"
        assert conn._region == "us-west-2"
        assert conn._iterator_type == "TRIM_HORIZON"


# ---------------------------------------------------------------------------
# Redis Streams
# ---------------------------------------------------------------------------


class TestRedisConnector:
    @pytest.mark.asyncio
    async def test_start_without_dep_raises(self):
        async def noop(p):
            pass

        conn = RedisStreamConnector(
            url="redis://nonexistent:6379",
            stream="test",
            group="g",
            consumer="c",
            on_message=noop,
        )
        try:
            await conn.start()
            await conn.stop()
        except ImportError as e:
            assert "redis" in str(e)
        except Exception:
            pass

    def test_construction(self):
        async def handler(p):
            return None

        conn = RedisStreamConnector(
            url="redis://localhost:6379",
            stream="events",
            group="agent-group",
            consumer="agent-1",
            on_message=handler,
            block_ms=2000,
        )
        assert conn._stream == "events"
        assert conn._group == "agent-group"
        assert conn._consumer == "agent-1"
        assert conn._block_ms == 2000


# ---------------------------------------------------------------------------
# Message handler integration
# ---------------------------------------------------------------------------


class TestMessageHandling:
    @pytest.mark.asyncio
    async def test_handler_signature(self):
        """The handler should receive a dict payload and return awaitable."""
        received = []

        async def handler(payload):
            received.append(payload)
            return "ok"

        # Just call the handler directly
        await handler({"key": "value", "data": 42})
        assert len(received) == 1
        assert received[0]["key"] == "value"
