"""Streaming input connectors — Kafka, Kinesis, Redis Streams.

Subscribe an agent to a real-time event stream. Each message fires the
registered pipeline with the message payload as input.

All three connectors implement the same StreamConnector protocol:
- start() — begin consuming
- stop() — stop consuming
- Messages fire the registered callback asynchronously

Optional dependencies:
- Kafka: ``pip install aiokafka``
- Kinesis: ``pip install aioboto3``
- Redis Streams: ``pip install redis``

Usage:
    from clawboss.streams import KafkaStreamConnector

    async def on_message(payload):
        # Run pipeline with the payload as input
        return await pipeline.run()

    connector = KafkaStreamConnector(
        bootstrap_servers="localhost:9092",
        topic="agent-events",
        group_id="my-agent",
        on_message=on_message,
    )
    await connector.start()
    # ... later ...
    await connector.stop()
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, runtime_checkable

MessageHandler = Callable[[Dict[str, Any]], Awaitable[Any]]


@runtime_checkable
class StreamConnector(Protocol):
    """Protocol for streaming input connectors."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...


# ---------------------------------------------------------------------------
# Kafka
# ---------------------------------------------------------------------------


class KafkaStreamConnector:
    """Consume messages from a Kafka topic and fire an agent pipeline.

    Uses aiokafka (optional dependency). Offsets are committed after
    successful handler execution — at-least-once delivery.

    Args:
        bootstrap_servers: Comma-separated Kafka broker list.
        topic: Topic to consume from.
        group_id: Consumer group ID for offset tracking.
        on_message: Async callback fired with each message payload.
        auto_offset_reset: "earliest" or "latest".
        value_deserializer: How to decode message bytes (default: JSON).
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        on_message: MessageHandler,
        auto_offset_reset: str = "latest",
        value_deserializer: Optional[Callable[[bytes], Any]] = None,
    ):
        self._bootstrap = bootstrap_servers
        self._topic = topic
        self._group_id = group_id
        self._on_message = on_message
        self._auto_offset_reset = auto_offset_reset
        self._deserializer = value_deserializer or (lambda b: json.loads(b.decode()))
        self._consumer: Any = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start consuming messages."""
        try:
            from aiokafka import AIOKafkaConsumer  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("aiokafka required: pip install aiokafka") from e

        self._consumer = AIOKafkaConsumer(
            self._topic,
            bootstrap_servers=self._bootstrap,
            group_id=self._group_id,
            auto_offset_reset=self._auto_offset_reset,
            value_deserializer=self._deserializer,
            enable_auto_commit=False,
        )
        await self._consumer.start()
        self._running = True
        self._task = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self) -> None:
        """Main consume loop — fires on_message for each record."""
        try:
            async for msg in self._consumer:
                if not self._running:
                    break
                payload = {
                    "key": msg.key.decode() if msg.key else None,
                    "value": msg.value,
                    "topic": msg.topic,
                    "partition": msg.partition,
                    "offset": msg.offset,
                    "timestamp": msg.timestamp,
                }
                try:
                    await self._on_message(payload)
                    await self._consumer.commit()
                except Exception:
                    # Don't commit — message will be redelivered
                    pass
        except Exception:
            pass

    async def stop(self) -> None:
        """Stop consuming."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None


# ---------------------------------------------------------------------------
# AWS Kinesis
# ---------------------------------------------------------------------------


class KinesisStreamConnector:
    """Consume records from an AWS Kinesis Data Stream.

    Uses aioboto3 (optional dependency). Polls shards in a loop and
    fires on_message for each record.

    Args:
        stream_name: Kinesis stream name.
        on_message: Async callback fired with each record payload.
        region_name: AWS region.
        shard_iterator_type: "LATEST" or "TRIM_HORIZON".
        poll_interval: Seconds between shard polls.
    """

    def __init__(
        self,
        stream_name: str,
        on_message: MessageHandler,
        region_name: str = "us-east-1",
        shard_iterator_type: str = "LATEST",
        poll_interval: float = 1.0,
    ):
        self._stream = stream_name
        self._on_message = on_message
        self._region = region_name
        self._iterator_type = shard_iterator_type
        self._poll_interval = poll_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._session: Any = None

    async def start(self) -> None:
        """Start consuming from all shards."""
        try:
            import aioboto3  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("aioboto3 required: pip install aioboto3") from e

        self._session = aioboto3.Session()
        self._running = True
        self._task = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self) -> None:
        """Poll Kinesis shards for records."""
        async with self._session.client("kinesis", region_name=self._region) as client:
            # Get shards
            desc = await client.describe_stream(StreamName=self._stream)
            shards = desc["StreamDescription"]["Shards"]

            # Get shard iterators
            iterators = {}
            for shard in shards:
                resp = await client.get_shard_iterator(
                    StreamName=self._stream,
                    ShardId=shard["ShardId"],
                    ShardIteratorType=self._iterator_type,
                )
                iterators[shard["ShardId"]] = resp["ShardIterator"]

            # Poll loop
            while self._running:
                for shard_id, iterator in list(iterators.items()):
                    if iterator is None:
                        continue
                    try:
                        resp = await client.get_records(ShardIterator=iterator, Limit=100)
                        for record in resp.get("Records", []):
                            payload = {
                                "data": json.loads(record["Data"])
                                if isinstance(record["Data"], bytes)
                                else record["Data"],
                                "partition_key": record.get("PartitionKey"),
                                "sequence_number": record.get("SequenceNumber"),
                                "shard_id": shard_id,
                            }
                            try:
                                await self._on_message(payload)
                            except Exception:
                                pass
                        iterators[shard_id] = resp.get("NextShardIterator")
                    except Exception:
                        pass
                await asyncio.sleep(self._poll_interval)

    async def stop(self) -> None:
        """Stop consuming."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass


# ---------------------------------------------------------------------------
# Redis Streams
# ---------------------------------------------------------------------------


class RedisStreamConnector:
    """Consume messages from a Redis Stream.

    Uses redis-py with consumer groups for at-least-once delivery.

    Args:
        url: Redis connection URL (e.g., "redis://localhost:6379").
        stream: Stream key name.
        group: Consumer group name.
        consumer: Consumer name within the group.
        on_message: Async callback fired with each message.
        block_ms: How long to block waiting for messages.
    """

    def __init__(
        self,
        url: str,
        stream: str,
        group: str,
        consumer: str,
        on_message: MessageHandler,
        block_ms: int = 5000,
    ):
        self._url = url
        self._stream = stream
        self._group = group
        self._consumer = consumer
        self._on_message = on_message
        self._block_ms = block_ms
        self._redis: Any = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start consuming from the Redis stream."""
        try:
            import redis.asyncio as redis  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("redis required: pip install redis") from e

        self._redis = redis.from_url(self._url, decode_responses=True)

        # Create consumer group (idempotent)
        try:
            await self._redis.xgroup_create(self._stream, self._group, id="$", mkstream=True)
        except Exception:
            pass  # group already exists

        self._running = True
        self._task = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self) -> None:
        """Main consume loop — uses XREADGROUP for consumer group semantics."""
        while self._running:
            try:
                messages = await self._redis.xreadgroup(
                    groupname=self._group,
                    consumername=self._consumer,
                    streams={self._stream: ">"},
                    count=10,
                    block=self._block_ms,
                )
                for stream_name, entries in messages:
                    for msg_id, fields in entries:
                        payload = {
                            "id": msg_id,
                            "stream": stream_name,
                            "fields": fields,
                        }
                        try:
                            await self._on_message(payload)
                            await self._redis.xack(self._stream, self._group, msg_id)
                        except Exception:
                            pass
            except Exception:
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop consuming."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._redis:
            await self._redis.close()
            self._redis = None
