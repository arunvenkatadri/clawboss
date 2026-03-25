"""Tests for clawboss.audit — AuditLog, AuditEntry, sinks."""

import json
import sys
from io import StringIO

import pytest

from clawboss.audit import (
    AuditEntry,
    AuditLog,
    AuditOutcome,
    AuditPhase,
    AuditSink,
    JsonlAuditSink,
    MemoryAuditSink,
)


# ---------------------------------------------------------------------------
# AuditEntry
# ---------------------------------------------------------------------------


class TestAuditEntry:
    def test_has_all_expected_fields(self):
        entry = AuditEntry(
            timestamp="2025-01-01T00:00:00+00:00",
            request_id="req-1",
            phase="tool_call",
            outcome="allowed",
            target="web_search",
            detail="called",
            metadata={"key": "value"},
        )
        assert entry.timestamp == "2025-01-01T00:00:00+00:00"
        assert entry.request_id == "req-1"
        assert entry.phase == "tool_call"
        assert entry.outcome == "allowed"
        assert entry.target == "web_search"
        assert entry.detail == "called"
        assert entry.metadata == {"key": "value"}

    def test_to_dict_omits_none_fields(self):
        entry = AuditEntry(
            timestamp="t",
            request_id="r",
            phase="tool_call",
            outcome="allowed",
        )
        d = entry.to_dict()
        assert "target" not in d
        assert "detail" not in d
        assert "metadata" not in d
        assert d["timestamp"] == "t"

    def test_to_dict_includes_non_none_fields(self):
        entry = AuditEntry(
            timestamp="t",
            request_id="r",
            phase="tool_call",
            outcome="allowed",
            target="web_search",
        )
        d = entry.to_dict()
        assert d["target"] == "web_search"

    def test_to_json_is_valid_json(self):
        entry = AuditEntry(
            timestamp="t",
            request_id="r",
            phase="tool_call",
            outcome="allowed",
            metadata={"count": 42},
        )
        parsed = json.loads(entry.to_json())
        assert parsed["metadata"]["count"] == 42


# ---------------------------------------------------------------------------
# AuditLog.noop
# ---------------------------------------------------------------------------


class TestAuditLogNoop:
    def test_noop_does_not_crash(self):
        log = AuditLog.noop()
        log.record(AuditPhase.TOOL_CALL, AuditOutcome.ALLOWED, target="test")
        # Should not raise

    def test_noop_has_request_id(self):
        log = AuditLog.noop()
        assert log.request_id == "noop"


# ---------------------------------------------------------------------------
# AuditLog with MemoryAuditSink
# ---------------------------------------------------------------------------


class TestAuditLogWithMemorySink:
    def test_records_entries(self):
        sink = MemoryAuditSink()
        log = AuditLog("req-123", sinks=[sink])
        log.record(AuditPhase.TOOL_CALL, AuditOutcome.ALLOWED, target="web_search")
        assert len(sink) == 1
        assert sink.entries[0].request_id == "req-123"

    def test_entry_has_correct_request_id(self):
        sink = MemoryAuditSink()
        log = AuditLog("req-abc", sinks=[sink])
        log.record(AuditPhase.BUDGET_CHECK, AuditOutcome.INFO)
        assert sink.entries[0].request_id == "req-abc"

    def test_entry_has_timestamp(self):
        sink = MemoryAuditSink()
        log = AuditLog("req-1", sinks=[sink])
        log.record(AuditPhase.TOOL_CALL, AuditOutcome.ALLOWED)
        entry = sink.entries[0]
        assert entry.timestamp  # not empty
        # Should be ISO format
        assert "T" in entry.timestamp

    def test_phase_and_outcome_are_strings(self):
        sink = MemoryAuditSink()
        log = AuditLog("req-1", sinks=[sink])
        log.record(AuditPhase.CIRCUIT_BREAKER, AuditOutcome.DENIED)
        entry = sink.entries[0]
        assert entry.phase == "circuit_breaker"
        assert entry.outcome == "denied"

    def test_multiple_entries(self):
        sink = MemoryAuditSink()
        log = AuditLog("req-1", sinks=[sink])
        log.record(AuditPhase.TOOL_CALL, AuditOutcome.ALLOWED, target="tool1")
        log.record(AuditPhase.TOOL_CALL, AuditOutcome.FAILED, target="tool2")
        log.record(AuditPhase.REQUEST_END, AuditOutcome.INFO)
        assert len(sink) == 3
        assert sink.entries[0].target == "tool1"
        assert sink.entries[1].target == "tool2"

    def test_metadata_is_preserved(self):
        sink = MemoryAuditSink()
        log = AuditLog("req-1", sinks=[sink])
        log.record(
            AuditPhase.TOOL_CALL,
            AuditOutcome.ALLOWED,
            metadata={"tokens": 500},
        )
        assert sink.entries[0].metadata == {"tokens": 500}


# ---------------------------------------------------------------------------
# Multiple sinks
# ---------------------------------------------------------------------------


class TestMultipleSinks:
    def test_both_sinks_receive_entry(self):
        sink1 = MemoryAuditSink()
        sink2 = MemoryAuditSink()
        log = AuditLog("req-1", sinks=[sink1, sink2])
        log.record(AuditPhase.TOOL_CALL, AuditOutcome.ALLOWED, target="test")
        assert len(sink1) == 1
        assert len(sink2) == 1
        assert sink1.entries[0].target == sink2.entries[0].target

    def test_failing_sink_does_not_block_others(self):
        class FailingSink(AuditSink):
            def write(self, entry):
                raise RuntimeError("sink broken")

        good_sink = MemoryAuditSink()
        bad_sink = FailingSink()
        log = AuditLog("req-1", sinks=[bad_sink, good_sink])
        log.record(AuditPhase.TOOL_CALL, AuditOutcome.ALLOWED)
        # good_sink should still receive the entry
        assert len(good_sink) == 1


# ---------------------------------------------------------------------------
# JsonlAuditSink — stdout
# ---------------------------------------------------------------------------


class TestJsonlAuditSinkStdout:
    def test_writes_to_stream(self):
        stream = StringIO()
        sink = JsonlAuditSink(stream)
        entry = AuditEntry(
            timestamp="t",
            request_id="r",
            phase="tool_call",
            outcome="allowed",
        )
        sink.write(entry)
        output = stream.getvalue()
        assert output.endswith("\n")
        parsed = json.loads(output.strip())
        assert parsed["request_id"] == "r"


# ---------------------------------------------------------------------------
# JsonlAuditSink — file
# ---------------------------------------------------------------------------


class TestJsonlAuditSinkFile:
    def test_writes_to_temp_file(self, tmp_path):
        filepath = str(tmp_path / "audit.jsonl")
        sink = JsonlAuditSink.file(filepath)
        entry = AuditEntry(
            timestamp="t",
            request_id="r",
            phase="tool_call",
            outcome="allowed",
            target="web_search",
        )
        sink.write(entry)
        # Close the underlying writer so data is flushed
        sink._writer.close()

        content = (tmp_path / "audit.jsonl").read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["target"] == "web_search"

    def test_appends_multiple_entries(self, tmp_path):
        filepath = str(tmp_path / "audit.jsonl")
        sink = JsonlAuditSink.file(filepath)
        for i in range(3):
            entry = AuditEntry(
                timestamp=f"t{i}",
                request_id=f"r{i}",
                phase="tool_call",
                outcome="allowed",
            )
            sink.write(entry)
        sink._writer.close()

        content = (tmp_path / "audit.jsonl").read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3

    def test_creates_parent_directories(self, tmp_path):
        filepath = str(tmp_path / "deep" / "nested" / "audit.jsonl")
        sink = JsonlAuditSink.file(filepath)
        entry = AuditEntry(
            timestamp="t",
            request_id="r",
            phase="tool_call",
            outcome="allowed",
        )
        sink.write(entry)
        sink._writer.close()
        assert (tmp_path / "deep" / "nested" / "audit.jsonl").exists()


# ---------------------------------------------------------------------------
# MemoryAuditSink
# ---------------------------------------------------------------------------


class TestMemoryAuditSink:
    def test_entries_returns_copy(self):
        sink = MemoryAuditSink()
        entry = AuditEntry(
            timestamp="t", request_id="r", phase="p", outcome="o"
        )
        sink.write(entry)
        entries = sink.entries
        entries.clear()
        # Original should be unaffected
        assert len(sink) == 1

    def test_len_reflects_entry_count(self):
        sink = MemoryAuditSink()
        assert len(sink) == 0
        entry = AuditEntry(
            timestamp="t", request_id="r", phase="p", outcome="o"
        )
        sink.write(entry)
        sink.write(entry)
        assert len(sink) == 2
