"""Session replay — reconstruct an agent's timeline from its audit log.

Take any stopped or failed session and walk through it step-by-step:
every tool call, every decision, every guardrail check, every state
change. Inspect the agent's state at any point in its history.

Built on top of the existing audit log and observer metrics — no new
storage required.

Usage:
    replay = SessionReplay(manager, session_id)
    for frame in replay.frames():
        print(frame.timestamp, frame.phase, frame.summary)

    # Jump to a specific point
    state = replay.state_at(frame_index=42)
    # {"iterations": 5, "tokens_used": 1200, "last_tool": "web_search", ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .session import SessionManager


@dataclass
class ReplayFrame:
    """A single frame in a session replay — one audit event + state at that point."""

    index: int
    timestamp: str
    phase: str
    outcome: str
    target: Optional[str] = None
    detail: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # Cumulative state at this point in the timeline
    iterations_so_far: int = 0
    tokens_so_far: int = 0
    tools_called: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "phase": self.phase,
            "outcome": self.outcome,
            "target": self.target,
            "detail": self.detail,
            "metadata": self.metadata,
            "iterations_so_far": self.iterations_so_far,
            "tokens_so_far": self.tokens_so_far,
            "tools_called": list(self.tools_called),
        }

    @property
    def summary(self) -> str:
        """One-line human-readable summary for log-style output."""
        ts = self.timestamp[11:19] if self.timestamp else ""
        tgt = f" {self.target}" if self.target else ""
        return f"[{ts}] #{self.index} {self.phase}/{self.outcome}{tgt}"


@dataclass
class ReplaySummary:
    """High-level overview of a replayed session."""

    session_id: str
    agent_id: str
    status: str
    total_frames: int
    total_tool_calls: int
    unique_tools: List[str]
    final_iterations: int
    final_tokens: int
    duration_ms: int
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "status": self.status,
            "total_frames": self.total_frames,
            "total_tool_calls": self.total_tool_calls,
            "unique_tools": self.unique_tools,
            "final_iterations": self.final_iterations,
            "final_tokens": self.final_tokens,
            "duration_ms": self.duration_ms,
            "failure_reason": self.failure_reason,
        }


class SessionReplay:
    """Time-travel through a session's history.

    Built on the audit log — replay is lossless because every action
    was already recorded for the audit trail.
    """

    def __init__(self, manager: SessionManager, session_id: str):
        self._manager = manager
        self._session_id = session_id
        self._checkpoint = manager.status(session_id)
        self._entries = manager.get_audit_entries(session_id)
        self._frames: Optional[List[ReplayFrame]] = None

    @property
    def exists(self) -> bool:
        return self._checkpoint is not None

    def frames(self) -> List[ReplayFrame]:
        """Build the replay frames from the audit log."""
        if self._frames is not None:
            return self._frames

        frames: List[ReplayFrame] = []
        iterations = 0
        tokens = 0
        tools_called: List[str] = []

        for i, entry in enumerate(self._entries):
            phase = entry.get("phase", "")
            outcome = entry.get("outcome", "")
            target = entry.get("target")
            detail = entry.get("detail")
            metadata = entry.get("metadata") or {}

            # Track cumulative state
            if phase == "iteration_check" and outcome == "allowed":
                iterations += 1
            if phase == "tool_call" and outcome == "info" and target:
                if target not in tools_called:
                    tools_called.append(target)
            if metadata.get("tokens_used") is not None:
                tokens = metadata["tokens_used"]

            frames.append(
                ReplayFrame(
                    index=i,
                    timestamp=entry.get("timestamp", ""),
                    phase=phase,
                    outcome=outcome,
                    target=target,
                    detail=detail,
                    metadata=metadata,
                    iterations_so_far=iterations,
                    tokens_so_far=tokens,
                    tools_called=list(tools_called),
                )
            )

        self._frames = frames
        return frames

    def state_at(self, frame_index: int) -> Dict[str, Any]:
        """Return the cumulative agent state at the given frame index."""
        frames = self.frames()
        if frame_index >= len(frames):
            frame_index = len(frames) - 1
        if frame_index < 0:
            return {"iterations": 0, "tokens": 0, "tools_called": []}
        f = frames[frame_index]
        return {
            "iterations": f.iterations_so_far,
            "tokens": f.tokens_so_far,
            "tools_called": list(f.tools_called),
            "phase": f.phase,
            "outcome": f.outcome,
            "target": f.target,
        }

    def filter(
        self,
        phase: Optional[str] = None,
        outcome: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> List[ReplayFrame]:
        """Filter frames by phase, outcome, or tool name."""
        result = self.frames()
        if phase is not None:
            result = [f for f in result if f.phase == phase]
        if outcome is not None:
            result = [f for f in result if f.outcome == outcome]
        if tool is not None:
            result = [f for f in result if f.target == tool]
        return result

    def summary(self) -> ReplaySummary:
        """High-level summary of the session."""
        cp = self._checkpoint
        frames = self.frames()
        tool_calls = [f for f in frames if f.phase == "tool_call" and f.outcome == "info"]
        unique_tools = sorted({f.target for f in tool_calls if f.target})

        # Duration = last frame timestamp - first frame timestamp
        duration_ms = 0
        if frames:
            try:
                from datetime import datetime

                first = datetime.fromisoformat(frames[0].timestamp.replace("Z", "+00:00"))
                last = datetime.fromisoformat(frames[-1].timestamp.replace("Z", "+00:00"))
                duration_ms = int((last - first).total_seconds() * 1000)
            except (ValueError, AttributeError):
                pass

        return ReplaySummary(
            session_id=self._session_id,
            agent_id=cp.agent_id if cp else "",
            status=cp.status.value if cp else "unknown",
            total_frames=len(frames),
            total_tool_calls=len(tool_calls),
            unique_tools=unique_tools,
            final_iterations=cp.iterations if cp else 0,
            final_tokens=cp.tokens_used if cp else 0,
            duration_ms=duration_ms,
            failure_reason=cp.failure_reason if cp and cp.failure_reason else None,
        )

    def to_timeline(self) -> List[Dict[str, Any]]:
        """Export the full replay as a list of frame dicts."""
        return [f.to_dict() for f in self.frames()]
