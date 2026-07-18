"""Context compression via Headroom — reduce token costs on tool outputs.

Integrates Netflix's Headroom compression library with AgentHandler's
supervision pipeline. Tool outputs are compressed before reaching the
LLM context, reducing token costs by 60-95% for JSON and 15-20% for code.

Usage:
    from agenthandler.compression import CompressedSupervisor

    sv = CompressedSupervisor(policy, compression="auto")
    result = await sv.call("search", search_fn, query="test")
    # result.output is compressed — fewer tokens when fed to LLM

Or wrap an existing supervisor:
    from agenthandler.compression import compress_output

    raw = await sv.call("search", search_fn, query="test")
    compressed = compress_output(raw.output)

Requires: pip install headroom-ai
"""

from __future__ import annotations

import json
from typing import Any, Callable, Coroutine, Dict, Optional

from .policy import Policy
from .supervisor import SupervisedResult, Supervisor

_headroom_available = False
try:
    from headroom import compress as _headroom_compress

    _headroom_available = True
except ImportError:
    pass


def compress_output(
    output: Any,
    model: str = "claude-sonnet-4-6",
    min_tokens: int = 100,
) -> Any:
    """Compress a tool output using Headroom.

    Only compresses string and dict outputs that exceed min_tokens
    (estimated). Small outputs pass through unchanged.

    Args:
        output: The tool output to compress.
        model: Target model (affects compression strategy).
        min_tokens: Skip compression for outputs shorter than this.

    Returns:
        Compressed output (string), or original if too small or not compressible.
    """
    if not _headroom_available:
        return output

    if output is None:
        return output

    if isinstance(output, dict):
        text = json.dumps(output, default=str)
    elif isinstance(output, str):
        text = output
    else:
        return output

    if len(text) < min_tokens * 4:
        return output

    try:
        messages = [{"role": "user", "content": text}]
        compressed = _headroom_compress(messages, model=model)
        if compressed and len(compressed) > 0:
            content = compressed[0].get("content", text)
            if isinstance(output, dict):
                try:
                    return json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    return content
            return content
    except Exception:
        pass

    return output


def compress_messages(
    messages: list[Dict[str, Any]],
    model: str = "claude-sonnet-4-6",
) -> list[Dict[str, Any]]:
    """Compress a list of chat messages using Headroom.

    Args:
        messages: OpenAI-format messages list.
        model: Target model.

    Returns:
        Compressed messages list.
    """
    if not _headroom_available:
        return messages

    try:
        result: list[Dict[str, Any]] = _headroom_compress(messages, model=model)
        return result
    except Exception:
        return messages


class CompressedSupervisor(Supervisor):
    """Supervisor that compresses tool outputs via Headroom.

    Every tool call result is passed through Headroom's compression
    pipeline before being returned. This reduces token costs when
    the output is fed into an LLM context.

    Args:
        policy: Supervision policy.
        compression: Compression mode — "auto" (compress large outputs),
                    "always" (compress everything), or "off".
        target_model: LLM model name (affects compression strategy).
        min_tokens: Minimum estimated tokens before compression kicks in.
        **kwargs: Passed to Supervisor.__init__.
    """

    def __init__(
        self,
        policy: Optional[Policy] = None,
        compression: str = "auto",
        target_model: str = "claude-sonnet-4-6",
        min_tokens: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy or Policy(), **kwargs)
        self._compression = compression
        self._target_model = target_model
        self._min_tokens = min_tokens

        if compression != "off" and not _headroom_available:
            import warnings

            warnings.warn(
                "Headroom not installed — compression disabled. "
                "Install with: pip install headroom-ai",
                stacklevel=2,
            )
            self._compression = "off"

    async def call(
        self,
        tool_name: str,
        fn: Callable[..., Coroutine[Any, Any, Any]],
        **kwargs: Any,
    ) -> SupervisedResult:
        """Call a tool with supervision and compress the output."""
        result = await super().call(tool_name, fn, **kwargs)

        if result.succeeded and self._compression != "off":
            if self._compression == "always" or self._should_compress(result.output):
                result = SupervisedResult(
                    output=compress_output(
                        result.output,
                        model=self._target_model,
                        min_tokens=self._min_tokens,
                    ),
                    error=result.error,
                    duration_ms=result.duration_ms,
                    budget=result.budget,
                )

        return result

    def _should_compress(self, output: Any) -> bool:
        """Decide whether to compress based on output size."""
        if output is None:
            return False
        if isinstance(output, dict):
            return len(json.dumps(output, default=str)) > self._min_tokens * 4
        if isinstance(output, str):
            return len(output) > self._min_tokens * 4
        return False
