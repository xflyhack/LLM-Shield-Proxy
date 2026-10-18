"""Native Anthropic passthrough: rehydrate_sse_stream(anthropic_native=True) must
preserve Anthropic event shapes (text_delta / tool_use / input_json_delta) while
still running PII rehydration on the values.

Reproduces the production bug: Claude Code -> LiteLLM gateway got tool_use blocks
rewritten into OpenAI chat.completion.chunk tool_calls, producing
"Part of the response never arrived" client errors.
"""

import asyncio
import json
from typing import Any, List, Optional

import pytest

from llm_shield_proxy.streaming.streaming import rehydrate_sse_stream


class FakeVault:
    """Minimal vault: no tokens, so rehydration is identity on values."""

    token_to_original: dict = {}

    def rehydrate(self, text: str, retention_length: int = 0, **kw: Any) -> str:
        return text


def _sse(events: List[tuple]) -> bytes:
    """Build a raw SSE byte stream from (event_name, payload_dict) tuples."""
    out = []
    for name, payload in events:
        out.append(f"event: {name}\n".encode())
        out.append(f"data: {json.dumps(payload)}\n\n".encode())
    return b"".join(out)


async def _collect(stream: Any) -> str:
    chunks = []
    async for chunk in stream:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
    return b"".join(chunks).decode("utf-8")


def _data_lines(text: str) -> List[dict]:
    objs = []
    for line in text.split("\n"):
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                objs.append(json.loads(payload))
    return objs


ANTHROPIC_TOOL_STREAM = [
    ("message_start", {"type": "message_start", "message": {"id": "msg_1", "model": "claude-sonnet-4-6", "content": []}}),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me check."}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"Paris\"}"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 1}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}}),
    ("message_stop", {"type": "message_stop"}),
]


@pytest.mark.parametrize("anthropic_native", [True, False])
def test_tool_stream_shape_by_mode(anthropic_native: bool) -> None:
    """native=True keeps every Anthropic event; native=False rewrites to OpenAI."""

    async def run() -> str:
        async def gen() -> Any:
            yield _sse(ANTHROPIC_TOOL_STREAM)

        return await _collect(
            rehydrate_sse_stream(gen(), FakeVault(), anthropic_native=anthropic_native)
        )

    out = asyncio.run(run())
    objs = _data_lines(out)

    if anthropic_native:
        # Native mode: event types preserved verbatim.
        types = [o.get("type") for o in objs]
        assert types.count("content_block_start") == 2
        assert types.count("content_block_stop") == 2
        assert types.count("message_stop") == 1
        # The tool_use opener keeps its native shape (id + name + type).
        openers = [o for o in objs if o.get("type") == "content_block_start"]
        tool_openers = [o for o in openers if o.get("content_block", {}).get("type") == "tool_use"]
        assert len(tool_openers) == 1
        assert tool_openers[0]["content_block"]["id"] == "toolu_1"
        assert tool_openers[0]["content_block"]["name"] == "get_weather"
        # Deltas stay native: text_delta and input_json_delta, never OpenAI.
        deltas = [o for o in objs if o.get("type") == "content_block_delta"]
        assert all(d["delta"].get("type") in ("text_delta", "input_json_delta") for d in deltas)
        text_deltas = [d for d in deltas if d["delta"]["type"] == "text_delta"]
        assert "".join(d["delta"]["text"] for d in text_deltas) == "Let me check."
        json_deltas = [d for d in deltas if d["delta"]["type"] == "input_json_delta"]
        assert "".join(d["delta"]["partial_json"] for d in json_deltas) == '{"city":"Paris"}'
        # No OpenAI shapes anywhere.
        assert "chat.completion.chunk" not in out
        assert "tool_calls" not in out
    else:
        # Legacy mode (translated client): OpenAI chunks, tool_calls present.
        assert "chat.completion.chunk" in out
        assert "tool_calls" in out


def test_native_watermark_is_anthropic_shaped() -> None:
    """Watermark text rides a native text_delta in native mode."""

    async def run() -> str:
        async def gen() -> Any:
            yield _sse(ANTHROPIC_TOOL_STREAM)

        return await _collect(
            rehydrate_sse_stream(
                gen(), FakeVault(), watermark_text="WM-TOKEN", anthropic_native=True
            )
        )

    out = asyncio.run(run())
    assert "WM-TOKEN" in out
    wm_events = [o for o in _data_lines(out) if o.get("delta", {}).get("text") == "WM-TOKEN"]
    assert len(wm_events) == 1
    assert wm_events[0]["type"] == "content_block_delta"
    assert wm_events[0]["delta"]["type"] == "text_delta"
    assert "chat.completion.chunk" not in json.dumps(wm_events[0])


def test_native_thinking_stream_passthrough() -> None:
    """thinking_delta / signature_delta events pass through untouched in native mode."""

    events = [
        ("message_start", {"type": "message_start", "message": {"id": "msg_2", "content": []}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "pondering"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig123"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Answer."}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_stop", {"type": "message_stop"}),
    ]

    async def run() -> str:
        async def gen() -> Any:
            yield _sse(events)

        return await _collect(
            rehydrate_sse_stream(gen(), FakeVault(), anthropic_native=True)
        )

    out = asyncio.run(run())
    assert '"type": "thinking_delta"' in out or '"type":"thinking_delta"' in out
    assert "pondering" in out
    assert "sig123" in out
    assert "chat.completion.chunk" not in out
