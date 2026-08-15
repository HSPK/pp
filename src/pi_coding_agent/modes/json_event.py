"""Wire shape for the JSON stdout protocol.

Ported from ``packages/coding-agent/src/modes/json-event.ts``.
"""

from __future__ import annotations

from typing import Any

from ..utils.wire import to_wire


def to_json_event(event: Any) -> dict[str, Any]:
    """Strip cumulative assistant snapshots from streaming wire events.

    ``message_start`` carries the initial message, the deltas build it, and
    ``message_end`` carries the final authoritative one, so the per-delta
    ``partial`` snapshot is redundant on the wire. Cumulative ``usage`` is
    kept, because its size is constant and dropping it leaves a streaming
    consumer with no token counts until the message ends.
    """
    payload = to_wire(event)
    if not isinstance(payload, dict):
        return {"type": getattr(event, "type", "unknown")}

    if payload.get("type") != "message_update":
        return payload

    message = getattr(event, "message", None)
    if message is not None and getattr(message, "role", None) != "assistant":
        raise ValueError("message_update message is not an assistant message")

    assistant_event = payload.get("assistantMessageEvent")
    if isinstance(assistant_event, dict) and "partial" in assistant_event:
        assistant_event = {key: value for key, value in assistant_event.items() if key != "partial"}

    # Cumulative usage stays on the wire even though the cumulative *message*
    # is stripped: its size is constant, and a consumer watching a stream
    # otherwise has no token counts until `message_end`.
    return {
        "type": "message_update",
        "usage": to_wire(getattr(message, "usage", None)) if message is not None else None,
        "assistantMessageEvent": assistant_event,
    }


__all__ = ["to_json_event"]
