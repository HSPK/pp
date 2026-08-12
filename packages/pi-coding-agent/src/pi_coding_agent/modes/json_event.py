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
    ``partial`` snapshot is redundant on the wire.
    """
    payload = to_wire(event)
    if not isinstance(payload, dict):
        return {"type": getattr(event, "type", "unknown")}

    if payload.get("type") != "message_update":
        return payload

    assistant_event = payload.get("assistantMessageEvent")
    if isinstance(assistant_event, dict) and "partial" in assistant_event:
        assistant_event = {key: value for key, value in assistant_event.items() if key != "partial"}
    return {"type": "message_update", "assistantMessageEvent": assistant_event}


__all__ = ["to_json_event"]
