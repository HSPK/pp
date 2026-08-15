"""Client-side transcript reconstruction from snapshots + progress events.

Python port of `packages/coding-agent/src/client/transcript.ts`. Wire values
are plain dicts (as validated by `pi_protocol.schemas`), not dataclasses, so
this module works with `dict[str, Any]` throughout rather than typed
`TranscriptItem`/`SessionSnapshot` classes.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any

JsonValue = str | int | float | bool | None | list[Any] | dict[str, Any]


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str, int)):
        return True
    if isinstance(value, float):
        # `Number.isFinite` in the TypeScript. `json.loads` accepts the `NaN`,
        # `Infinity` and `-Infinity` literals that `JSON.parse` rejects, so a
        # streamed buffer containing one must stay a raw string prefix.
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _parse_partial_tool_input(value: str) -> JsonValue:
    """Tool arguments are incomplete while streaming; keep the raw prefix until valid JSON forms."""
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value
    if _is_json_value(parsed):
        return parsed
    return value


@dataclass
class TranscriptState:
    snapshot: dict[str, Any]
    progress_items: dict[str, dict[str, Any]] = field(default_factory=dict)
    progress_order: list[str] = field(default_factory=list)
    tool_call_buffers: dict[str, str] = field(default_factory=dict)


def create_transcript_state(snapshot: dict[str, Any]) -> TranscriptState:
    return TranscriptState(snapshot=copy.deepcopy(snapshot))


def apply_transcript_snapshot(state: TranscriptState, snapshot: dict[str, Any]) -> TranscriptState:
    if state.snapshot["id"] == snapshot["id"] and snapshot["revision"] < state.snapshot["revision"]:
        return state
    return create_transcript_state(snapshot)


def _set_progress_item(state: TranscriptState, item: dict[str, Any]) -> TranscriptState:
    progress_items = dict(state.progress_items)
    progress_order = list(state.progress_order)
    if item["id"] not in progress_items:
        progress_order.append(item["id"])
    progress_items[item["id"]] = copy.deepcopy(item)
    return TranscriptState(
        snapshot=state.snapshot,
        progress_items=progress_items,
        progress_order=progress_order,
        tool_call_buffers=state.tool_call_buffers,
    )


def apply_transcript_progress(state: TranscriptState, progress: dict[str, Any]) -> TranscriptState:
    progress_type = progress["type"]
    if progress_type in ("item_started", "item_updated"):
        return _set_progress_item(state, progress["item"])

    if progress_type == "item_finished":
        item = progress["item"]
        prefix = f"{item['id']}:"
        tool_call_buffers = {key: value for key, value in state.tool_call_buffers.items() if not key.startswith(prefix)}
        return _set_progress_item(
            TranscriptState(
                snapshot=state.snapshot,
                progress_items=state.progress_items,
                progress_order=state.progress_order,
                tool_call_buffers=tool_call_buffers,
            ),
            item,
        )

    # assistant_delta
    message_id = progress["messageId"]
    item = state.progress_items.get(message_id) or next(
        (entry for entry in state.snapshot["transcript"] if entry["id"] == message_id), None
    )
    if item is None or item["role"] != "assistant":
        return state

    tool_call_buffers = state.tool_call_buffers
    content_index = progress["contentIndex"]
    kind = progress["kind"]
    delta = progress["delta"]
    new_content: list[dict[str, Any]] = []
    for index, part in enumerate(item["content"]):
        if index != content_index:
            new_content.append(copy.deepcopy(part))
            continue
        if kind == "text" and part["type"] == "text":
            new_content.append({**part, "text": part["text"] + delta})
        elif kind == "thinking" and part["type"] == "thinking":
            new_content.append({**part, "thinking": part["thinking"] + delta})
        elif kind == "toolCall" and part["type"] == "toolCall":
            key = f"{message_id}:{content_index}"
            existing = state.tool_call_buffers.get(key) or (part["input"] if isinstance(part["input"], str) else "")
            buffer = existing + delta
            tool_call_buffers = {**state.tool_call_buffers, key: buffer}
            new_content.append({**part, "input": _parse_partial_tool_input(buffer)})
        else:
            new_content.append(copy.deepcopy(part))

    return _set_progress_item(
        TranscriptState(
            snapshot=state.snapshot,
            progress_items=state.progress_items,
            progress_order=state.progress_order,
            tool_call_buffers=tool_call_buffers,
        ),
        {**item, "content": new_content},
    )


def select_transcript(state: TranscriptState) -> list[dict[str, Any]]:
    transcript = [state.progress_items.get(item["id"], item) for item in state.snapshot["transcript"]]
    ids = {item["id"] for item in transcript}
    for item_id in state.progress_order:
        if item_id in ids:
            continue
        item = state.progress_items.get(item_id)
        if item is not None:
            transcript.append(item)
            ids.add(item_id)
    for item in state.snapshot["queuedSteer"]:
        if item["id"] in ids:
            continue
        transcript.append(item)
        ids.add(item["id"])
    return transcript


__all__ = [
    "TranscriptState",
    "apply_transcript_progress",
    "apply_transcript_snapshot",
    "create_transcript_state",
    "select_transcript",
]
