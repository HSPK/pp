"""Protocol conversion helpers.

Python port of `packages/server/src/protocol.ts`, including the transcript
conversion functions (`toProtocolUserMessage` / `toProtocolAssistantMessage` /
`toProtocolToolResultMessage`) that turn live `pi_ai` messages into wire
transcript items. Those functions are only exercised by a real
`PiSessionRuntime` adapter backed by `pi_agent`'s harness; that adapter now
lives in `pi_coding_agent.core.agent_session_runtime`, which is why this
module (rather than a new module in `pi_coding_agent`) is the right home for
them: `pi_server` already depends on `pi_ai` for `to_protocol_usage` /
`to_protocol_model_metadata`, so importing `pi_ai.types.UserMessage` /
`AssistantMessage` / `ToolResultMessage` / `ToolCall` here adds no new
dependency direction, keeps every wire-shaping function in one file (matching
the TS source), and lets `pi_coding_agent` depend on `pi_server` (for the
`PiSessionRuntime` protocol) without `pi_server` depending back on
`pi_coding_agent`.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Any

from pi_ai.models import get_supported_thinking_levels
from pi_ai.types import AssistantMessage, Model, ToolCall, ToolResultMessage, Usage, UserMessage

JsonValue = str | int | float | bool | None | list[Any] | dict[str, Any]


class _Undefined:
    """Stand-in for TypeScript's `undefined`.

    `sanitizeProtocolDetails` returns `undefined` for values it drops entirely
    (functions, symbols, `undefined` itself) and `null` for JSON null, and its
    callers treat the two differently: a dropped value disappears from its
    object, a null is kept as a null. Python's `None` is JSON null, so a
    separate marker is needed or `{"a": None}` would silently lose its key.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNDEFINED"

    def __bool__(self) -> bool:
        return False


UNDEFINED = _Undefined()


def _non_negative_integer(value: float | None) -> int | None:
    if value is None or not math.isfinite(value):
        return None
    return max(0, math.floor(value))


def _non_negative_number(value: float) -> float:
    return max(0.0, value) if math.isfinite(value) else 0.0


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) == 0:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _timestamp(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError("Protocol timestamps must be non-negative integers")
    return value


def to_protocol_json_value(value: Any, seen: set[int] | None = None) -> JsonValue:
    """Validate and copy a value from an execution boundary into the protocol's JSON-compatible subset."""
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise TypeError("Protocol JSON numbers must be finite")
        return value
    if not isinstance(value, (list, dict)):
        raise TypeError(f"Unsupported protocol JSON value: {type(value).__name__}")
    if id(value) in seen:
        raise TypeError("Protocol JSON values must not contain circular references")
    seen = seen | {id(value)}
    if isinstance(value, list):
        return [to_protocol_json_value(entry, seen) for entry in value]
    return {key: to_protocol_json_value(entry, seen) for key, entry in value.items()}


def sanitize_protocol_details(value: Any, seen: set[int] | None = None) -> JsonValue | _Undefined:
    """Lossily sanitize diagnostic tool details that must not affect execution semantics.

    Returns `UNDEFINED` for values that have no JSON representation at all, so
    callers can drop them; `None` means JSON null and is preserved.
    """
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else str(value)
    if callable(value):
        return UNDEFINED
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if not isinstance(value, (list, dict)):
        return str(value)
    if id(value) in seen:
        return "[Circular]"
    seen = seen | {id(value)}
    if isinstance(value, list):
        return [
            sanitized if not isinstance(sanitized := sanitize_protocol_details(entry, seen), _Undefined) else None
            for entry in value
        ]
    result: dict[str, JsonValue] = {}
    for key, entry in value.items():
        normalized = sanitize_protocol_details(entry, seen)
        if not isinstance(normalized, _Undefined):
            result[key] = normalized
    return result


def to_protocol_usage(usage: Usage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    reasoning = _non_negative_integer(usage.reasoning)
    result: dict[str, Any] = {
        "input": _non_negative_integer(usage.input) or 0,
        "output": _non_negative_integer(usage.output) or 0,
        "cacheRead": _non_negative_integer(usage.cache_read) or 0,
        "cacheWrite": _non_negative_integer(usage.cache_write) or 0,
    }
    if reasoning is not None:
        result["reasoning"] = reasoning
    result["totalTokens"] = _non_negative_integer(usage.total_tokens) or 0
    result["cost"] = {
        "input": _non_negative_number(usage.cost.input),
        "output": _non_negative_number(usage.cost.output),
        "cacheRead": _non_negative_number(usage.cost.cache_read),
        "cacheWrite": _non_negative_number(usage.cost.cache_write),
        "total": _non_negative_number(usage.cost.total),
    }
    return result


def to_protocol_model_metadata(model: Model, authenticated: bool) -> dict[str, Any]:
    return {
        "provider": _identifier(model.provider, "Model provider"),
        "id": _identifier(model.id, "Model id"),
        "name": _identifier(model.name, "Model name"),
        "api": _identifier(model.api, "Model API"),
        "reasoning": model.reasoning,
        "input": list(model.input),
        "contextWindow": max(1, math.floor(model.context_window)),
        "maxTokens": max(1, math.floor(model.max_tokens)),
        "cost": {
            "input": _non_negative_number(model.cost.input),
            "output": _non_negative_number(model.cost.output),
            "cacheRead": _non_negative_number(model.cost.cache_read),
            "cacheWrite": _non_negative_number(model.cost.cache_write),
        },
        "supportedThinkingLevels": get_supported_thinking_levels(model),
        "authenticated": authenticated,
    }


@dataclass
class UserTranscriptOptions:
    id: str


@dataclass
class AssistantTranscriptOptions:
    id: str


@dataclass
class ToolTranscriptOptions:
    id: str
    call: ToolCall


def _to_protocol_user_content(content: str | list[Any]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    result: list[dict[str, Any]] = []
    for part in content:
        if part.type == "text":
            result.append({"type": "text", "text": part.text})
        elif part.type == "image":
            result.append({"type": "image", "data": part.data, "mimeType": part.mime_type})
        else:
            raise TypeError(f"Unsupported user content part: {part.type}")
    return result


def to_protocol_user_message(message: UserMessage, options: UserTranscriptOptions) -> dict[str, Any]:
    return {
        "id": _identifier(options.id, "Transcript item id"),
        "role": "user",
        "content": _to_protocol_user_content(message.content),
        "timestamp": _timestamp(message.timestamp),
    }


def _to_protocol_assistant_content(message: AssistantMessage) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for part in message.content:
        if part.type == "text":
            result.append({"type": "text", "text": part.text})
        elif part.type == "thinking":
            item: dict[str, Any] = {"type": "thinking", "thinking": part.thinking}
            if part.redacted is not None:
                item["redacted"] = part.redacted
            result.append(item)
        elif part.type == "toolCall":
            result.append(
                {
                    "type": "toolCall",
                    "toolCallId": _identifier(part.id, "Tool call id"),
                    "toolName": _identifier(part.name, "Tool call name"),
                    "input": to_protocol_json_value(part.arguments),
                }
            )
        else:
            raise TypeError(f"Unsupported assistant content part: {part.type}")
    return result


def to_protocol_assistant_message(message: AssistantMessage, options: AssistantTranscriptOptions) -> dict[str, Any]:
    usage = to_protocol_usage(message.usage)
    common: dict[str, Any] = {
        "id": _identifier(options.id, "Transcript item id"),
        "role": "assistant",
        "content": _to_protocol_assistant_content(message),
        "model": {
            "provider": _identifier(message.provider, "Assistant provider"),
            "id": _identifier(message.model, "Assistant model"),
        },
    }
    if message.response_model is not None:
        common["responseModel"] = _identifier(message.response_model, "Assistant response model")
    if usage is not None:
        common["usage"] = usage
    common["timestamp"] = _timestamp(message.timestamp)

    stop_reason = message.stop_reason
    if stop_reason == "pending":
        return {**common, "status": "streaming"}
    if stop_reason in ("stop", "length", "toolUse"):
        return {**common, "status": "complete", "stopReason": stop_reason}
    if stop_reason == "deferred":
        raise TypeError("Deferred assistant messages are not supported by protocol v1")
    if stop_reason == "error":
        if message.error_message is not None and len(message.error_message) == 0:
            raise TypeError("Assistant error messages must not be empty")
        result = {**common, "status": "error", "stopReason": "error"}
        if message.error_message is not None:
            result["errorMessage"] = message.error_message
        return result
    if stop_reason == "aborted":
        result = {**common, "status": "aborted", "stopReason": "aborted"}
        if message.error_message is not None:
            result["errorMessage"] = message.error_message
        return result
    raise TypeError(f"Unsupported assistant stop reason: {stop_reason}")


def _to_protocol_tool_content(content: list[Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for part in content:
        if part.type == "text":
            result.append({"type": "text", "text": part.text})
        elif part.type == "image":
            result.append({"type": "image", "data": part.data, "mimeType": part.mime_type})
        else:
            raise TypeError(f"Unsupported tool content part: {part.type}")
    return result


def to_protocol_tool_result_message(message: ToolResultMessage, options: ToolTranscriptOptions) -> dict[str, Any]:
    call_id = _identifier(options.call.id, "Tool call id")
    call_name = _identifier(options.call.name, "Tool call name")
    if _identifier(message.tool_call_id, "Tool result call id") != call_id:
        raise TypeError(f"Tool result {message.tool_call_id} does not match tool call {call_id}")
    if _identifier(message.tool_name, "Tool result name") != call_name:
        raise TypeError(f"Tool result {message.tool_name} does not match tool call {call_name}")
    # `ToolResultMessage.details` defaults to `None`, so an absent value and an
    # explicit JSON null are indistinguishable here; the default is treated as
    # TypeScript's absent `details`, which the wire item omits.
    details = UNDEFINED if message.details is None else sanitize_protocol_details(message.details)
    usage = to_protocol_usage(message.usage)
    common: dict[str, Any] = {
        "id": _identifier(options.id, "Transcript item id"),
        "role": "tool",
        "toolCallId": call_id,
        "toolName": call_name,
        "input": to_protocol_json_value(options.call.arguments),
        "content": _to_protocol_tool_content(message.content),
    }
    if not isinstance(details, _Undefined):
        common["details"] = details
    if usage is not None:
        common["usage"] = usage
    common["timestamp"] = _timestamp(message.timestamp)
    if message.is_error:
        return {**common, "status": "error", "isError": True}
    return {**common, "status": "complete", "isError": False}
