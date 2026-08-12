"""Wire schemas for the RPC protocol.

Python port of `packages/protocol/src/schemas.ts`. The TypeScript version builds
TypeBox schemas; this port emits the equivalent JSON Schema (draft 2020-12)
dicts, which :mod:`jsonschema` validates in :mod:`pi_protocol.codec`.

Field names are the on-the-wire names and stay camelCase, matching TypeScript.
"""

from __future__ import annotations

from typing import Any

PROTOCOL_VERSION = 1

_ID = {"type": "string", "minLength": 1}
_TIMESTAMP = {"type": "integer", "minimum": 0}
_NON_NEGATIVE_INT = {"type": "integer", "minimum": 0}
_NON_NEGATIVE_NUMBER = {"type": "number", "minimum": 0}

JSON_VALUE_REF = {"$ref": "#/$defs/JsonValue"}

JSON_VALUE_DEFS: dict[str, Any] = {
    "JsonValue": {
        "anyOf": [
            {"type": "null"},
            {"type": "boolean"},
            {"type": "number"},
            {"type": "string"},
            {"type": "array", "items": {"$ref": "#/$defs/JsonValue"}},
            {"type": "object", "additionalProperties": {"$ref": "#/$defs/JsonValue"}},
        ]
    }
}


def strict_object(properties: dict[str, Any], optional: tuple[str, ...] = ()) -> dict[str, Any]:
    """A closed object schema. Everything not in ``optional`` is required."""
    return {
        "type": "object",
        "properties": properties,
        "required": [name for name in properties if name not in optional],
        "additionalProperties": False,
    }


def literal(value: Any) -> dict[str, Any]:
    return {"const": value}


def union(*schemas: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": list(schemas)}


def enum(*values: str) -> dict[str, Any]:
    return {"enum": list(values)}


THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
THINKING_LEVEL_SCHEMA = enum(*THINKING_LEVELS)

SESSION_PHASES = ("idle", "turn", "compaction", "branch_summary", "retry")
SESSION_PHASE_SCHEMA = enum(*SESSION_PHASES)
"""Matches AgentHarnessPhase so adapters do not need a second phase vocabulary."""

MODEL_REF_SCHEMA = strict_object({"provider": _ID, "id": _ID})

MODEL_COST_SCHEMA = strict_object(
    {
        "input": _NON_NEGATIVE_NUMBER,
        "output": _NON_NEGATIVE_NUMBER,
        "cacheRead": _NON_NEGATIVE_NUMBER,
        "cacheWrite": _NON_NEGATIVE_NUMBER,
    }
)

MODEL_METADATA_SCHEMA = strict_object(
    {
        "provider": _ID,
        "id": _ID,
        "name": {"type": "string", "minLength": 1},
        "api": _ID,
        "reasoning": {"type": "boolean"},
        "input": {"type": "array", "items": enum("text", "image")},
        "contextWindow": {"type": "integer", "minimum": 1},
        "maxTokens": {"type": "integer", "minimum": 1},
        "cost": MODEL_COST_SCHEMA,
        "supportedThinkingLevels": {"type": "array", "items": THINKING_LEVEL_SCHEMA, "minItems": 1},
        "authenticated": {"type": "boolean"},
    }
)

TEXT_CONTENT_SCHEMA = strict_object({"type": literal("text"), "text": {"type": "string"}})
THINKING_CONTENT_SCHEMA = strict_object(
    {"type": literal("thinking"), "thinking": {"type": "string"}, "redacted": {"type": "boolean"}},
    optional=("redacted",),
)
IMAGE_CONTENT_SCHEMA = strict_object(
    {"type": literal("image"), "data": {"type": "string"}, "mimeType": {"type": "string", "minLength": 1}}
)
TOOL_CALL_CONTENT_SCHEMA = strict_object(
    {"type": literal("toolCall"), "toolCallId": _ID, "toolName": _ID, "input": JSON_VALUE_REF}
)

USER_CONTENT_SCHEMA = union(TEXT_CONTENT_SCHEMA, IMAGE_CONTENT_SCHEMA)
ASSISTANT_CONTENT_SCHEMA = union(TEXT_CONTENT_SCHEMA, THINKING_CONTENT_SCHEMA, TOOL_CALL_CONTENT_SCHEMA)
TOOL_CONTENT_SCHEMA = union(TEXT_CONTENT_SCHEMA, IMAGE_CONTENT_SCHEMA)

USAGE_SCHEMA = strict_object(
    {
        "input": _NON_NEGATIVE_INT,
        "output": _NON_NEGATIVE_INT,
        "cacheRead": _NON_NEGATIVE_INT,
        "cacheWrite": _NON_NEGATIVE_INT,
        "reasoning": _NON_NEGATIVE_INT,
        "totalTokens": _NON_NEGATIVE_INT,
        "cost": strict_object(
            {
                "input": _NON_NEGATIVE_NUMBER,
                "output": _NON_NEGATIVE_NUMBER,
                "cacheRead": _NON_NEGATIVE_NUMBER,
                "cacheWrite": _NON_NEGATIVE_NUMBER,
                "total": _NON_NEGATIVE_NUMBER,
            }
        ),
    },
    optional=("reasoning",),
)

USER_TRANSCRIPT_ITEM_SCHEMA = strict_object(
    {
        "id": _ID,
        "role": literal("user"),
        "content": {"type": "array", "items": USER_CONTENT_SCHEMA},
        "timestamp": _TIMESTAMP,
    }
)

_ASSISTANT_ITEM_PROPERTIES = {
    "id": _ID,
    "role": literal("assistant"),
    "content": {"type": "array", "items": ASSISTANT_CONTENT_SCHEMA},
    "model": MODEL_REF_SCHEMA,
    "responseModel": {"type": "string", "minLength": 1},
    "usage": USAGE_SCHEMA,
    "timestamp": _TIMESTAMP,
}
_ASSISTANT_OPTIONAL = ("responseModel", "usage")

STREAMING_ASSISTANT_ITEM_SCHEMA = strict_object(
    {**_ASSISTANT_ITEM_PROPERTIES, "status": literal("streaming")}, optional=_ASSISTANT_OPTIONAL
)
COMPLETE_ASSISTANT_ITEM_SCHEMA = strict_object(
    {
        **_ASSISTANT_ITEM_PROPERTIES,
        "status": literal("complete"),
        "stopReason": enum("stop", "length", "toolUse"),
    },
    optional=_ASSISTANT_OPTIONAL,
)
ERROR_ASSISTANT_ITEM_SCHEMA = strict_object(
    {
        **_ASSISTANT_ITEM_PROPERTIES,
        "status": literal("error"),
        "stopReason": literal("error"),
        "errorMessage": {"type": "string", "minLength": 1},
    },
    optional=(*_ASSISTANT_OPTIONAL, "errorMessage"),
)
ABORTED_ASSISTANT_ITEM_SCHEMA = strict_object(
    {
        **_ASSISTANT_ITEM_PROPERTIES,
        "status": literal("aborted"),
        "stopReason": literal("aborted"),
        "errorMessage": {"type": "string"},
    },
    optional=(*_ASSISTANT_OPTIONAL, "errorMessage"),
)
ASSISTANT_TRANSCRIPT_ITEM_SCHEMA = union(
    STREAMING_ASSISTANT_ITEM_SCHEMA,
    COMPLETE_ASSISTANT_ITEM_SCHEMA,
    ERROR_ASSISTANT_ITEM_SCHEMA,
    ABORTED_ASSISTANT_ITEM_SCHEMA,
)

_TOOL_ITEM_PROPERTIES = {
    "id": _ID,
    "role": literal("tool"),
    "toolCallId": _ID,
    "toolName": _ID,
    "input": JSON_VALUE_REF,
    "content": {"type": "array", "items": TOOL_CONTENT_SCHEMA},
    "details": JSON_VALUE_REF,
    "usage": USAGE_SCHEMA,
    "timestamp": _TIMESTAMP,
}
_TOOL_OPTIONAL = ("details", "usage")

RUNNING_TOOL_ITEM_SCHEMA = strict_object(
    {**_TOOL_ITEM_PROPERTIES, "status": literal("running"), "isError": literal(False)}, optional=_TOOL_OPTIONAL
)
COMPLETE_TOOL_ITEM_SCHEMA = strict_object(
    {**_TOOL_ITEM_PROPERTIES, "status": literal("complete"), "isError": literal(False)}, optional=_TOOL_OPTIONAL
)
ERROR_TOOL_ITEM_SCHEMA = strict_object(
    {**_TOOL_ITEM_PROPERTIES, "status": literal("error"), "isError": literal(True)}, optional=_TOOL_OPTIONAL
)
TOOL_TRANSCRIPT_ITEM_SCHEMA = union(RUNNING_TOOL_ITEM_SCHEMA, COMPLETE_TOOL_ITEM_SCHEMA, ERROR_TOOL_ITEM_SCHEMA)

TRANSCRIPT_ITEM_SCHEMA = union(
    USER_TRANSCRIPT_ITEM_SCHEMA, ASSISTANT_TRANSCRIPT_ITEM_SCHEMA, TOOL_TRANSCRIPT_ITEM_SCHEMA
)

TRANSCRIPT_PROGRESS_SCHEMA = union(
    strict_object({"type": literal("item_started"), "item": TRANSCRIPT_ITEM_SCHEMA}),
    strict_object(
        {
            "type": literal("assistant_delta"),
            "messageId": _ID,
            "contentIndex": _NON_NEGATIVE_INT,
            "kind": enum("text", "thinking", "toolCall"),
            "delta": {"type": "string"},
        }
    ),
    strict_object(
        {
            "type": literal("item_updated"),
            "item": union(ASSISTANT_TRANSCRIPT_ITEM_SCHEMA, TOOL_TRANSCRIPT_ITEM_SCHEMA),
        }
    ),
    strict_object(
        {
            "type": literal("item_finished"),
            "item": union(
                COMPLETE_ASSISTANT_ITEM_SCHEMA,
                ERROR_ASSISTANT_ITEM_SCHEMA,
                ABORTED_ASSISTANT_ITEM_SCHEMA,
                COMPLETE_TOOL_ITEM_SCHEMA,
                ERROR_TOOL_ITEM_SCHEMA,
            ),
        }
    ),
)
"""Normalized incremental activity. Snapshots remain authoritative."""

SESSION_METADATA_SCHEMA = strict_object(
    {
        "id": _ID,
        "createdAt": _TIMESTAMP,
        "updatedAt": _TIMESTAMP,
        "parentSessionId": _ID,
        "sessionName": {"type": "string"},
        "cwd": {"type": "string", "minLength": 1},
    },
    optional=("updatedAt", "parentSessionId", "sessionName", "cwd"),
)

SESSION_SNAPSHOT_SCHEMA = strict_object(
    {
        "id": _ID,
        "name": {"type": "string"},
        "cwd": {"type": "string", "minLength": 1},
        "createdAt": _TIMESTAMP,
        "updatedAt": _TIMESTAMP,
        "phase": SESSION_PHASE_SCHEMA,
        "model": MODEL_REF_SCHEMA,
        "thinkingLevel": THINKING_LEVEL_SCHEMA,
        "attached": {"type": "boolean"},
        "locked": {"type": "boolean"},
        "revision": _NON_NEGATIVE_INT,
        "transcript": {"type": "array", "items": TRANSCRIPT_ITEM_SCHEMA},
        "queuedSteer": {"type": "array", "items": USER_TRANSCRIPT_ITEM_SCHEMA},
        "queuedSteerCount": _NON_NEGATIVE_INT,
    },
    optional=("name",),
)

SERVER_SNAPSHOT_SCHEMA = strict_object(
    {
        "serverId": _ID,
        "protocolVersion": literal(PROTOCOL_VERSION),
        "revision": _NON_NEGATIVE_INT,
        "sessions": {"type": "array", "items": SESSION_METADATA_SCHEMA},
        "models": {"type": "array", "items": MODEL_METADATA_SCHEMA},
    }
)

PROTOCOL_ERROR_CODES = (
    "version",
    "busy",
    "session_locked",
    "not_found",
    "invalid_request",
    "not_implemented",
    "internal_error",
)
PROTOCOL_ERROR_CODE_SCHEMA = enum(*PROTOCOL_ERROR_CODES)
PROTOCOL_ERROR_SCHEMA = strict_object(
    {"code": PROTOCOL_ERROR_CODE_SCHEMA, "message": {"type": "string"}, "details": JSON_VALUE_REF},
    optional=("details",),
)

_PROMPT_PAYLOAD = {"sessionId": _ID, "text": {"type": "string"}}

LIST_COMMAND_SCHEMA = strict_object({"command": literal("list")})
CREATE_COMMAND_SCHEMA = strict_object(
    {
        "command": literal("create"),
        "cwd": {"type": "string", "minLength": 1},
        "name": {"type": "string"},
        "model": MODEL_REF_SCHEMA,
        "thinkingLevel": THINKING_LEVEL_SCHEMA,
    },
    optional=("cwd", "name", "model", "thinkingLevel"),
)
ATTACH_COMMAND_SCHEMA = strict_object({"command": literal("attach"), "sessionId": _ID})
DETACH_COMMAND_SCHEMA = strict_object({"command": literal("detach"), "sessionId": _ID})
PROMPT_COMMAND_SCHEMA = strict_object({"command": literal("prompt"), **_PROMPT_PAYLOAD})
STEER_COMMAND_SCHEMA = strict_object({"command": literal("steer"), **_PROMPT_PAYLOAD})
ABORT_COMMAND_SCHEMA = strict_object({"command": literal("abort"), "sessionId": _ID})
SET_MODEL_COMMAND_SCHEMA = strict_object({"command": literal("set_model"), "sessionId": _ID, "model": MODEL_REF_SCHEMA})
SET_THINKING_COMMAND_SCHEMA = strict_object(
    {"command": literal("set_thinking"), "sessionId": _ID, "thinkingLevel": THINKING_LEVEL_SCHEMA}
)

COMMAND_SCHEMA = union(
    LIST_COMMAND_SCHEMA,
    CREATE_COMMAND_SCHEMA,
    ATTACH_COMMAND_SCHEMA,
    DETACH_COMMAND_SCHEMA,
    PROMPT_COMMAND_SCHEMA,
    STEER_COMMAND_SCHEMA,
    ABORT_COMMAND_SCHEMA,
    SET_MODEL_COMMAND_SCHEMA,
    SET_THINKING_COMMAND_SCHEMA,
)

COMMAND_NAMES = (
    "list",
    "create",
    "attach",
    "detach",
    "prompt",
    "steer",
    "abort",
    "set_model",
    "set_thinking",
)


def _session_result(command: str) -> dict[str, Any]:
    return strict_object({"command": literal(command), "session": SESSION_SNAPSHOT_SCHEMA})


CREATE_RESULT_SCHEMA = _session_result("create")
ATTACH_RESULT_SCHEMA = _session_result("attach")
PROMPT_RESULT_SCHEMA = _session_result("prompt")
STEER_RESULT_SCHEMA = _session_result("steer")
ABORT_RESULT_SCHEMA = _session_result("abort")
SET_MODEL_RESULT_SCHEMA = _session_result("set_model")
SET_THINKING_RESULT_SCHEMA = _session_result("set_thinking")
LIST_RESULT_SCHEMA = strict_object(
    {"command": literal("list"), "sessions": {"type": "array", "items": SESSION_METADATA_SCHEMA}}
)
DETACH_RESULT_SCHEMA = strict_object({"command": literal("detach"), "sessionId": _ID})

COMMAND_RESULT_SCHEMA = union(
    LIST_RESULT_SCHEMA,
    CREATE_RESULT_SCHEMA,
    ATTACH_RESULT_SCHEMA,
    DETACH_RESULT_SCHEMA,
    PROMPT_RESULT_SCHEMA,
    STEER_RESULT_SCHEMA,
    ABORT_RESULT_SCHEMA,
    SET_MODEL_RESULT_SCHEMA,
    SET_THINKING_RESULT_SCHEMA,
)

CLIENT_HELLO_SCHEMA = strict_object({"type": literal("hello"), "version": {"type": "integer", "minimum": 0}})
"""Must be the first frame a client sends. Version is an integer, not a coercible string."""

REQUEST_ENVELOPE_SCHEMA = strict_object({"type": literal("request"), "id": _ID, "request": COMMAND_SCHEMA})

CLIENT_MESSAGE_SCHEMA: dict[str, Any] = {
    "$defs": JSON_VALUE_DEFS,
    **union(CLIENT_HELLO_SCHEMA, REQUEST_ENVELOPE_SCHEMA),
}

SERVER_EVENT_SCHEMA = union(
    strict_object({"type": literal("server_snapshot"), "snapshot": SERVER_SNAPSHOT_SCHEMA}),
    strict_object({"type": literal("session_snapshot"), "snapshot": SESSION_SNAPSHOT_SCHEMA}),
    strict_object({"type": literal("session_progress"), "sessionId": _ID, "progress": TRANSCRIPT_PROGRESS_SCHEMA}),
    strict_object({"type": literal("session_removed"), "sessionId": _ID}),
)

SERVER_HELLO_SCHEMA = strict_object(
    {
        "type": literal("hello"),
        "version": literal(PROTOCOL_VERSION),
        "connectionId": _ID,
        "snapshot": SERVER_SNAPSHOT_SCHEMA,
    }
)
SERVER_HELLO_ERROR_SCHEMA = strict_object({"type": literal("hello_error"), "error": PROTOCOL_ERROR_SCHEMA})
RESPONSE_ENVELOPE_SCHEMA = union(
    strict_object({"type": literal("response"), "id": _ID, "ok": literal(True), "result": COMMAND_RESULT_SCHEMA}),
    strict_object({"type": literal("response"), "id": _ID, "ok": literal(False), "error": PROTOCOL_ERROR_SCHEMA}),
)
EVENT_ENVELOPE_SCHEMA = strict_object({"type": literal("event"), "event": SERVER_EVENT_SCHEMA})

SERVER_MESSAGE_SCHEMA: dict[str, Any] = {
    "$defs": JSON_VALUE_DEFS,
    **union(
        SERVER_HELLO_SCHEMA,
        SERVER_HELLO_ERROR_SCHEMA,
        RESPONSE_ENVELOPE_SCHEMA,
        EVENT_ENVELOPE_SCHEMA,
    ),
}
