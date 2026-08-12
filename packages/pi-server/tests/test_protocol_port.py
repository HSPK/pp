"""Python port of `packages/server/test/protocol.test.ts`.

`tests/test_protocol.py` already exists in the port with a broader port-only
suite, so this file keeps the upstream name with a `_port` suffix.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from pi_ai.types import (
    AssistantMessage,
    Cost,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_protocol import PROTOCOL_VERSION, encode_server_message
from pi_server.protocol import (
    AssistantTranscriptOptions,
    ToolTranscriptOptions,
    UserTranscriptOptions,
    sanitize_protocol_details,
    to_protocol_assistant_message,
    to_protocol_json_value,
    to_protocol_model_metadata,
    to_protocol_tool_result_message,
    to_protocol_user_message,
)

MODEL = Model(
    id="model-1",
    name="Model One",
    api="test-api",
    provider="test-provider",
    base_url="https://example.test",
    reasoning=True,
    input=["text", "image"],
    cost=ModelCost(input=1, output=2, cache_read=0.1, cache_write=0.2),
    context_window=100_000,
    max_tokens=10_000,
)


def assert_valid_server_payload(item: dict[str, Any]) -> None:
    encode_server_message(
        {
            "type": "hello",
            "version": PROTOCOL_VERSION,
            "connectionId": "connection-1",
            "snapshot": {
                "serverId": "server-1",
                "protocolVersion": PROTOCOL_VERSION,
                "revision": 0,
                "sessions": [
                    {
                        "id": "session-1",
                        "createdAt": 1,
                        "updatedAt": 1,
                        "sessionName": "Session one",
                        "cwd": "/workspace",
                    }
                ],
                "models": [to_protocol_model_metadata(MODEL, True)],
            },
        }
    )

    encode_server_message(
        {
            "type": "event",
            "event": {
                "type": "session_snapshot",
                "snapshot": {
                    "id": "session-1",
                    "cwd": "/workspace",
                    "createdAt": 1,
                    "updatedAt": 1,
                    "phase": "idle",
                    "model": {"provider": "test-provider", "id": "model-1"},
                    "thinkingLevel": "off",
                    "attached": True,
                    "locked": True,
                    "revision": 1,
                    "transcript": [item],
                    "queuedSteer": [],
                    "queuedSteerCount": 0,
                },
            },
        }
    )


def test_maps_model_metadata_and_produces_protocol_valid_output():
    result = to_protocol_model_metadata(MODEL, True)

    assert result["provider"] == "test-provider"
    assert result["id"] == "model-1"
    assert result["api"] == "test-api"
    assert result["input"] == ["text", "image"]
    assert result["authenticated"] is True
    assert "off" in result["supportedThinkingLevels"]


def test_exhaustively_maps_assistant_content_and_stop_reasons():
    message = AssistantMessage(
        content=[
            TextContent(text="hello"),
            ThinkingContent(thinking="hmm", redacted=False),
            ToolCall(id="call-1", name="read", arguments={"path": "README.md"}),
        ],
        api="test-api",
        provider="test-provider",
        model="model-1",
        usage=Usage(
            input=1,
            output=2,
            cache_read=3,
            cache_write=4,
            total_tokens=10,
            cost=Cost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
        ),
        stop_reason="toolUse",
        timestamp=123,
    )

    result = to_protocol_assistant_message(message, AssistantTranscriptOptions(id="message-1"))

    assert result["id"] == "message-1"
    assert result["status"] == "complete"
    assert result["stopReason"] == "toolUse"
    assert result["model"] == {"provider": "test-provider", "id": "model-1"}
    assert result["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "thinking", "thinking": "hmm", "redacted": False},
        {"type": "toolCall", "toolCallId": "call-1", "toolName": "read", "input": {"path": "README.md"}},
    ]
    assert_valid_server_payload(result)


def test_maps_user_and_tool_messages_without_leaking_non_json_details():
    user = UserMessage(content="hello", timestamp=1)
    circular: dict[str, Any] = {}
    circular["self"] = circular
    tool = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="read",
        content=[TextContent(text="result")],
        details=circular,
        is_error=False,
        timestamp=2,
    )
    call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})

    user_result = to_protocol_user_message(user, UserTranscriptOptions(id="user-1"))
    assert user_result["id"] == "user-1"
    assert user_result["content"] == [{"type": "text", "text": "hello"}]
    assert_valid_server_payload(user_result)

    tool_result = to_protocol_tool_result_message(tool, ToolTranscriptOptions(id="tool-1", call=call))
    assert tool_result["id"] == "tool-1"
    assert tool_result["toolName"] == "read"
    assert tool_result["input"] == {"path": "README.md"}
    assert tool_result["details"] == {"self": "[Circular]"}
    assert tool_result["status"] == "complete"
    assert_valid_server_payload(tool_result)


def test_rejects_tool_results_associated_with_a_different_call():
    call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    result = ToolResultMessage(
        tool_call_id="call-2",
        tool_name="read",
        content=[TextContent(text="result")],
        is_error=False,
        timestamp=2,
    )

    with pytest.raises(TypeError, match=r"(?i)tool call"):
        to_protocol_tool_result_message(result, ToolTranscriptOptions(id="tool-1", call=call))

    mismatched_name = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="write",
        content=[TextContent(text="result")],
        is_error=False,
        timestamp=2,
    )
    with pytest.raises(TypeError, match=r"(?i)tool call"):
        to_protocol_tool_result_message(mismatched_name, ToolTranscriptOptions(id="tool-1", call=call))


def test_derives_streaming_status_from_a_pending_stop_reason():
    message = AssistantMessage(
        content=[TextContent(text="partial")],
        api="test-api",
        provider="test-provider",
        model="model-1",
        usage=Usage(
            input=0,
            output=0,
            cache_read=0,
            cache_write=0,
            total_tokens=0,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        ),
        stop_reason="pending",
        timestamp=123,
    )

    result = to_protocol_assistant_message(message, AssistantTranscriptOptions(id="message-pending"))
    assert result["status"] == "streaming"
    assert "stopReason" not in result
    assert_valid_server_payload(result)


def test_preserves_optional_non_empty_assistant_error_messages():
    def make_message(error_message: str | None = None) -> AssistantMessage:
        return AssistantMessage(
            content=[],
            api="test-api",
            provider="test-provider",
            model="model-1",
            usage=Usage(
                input=0,
                output=0,
                cache_read=0,
                cache_write=0,
                total_tokens=0,
                cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            ),
            stop_reason="error",
            timestamp=123,
            error_message=error_message,
        )

    result_without_message = to_protocol_assistant_message(
        make_message(), AssistantTranscriptOptions(id="message-error")
    )
    assert result_without_message["status"] == "error"
    assert result_without_message["stopReason"] == "error"
    assert "errorMessage" not in result_without_message
    assert_valid_server_payload(result_without_message)

    with pytest.raises(TypeError):
        to_protocol_assistant_message(make_message(""), AssistantTranscriptOptions(id="message-error"))

    result_with_message = to_protocol_assistant_message(
        make_message("failed"), AssistantTranscriptOptions(id="message-error")
    )
    assert result_with_message["status"] == "error"
    assert result_with_message["stopReason"] == "error"
    assert result_with_message["errorMessage"] == "failed"
    assert_valid_server_payload(result_with_message)


def test_rejects_invalid_source_identifiers_and_timestamps():
    message = AssistantMessage(
        content=[ToolCall(id="", name="read", arguments={})],
        api="test-api",
        provider="test-provider",
        model="model-1",
        usage=Usage(
            input=0,
            output=0,
            cache_read=0,
            cache_write=0,
            total_tokens=0,
            cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
        ),
        stop_reason="toolUse",
        timestamp=1,
    )

    with pytest.raises(TypeError, match=r"(?i)tool call id"):
        to_protocol_assistant_message(message, AssistantTranscriptOptions(id="assistant-1"))

    # `Number.NaN` upstream; Python timestamps are `int`, so the equivalent
    # out-of-domain value is a negative one -- both fail `_timestamp`.
    with pytest.raises(TypeError, match=r"(?i)timestamp"):
        to_protocol_user_message(UserMessage(content="hello", timestamp=-1), UserTranscriptOptions(id="user-1"))


def test_rejects_lossy_tool_input_conversions():
    circular: dict[str, Any] = {}
    circular["self"] = circular

    with pytest.raises(TypeError):
        to_protocol_json_value(math.inf)
    # Upstream also rejects `1n`; Python `int` is already arbitrary precision and
    # is a valid protocol number, so the equivalent "unsupported runtime type"
    # rejection is asserted with a value that has no JSON counterpart at all.
    with pytest.raises(TypeError):
        to_protocol_json_value({1, 2})
    # Upstream's `undefined` case has no Python analogue: `None` is JSON `null`,
    # which is a legal protocol value and must not raise.
    assert to_protocol_json_value(None) is None
    with pytest.raises(TypeError):
        to_protocol_json_value(circular)


def test_rejects_sparse_execution_data_and_normalizes_sparse_diagnostic_arrays():
    # Upstream builds `new Array(2)` with a hole at index 0; a hole reads back as
    # `undefined`, which `toProtocolJsonValue` rejects. Python lists cannot have
    # holes, so the closest value is an explicit `None`, which is legal JSON
    # `null` and is preserved by both helpers rather than rejected.
    sparse: list[Any] = [None, "value"]

    assert to_protocol_json_value(sparse) == [None, "value"]
    assert sanitize_protocol_details(sparse) == [None, "value"]
