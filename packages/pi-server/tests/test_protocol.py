"""Tests for the protocol conversion helpers.

These functions decide what crosses the wire, so the tests assert both the
converted value and — where the result is a whole protocol object — that it
validates against the real schema from :mod:`pi_protocol`.
"""

from __future__ import annotations

import datetime
import math

import pytest
from jsonschema import Draft202012Validator
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
from pi_protocol.schemas import JSON_VALUE_DEFS, MODEL_METADATA_SCHEMA, USAGE_SCHEMA
from pi_server.protocol import (
    UNDEFINED,
    AssistantTranscriptOptions,
    ToolTranscriptOptions,
    UserTranscriptOptions,
    sanitize_protocol_details,
    to_protocol_assistant_message,
    to_protocol_json_value,
    to_protocol_model_metadata,
    to_protocol_tool_result_message,
    to_protocol_usage,
    to_protocol_user_message,
)

_usage_validator = Draft202012Validator({"$defs": JSON_VALUE_DEFS, **USAGE_SCHEMA})
_model_validator = Draft202012Validator({"$defs": JSON_VALUE_DEFS, **MODEL_METADATA_SCHEMA})


# --------------------------------------------------------------------------
# to_protocol_json_value
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, True, False, "text", 0, -1, 1.5])
def test_json_value_passes_through_scalars(value):
    assert to_protocol_json_value(value) == value


def test_json_value_copies_containers_deeply():
    source = {"a": [1, {"b": "c"}]}
    result = to_protocol_json_value(source)
    assert result == source
    assert result is not source
    assert result["a"] is not source["a"]
    assert result["a"][1] is not source["a"][1]


def test_json_value_rejects_non_finite_numbers():
    for bad in (math.inf, -math.inf, math.nan):
        with pytest.raises(TypeError, match="must be finite"):
            to_protocol_json_value(bad)


def test_json_value_rejects_unsupported_types():
    with pytest.raises(TypeError, match="Unsupported protocol JSON value"):
        to_protocol_json_value({1, 2})
    with pytest.raises(TypeError, match="Unsupported protocol JSON value"):
        to_protocol_json_value(object())


def test_json_value_rejects_cycles():
    cyclic: list = [1]
    cyclic.append(cyclic)
    with pytest.raises(TypeError, match="circular references"):
        to_protocol_json_value(cyclic)

    cyclic_dict: dict = {}
    cyclic_dict["self"] = cyclic_dict
    with pytest.raises(TypeError, match="circular references"):
        to_protocol_json_value(cyclic_dict)


def test_json_value_allows_repeated_siblings():
    shared = {"a": 1}
    assert to_protocol_json_value([shared, shared]) == [{"a": 1}, {"a": 1}]


def test_json_value_rejects_a_nested_bad_value():
    with pytest.raises(TypeError):
        to_protocol_json_value({"ok": 1, "bad": object()})


# --------------------------------------------------------------------------
# sanitize_protocol_details
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, True, "text", 3, 2.5])
def test_sanitize_passes_through_scalars(value):
    assert sanitize_protocol_details(value) == value


def test_sanitize_stringifies_non_finite_numbers():
    assert sanitize_protocol_details(math.inf) == "inf"
    assert sanitize_protocol_details(math.nan) == "nan"


def test_sanitize_drops_callables():
    # `UNDEFINED`, not `None`: a dropped value is TypeScript's `undefined`, and
    # `None` is reserved for JSON null (see `test_sanitize_keeps_json_nulls`).
    assert sanitize_protocol_details(lambda: None) is UNDEFINED
    assert sanitize_protocol_details({"fn": lambda: None, "keep": 1}) == {"keep": 1}


def test_sanitize_keeps_json_nulls():
    assert sanitize_protocol_details(None) is None
    assert sanitize_protocol_details({"a": None, "b": 1}) == {"a": None, "b": 1}
    assert sanitize_protocol_details([None, 1]) == [None, 1]


def test_sanitize_stringifies_unknown_objects():
    class Thing:
        def __str__(self) -> str:
            return "a thing"

    assert sanitize_protocol_details(Thing()) == "a thing"
    assert sanitize_protocol_details({1, 2}) in ("{1, 2}", "{2, 1}")


def test_sanitize_renders_datetimes_as_iso_8601():
    # TypeScript special-cases `Date` with `toISOString()`; plain `str()` on a
    # `datetime` is not ISO 8601 (it uses a space separator).
    moment = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)
    assert sanitize_protocol_details(moment) == "1970-01-01T00:00:00+00:00"


def test_sanitize_marks_cycles_instead_of_raising():
    cyclic: dict = {"name": "root"}
    cyclic["self"] = cyclic
    assert sanitize_protocol_details(cyclic) == {"name": "root", "self": "[Circular]"}

    cyclic_list: list = [1]
    cyclic_list.append(cyclic_list)
    assert sanitize_protocol_details(cyclic_list) == [1, "[Circular]"]


def test_sanitize_keeps_list_positions_for_dropped_entries():
    assert sanitize_protocol_details([1, lambda: None, 3]) == [1, None, 3]


def test_sanitize_omits_dropped_dict_keys():
    result = sanitize_protocol_details({"a": 1, "b": lambda: None})
    assert result == {"a": 1}
    assert "b" not in result


def test_sanitize_recurses_into_nested_structures():
    value = {"outer": {"inner": [1, {"fn": lambda: None, "keep": "x"}]}}
    assert sanitize_protocol_details(value) == {"outer": {"inner": [1, {"keep": "x"}]}}


# --------------------------------------------------------------------------
# to_protocol_usage
# --------------------------------------------------------------------------


def test_usage_none_returns_none():
    assert to_protocol_usage(None) is None


def test_usage_converts_and_validates_against_the_schema():
    usage = Usage(
        input=10,
        output=5,
        cache_read=2,
        cache_write=1,
        total_tokens=18,
        cost=Cost(input=0.1, output=0.2, cache_read=0.01, cache_write=0.02, total=0.33),
    )
    result = to_protocol_usage(usage)

    assert result["input"] == 10
    assert result["cacheRead"] == 2
    assert result["totalTokens"] == 18
    assert result["cost"]["total"] == pytest.approx(0.33)
    assert "reasoning" not in result
    _usage_validator.validate(result)


def test_usage_includes_reasoning_when_reported():
    usage = Usage(input=1, output=1, total_tokens=2, reasoning=7)
    result = to_protocol_usage(usage)
    assert result["reasoning"] == 7
    _usage_validator.validate(result)


def test_usage_clamps_negative_counts_and_costs():
    usage = Usage(
        input=-5,
        output=-1,
        cache_read=-2,
        cache_write=-3,
        total_tokens=-9,
        reasoning=-4,
        cost=Cost(input=-1.0, output=-2.0, cache_read=-3.0, cache_write=-4.0, total=-5.0),
    )
    result = to_protocol_usage(usage)

    assert result["input"] == 0
    assert result["output"] == 0
    assert result["cacheRead"] == 0
    assert result["cacheWrite"] == 0
    assert result["totalTokens"] == 0
    assert result["reasoning"] == 0
    assert result["cost"] == {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0}
    _usage_validator.validate(result)


def test_usage_floors_fractional_token_counts():
    usage = Usage(input=10.9, output=1, total_tokens=11.9)
    result = to_protocol_usage(usage)
    assert result["input"] == 10
    assert result["totalTokens"] == 11
    _usage_validator.validate(result)


def test_usage_drops_non_finite_reasoning_and_zeroes_non_finite_costs():
    usage = Usage(input=1, output=1, total_tokens=2, reasoning=math.inf)
    usage.cost = Cost(input=math.nan, output=math.inf, cache_read=0.0, cache_write=0.0, total=math.inf)
    result = to_protocol_usage(usage)

    assert "reasoning" not in result
    assert result["cost"]["input"] == 0.0
    assert result["cost"]["output"] == 0.0
    assert result["cost"]["total"] == 0.0
    _usage_validator.validate(result)


# --------------------------------------------------------------------------
# to_protocol_model_metadata
# --------------------------------------------------------------------------


def make_model(**overrides) -> Model:
    defaults = dict(
        id="gpt-test",
        name="GPT Test",
        api="openai-completions",
        provider="openai",
        base_url="https://example.invalid",
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=1.0, output=2.0, cache_read=0.5, cache_write=1.5),
        context_window=128_000,
        max_tokens=4096,
    )
    defaults.update(overrides)
    return Model(**defaults)


def test_model_metadata_converts_and_validates_against_the_schema():
    result = to_protocol_model_metadata(make_model(), authenticated=True)

    assert result["provider"] == "openai"
    assert result["id"] == "gpt-test"
    assert result["contextWindow"] == 128_000
    assert result["maxTokens"] == 4096
    assert result["authenticated"] is True
    assert result["supportedThinkingLevels"] == ["off"]
    _model_validator.validate(result)


def test_model_metadata_reports_supported_thinking_levels_for_reasoning_models():
    model = make_model(reasoning=True, thinking_level_map={"minimal": None, "xhigh": "ultra"})
    result = to_protocol_model_metadata(model, authenticated=False)

    assert "minimal" not in result["supportedThinkingLevels"]
    assert "xhigh" in result["supportedThinkingLevels"]
    assert "max" not in result["supportedThinkingLevels"]
    assert result["authenticated"] is False
    _model_validator.validate(result)


def test_model_metadata_preserves_image_input():
    result = to_protocol_model_metadata(make_model(input=["text", "image"]), authenticated=True)
    assert result["input"] == ["text", "image"]
    _model_validator.validate(result)


def test_model_metadata_clamps_costs_and_windows_to_schema_minimums():
    model = make_model(
        context_window=0,
        max_tokens=-5,
        cost=ModelCost(input=-1.0, output=-2.0, cache_read=-3.0, cache_write=-4.0),
    )
    result = to_protocol_model_metadata(model, authenticated=True)

    assert result["contextWindow"] == 1
    assert result["maxTokens"] == 1
    assert result["cost"] == {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0}
    _model_validator.validate(result)


def test_model_metadata_floors_fractional_windows():
    result = to_protocol_model_metadata(make_model(context_window=1000.9, max_tokens=10.9), authenticated=True)
    assert result["contextWindow"] == 1000
    assert result["maxTokens"] == 10


@pytest.mark.parametrize(
    ("field", "label"),
    [("provider", "Model provider"), ("id", "Model id"), ("api", "Model API")],
)
def test_model_metadata_rejects_empty_identifiers(field, label):
    model = make_model(**{field: ""})
    with pytest.raises(TypeError, match=label):
        to_protocol_model_metadata(model, authenticated=True)


def test_model_metadata_name_defaults_to_the_id():
    # Model.__post_init__ backfills an empty name from the id, so the name can
    # never reach the identifier check empty.
    result = to_protocol_model_metadata(make_model(name=""), authenticated=True)
    assert result["name"] == "gpt-test"


def test_model_metadata_rejects_a_name_blanked_after_construction():
    model = make_model()
    model.name = ""
    with pytest.raises(TypeError, match="Model name"):
        to_protocol_model_metadata(model, authenticated=True)


# --------------------------------------------------------------------------
# transcript conversion: to_protocol_user_message / to_protocol_assistant_message
# / to_protocol_tool_result_message
# --------------------------------------------------------------------------


def _assert_valid_server_payload(item: dict) -> None:
    """A converted transcript item must fit inside a real server snapshot payload."""
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
                "models": [to_protocol_model_metadata(make_model(), authenticated=True)],
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


def test_assistant_message_exhaustively_maps_content_and_stop_reasons():
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
    _assert_valid_server_payload(result)


def test_user_and_tool_messages_map_without_leaking_non_json_details():
    user = UserMessage(content="hello", timestamp=1)
    circular: dict = {}
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
    _assert_valid_server_payload(user_result)

    tool_result = to_protocol_tool_result_message(tool, ToolTranscriptOptions(id="tool-1", call=call))
    assert tool_result["id"] == "tool-1"
    assert tool_result["toolName"] == "read"
    assert tool_result["input"] == {"path": "README.md"}
    assert tool_result["details"] == {"self": "[Circular]"}
    assert tool_result["status"] == "complete"
    _assert_valid_server_payload(tool_result)


def test_tool_result_details_absent_versus_dropped_versus_null():
    call = ToolCall(id="call-1", name="read", arguments={})

    def item(details: object) -> dict:
        message = ToolResultMessage(
            tool_call_id="call-1",
            tool_name="read",
            content=[TextContent(text="result")],
            details=details,
            is_error=False,
            timestamp=2,
        )
        return to_protocol_tool_result_message(message, ToolTranscriptOptions(id="tool-1", call=call))

    assert "details" not in item(None)
    assert "details" not in item(lambda: None)
    # A nested null is JSON null and survives; only `undefined` disappears.
    assert item({"a": None, "fn": lambda: None})["details"] == {"a": None}


def test_tool_result_rejects_mismatched_call():
    call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})
    result = ToolResultMessage(
        tool_call_id="call-2",
        tool_name="read",
        content=[TextContent(text="result")],
        is_error=False,
        timestamp=2,
    )
    with pytest.raises(TypeError, match="tool call"):
        to_protocol_tool_result_message(result, ToolTranscriptOptions(id="tool-1", call=call))

    result.tool_call_id = "call-1"
    result.tool_name = "write"
    with pytest.raises(TypeError, match="tool call"):
        to_protocol_tool_result_message(result, ToolTranscriptOptions(id="tool-1", call=call))


def test_assistant_message_derives_streaming_status_from_pending_stop_reason():
    message = AssistantMessage(
        content=[TextContent(text="partial")],
        api="test-api",
        provider="test-provider",
        model="model-1",
        usage=Usage(input=0, output=0, total_tokens=0),
        stop_reason="pending",
        timestamp=123,
    )
    result = to_protocol_assistant_message(message, AssistantTranscriptOptions(id="message-pending"))
    assert result["status"] == "streaming"
    assert "stopReason" not in result
    _assert_valid_server_payload(result)


def test_assistant_message_preserves_optional_non_empty_error_messages():
    message = AssistantMessage(
        content=[],
        api="test-api",
        provider="test-provider",
        model="model-1",
        usage=Usage(input=0, output=0, total_tokens=0),
        stop_reason="error",
        timestamp=123,
    )
    result_without_message = to_protocol_assistant_message(message, AssistantTranscriptOptions(id="message-error"))
    assert result_without_message["status"] == "error"
    assert result_without_message["stopReason"] == "error"
    assert "errorMessage" not in result_without_message
    _assert_valid_server_payload(result_without_message)

    message.error_message = ""
    with pytest.raises(TypeError):
        to_protocol_assistant_message(message, AssistantTranscriptOptions(id="message-error"))

    message.error_message = "failed"
    result_with_message = to_protocol_assistant_message(message, AssistantTranscriptOptions(id="message-error"))
    assert result_with_message["status"] == "error"
    assert result_with_message["stopReason"] == "error"
    assert result_with_message["errorMessage"] == "failed"
    _assert_valid_server_payload(result_with_message)


def test_transcript_conversion_rejects_invalid_source_identifiers_and_timestamps():
    message = AssistantMessage(
        content=[ToolCall(id="", name="read", arguments={})],
        api="test-api",
        provider="test-provider",
        model="model-1",
        usage=Usage(input=0, output=0, total_tokens=0),
        stop_reason="toolUse",
        timestamp=1,
    )
    with pytest.raises(TypeError, match="Tool call id"):
        to_protocol_assistant_message(message, AssistantTranscriptOptions(id="assistant-1"))

    with pytest.raises(TypeError, match=r"[Tt]imestamp"):
        to_protocol_user_message(UserMessage(content="hello", timestamp=-1), UserTranscriptOptions(id="user-1"))
