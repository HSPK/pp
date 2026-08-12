import pytest
from pi_protocol.codec import (
    ClientMessageDecoder,
    ProtocolValidationError,
    ServerMessageDecoder,
    create_client_message_decoder,
    create_server_message_decoder,
    encode_client_message,
    encode_server_message,
    is_protocol_value,
    is_supported_protocol_version,
    parse_client_message,
    parse_server_message,
)
from pi_protocol.schemas import PROTOCOL_VERSION


def model_ref() -> dict:
    return {"provider": "openai", "id": "gpt-4o"}


def usage() -> dict:
    return {
        "input": 10,
        "output": 5,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 15,
        "cost": {"input": 0.1, "output": 0.2, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.3},
    }


def session_snapshot() -> dict:
    return {
        "id": "s1",
        "cwd": "/work",
        "createdAt": 1,
        "updatedAt": 2,
        "phase": "idle",
        "model": model_ref(),
        "thinkingLevel": "off",
        "attached": True,
        "locked": False,
        "revision": 3,
        "transcript": [],
        "queuedSteer": [],
        "queuedSteerCount": 0,
    }


def server_snapshot() -> dict:
    return {
        "serverId": "srv",
        "protocolVersion": PROTOCOL_VERSION,
        "revision": 1,
        "sessions": [],
        "models": [],
    }


# --------------------------------------------------------------------------
# is_protocol_value
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, True, 1, 1.5, "s", [], {}, [1, {"a": [None]}]])
def test_is_protocol_value_accepts_json_values(value):
    assert is_protocol_value(value) is True


def test_is_protocol_value_rejects_non_json_types():
    assert is_protocol_value({1, 2}) is False
    assert is_protocol_value(object()) is False
    assert is_protocol_value({1: "int key"}) is False


def test_is_protocol_value_rejects_cycles():
    cyclic: list = [1]
    cyclic.append(cyclic)
    assert is_protocol_value(cyclic) is False

    cyclic_dict: dict = {}
    cyclic_dict["self"] = cyclic_dict
    assert is_protocol_value(cyclic_dict) is False


def test_is_protocol_value_allows_repeated_siblings():
    shared = {"a": 1}
    assert is_protocol_value([shared, shared]) is True


# --------------------------------------------------------------------------
# client messages
# --------------------------------------------------------------------------


def test_parse_client_hello():
    message = {"type": "hello", "version": 1}
    assert parse_client_message(message) is message


def test_client_hello_version_must_be_an_integer():
    with pytest.raises(ProtocolValidationError):
        parse_client_message({"type": "hello", "version": "1"})
    with pytest.raises(ProtocolValidationError):
        parse_client_message({"type": "hello", "version": 1.5})


def test_client_hello_rejects_unknown_properties():
    with pytest.raises(ProtocolValidationError):
        parse_client_message({"type": "hello", "version": 1, "extra": True})


@pytest.mark.parametrize(
    "command",
    [
        {"command": "list"},
        {"command": "create"},
        {"command": "create", "cwd": "/w", "name": "n", "model": {"provider": "p", "id": "m"}, "thinkingLevel": "high"},
        {"command": "attach", "sessionId": "s"},
        {"command": "detach", "sessionId": "s"},
        {"command": "prompt", "sessionId": "s", "text": "hi"},
        {"command": "steer", "sessionId": "s", "text": "stop"},
        {"command": "abort", "sessionId": "s"},
        {"command": "set_model", "sessionId": "s", "model": {"provider": "p", "id": "m"}},
        {"command": "set_thinking", "sessionId": "s", "thinkingLevel": "max"},
    ],
)
def test_parse_every_command(command):
    parse_client_message({"type": "request", "id": "r1", "request": command})


def test_rejects_unknown_command():
    with pytest.raises(ProtocolValidationError):
        parse_client_message({"type": "request", "id": "r", "request": {"command": "nope"}})


def test_rejects_command_with_missing_field():
    with pytest.raises(ProtocolValidationError):
        parse_client_message({"type": "request", "id": "r", "request": {"command": "prompt", "sessionId": "s"}})


def test_rejects_empty_id():
    with pytest.raises(ProtocolValidationError):
        parse_client_message({"type": "request", "id": "", "request": {"command": "list"}})


def test_rejects_invalid_thinking_level():
    with pytest.raises(ProtocolValidationError):
        parse_client_message(
            {
                "type": "request",
                "id": "r",
                "request": {"command": "set_thinking", "sessionId": "s", "thinkingLevel": "ultra"},
            }
        )


def test_rejects_a_server_message_as_a_client_message():
    with pytest.raises(ProtocolValidationError):
        parse_client_message({"type": "event", "event": {"type": "session_removed", "sessionId": "s"}})


# --------------------------------------------------------------------------
# server messages
# --------------------------------------------------------------------------


def test_parse_server_hello():
    parse_server_message(
        {"type": "hello", "version": PROTOCOL_VERSION, "connectionId": "c1", "snapshot": server_snapshot()}
    )


def test_server_hello_version_must_match_the_protocol_version():
    with pytest.raises(ProtocolValidationError):
        parse_server_message({"type": "hello", "version": 99, "connectionId": "c", "snapshot": server_snapshot()})


def test_parse_server_hello_error():
    parse_server_message({"type": "hello_error", "error": {"code": "version", "message": "too old"}})


def test_rejects_unknown_error_code():
    with pytest.raises(ProtocolValidationError):
        parse_server_message({"type": "hello_error", "error": {"code": "teapot", "message": "no"}})


def test_parse_ok_and_error_responses():
    parse_server_message({"type": "response", "id": "r", "ok": True, "result": {"command": "list", "sessions": []}})
    parse_server_message(
        {"type": "response", "id": "r", "ok": False, "error": {"code": "not_found", "message": "gone"}}
    )


def test_ok_response_must_carry_a_result_not_an_error():
    with pytest.raises(ProtocolValidationError):
        parse_server_message({"type": "response", "id": "r", "ok": True, "error": {"code": "busy", "message": "x"}})


def test_parse_session_events():
    parse_server_message({"type": "event", "event": {"type": "server_snapshot", "snapshot": server_snapshot()}})
    parse_server_message({"type": "event", "event": {"type": "session_snapshot", "snapshot": session_snapshot()}})
    parse_server_message({"type": "event", "event": {"type": "session_removed", "sessionId": "s"}})


def test_parse_session_progress_events():
    user_item = {"id": "i1", "role": "user", "content": [{"type": "text", "text": "hi"}], "timestamp": 1}
    parse_server_message(
        {
            "type": "event",
            "event": {
                "type": "session_progress",
                "sessionId": "s",
                "progress": {"type": "item_started", "item": user_item},
            },
        }
    )
    parse_server_message(
        {
            "type": "event",
            "event": {
                "type": "session_progress",
                "sessionId": "s",
                "progress": {
                    "type": "assistant_delta",
                    "messageId": "m",
                    "contentIndex": 0,
                    "kind": "text",
                    "delta": "hello",
                },
            },
        }
    )


def test_assistant_item_status_and_stop_reason_must_agree():
    base = {
        "id": "a1",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "model": model_ref(),
        "timestamp": 1,
    }
    parse_server_message(
        {
            "type": "event",
            "event": {
                "type": "session_progress",
                "sessionId": "s",
                "progress": {"type": "item_finished", "item": {**base, "status": "complete", "stopReason": "stop"}},
            },
        }
    )
    with pytest.raises(ProtocolValidationError):
        parse_server_message(
            {
                "type": "event",
                "event": {
                    "type": "session_progress",
                    "sessionId": "s",
                    "progress": {
                        "type": "item_finished",
                        "item": {**base, "status": "complete", "stopReason": "error"},
                    },
                },
            }
        )


def test_streaming_assistant_item_is_not_a_finished_item():
    streaming = {
        "id": "a1",
        "role": "assistant",
        "content": [],
        "model": model_ref(),
        "timestamp": 1,
        "status": "streaming",
    }
    # Valid as an item_started payload...
    parse_server_message(
        {
            "type": "event",
            "event": {
                "type": "session_progress",
                "sessionId": "s",
                "progress": {"type": "item_started", "item": streaming},
            },
        }
    )
    # ...but not as an item_finished payload.
    with pytest.raises(ProtocolValidationError):
        parse_server_message(
            {
                "type": "event",
                "event": {
                    "type": "session_progress",
                    "sessionId": "s",
                    "progress": {"type": "item_finished", "item": streaming},
                },
            }
        )


def test_tool_item_error_flag_must_match_status():
    base = {
        "id": "t1",
        "role": "tool",
        "toolCallId": "c1",
        "toolName": "read",
        "input": {"path": "a"},
        "content": [{"type": "text", "text": "ok"}],
        "timestamp": 1,
    }
    parse_server_message(
        {
            "type": "event",
            "event": {
                "type": "session_progress",
                "sessionId": "s",
                "progress": {"type": "item_finished", "item": {**base, "status": "error", "isError": True}},
            },
        }
    )
    with pytest.raises(ProtocolValidationError):
        parse_server_message(
            {
                "type": "event",
                "event": {
                    "type": "session_progress",
                    "sessionId": "s",
                    "progress": {"type": "item_finished", "item": {**base, "status": "error", "isError": False}},
                },
            }
        )


def test_usage_accepts_optional_reasoning():
    item = {
        "id": "a1",
        "role": "assistant",
        "content": [],
        "model": model_ref(),
        "timestamp": 1,
        "status": "complete",
        "stopReason": "stop",
        "usage": {**usage(), "reasoning": 3},
    }
    parse_server_message(
        {
            "type": "event",
            "event": {
                "type": "session_progress",
                "sessionId": "s",
                "progress": {"type": "item_finished", "item": item},
            },
        }
    )


def test_rejects_negative_token_counts():
    bad = {**usage(), "input": -1}
    item = {
        "id": "a1",
        "role": "assistant",
        "content": [],
        "model": model_ref(),
        "timestamp": 1,
        "status": "complete",
        "stopReason": "stop",
        "usage": bad,
    }
    with pytest.raises(ProtocolValidationError):
        parse_server_message(
            {
                "type": "event",
                "event": {
                    "type": "session_progress",
                    "sessionId": "s",
                    "progress": {"type": "item_finished", "item": item},
                },
            }
        )


# --------------------------------------------------------------------------
# framing round-trips
# --------------------------------------------------------------------------


def test_client_message_round_trip():
    message = {"type": "request", "id": "r1", "request": {"command": "prompt", "sessionId": "s", "text": "hi"}}
    decoder = ClientMessageDecoder()
    assert decoder.push(encode_client_message(message)) == [message]
    decoder.end()


def test_server_message_round_trip():
    message = {"type": "event", "event": {"type": "session_snapshot", "snapshot": session_snapshot()}}
    decoder = ServerMessageDecoder()
    assert decoder.push(encode_server_message(message)) == [message]
    decoder.end()


def test_decoder_reassembles_a_message_split_across_chunks():
    message = {"type": "hello", "version": 1}
    frame = encode_client_message(message)
    decoder = ClientMessageDecoder()
    decoded = []
    for index in range(len(frame)):
        decoded.extend(decoder.push(frame[index : index + 1]))
    decoder.end()
    assert decoded == [message]


def test_decoder_handles_several_messages_in_one_chunk():
    first = {"type": "hello", "version": 1}
    second = {"type": "request", "id": "r", "request": {"command": "list"}}
    decoder = ClientMessageDecoder()
    assert decoder.push(encode_client_message(first) + encode_client_message(second)) == [first, second]


def test_encoding_rejects_an_invalid_message():
    with pytest.raises(ProtocolValidationError, match="Invalid client protocol message"):
        encode_client_message({"type": "hello"})
    with pytest.raises(ProtocolValidationError, match="Invalid server protocol message"):
        encode_server_message({"type": "nope"})


def test_encoding_rejects_a_message_that_exceeds_the_frame_limit():
    message = {"type": "request", "id": "r", "request": {"command": "prompt", "sessionId": "s", "text": "x" * 500}}
    with pytest.raises(ProtocolValidationError, match="Unable to encode client protocol message"):
        encode_client_message(message, max_frame_length=32)


def test_decoder_rejects_a_frame_carrying_an_invalid_message():
    from pi_protocol.cbor import encode_cbor
    from pi_protocol.framing import encode_frame

    bad_frame = encode_frame(encode_cbor({"type": "hello"}))
    decoder = ClientMessageDecoder()
    with pytest.raises(ProtocolValidationError, match="Invalid client protocol message"):
        decoder.push(bad_frame)


def test_decoder_rejects_non_cbor_payloads():
    from pi_protocol.framing import encode_frame

    decoder = ClientMessageDecoder()
    with pytest.raises(ProtocolValidationError, match="Invalid client protocol frame"):
        decoder.push(encode_frame(b"\xff\xff\xff"))


def test_decoder_is_unusable_after_a_failure():
    from pi_protocol.framing import encode_frame

    decoder = ClientMessageDecoder()
    with pytest.raises(ProtocolValidationError):
        decoder.push(encode_frame(b"\xff\xff\xff"))
    with pytest.raises(ProtocolValidationError, match="has failed"):
        decoder.push(b"x")
    with pytest.raises(ProtocolValidationError, match="has failed"):
        decoder.end()


def test_decoder_end_rejects_a_truncated_frame():
    frame = encode_client_message({"type": "hello", "version": 1})
    decoder = ClientMessageDecoder()
    decoder.push(frame[:-1])
    with pytest.raises(ProtocolValidationError, match="Invalid client protocol framing"):
        decoder.end()


def test_decoder_factories_build_working_decoders():
    frame = encode_client_message({"type": "hello", "version": 1})
    assert create_client_message_decoder().push(frame) == [{"type": "hello", "version": 1}]

    server_frame = encode_server_message({"type": "hello_error", "error": {"code": "busy", "message": "m"}})
    assert create_server_message_decoder().push(server_frame)[0]["type"] == "hello_error"


# --------------------------------------------------------------------------
# version negotiation
# --------------------------------------------------------------------------


def test_is_supported_protocol_version():
    assert is_supported_protocol_version(PROTOCOL_VERSION) is True
    assert is_supported_protocol_version(PROTOCOL_VERSION + 1) is False
    assert is_supported_protocol_version(1.0) is False
    assert is_supported_protocol_version(True) is False
    assert is_supported_protocol_version("1") is False
