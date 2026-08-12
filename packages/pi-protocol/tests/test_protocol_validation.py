"""Python port of `packages/protocol/test/protocol.test.ts`."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pi_protocol import (
    PROTOCOL_VERSION,
    ClientMessageDecoder,
    FrameDecoder,
    ProtocolValidationError,
    ServerMessageDecoder,
    decode_cbor,
    encode_cbor,
    encode_client_message,
    encode_frame,
    encode_server_message,
    is_supported_protocol_version,
    parse_client_message,
    parse_server_message,
)

EMPTY_SERVER_SNAPSHOT: dict[str, Any] = {
    "serverId": "server-1",
    "protocolVersion": PROTOCOL_VERSION,
    "revision": 0,
    "sessions": [],
    "models": [],
}

CLIENT_HELLO: dict[str, Any] = {"type": "hello", "version": PROTOCOL_VERSION}

SERVER_HELLO: dict[str, Any] = {
    "type": "hello",
    "version": PROTOCOL_VERSION,
    "connectionId": "connection-1",
    "snapshot": EMPTY_SERVER_SNAPSHOT,
}


def item_message(item: Any, type_: str = "item_finished") -> dict[str, Any]:
    return {
        "type": "event",
        "event": {
            "type": "session_progress",
            "sessionId": "session-1",
            "progress": {"type": type_, "item": item},
        },
    }


def test_uses_protocol_version_1() -> None:
    assert PROTOCOL_VERSION == 1
    assert is_supported_protocol_version(1) is True
    assert is_supported_protocol_version(2) is False
    assert is_supported_protocol_version(2.5) is False


@pytest.mark.parametrize("version", [0, PROTOCOL_VERSION, PROTOCOL_VERSION + 1])
def test_accepts_integer_client_hello_version_for_negotiation(version: int) -> None:
    message = {**CLIENT_HELLO, "version": version}
    assert parse_client_message(message) == message


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("string version", {"type": "hello", "version": str(PROTOCOL_VERSION)}),
        ("fractional version", {"type": "hello", "version": PROTOCOL_VERSION + 0.5}),
        ("credential field", {"type": "hello", "version": PROTOCOL_VERSION, "token": "secret"}),
        ("unknown field", {"type": "hello", "version": PROTOCOL_VERSION, "extra": True}),
    ],
)
def test_rejects_a_handshake(label: str, message: dict[str, Any]) -> None:
    with pytest.raises(ProtocolValidationError):
        parse_client_message(message)


def test_does_not_parse_json_strings_as_wire_messages() -> None:
    with pytest.raises(ProtocolValidationError):
        parse_client_message(json.dumps(CLIENT_HELLO))
    with pytest.raises(ProtocolValidationError):
        parse_server_message(json.dumps(SERVER_HELLO))


def test_rejects_image_input_while_the_mvp_remains_text_only() -> None:
    with pytest.raises(ProtocolValidationError):
        parse_client_message(
            {
                "type": "request",
                "id": "request-1",
                "request": {
                    "command": "prompt",
                    "sessionId": "session-1",
                    "text": "inspect",
                    "images": [{"type": "image", "data": "abc", "mimeType": "image/png"}],
                },
            }
        )


def test_parses_a_server_handshake_snapshot() -> None:
    assert parse_server_message(SERVER_HELLO) == SERVER_HELLO


def test_represents_listed_sessions_as_durable_metadata() -> None:
    message: dict[str, Any] = {
        "type": "response",
        "id": "request-1",
        "ok": True,
        "result": {
            "command": "list",
            "sessions": [
                {
                    "id": "session-1",
                    "createdAt": 1,
                    "updatedAt": 2,
                    "parentSessionId": "parent-1",
                    "sessionName": "Named session",
                    "cwd": "/workspace",
                }
            ],
        },
    }

    assert parse_server_message(message) == message
    with pytest.raises(ProtocolValidationError):
        parse_server_message(
            {
                **message,
                "result": {
                    **message["result"],
                    "sessions": [{"id": "session-1", "createdAt": 1, "phase": "idle"}],
                },
            }
        )


@pytest.mark.parametrize("code", ["not_implemented", "internal_error"])
def test_accepts_the_error_code(code: str) -> None:
    message: dict[str, Any] = {
        "type": "response",
        "id": "request-1",
        "ok": False,
        "error": {"code": code, "message": "safe"},
    }
    assert parse_server_message(message) == message


@pytest.mark.parametrize(
    "wire",
    [
        {
            "type": "hello",
            "version": PROTOCOL_VERSION + 1,
            "connectionId": "connection-1",
            "snapshot": EMPTY_SERVER_SNAPSHOT,
        },
        {"type": "hello_error", "error": {"code": "auth", "message": "Authentication failed"}},
        {"type": "response", "id": "request-1", "ok": True, "result": {"command": "unknown"}},
        {"type": "event", "event": {"type": "session_removed", "sessionId": 42}},
    ],
)
def test_rejects_invalid_server_messages(wire: dict[str, Any]) -> None:
    with pytest.raises(ProtocolValidationError):
        parse_server_message(wire)


def test_validates_nested_json_tool_details() -> None:
    message = {
        "type": "event",
        "event": {
            "type": "session_progress",
            "sessionId": "session-1",
            "progress": {
                "type": "item_finished",
                "item": {
                    "id": "tool-1",
                    "role": "tool",
                    "toolCallId": "call-1",
                    "toolName": "read",
                    "input": {"path": "/tmp/file"},
                    "content": [{"type": "text", "text": "done"}],
                    "details": {"lines": [1, 2, 3], "cached": False},
                    "status": "complete",
                    "isError": False,
                    "timestamp": 1,
                },
            },
        },
    }
    assert parse_server_message(message) == message


@pytest.mark.parametrize(
    "state",
    [
        {"status": "streaming"},
        {"status": "complete", "stopReason": "stop"},
        {"status": "error", "stopReason": "error"},
        {"status": "error", "stopReason": "error", "errorMessage": "failed"},
        {"status": "aborted", "stopReason": "aborted"},
    ],
)
def test_accepts_a_consistent_assistant_item(state: dict[str, Any]) -> None:
    message = item_message(
        {
            "id": "assistant-1",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": {"provider": "test", "id": "model"},
            "timestamp": 1,
            **state,
        },
        "item_updated" if state["status"] == "streaming" else "item_finished",
    )
    assert parse_server_message(message) == message


@pytest.mark.parametrize(
    "state",
    [
        {"status": "streaming", "stopReason": "stop"},
        {"status": "complete"},
        {"status": "complete", "stopReason": "error"},
        {"status": "error", "stopReason": "error", "errorMessage": ""},
        {"status": "aborted", "stopReason": "stop"},
    ],
)
def test_rejects_an_inconsistent_assistant_item(state: dict[str, Any]) -> None:
    with pytest.raises(ProtocolValidationError):
        parse_server_message(
            item_message(
                {
                    "id": "assistant-1",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "model": {"provider": "test", "id": "model"},
                    "timestamp": 1,
                    **state,
                }
            )
        )


@pytest.mark.parametrize(
    "state",
    [
        {"status": "running", "isError": False},
        {"status": "complete", "isError": False},
        {"status": "error", "isError": True},
    ],
)
def test_accepts_a_consistent_tool_item(state: dict[str, Any]) -> None:
    message = item_message(
        {
            "id": "tool-1",
            "role": "tool",
            "toolCallId": "call-1",
            "toolName": "read",
            "input": {},
            "content": [],
            "timestamp": 1,
            **state,
        },
        "item_updated" if state["status"] == "running" else "item_finished",
    )
    assert parse_server_message(message) == message


def test_rejects_nonterminal_items_reported_as_finished() -> None:
    assistant = {
        "id": "assistant-1",
        "role": "assistant",
        "content": [],
        "model": {"provider": "test", "id": "model"},
        "status": "streaming",
        "timestamp": 1,
    }
    tool = {
        "id": "tool-1",
        "role": "tool",
        "toolCallId": "call-1",
        "toolName": "read",
        "input": {},
        "content": [],
        "status": "running",
        "isError": False,
        "timestamp": 1,
    }

    with pytest.raises(ProtocolValidationError):
        parse_server_message(item_message(assistant))
    with pytest.raises(ProtocolValidationError):
        parse_server_message(item_message(tool))


@pytest.mark.parametrize(
    "state",
    [
        {"status": "running", "isError": True},
        {"status": "complete", "isError": True},
        {"status": "error", "isError": False},
    ],
)
def test_rejects_an_inconsistent_tool_item(state: dict[str, Any]) -> None:
    with pytest.raises(ProtocolValidationError):
        parse_server_message(
            item_message(
                {
                    "id": "tool-1",
                    "role": "tool",
                    "toolCallId": "call-1",
                    "toolName": "read",
                    "input": {},
                    "content": [],
                    "timestamp": 1,
                    **state,
                }
            )
        )


def test_rejects_cyclic_protocol_values_with_a_protocol_validation_error() -> None:
    details: dict[str, Any] = {}
    details["self"] = details
    message = {
        "type": "response",
        "id": "request-1",
        "ok": False,
        "error": {"code": "invalid_request", "message": "invalid", "details": details},
    }

    with pytest.raises(ProtocolValidationError):
        parse_server_message(message)


def test_validation_errors_do_not_retain_rejected_payloads() -> None:
    thrown: BaseException | None = None
    try:
        parse_client_message({"type": "hello", "version": str(PROTOCOL_VERSION), "extra": "x" * 2_000_000})
    except Exception as error:
        thrown = error
    assert isinstance(thrown, ProtocolValidationError)
    assert not hasattr(thrown, "value")
    assert len(str(thrown)) < 1_000


def test_encodes_complete_client_and_server_frames() -> None:
    client_frames = FrameDecoder().push(encode_client_message(CLIENT_HELLO))
    assert len(client_frames) == 1
    assert parse_client_message(decode_cbor(client_frames[0])) == CLIENT_HELLO

    server_frames = FrameDecoder().push(encode_server_message(SERVER_HELLO))
    assert len(server_frames) == 1
    assert parse_server_message(decode_cbor(server_frames[0])) == SERVER_HELLO


def test_enforces_an_outbound_frame_limit_before_returning_encoded_bytes() -> None:
    with pytest.raises(ProtocolValidationError):
        encode_client_message(CLIENT_HELLO, max_frame_length=8)
    with pytest.raises(ProtocolValidationError):
        encode_server_message(SERVER_HELLO, max_frame_length=8)


def test_validates_messages_before_encoding() -> None:
    with pytest.raises(ProtocolValidationError):
        encode_client_message({"type": "hello", "version": PROTOCOL_VERSION + 0.5})


# TypeScript strips properties explicitly set to `undefined` before encoding.
# Python has no `undefined`: an absent optional key is simply absent, and an
# explicit `None` is a JSON null that the schema rejects. Both halves are
# asserted here.
def test_absent_optional_properties_are_absent_on_the_wire() -> None:
    message: dict[str, Any] = {
        "type": "request",
        "id": "request-1",
        "request": {"command": "create"},
    }
    payload = FrameDecoder().push(encode_client_message(message))[0]
    assert decode_cbor(payload) == {
        "type": "request",
        "id": "request-1",
        "request": {"command": "create"},
    }

    with pytest.raises(ProtocolValidationError):
        encode_client_message(
            {
                "type": "request",
                "id": "request-1",
                "request": {"command": "create", "cwd": None, "name": None},
            }
        )


def test_incrementally_decodes_fragmented_and_coalesced_client_messages() -> None:
    request: dict[str, Any] = {"type": "request", "id": "request-1", "request": {"command": "list"}}
    wire = encode_client_message(CLIENT_HELLO) + encode_client_message(request)

    for split in range(len(wire) + 1):
        decoder = ClientMessageDecoder()
        messages = [*decoder.push(wire[:split]), *decoder.push(wire[split:])]
        decoder.end()
        assert messages == [CLIENT_HELLO, request]


def test_incrementally_decodes_server_messages() -> None:
    error_message: dict[str, Any] = {
        "type": "hello_error",
        "error": {"code": "version", "message": "Unsupported protocol version"},
    }
    decoder = ServerMessageDecoder()
    assert decoder.push(encode_server_message(error_message)) == [error_message]
    decoder.end()


@pytest.mark.parametrize(
    ("label", "wire"),
    [
        ("empty CBOR payload", encode_frame(b"")),
        ("malformed CBOR", encode_frame(bytes([0xFF]))),
        (
            "schema-invalid CBOR",
            encode_frame(encode_cbor({"type": "hello", "version": PROTOCOL_VERSION, "extra": True})),
        ),
    ],
)
def test_rejects_invalid_framed_client_input(label: str, wire: bytes) -> None:
    decoder = ClientMessageDecoder()
    with pytest.raises(ProtocolValidationError):
        decoder.push(wire)
    with pytest.raises(ProtocolValidationError, match=r"(?i)failed"):
        decoder.push(encode_client_message(CLIENT_HELLO))


def test_rejects_cbor_byte_strings_nested_in_json_valued_fields() -> None:
    wire = encode_frame(
        encode_cbor(
            {
                "type": "response",
                "id": "request-1",
                "ok": False,
                "error": {
                    "code": "invalid_request",
                    "message": "invalid",
                    "details": {"nested": bytes([1, 2, 3])},
                },
            }
        )
    )
    with pytest.raises(ProtocolValidationError):
        ServerMessageDecoder().push(wire)


def test_rejects_truncated_and_oversized_framing_through_the_validated_decoder() -> None:
    truncated = ServerMessageDecoder()
    assert truncated.push(bytes([0, 0, 0, 2, 1])) == []
    with pytest.raises(ProtocolValidationError):
        truncated.end()

    oversized = ClientMessageDecoder(max_frame_length=3)
    with pytest.raises(ProtocolValidationError):
        oversized.push(bytes([0, 0, 0, 4]))
