"""RPC protocol primitives (Python port of ``@earendil-works/pi-protocol``)."""

from __future__ import annotations

from .cbor import CborError, decode_cbor, encode_cbor
from .codec import (
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
from .framing import (
    DEFAULT_MAX_FRAME_LENGTH,
    FRAME_HEADER_LENGTH,
    FrameDecoder,
    FrameError,
    assert_complete_frame,
    encode_frame,
)
from .schemas import (
    CLIENT_MESSAGE_SCHEMA,
    COMMAND_NAMES,
    PROTOCOL_ERROR_CODES,
    PROTOCOL_VERSION,
    SERVER_MESSAGE_SCHEMA,
    SESSION_PHASES,
    THINKING_LEVELS,
)

__all__ = [
    "CLIENT_MESSAGE_SCHEMA",
    "COMMAND_NAMES",
    "DEFAULT_MAX_FRAME_LENGTH",
    "FRAME_HEADER_LENGTH",
    "PROTOCOL_ERROR_CODES",
    "PROTOCOL_VERSION",
    "SERVER_MESSAGE_SCHEMA",
    "SESSION_PHASES",
    "THINKING_LEVELS",
    "CborError",
    "ClientMessageDecoder",
    "FrameDecoder",
    "FrameError",
    "ProtocolValidationError",
    "ServerMessageDecoder",
    "assert_complete_frame",
    "create_client_message_decoder",
    "create_server_message_decoder",
    "decode_cbor",
    "encode_cbor",
    "encode_client_message",
    "encode_frame",
    "encode_server_message",
    "is_protocol_value",
    "is_supported_protocol_version",
    "parse_client_message",
    "parse_server_message",
]
