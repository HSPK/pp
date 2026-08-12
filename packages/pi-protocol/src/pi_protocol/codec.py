"""Validated, framed protocol message codec.

Python port of `packages/protocol/src/codec.ts`: a message is validated against
its schema, encoded as CBOR, and wrapped in a length-prefixed frame. Decoding
reverses that and validates again, so a peer can never inject a value the
schema does not allow.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from .cbor import decode_cbor, encode_cbor
from .framing import (
    DEFAULT_MAX_FRAME_LENGTH,
    FrameDecoder,
    assert_complete_frame,
    encode_frame,
)
from .schemas import CLIENT_MESSAGE_SCHEMA, PROTOCOL_VERSION, SERVER_MESSAGE_SCHEMA

MAX_ERROR_MESSAGE_CHARS = 500


class ProtocolValidationError(Exception):
    """A message does not satisfy the protocol schema, or a frame is malformed."""


_client_validator = Draft202012Validator(CLIENT_MESSAGE_SCHEMA)
_server_validator = Draft202012Validator(SERVER_MESSAGE_SCHEMA)


def is_protocol_value(value: Any, _ancestors: set[int] | None = None) -> bool:
    """Whether ``value`` is a plain, acyclic JSON value.

    Guards against cycles and non-JSON objects before schema validation, which
    would otherwise recurse forever or accept a class instance.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if not isinstance(value, (list, dict)):
        return False

    ancestors = _ancestors if _ancestors is not None else set()
    if id(value) in ancestors:
        return False
    ancestors.add(id(value))
    try:
        if isinstance(value, list):
            return all(is_protocol_value(item, ancestors) for item in value)
        return all(isinstance(key, str) and is_protocol_value(item, ancestors) for key, item in value.items())
    finally:
        ancestors.discard(id(value))


def parse_client_message(value: Any) -> dict[str, Any]:
    """Validate ``value`` as a client message, returning it unchanged."""
    if not is_protocol_value(value) or not _client_validator.is_valid(value):
        raise ProtocolValidationError("Invalid client protocol message")
    return value


def parse_server_message(value: Any) -> dict[str, Any]:
    """Validate ``value`` as a server message, returning it unchanged."""
    if not is_protocol_value(value) or not _server_validator.is_valid(value):
        raise ProtocolValidationError("Invalid server protocol message")
    return value


def _bounded_error_message(error: BaseException) -> str:
    message = str(error)
    if len(message) <= MAX_ERROR_MESSAGE_CHARS:
        return message
    return f"{message[: MAX_ERROR_MESSAGE_CHARS - 3]}..."


def _encode_protocol_message(value: Any, parse: Any, kind: str, max_frame_length: int | None) -> bytes:
    validated = parse(value)
    limit = DEFAULT_MAX_FRAME_LENGTH if max_frame_length is None else max_frame_length
    try:
        frame = encode_frame(encode_cbor(validated, max_byte_length=limit))
        assert_complete_frame(frame, max_frame_length=limit)
        return frame
    except ProtocolValidationError:
        raise
    except Exception as error:
        raise ProtocolValidationError(
            f"Unable to encode {kind} protocol message: {_bounded_error_message(error)}"
        ) from error


def encode_client_message(message: Any, max_frame_length: int | None = None) -> bytes:
    """Validate and encode one complete length-prefixed client message."""
    return _encode_protocol_message(message, parse_client_message, "client", max_frame_length)


def encode_server_message(message: Any, max_frame_length: int | None = None) -> bytes:
    """Validate and encode one complete length-prefixed server message."""
    return _encode_protocol_message(message, parse_server_message, "server", max_frame_length)


class _ValidatedMessageDecoder:
    def __init__(self, kind: str, parse: Any, max_frame_length: int | None = None) -> None:
        self._kind = kind
        self._parse = parse
        self._frames = FrameDecoder(max_frame_length)
        self._max_frame_length = DEFAULT_MAX_FRAME_LENGTH if max_frame_length is None else max_frame_length
        self._failed = False

    def push(self, chunk: bytes) -> list[dict[str, Any]]:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        try:
            return [
                self._parse(decode_cbor(frame, max_byte_length=self._max_frame_length))
                for frame in self._frames.push(chunk)
            ]
        except ProtocolValidationError:
            self._failed = True
            raise
        except Exception as error:
            self._failed = True
            raise ProtocolValidationError(
                f"Invalid {self._kind} protocol frame: {_bounded_error_message(error)}"
            ) from error

    def end(self) -> None:
        if self._failed:
            raise ProtocolValidationError(f"{self._kind} message decoder has failed")
        try:
            self._frames.end()
        except Exception as error:
            self._failed = True
            raise ProtocolValidationError(
                f"Invalid {self._kind} protocol framing: {_bounded_error_message(error)}"
            ) from error


class ClientMessageDecoder:
    """Incrementally decodes and validates framed client messages."""

    def __init__(self, max_frame_length: int | None = None) -> None:
        self._decoder = _ValidatedMessageDecoder("client", parse_client_message, max_frame_length)

    def push(self, chunk: bytes) -> list[dict[str, Any]]:
        return self._decoder.push(chunk)

    def end(self) -> None:
        self._decoder.end()


class ServerMessageDecoder:
    """Incrementally decodes and validates framed server messages."""

    def __init__(self, max_frame_length: int | None = None) -> None:
        self._decoder = _ValidatedMessageDecoder("server", parse_server_message, max_frame_length)

    def push(self, chunk: bytes) -> list[dict[str, Any]]:
        return self._decoder.push(chunk)

    def end(self) -> None:
        self._decoder.end()


def create_client_message_decoder(max_frame_length: int | None = None) -> ClientMessageDecoder:
    return ClientMessageDecoder(max_frame_length)


def create_server_message_decoder(max_frame_length: int | None = None) -> ServerMessageDecoder:
    return ServerMessageDecoder(max_frame_length)


def is_supported_protocol_version(version: Any) -> bool:
    return isinstance(version, int) and not isinstance(version, bool) and version == PROTOCOL_VERSION
