"""CBOR encoder.

Python port of `packages/protocol/src/cbor/encoder.ts`. Encodes the protocol's
strict, definite-length RFC 8949 subset: integers, float64, byte strings, text
strings, arrays, string-keyed maps, booleans and null.
"""

from __future__ import annotations

import math
import struct
from typing import Any

from .options import (
    MAX_SAFE_INTEGER,
    MAX_UINT32,
    CborError,
    ResolvedCborOptions,
    resolve_options,
)


class _CborWriter:
    def __init__(self, max_byte_length: int) -> None:
        self._max_byte_length = max_byte_length
        self._parts: list[bytes] = []
        self._length = 0

    def write(self, data: bytes) -> None:
        if self._length + len(data) > self._max_byte_length:
            raise CborError(f"CBOR byte length exceeds configured limit of {self._max_byte_length}")
        self._parts.append(data)
        self._length += len(data)

    def write_byte(self, value: int) -> None:
        self.write(bytes((value,)))

    def write_float64(self, value: float) -> None:
        self.write(b"\xfb" + struct.pack(">d", value))

    def finish(self) -> bytes:
        return b"".join(self._parts)


def _write_argument(writer: _CborWriter, major_type: int, value: int) -> None:
    prefix = major_type << 5
    if value < 24:
        writer.write_byte(prefix | value)
    elif value <= 0xFF:
        writer.write_byte(prefix | 24)
        writer.write_byte(value)
    elif value <= 0xFFFF:
        writer.write_byte(prefix | 25)
        writer.write(value.to_bytes(2, "big"))
    elif value <= MAX_UINT32:
        writer.write_byte(prefix | 26)
        writer.write(value.to_bytes(4, "big"))
    else:
        writer.write_byte(prefix | 27)
        writer.write(value.to_bytes(8, "big"))


def _encode_text(writer: _CborWriter, value: str, options: ResolvedCborOptions) -> None:
    try:
        data = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CborError("CBOR text strings must contain valid Unicode scalar values") from error
    if len(data) > options.max_byte_length:
        raise CborError(f"CBOR text string length exceeds configured limit of {options.max_byte_length}")
    _write_argument(writer, 3, len(data))
    writer.write(data)


def _is_negative_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) < 0


def _encode_integer(writer: _CborWriter, value: int) -> None:
    if abs(value) > MAX_SAFE_INTEGER:
        raise CborError("CBOR integers must be safe JavaScript integers")
    if value >= 0:
        _write_argument(writer, 0, value)
    else:
        _write_argument(writer, 1, -1 - value)


def _encode_value(
    writer: _CborWriter,
    value: Any,
    options: ResolvedCborOptions,
    depth: int,
    ancestors: set[int],
) -> None:
    if depth > options.max_depth:
        raise CborError(f"CBOR nesting depth exceeds configured limit of {options.max_depth}")

    if value is None:
        writer.write_byte(0xF6)
        return
    if isinstance(value, bool):
        writer.write_byte(0xF5 if value else 0xF4)
        return
    if isinstance(value, int):
        _encode_integer(writer, value)
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CborError("CBOR numbers must be finite")
        if value.is_integer() and not _is_negative_zero(value):
            _encode_integer(writer, int(value))
        else:
            writer.write_float64(value)
        return
    if isinstance(value, str):
        _encode_text(writer, value, options)
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        if len(data) > options.max_byte_length:
            raise CborError(f"CBOR byte string length exceeds configured limit of {options.max_byte_length}")
        _write_argument(writer, 2, len(data))
        writer.write(data)
        return
    if isinstance(value, (list, tuple)):
        if id(value) in ancestors:
            raise CborError("CBOR values must not contain cycles")
        if len(value) > options.max_container_length:
            raise CborError(f"CBOR array length exceeds configured limit of {options.max_container_length}")
        ancestors.add(id(value))
        try:
            _write_argument(writer, 4, len(value))
            for item in value:
                _encode_value(writer, item, options, depth + 1, ancestors)
        finally:
            ancestors.discard(id(value))
        return
    if isinstance(value, dict):
        if id(value) in ancestors:
            raise CborError("CBOR values must not contain cycles")
        for key in value:
            if not isinstance(key, str):
                raise CborError("CBOR map keys must be strings")
        if len(value) > options.max_container_length:
            raise CborError(f"CBOR map length exceeds configured limit of {options.max_container_length}")
        ancestors.add(id(value))
        try:
            _write_argument(writer, 5, len(value))
            for key, entry_value in value.items():
                _encode_text(writer, key, options)
                _encode_value(writer, entry_value, options, depth + 1, ancestors)
        finally:
            ancestors.discard(id(value))
        return

    raise CborError(f"Unsupported CBOR value type: {type(value).__name__}")


def encode_cbor(
    value: Any,
    max_byte_length: int | None = None,
    max_container_length: int | None = None,
    max_depth: int | None = None,
) -> bytes:
    """Encode the protocol's strict, definite-length RFC 8949 subset."""
    options = resolve_options(max_byte_length, max_container_length, max_depth)
    writer = _CborWriter(options.max_byte_length)
    _encode_value(writer, value, options, 0, set())
    return writer.finish()
