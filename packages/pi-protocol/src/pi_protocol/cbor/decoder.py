"""CBOR decoder.

Python port of `packages/protocol/src/cbor/decoder.ts`. Decodes exactly one
item from the protocol's strict RFC 8949 subset: no tags, no indefinite-length
items, no break markers, and map keys must be unique strings.
"""

from __future__ import annotations

import math
import struct
from typing import Any

from .options import (
    MAX_SAFE_INTEGER,
    UINT32_BASE,
    CborError,
    ResolvedCborOptions,
    resolve_options,
)


class _CborReader:
    def __init__(self, data: bytes, options: ResolvedCborOptions) -> None:
        self._data = data
        self._offset = 0
        self._options = options

    def decode(self) -> Any:
        value = self._read_item(0)
        if self._offset != len(self._data):
            raise CborError("CBOR payload contains trailing data")
        return value

    def _read_item(self, depth: int) -> Any:
        if depth > self._options.max_depth:
            raise CborError(f"CBOR nesting depth exceeds configured limit of {self._options.max_depth}")
        initial = self._read_byte()
        major_type = initial >> 5
        additional = initial & 0x1F

        if major_type == 0:
            return self._read_argument(additional)
        if major_type == 1:
            value = -1 - self._read_argument(additional)
            if abs(value) > MAX_SAFE_INTEGER:
                raise CborError("Decoded CBOR integer is outside the safe range")
            return value
        if major_type == 2:
            length = self._read_length(additional, "byte string", self._options.max_byte_length)
            return self._read_bytes(length)
        if major_type == 3:
            length = self._read_length(additional, "text string", self._options.max_byte_length)
            raw = self._read_bytes(length)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError as error:
                raise CborError("CBOR text string contains invalid UTF-8") from error
        if major_type == 4:
            length = self._read_length(additional, "array", self._options.max_container_length)
            return [self._read_item(depth + 1) for _ in range(length)]
        if major_type == 5:
            length = self._read_length(additional, "map", self._options.max_container_length)
            result: dict[str, Any] = {}
            for _ in range(length):
                key = self._read_item(depth + 1)
                if not isinstance(key, str):
                    raise CborError("CBOR map keys must be strings")
                if key in result:
                    raise CborError("CBOR map contains a duplicate key")
                result[key] = self._read_item(depth + 1)
            return result
        if major_type == 6:
            raise CborError("CBOR tags are not supported")
        if major_type == 7:
            return self._read_simple(additional)
        raise CborError("Malformed CBOR major type")

    def _read_simple(self, additional: int) -> Any:
        if additional == 20:
            return False
        if additional == 21:
            return True
        if additional == 22:
            return None
        if additional == 27:
            value = struct.unpack(">d", self._read_bytes(8))[0]
            if not math.isfinite(value):
                raise CborError("Decoded CBOR number must be finite")
            if value.is_integer() and abs(value) > MAX_SAFE_INTEGER:
                raise CborError("Decoded CBOR integer is outside the safe range")
            return value
        if additional == 31:
            raise CborError("CBOR break marker is not supported")
        raise CborError("Unsupported CBOR simple value or floating-point width")

    def _read_length(self, additional: int, kind: str, limit: int) -> int:
        if additional == 31:
            raise CborError(f"Indefinite-length CBOR {kind}s are not supported")
        length = self._read_argument(additional)
        if length > limit:
            raise CborError(f"CBOR {kind} length exceeds configured limit of {limit}")
        return length

    def _read_argument(self, additional: int) -> int:
        if additional < 24:
            return additional
        if additional == 24:
            return self._read_byte()
        if additional == 25:
            return int.from_bytes(self._read_bytes(2), "big")
        if additional == 26:
            return int.from_bytes(self._read_bytes(4), "big")
        if additional == 27:
            high = self._read_argument(26)
            low = self._read_argument(26)
            if high > 0x1F_FFFF:
                raise CborError("Decoded CBOR integer or length is outside the safe range")
            return high * UINT32_BASE + low
        if additional == 31:
            raise CborError("Indefinite-length CBOR items are not supported")
        raise CborError("Malformed CBOR additional information")

    def _read_byte(self) -> int:
        if self._offset >= len(self._data):
            raise CborError("Truncated CBOR payload")
        value = self._data[self._offset]
        self._offset += 1
        return value

    def _read_bytes(self, length: int) -> bytes:
        if length > len(self._data) - self._offset:
            raise CborError("Truncated CBOR payload")
        value = self._data[self._offset : self._offset + length]
        self._offset += length
        return value


def decode_cbor(
    data: bytes,
    max_byte_length: int | None = None,
    max_container_length: int | None = None,
    max_depth: int | None = None,
) -> Any:
    """Decode exactly one item from the protocol's strict RFC 8949 subset."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("CBOR input must be bytes")
    data = bytes(data)
    options = resolve_options(max_byte_length, max_container_length, max_depth)
    if len(data) > options.max_byte_length:
        raise CborError(f"CBOR byte length exceeds configured limit of {options.max_byte_length}")
    return _CborReader(data, options).decode()
