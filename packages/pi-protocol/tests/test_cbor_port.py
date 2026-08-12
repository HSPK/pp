"""Python port of `packages/protocol/test/cbor/cbor.test.ts`.

Named ``test_cbor_port`` because ``test_cbor.py`` already holds this port's own
RFC 8949 Appendix A suite; this file is the direct translation of the upstream
test so both sides can be compared case by case.
"""

from __future__ import annotations

import datetime
import math
from typing import Any

import pytest
from pi_protocol.cbor import (
    DEFAULT_MAX_CBOR_BYTE_LENGTH,
    DEFAULT_MAX_CBOR_CONTAINER_LENGTH,
    DEFAULT_MAX_CBOR_DEPTH,
    CborError,
    decode_cbor,
    encode_cbor,
)

MAX_SAFE_INTEGER = 2**53 - 1
MIN_SAFE_INTEGER = -(2**53) + 1


def from_hex(hex_text: str) -> bytes:
    if len(hex_text) % 2 != 0:
        raise ValueError("Hex fixture must contain whole bytes")
    return bytes.fromhex(hex_text)


def to_hex(data: bytes) -> str:
    return data.hex()


KNOWN_VECTORS: list[tuple[object, str]] = [
    (None, "f6"),
    (False, "f4"),
    (True, "f5"),
    (0, "00"),
    (1, "01"),
    (10, "0a"),
    (23, "17"),
    (24, "1818"),
    (25, "1819"),
    (100, "1864"),
    (1000, "1903e8"),
    (1_000_000, "1a000f4240"),
    (1_000_000_000_000, "1b000000e8d4a51000"),
    (MAX_SAFE_INTEGER, "1b001fffffffffffff"),
    (-1, "20"),
    (-10, "29"),
    (-24, "37"),
    (-25, "3818"),
    (-100, "3863"),
    (-1000, "3903e7"),
    (-1_000_000, "3a000f423f"),
    (MIN_SAFE_INTEGER, "3b001ffffffffffffe"),
    (1.1, "fb3ff199999999999a"),
    (-0.0, "fb8000000000000000"),
    (bytes([1, 2, 3, 4]), "4401020304"),
    ("", "60"),
    ("IETF", "6449455446"),
    ("\u00fc", "62c3bc"),
    ("\u6c34", "63e6b0b4"),
    ("\U00010151", "64f0908591"),
    ([], "80"),
    ([1, 2, 3], "83010203"),
    ([1, [2, 3], [4, 5]], "8301820203820405"),
    ({"a": 1, "b": [2, 3]}, "a26161016162820203"),
]


def _is_negative_zero(value: object) -> bool:
    return isinstance(value, float) and value == 0.0 and math.copysign(1.0, value) < 0


@pytest.mark.parametrize(("value", "wire"), KNOWN_VECTORS)
def test_encodes_and_decodes_rfc_8949_vector(value: object, wire: str) -> None:
    assert to_hex(encode_cbor(value)) == wire
    decoded = decode_cbor(from_hex(wire))
    if _is_negative_zero(value):
        assert _is_negative_zero(decoded)
    else:
        # `toEqual` in TypeScript distinguishes `0` from `false`; in Python
        # `0 == False`, so the decoded type is asserted as well or the bool
        # vectors would pass against an integer decoder.
        assert type(decoded) is type(value)
        assert decoded == value


# The TypeScript case "omits undefined object properties without omitting
# falsey values" has no Python analogue: JavaScript distinguishes a property
# explicitly set to `undefined` from a missing one, Python does not. The
# falsey half of the assertion is kept, since that is the part that matters on
# the wire.
def test_encodes_falsey_values_rather_than_omitting_them() -> None:
    value = {"zero": 0, "empty": "", "no": False, "nil": None}
    decoded = decode_cbor(encode_cbor(value))
    assert decoded == value
    # `==` on dicts would accept `{"no": 0}` for `{"no": False}`, which the
    # TypeScript `toEqual` would not, so the value types are pinned too.
    assert [type(decoded[key]) for key in value] == [int, str, bool, type(None)]
    assert list(decoded) == ["zero", "empty", "no", "nil"]


def test_preserves_a_leading_unicode_bom_and_treats_dunder_proto_as_data() -> None:
    assert decode_cbor(from_hex("63efbbbf")) == "\ufeff"
    value = {"__proto__": "safe"}
    decoded = decode_cbor(encode_cbor(value))
    assert decoded == value
    assert type(decoded) is dict
    assert "__proto__" in decoded


class _Unsupported:
    pass


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("NaN", math.nan),
        ("positive infinity", math.inf),
        ("negative infinity", -math.inf),
        ("unsafe positive integer", MAX_SAFE_INTEGER + 1),
        ("unsafe negative integer", MIN_SAFE_INTEGER - 1),
        ("set", set()),
        ("class instance", _Unsupported()),
        ("function", lambda: None),
        ("datetime", datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)),
        ("complex", complex(1, 2)),
    ],
)
def test_rejects_unsupported_encoder_value(label: str, value: object) -> None:
    with pytest.raises(CborError):
        encode_cbor(value)


# TypeScript rejects a map with enumerable symbol keys; Python's analogue of a
# key that is not a string is any non-``str`` mapping key.
def test_rejects_maps_with_non_string_keys() -> None:
    with pytest.raises(CborError):
        encode_cbor({"valid": True, 1: False})


def test_rejects_lossy_strings_cycles_and_excessive_encoder_depth() -> None:
    with pytest.raises(CborError, match=r"(?i)unicode"):
        encode_cbor("\ud800")

    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(CborError, match=r"(?i)cycles"):
        encode_cbor(cyclic)

    too_deep: object = None
    for _ in range(DEFAULT_MAX_CBOR_DEPTH + 1):
        too_deep = [too_deep]
    with pytest.raises(CborError, match=r"(?i)depth"):
        encode_cbor(too_deep)


@pytest.mark.parametrize(
    ("label", "wire"),
    [
        ("empty input", ""),
        ("truncated integer", "18"),
        ("reserved additional information", "1c"),
        ("indefinite byte string", "5f"),
        ("indefinite text string", "7f"),
        ("indefinite array", "9f"),
        ("indefinite map", "bf"),
        ("tag", "c000"),
        ("undefined", "f7"),
        ("unsupported simple value", "e0"),
        ("break outside an indefinite item", "ff"),
        ("float16", "f93c00"),
        ("float32", "fa3f800000"),
        ("positive infinity", "fb7ff0000000000000"),
        ("NaN", "fb7ff8000000000000"),
        ("truncated float64", "fb3ff00000"),
        ("truncated byte string", "44010203"),
        ("truncated text string", "636162"),
        ("truncated array", "8201"),
        ("truncated map", "a16161"),
        ("trailing data", "0000"),
        ("non-string map key", "a10102"),
        ("duplicate map key", "a2616101616102"),
        ("invalid UTF-8 byte", "61ff"),
        ("overlong UTF-8", "62c080"),
        ("UTF-8 surrogate", "63eda080"),
        ("unsafe positive integer", "1b0020000000000000"),
        ("unsafe negative integer", "3b001fffffffffffff"),
        ("unsafe integer encoded as float64", "fb4340000000000000"),
    ],
)
def test_rejects_invalid_decoder_input(label: str, wire: str) -> None:
    with pytest.raises(CborError):
        decode_cbor(from_hex(wire))


def test_enforces_depth_and_declared_length_limits_before_traversing_values() -> None:
    too_deep = bytearray(DEFAULT_MAX_CBOR_DEPTH + 2)
    for index in range(len(too_deep) - 1):
        too_deep[index] = 0x81
    too_deep[-1] = 0xF6
    with pytest.raises(CborError, match=r"(?i)depth"):
        decode_cbor(bytes(too_deep))

    oversized_bytes = from_hex(f"5a{DEFAULT_MAX_CBOR_BYTE_LENGTH + 1:08x}")
    oversized_text = from_hex(f"7a{DEFAULT_MAX_CBOR_BYTE_LENGTH + 1:08x}")
    oversized_array = from_hex(f"9a{DEFAULT_MAX_CBOR_CONTAINER_LENGTH + 1:08x}")
    oversized_map = from_hex(f"ba{DEFAULT_MAX_CBOR_CONTAINER_LENGTH + 1:08x}")
    for wire in (oversized_bytes, oversized_text, oversized_array, oversized_map):
        with pytest.raises(CborError, match=r"(?i)limit"):
            decode_cbor(wire)


def test_supports_stricter_caller_provided_limits() -> None:
    with pytest.raises(CborError, match=r"(?i)limit"):
        decode_cbor(from_hex("83010203"), max_container_length=2)
    with pytest.raises(CborError, match=r"(?i)limit"):
        decode_cbor(from_hex("626162"), max_byte_length=2)
    with pytest.raises(CborError, match=r"(?i)limit"):
        encode_cbor([1, 2, 3], max_container_length=2)
    with pytest.raises(CborError, match=r"(?i)limit"):
        encode_cbor("ab", max_byte_length=2)


def test_round_trips_nested_structures() -> None:
    value: dict[str, Any] = {"a": [1, {"b": None}], "c": b"\x00\xff"}
    assert decode_cbor(encode_cbor(value)) == value
