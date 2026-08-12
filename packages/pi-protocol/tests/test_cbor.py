"""CBOR tests, anchored on the RFC 8949 Appendix A test vectors.

The vectors are the normative examples from the specification, so they verify
the encoder and decoder against the standard itself rather than against another
implementation.
"""

from __future__ import annotations

import pytest
from pi_protocol.cbor import (
    DEFAULT_MAX_CBOR_BYTE_LENGTH,
    CborError,
    decode_cbor,
    encode_cbor,
    resolve_options,
)

# (value, hex encoding) from RFC 8949 Appendix A, restricted to the subset the
# protocol supports (no tags, no indefinite lengths, no half/single floats).
RFC_VECTORS = [
    (0, "00"),
    (1, "01"),
    (10, "0a"),
    (23, "17"),
    (24, "1818"),
    (25, "1819"),
    (100, "1864"),
    (1000, "1903e8"),
    (1000000, "1a000f4240"),
    (1000000000000, "1b000000e8d4a51000"),
    (-1, "20"),
    (-10, "29"),
    (-100, "3863"),
    (-1000, "3903e7"),
    (1.1, "fb3ff199999999999a"),
    (-4.1, "fbc010666666666666"),
    (False, "f4"),
    (True, "f5"),
    (None, "f6"),
    (b"", "40"),
    (b"\x01\x02\x03\x04", "4401020304"),
    ("", "60"),
    ("a", "6161"),
    ("IETF", "6449455446"),
    ('"\\', "62225c"),
    ("\u00fc", "62c3bc"),
    ("\u6c34", "63e6b0b4"),
    ("\U00010151", "64f0908591"),
    ([], "80"),
    ([1, 2, 3], "83010203"),
    ([1, [2, 3], [4, 5]], "8301820203820405"),
    (list(range(1, 26)), "98190102030405060708090a0b0c0d0e0f101112131415161718181819"),
    ({}, "a0"),
    ({"a": 1, "b": [2, 3]}, "a26161016162820203"),
    (["a", {"b": "c"}], "826161a161626163"),
    (
        {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"},
        "a56161614161626142616361436164614461656145",
    ),
]


@pytest.mark.parametrize(("value", "expected_hex"), RFC_VECTORS, ids=[v[1] for v in RFC_VECTORS])
def test_encodes_rfc_8949_vectors(value, expected_hex):
    assert encode_cbor(value).hex() == expected_hex


@pytest.mark.parametrize(("value", "expected_hex"), RFC_VECTORS, ids=[v[1] for v in RFC_VECTORS])
def test_decodes_rfc_8949_vectors(value, expected_hex):
    decoded = decode_cbor(bytes.fromhex(expected_hex))
    assert decoded == value
    assert type(decoded) is type(value)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -1,
        2**53 - 1,
        -(2**53 - 1),
        0.5,
        -0.5,
        3.141592653589793,
        "",
        "unicode 你好 🎉",
        b"\x00\xff",
        [],
        [1, "two", [3, {"four": 4}]],
        {},
        {"nested": {"deep": {"deeper": [1, 2, 3]}}},
    ],
)
def test_round_trip(value):
    assert decode_cbor(encode_cbor(value)) == value


def test_float_that_is_integral_encodes_as_an_integer():
    # JavaScript has one number type, so 2.0 must encode the same as 2.
    assert encode_cbor(2.0) == encode_cbor(2)
    assert decode_cbor(encode_cbor(2.0)) == 2


def test_negative_zero_stays_a_float():
    assert encode_cbor(-0.0).hex() == "fb8000000000000000"


def test_bool_is_not_treated_as_an_integer():
    assert encode_cbor(True).hex() == "f5"
    assert encode_cbor(1).hex() == "01"


# --------------------------------------------------------------------------
# encoder limits and rejections
# --------------------------------------------------------------------------


def test_rejects_unsupported_types():
    with pytest.raises(CborError, match="Unsupported CBOR value type"):
        encode_cbor({1, 2})
    with pytest.raises(CborError, match="Unsupported CBOR value type"):
        encode_cbor(object())


def test_rejects_non_string_map_keys():
    with pytest.raises(CborError, match="map keys must be strings"):
        encode_cbor({1: "a"})


def test_rejects_non_finite_numbers():
    with pytest.raises(CborError, match="must be finite"):
        encode_cbor(float("inf"))
    with pytest.raises(CborError, match="must be finite"):
        encode_cbor(float("nan"))


def test_rejects_integers_outside_the_javascript_safe_range():
    with pytest.raises(CborError, match="safe JavaScript integers"):
        encode_cbor(2**53)
    with pytest.raises(CborError, match="safe JavaScript integers"):
        encode_cbor(-(2**53))


def test_rejects_cycles_in_lists_and_dicts():
    cyclic_list: list = [1]
    cyclic_list.append(cyclic_list)
    with pytest.raises(CborError, match="must not contain cycles"):
        encode_cbor(cyclic_list)

    cyclic_dict: dict = {}
    cyclic_dict["self"] = cyclic_dict
    with pytest.raises(CborError, match="must not contain cycles"):
        encode_cbor(cyclic_dict)


def test_repeated_sibling_references_are_not_cycles():
    shared = {"a": 1}
    assert decode_cbor(encode_cbor([shared, shared])) == [{"a": 1}, {"a": 1}]


def test_enforces_the_byte_length_limit():
    with pytest.raises(CborError, match="text string length exceeds configured limit"):
        encode_cbor("x" * 100, max_byte_length=10)
    with pytest.raises(CborError, match="byte string length exceeds configured limit"):
        encode_cbor(b"x" * 100, max_byte_length=10)
    with pytest.raises(CborError, match="byte length exceeds configured limit"):
        encode_cbor([1] * 100, max_byte_length=10)


def test_enforces_the_container_length_limit():
    with pytest.raises(CborError, match="array length exceeds configured limit"):
        encode_cbor([1, 2, 3], max_container_length=2)
    with pytest.raises(CborError, match="map length exceeds configured limit"):
        encode_cbor({"a": 1, "b": 2}, max_container_length=1)


def test_enforces_the_depth_limit():
    nested: object = 1
    for _ in range(10):
        nested = [nested]
    with pytest.raises(CborError, match="nesting depth exceeds configured limit"):
        encode_cbor(nested, max_depth=5)


def test_deeply_nested_within_the_limit_is_allowed():
    nested: object = 1
    for _ in range(5):
        nested = [nested]
    assert decode_cbor(encode_cbor(nested, max_depth=10)) == nested


# --------------------------------------------------------------------------
# decoder rejections
# --------------------------------------------------------------------------


def test_rejects_trailing_data():
    with pytest.raises(CborError, match="trailing data"):
        decode_cbor(bytes.fromhex("0101"))


def test_rejects_truncated_payloads():
    with pytest.raises(CborError, match="Truncated"):
        decode_cbor(bytes.fromhex("18"))
    with pytest.raises(CborError, match="Truncated"):
        decode_cbor(bytes.fromhex("4401"))
    with pytest.raises(CborError, match="Truncated"):
        decode_cbor(b"")


def test_rejects_tags():
    # Tag 0 (standard date/time string) applied to a text string.
    with pytest.raises(CborError, match="tags are not supported"):
        decode_cbor(bytes.fromhex("c06161"))


def test_rejects_indefinite_length_items():
    with pytest.raises(CborError, match="Indefinite-length"):
        decode_cbor(bytes.fromhex("9f01ff"))
    with pytest.raises(CborError, match="Indefinite-length"):
        decode_cbor(bytes.fromhex("5f42010243030405ff"))


def test_rejects_break_marker():
    with pytest.raises(CborError, match="break marker"):
        decode_cbor(bytes.fromhex("ff"))


def test_rejects_half_and_single_precision_floats():
    with pytest.raises(CborError, match="Unsupported CBOR simple value"):
        decode_cbor(bytes.fromhex("f93c00"))  # half float 1.0
    with pytest.raises(CborError, match="Unsupported CBOR simple value"):
        decode_cbor(bytes.fromhex("fa47c35000"))  # single float 100000.0


def test_rejects_undefined_and_other_simple_values():
    with pytest.raises(CborError, match="Unsupported CBOR simple value"):
        decode_cbor(bytes.fromhex("f7"))  # undefined
    with pytest.raises(CborError, match="Unsupported CBOR simple value"):
        decode_cbor(bytes.fromhex("f0"))  # simple(16)


def test_rejects_non_string_map_keys_when_decoding():
    with pytest.raises(CborError, match="map keys must be strings"):
        decode_cbor(bytes.fromhex("a10101"))


def test_rejects_duplicate_map_keys():
    with pytest.raises(CborError, match="duplicate key"):
        decode_cbor(bytes.fromhex("a2616101616102"))


def test_rejects_invalid_utf8_text():
    with pytest.raises(CborError, match="invalid UTF-8"):
        decode_cbor(bytes.fromhex("62c328"))


def test_rejects_non_finite_decoded_floats():
    with pytest.raises(CborError, match="must be finite"):
        decode_cbor(bytes.fromhex("fb7ff0000000000000"))  # +Infinity


def test_rejects_decoded_integers_outside_the_safe_range():
    with pytest.raises(CborError, match="outside the safe range"):
        decode_cbor(bytes.fromhex("1bffffffffffffffff"))


def test_rejects_non_bytes_input():
    with pytest.raises(TypeError, match="must be bytes"):
        decode_cbor("not bytes")


def test_enforces_the_decode_byte_limit():
    payload = encode_cbor("x" * 100)
    with pytest.raises(CborError, match="byte length exceeds configured limit"):
        decode_cbor(payload, max_byte_length=10)


def test_decoder_enforces_the_container_limit_without_allocating():
    # Declares a 1,000,000-element array but supplies no elements.
    with pytest.raises(CborError, match="array length exceeds configured limit"):
        decode_cbor(bytes.fromhex("9a000f4240"), max_container_length=10)


# --------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------


def test_resolve_options_defaults():
    options = resolve_options()
    assert options.max_byte_length == DEFAULT_MAX_CBOR_BYTE_LENGTH
    assert options.max_container_length == 1_000_000
    assert options.max_depth == 64


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_byte_length": -1}, "max_byte_length"),
        ({"max_byte_length": 2**32}, "max_byte_length"),
        ({"max_container_length": -1}, "max_container_length"),
        ({"max_depth": 513}, "max_depth"),
    ],
)
def test_resolve_options_rejects_out_of_range_limits(kwargs, message):
    with pytest.raises(ValueError, match=message):
        resolve_options(**kwargs)
