"""Python port of `packages/protocol/test/framing.test.ts`.

Named ``test_framing_port`` because ``test_framing.py`` already holds this
port's own framing suite; this file is the direct translation of the upstream
test.
"""

from __future__ import annotations

import math

import pytest
from pi_protocol.framing import (
    DEFAULT_MAX_FRAME_LENGTH,
    FrameDecoder,
    FrameError,
    assert_complete_frame,
    encode_frame,
)


def test_prefixes_payloads_with_a_four_byte_big_endian_length() -> None:
    assert encode_frame(bytes([0xAA, 0xBB, 0xCC])) == bytes([0x00, 0x00, 0x00, 0x03, 0xAA, 0xBB, 0xCC])
    assert encode_frame(b"") == bytes([0, 0, 0, 0])


def test_validates_one_complete_bounded_frame() -> None:
    assert_complete_frame(bytes([0, 0, 0, 2, 1, 2]), max_frame_length=2)
    with pytest.raises(FrameError, match=r"(?i)complete"):
        assert_complete_frame(bytes([0, 0, 0, 2, 1]))
    with pytest.raises(FrameError, match=r"(?i)exactly"):
        assert_complete_frame(bytes([0, 0, 0, 1, 1, 2]))
    with pytest.raises(FrameError, match=r"(?i)limit"):
        assert_complete_frame(bytes([0, 0, 0, 3, 1, 2, 3]), max_frame_length=2)


def test_decodes_fragmented_coalesced_and_empty_frames_in_order() -> None:
    wire = encode_frame(bytes([1, 2, 3])) + encode_frame(b"") + encode_frame(bytes([4]))
    decoder = FrameDecoder()
    frames: list[bytes] = []
    for byte in wire:
        frames.extend(decoder.push(bytes([byte])))
    decoder.end()
    assert frames == [bytes([1, 2, 3]), b"", bytes([4])]

    coalesced = FrameDecoder()
    assert coalesced.push(wire) == frames
    coalesced.end()


def test_assembles_payloads_spanning_multiple_internal_blocks() -> None:
    payload = bytes(index % 251 for index in range(70_000))
    wire = encode_frame(payload)
    decoder = FrameDecoder()
    frames = [
        *decoder.push(wire[:101]),
        *decoder.push(wire[101:65_541]),
        *decoder.push(wire[65_541:]),
    ]
    decoder.end()
    assert frames == [payload]


def test_handles_every_split_point_across_a_frame() -> None:
    wire = encode_frame(bytes([10, 20, 30, 40]))
    for split in range(len(wire) + 1):
        decoder = FrameDecoder()
        frames = [*decoder.push(wire[:split]), *decoder.push(wire[split:])]
        decoder.end()
        assert frames == [bytes([10, 20, 30, 40])]


def test_copies_payload_bytes_instead_of_aliasing_input_chunks() -> None:
    chunk = bytearray(encode_frame(bytes([1, 2, 3])))
    decoder = FrameDecoder()
    frames = decoder.push(chunk)
    for index in range(len(chunk)):
        chunk[index] = 9
    assert frames == [bytes([1, 2, 3])]
    # `bytes` and `bytearray` compare equal in Python, so the immutable type is
    # pinned explicitly: a returned `bytearray` view would still satisfy `==`
    # while aliasing the decoder's buffer.
    assert [type(frame) for frame in frames] == [bytes]


def test_accepts_empty_chunks_and_a_clean_empty_stream() -> None:
    decoder = FrameDecoder()
    assert decoder.push(b"") == []
    decoder.end()


@pytest.mark.parametrize(
    ("label", "wire"),
    [("partial header", bytes([0, 0, 0])), ("partial payload", bytes([0, 0, 0, 2, 1]))],
)
def test_rejects_a_truncated_stream_at_end(label: str, wire: bytes) -> None:
    decoder = FrameDecoder()
    assert decoder.push(wire) == []
    with pytest.raises(FrameError):
        decoder.end()


def test_rejects_an_oversized_declared_length_as_soon_as_its_header_is_complete() -> None:
    decoder = FrameDecoder(max_frame_length=3)
    with pytest.raises(FrameError, match=r"(?i)limit"):
        decoder.push(bytes([0, 0, 0, 4]))
    with pytest.raises(FrameError, match=r"(?i)failed"):
        decoder.push(bytes([1]))


def test_accepts_a_frame_exactly_at_the_configured_maximum() -> None:
    decoder = FrameDecoder(max_frame_length=3)
    assert decoder.push(encode_frame(bytes([1, 2, 3]))) == [bytes([1, 2, 3])]
    decoder.end()


def test_cannot_be_pushed_after_end() -> None:
    decoder = FrameDecoder()
    decoder.end()
    with pytest.raises(FrameError, match=r"(?i)ended"):
        decoder.push(b"")
    with pytest.raises(FrameError, match=r"(?i)ended"):
        decoder.end()


# TypeScript throws `RangeError`; the Python port raises `ValueError`, which is
# the standard-library analogue for an out-of-range argument.
@pytest.mark.parametrize("max_frame_length", [-1, 1.5, math.nan, DEFAULT_MAX_FRAME_LENGTH * 1_000])
def test_rejects_invalid_maximum_frame_length(max_frame_length: float) -> None:
    with pytest.raises(ValueError):
        FrameDecoder(max_frame_length=max_frame_length)
