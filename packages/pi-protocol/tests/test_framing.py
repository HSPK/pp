import pytest
from pi_protocol.framing import (
    DEFAULT_MAX_FRAME_LENGTH,
    FRAME_HEADER_LENGTH,
    FrameDecoder,
    FrameError,
    assert_complete_frame,
    encode_frame,
)


def test_encode_frame_prefixes_a_big_endian_length():
    assert encode_frame(b"hi") == b"\x00\x00\x00\x02hi"
    assert encode_frame(b"") == b"\x00\x00\x00\x00"


def test_encode_frame_handles_a_length_needing_all_four_bytes():
    payload = b"x" * 300
    frame = encode_frame(payload)
    assert frame[:FRAME_HEADER_LENGTH] == (300).to_bytes(4, "big")
    assert frame[FRAME_HEADER_LENGTH:] == payload


def test_encode_frame_rejects_non_bytes():
    with pytest.raises(TypeError, match="must be bytes"):
        encode_frame("hi")


def test_assert_complete_frame_accepts_exactly_one_frame():
    assert_complete_frame(encode_frame(b"payload"))
    assert_complete_frame(encode_frame(b""))


def test_assert_complete_frame_rejects_a_short_header():
    with pytest.raises(FrameError, match="complete length prefix"):
        assert_complete_frame(b"\x00\x00")


def test_assert_complete_frame_rejects_extra_or_missing_bytes():
    with pytest.raises(FrameError, match="exactly one complete payload"):
        assert_complete_frame(encode_frame(b"hi") + b"extra")
    with pytest.raises(FrameError, match="exactly one complete payload"):
        assert_complete_frame(encode_frame(b"hello")[:-1])


def test_assert_complete_frame_enforces_the_limit():
    with pytest.raises(FrameError, match="exceeds configured limit"):
        assert_complete_frame(encode_frame(b"x" * 10), max_frame_length=5)


def test_assert_complete_frame_rejects_non_bytes():
    with pytest.raises(TypeError, match="must be bytes"):
        assert_complete_frame("nope")


def test_decoder_splits_concatenated_frames():
    decoder = FrameDecoder()
    stream = encode_frame(b"one") + encode_frame(b"") + encode_frame(b"three")
    assert decoder.push(stream) == [b"one", b"", b"three"]
    decoder.end()


def test_decoder_reassembles_frames_split_byte_by_byte():
    decoder = FrameDecoder()
    stream = encode_frame(b"hello") + encode_frame(b"world")
    frames = []
    for index in range(len(stream)):
        frames.extend(decoder.push(stream[index : index + 1]))
    decoder.end()
    assert frames == [b"hello", b"world"]


def test_decoder_handles_a_header_split_across_chunks():
    decoder = FrameDecoder()
    stream = encode_frame(b"payload")
    assert decoder.push(stream[:2]) == []
    assert decoder.push(stream[2:6]) == []
    assert decoder.push(stream[6:]) == [b"payload"]


def test_decoder_handles_a_payload_larger_than_one_chunk():
    payload = bytes(range(256)) * 500
    decoder = FrameDecoder()
    stream = encode_frame(payload)
    frames = []
    for index in range(0, len(stream), 997):
        frames.extend(decoder.push(stream[index : index + 997]))
    decoder.end()
    assert frames == [payload]


def test_decoder_returns_nothing_for_an_empty_chunk():
    decoder = FrameDecoder()
    assert decoder.push(b"") == []


def test_decoder_rejects_a_frame_over_the_limit():
    decoder = FrameDecoder(max_frame_length=4)
    with pytest.raises(FrameError, match="exceeds configured limit"):
        decoder.push(encode_frame(b"toolong"))


def test_decoder_is_unusable_after_a_failure():
    decoder = FrameDecoder(max_frame_length=4)
    with pytest.raises(FrameError):
        decoder.push(encode_frame(b"toolong"))
    with pytest.raises(FrameError, match="has failed"):
        decoder.push(b"more")
    with pytest.raises(FrameError, match="has failed"):
        decoder.end()


def test_decoder_end_rejects_a_truncated_header():
    decoder = FrameDecoder()
    decoder.push(b"\x00\x00")
    with pytest.raises(FrameError, match="Truncated frame"):
        decoder.end()


def test_decoder_end_rejects_a_truncated_payload():
    decoder = FrameDecoder()
    decoder.push(encode_frame(b"hello")[:-2])
    with pytest.raises(FrameError, match="Truncated frame"):
        decoder.end()


def test_decoder_cannot_be_used_after_end():
    decoder = FrameDecoder()
    decoder.end()
    with pytest.raises(FrameError, match="has ended"):
        decoder.push(b"x")
    with pytest.raises(FrameError, match="has ended"):
        decoder.end()


def test_decoder_rejects_non_bytes_chunks():
    with pytest.raises(TypeError, match="must be bytes"):
        FrameDecoder().push("nope")


@pytest.mark.parametrize("bad_limit", [-1, 2**32, 1.5, True])
def test_decoder_rejects_an_out_of_range_limit(bad_limit):
    with pytest.raises(ValueError, match="max_frame_length"):
        FrameDecoder(max_frame_length=bad_limit)


def test_zero_limit_only_allows_empty_frames():
    decoder = FrameDecoder(max_frame_length=0)
    assert decoder.push(encode_frame(b"")) == [b""]
    with pytest.raises(FrameError, match="exceeds configured limit"):
        decoder.push(encode_frame(b"x"))


def test_default_limit_is_16_mib():
    assert DEFAULT_MAX_FRAME_LENGTH == 16 * 1024 * 1024


def test_round_trip_through_encode_and_decode():
    payloads = [b"", b"a", b"x" * 70_000, bytes(range(256))]
    decoder = FrameDecoder()
    stream = b"".join(encode_frame(payload) for payload in payloads)
    assert decoder.push(stream) == payloads
    decoder.end()
