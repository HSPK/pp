"""Length-prefixed frame encoding and incremental decoding.

Python port of `packages/protocol/src/framing.ts`. Every protocol message is a
CBOR payload prefixed with its unsigned 32-bit big-endian byte length.
"""

from __future__ import annotations

FRAME_HEADER_LENGTH = 4
MAX_UINT32 = 0xFFFF_FFFF
DEFAULT_MAX_FRAME_LENGTH = 16 * 1024 * 1024
"""Default upper bound for one framed CBOR payload."""


class FrameError(Exception):
    """A frame is malformed, truncated, or exceeds the configured limit."""


def _resolve_max_frame_length(max_frame_length: int | None) -> int:
    value = DEFAULT_MAX_FRAME_LENGTH if max_frame_length is None else max_frame_length
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > MAX_UINT32:
        raise ValueError(f"max_frame_length must be an integer between 0 and {MAX_UINT32}")
    return value


def encode_frame(payload: bytes) -> bytes:
    """Prefix ``payload`` with its unsigned 32-bit big-endian byte length."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("Frame payload must be bytes")
    payload = bytes(payload)
    if len(payload) > MAX_UINT32:
        raise ValueError("Frame payload exceeds the unsigned 32-bit length limit")
    return len(payload).to_bytes(FRAME_HEADER_LENGTH, "big") + payload


def assert_complete_frame(frame: bytes, max_frame_length: int | None = None) -> None:
    """Validate that ``frame`` holds exactly one complete payload within the limit."""
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise TypeError("Frame must be bytes")
    frame = bytes(frame)
    if len(frame) < FRAME_HEADER_LENGTH:
        raise FrameError("Frame does not contain a complete length prefix")
    length = int.from_bytes(frame[:FRAME_HEADER_LENGTH], "big")
    limit = _resolve_max_frame_length(max_frame_length)
    if length > limit:
        raise FrameError(f"Frame length {length} exceeds configured limit of {limit}")
    if len(frame) != FRAME_HEADER_LENGTH + length:
        raise FrameError("Frame must contain exactly one complete payload")


class FrameDecoder:
    """Incrementally splits arbitrary byte chunks into length-prefixed payloads.

    Payload bytes are accumulated as received slices and joined only once the
    frame is complete, so a declared length cannot force a large allocation
    before the bytes actually arrive.
    """

    def __init__(self, max_frame_length: int | None = None) -> None:
        self._max_frame_length = _resolve_max_frame_length(max_frame_length)
        self._header = bytearray()
        self._payload_chunks: list[bytes] = []
        self._expected_payload_length: int | None = None
        self._payload_length = 0
        self._state = "open"

    def push(self, chunk: bytes) -> list[bytes]:
        """Feed bytes in and return every payload that became complete."""
        if self._state == "ended":
            raise FrameError("Frame decoder has ended")
        if self._state == "failed":
            raise FrameError("Frame decoder has failed")
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("Frame chunk must be bytes")
        chunk = bytes(chunk)

        frames: list[bytes] = []
        offset = 0
        while offset < len(chunk):
            if self._expected_payload_length is None:
                header_bytes = min(FRAME_HEADER_LENGTH - len(self._header), len(chunk) - offset)
                self._header.extend(chunk[offset : offset + header_bytes])
                offset += header_bytes
                if len(self._header) < FRAME_HEADER_LENGTH:
                    continue

                frame_length = int.from_bytes(self._header, "big")
                self._header.clear()
                if frame_length > self._max_frame_length:
                    self._fail(f"Frame length {frame_length} exceeds configured limit of {self._max_frame_length}")
                if frame_length == 0:
                    frames.append(b"")
                    continue
                self._expected_payload_length = frame_length
                self._payload_chunks = []
                self._payload_length = 0

            expected = self._expected_payload_length
            if expected is None:
                continue

            if offset < len(chunk) and self._payload_length < expected:
                payload_bytes = min(expected - self._payload_length, len(chunk) - offset)
                self._payload_chunks.append(chunk[offset : offset + payload_bytes])
                self._payload_length += payload_bytes
                offset += payload_bytes

            if self._payload_length == expected:
                frames.append(
                    self._payload_chunks[0] if len(self._payload_chunks) == 1 else b"".join(self._payload_chunks)
                )
                self._payload_chunks = []
                self._expected_payload_length = None
                self._payload_length = 0

        return frames

    def end(self) -> None:
        """Assert the stream ended on a frame boundary."""
        if self._state == "ended":
            raise FrameError("Frame decoder has ended")
        if self._state == "failed":
            raise FrameError("Frame decoder has failed")
        if self._header or self._expected_payload_length is not None:
            self._fail("Truncated frame at end of stream")
        self._state = "ended"

    def _fail(self, message: str) -> None:
        self._state = "failed"
        self._header.clear()
        self._payload_chunks = []
        self._expected_payload_length = None
        self._payload_length = 0
        raise FrameError(message)
