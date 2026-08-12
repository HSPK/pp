"""Image MIME-type sniffing by file signature.

Python port of `packages/coding-agent/src/utils/mime.ts`.

Sniffing is deliberately strict: a PNG must carry a valid `IHDR` and must not
be animated (APNG), a JPEG must not be the JPEG-LS variant, and a BMP header
must be internally consistent. Anything else is treated as text, which is what
the read tool and `@file` processing fall back to.
"""

from __future__ import annotations

IMAGE_TYPE_SNIFF_BYTES = 4100
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_BMP_BITS_PER_PIXEL = (1, 4, 8, 16, 24, 32)


def _starts_with_ascii(data: bytes, offset: int, text: str) -> bool:
    return data[offset : offset + len(text)] == text.encode("ascii")


def _read_uint16_le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], "little")


def _read_uint32_be(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "big")


def _read_uint32_le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], "little")


def _is_png(data: bytes) -> bool:
    return len(data) >= 16 and _read_uint32_be(data, len(_PNG_SIGNATURE)) == 13 and _starts_with_ascii(data, 12, "IHDR")


def _is_animated_png(data: bytes) -> bool:
    offset = len(_PNG_SIGNATURE)
    while offset + 8 <= len(data):
        chunk_length = _read_uint32_be(data, offset)
        chunk_type_offset = offset + 4
        if _starts_with_ascii(data, chunk_type_offset, "acTL"):
            return True
        if _starts_with_ascii(data, chunk_type_offset, "IDAT"):
            return False

        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > len(data):
            return False
        offset = next_offset
    return False


def _is_bmp(data: bytes) -> bool:
    if len(data) < 26:
        return False

    declared_file_size = _read_uint32_le(data, 2)
    pixel_data_offset = _read_uint32_le(data, 10)
    dib_header_size = _read_uint32_le(data, 14)
    if declared_file_size != 0 and declared_file_size < 26:
        return False
    if pixel_data_offset < 14 + dib_header_size:
        return False
    if declared_file_size != 0 and pixel_data_offset >= declared_file_size:
        return False

    if dib_header_size == 12:
        color_planes = _read_uint16_le(data, 22)
        bits_per_pixel = _read_uint16_le(data, 24)
    elif 40 <= dib_header_size <= 124:
        if len(data) < 30:
            return False
        color_planes = _read_uint16_le(data, 26)
        bits_per_pixel = _read_uint16_le(data, 28)
    else:
        return False

    return color_planes == 1 and bits_per_pixel in _BMP_BITS_PER_PIXEL


def detect_supported_image_mime_type(data: bytes) -> str | None:
    """Sniff an image MIME type from the leading bytes. `None` when unsupported."""
    if data.startswith(b"\xff\xd8\xff"):
        return None if len(data) > 3 and data[3] == 0xF7 else "image/jpeg"
    if data.startswith(_PNG_SIGNATURE):
        return "image/png" if _is_png(data) and not _is_animated_png(data) else None
    if _starts_with_ascii(data, 0, "GIF"):
        return "image/gif"
    if _starts_with_ascii(data, 0, "RIFF") and _starts_with_ascii(data, 8, "WEBP"):
        return "image/webp"
    if _starts_with_ascii(data, 0, "BM") and _is_bmp(data):
        return "image/bmp"
    return None


def detect_image_mime_type_from_bytes(data: bytes) -> str | None:
    """Alias kept for callers that predate the `detect_supported_image_mime_type` name."""
    return detect_supported_image_mime_type(data)


def detect_supported_image_mime_type_from_file(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            header = f.read(IMAGE_TYPE_SNIFF_BYTES)
    except OSError:
        return None
    return detect_supported_image_mime_type(header)


__all__ = [
    "IMAGE_TYPE_SNIFF_BYTES",
    "detect_image_mime_type_from_bytes",
    "detect_supported_image_mime_type",
    "detect_supported_image_mime_type_from_file",
]
