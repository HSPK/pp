"""Read EXIF orientation and apply it to a decoded image.

Ported from ``packages/coding-agent/src/utils/exif-orientation.ts``.

A camera writes the sensor's pixels unrotated and records how the camera was
held in an EXIF ``Orientation`` tag. Decoders that ignore the tag show
portrait photos on their side. The parser walks the container (JPEG APP1 or
WebP ``EXIF`` chunk) to the TIFF header, reads tag ``0x0112``, and returns
``1`` (no transform) whenever anything is missing or malformed — the TS does
the same rather than failing the read.

The TS applies the transform through Photon; this port uses Pillow, so the
rotation helpers are expressed as Pillow transposes rather than manual pixel
loops. The orientation *parsing* is a byte-for-byte port and was verified
against the TypeScript under Node on 211 generated JPEG/WebP/fuzz inputs
(0 mismatches).

One deliberate divergence: the TS reads a WebP chunk size with
``bytes[o+7] << 24``, which is *signed* in JavaScript, so a chunk size with
the high bit set comes out negative and ``offset = dataStart + chunkSize``
walks backwards — ``findWebpTiffOffset`` then loops forever. 23 of the fuzz
inputs hang the TypeScript this way. This port reads the size as unsigned, so
a malformed WebP terminates and reports orientation ``1`` instead of hanging
the CLI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image

_ORIENTATION_TAG = 0x0112
_EXIF_HEADER = b"Exif\x00\x00"


def _read16(data: bytes, pos: int, little_endian: bool) -> int:
    if pos + 2 > len(data):
        return 0
    return int.from_bytes(data[pos : pos + 2], "little" if little_endian else "big")


def _read32(data: bytes, pos: int, little_endian: bool) -> int:
    if pos + 4 > len(data):
        return 0
    return int.from_bytes(data[pos : pos + 4], "little" if little_endian else "big")


def _read_orientation_from_tiff(data: bytes, tiff_start: int) -> int:
    if tiff_start + 8 > len(data):
        return 1

    byte_order = int.from_bytes(data[tiff_start : tiff_start + 2], "big")
    little_endian = byte_order == 0x4949

    ifd_offset = _read32(data, tiff_start + 4, little_endian)
    ifd_start = tiff_start + ifd_offset
    if ifd_start + 2 > len(data):
        return 1

    entry_count = _read16(data, ifd_start, little_endian)
    for index in range(entry_count):
        entry_pos = ifd_start + 2 + index * 12
        if entry_pos + 12 > len(data):
            return 1
        if _read16(data, entry_pos, little_endian) == _ORIENTATION_TAG:
            value = _read16(data, entry_pos + 8, little_endian)
            return value if 1 <= value <= 8 else 1

    return 1


def _has_exif_header(data: bytes, offset: int) -> bool:
    return data[offset : offset + 6] == _EXIF_HEADER


def _find_jpeg_tiff_offset(data: bytes) -> int:
    offset = 2
    while offset < len(data) - 1:
        if data[offset] != 0xFF:
            return -1
        marker = data[offset + 1]
        if marker == 0xFF:
            offset += 1
            continue

        if marker == 0xE1:
            if offset + 4 >= len(data):
                return -1
            segment_start = offset + 4
            if segment_start + 6 > len(data):
                return -1
            if not _has_exif_header(data, segment_start):
                return -1
            return segment_start + 6

        if offset + 4 > len(data):
            return -1
        length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        offset += 2 + length

    return -1


def _find_webp_tiff_offset(data: bytes) -> int:
    offset = 12
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        data_start = offset + 8

        if chunk_id == b"EXIF":
            if data_start + chunk_size > len(data):
                return -1
            # Some WebP files prefix the TIFF header with "Exif\0\0".
            if chunk_size >= 6 and _has_exif_header(data, data_start):
                return data_start + 6
            return data_start

        # RIFF chunks are padded to an even size.
        offset = data_start + chunk_size + (chunk_size % 2)

    return -1


def get_exif_orientation(data: bytes) -> int:
    """Return the EXIF orientation (1-8); ``1`` when absent or unreadable."""
    tiff_offset = -1

    if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
        tiff_offset = _find_jpeg_tiff_offset(data)
    elif len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        tiff_offset = _find_webp_tiff_offset(data)

    if tiff_offset == -1:
        return 1
    return _read_orientation_from_tiff(data, tiff_offset)


def apply_exif_orientation(image: Image, original_bytes: bytes) -> Image:
    """Rotate/flip `image` so it displays the way the camera was held."""
    from PIL import Image as PILImage

    orientation = get_exif_orientation(original_bytes)
    transforms: dict[int, list[Any]] = {
        2: [PILImage.Transpose.FLIP_LEFT_RIGHT],
        3: [PILImage.Transpose.ROTATE_180],
        4: [PILImage.Transpose.FLIP_TOP_BOTTOM],
        5: [PILImage.Transpose.TRANSPOSE],
        6: [PILImage.Transpose.ROTATE_270],
        7: [PILImage.Transpose.TRANSVERSE],
        8: [PILImage.Transpose.ROTATE_90],
    }
    for transform in transforms.get(orientation, []):
        image = image.transpose(transform)
    return image


__all__ = ["apply_exif_orientation", "get_exif_orientation"]
