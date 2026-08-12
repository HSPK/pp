"""Python port of `packages/coding-agent/test/image-process.test.ts`."""

from __future__ import annotations

import base64
import struct

from pi_coding_agent.utils.image_process import process_image
from pi_coding_agent.utils.mime import detect_supported_image_mime_type


def create_tiny_bmp_1x1_red_24bpp() -> bytes:
    buffer = bytearray(58)
    buffer[0:2] = b"BM"
    struct.pack_into("<I", buffer, 2, len(buffer))
    struct.pack_into("<I", buffer, 10, 54)
    struct.pack_into("<I", buffer, 14, 40)
    struct.pack_into("<i", buffer, 18, 1)
    struct.pack_into("<i", buffer, 22, 1)
    struct.pack_into("<H", buffer, 26, 1)
    struct.pack_into("<H", buffer, 28, 24)
    struct.pack_into("<I", buffer, 30, 0)
    struct.pack_into("<I", buffer, 34, 4)
    buffer[56] = 0xFF
    return bytes(buffer)


def expect_png_magic(base64_data: str) -> None:
    data = base64.b64decode(base64_data)
    assert data[0] == 0x89
    assert data[1] == 0x50
    assert data[2] == 0x4E
    assert data[3] == 0x47


def test_detects_bmp_files_from_magic_bytes() -> None:
    assert detect_supported_image_mime_type(create_tiny_bmp_1x1_red_24bpp()) == "image/bmp"


def test_converts_bmp_to_png_when_auto_resize_is_disabled() -> None:
    result = process_image(create_tiny_bmp_1x1_red_24bpp(), "image/bmp", auto_resize_images=False)

    assert result.ok is True
    assert result.mime_type == "image/png"
    assert "[Image converted from image/bmp to image/png.]" in result.hints
    expect_png_magic(result.data)


def test_converts_bmp_before_auto_resizing() -> None:
    result = process_image(create_tiny_bmp_1x1_red_24bpp(), "image/bmp")

    assert result.ok is True
    assert result.mime_type == "image/png"
    assert "[Image converted from image/bmp to image/png.]" in result.hints
    expect_png_magic(result.data)
