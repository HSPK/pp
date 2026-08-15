"""Python port of `packages/coding-agent/test/tool-result-images.test.ts`."""

from __future__ import annotations

import base64
import io
import struct
import zlib

from pi_ai.types import ImageContent, TextContent
from PIL import Image

from pi_coding_agent.core.tool_result_images import (
    NormalizeToolResultImagesOptions,
    ToolResultContent,
    normalize_tool_result_images,
)

TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="


def _png_chunk(chunk_type: bytes, body: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + chunk_type + body + struct.pack(">I", checksum)


def _create_png(width: int, height: int) -> bytes:
    """8-bit grayscale PNG of arbitrary dimensions, built without an encoder."""
    ihdr = struct.pack(">II", width, height) + bytes([8, 0, 0, 0, 0])
    raw = bytearray((width + 1) * height)
    for row in range(height):
        start = row * (width + 1) + 1
        for index in range(start, (row + 1) * (width + 1)):
            raw[index] = row % 256
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw)))
        + _png_chunk(b"IEND", b"")
    )


def _read_png_dimensions(base64_data: str) -> tuple[int, int]:
    buffer = base64.b64decode(base64_data)
    with Image.open(io.BytesIO(buffer)) as image:
        return image.width, image.height


def _create_tiny_bmp_1x1_red_24bpp() -> bytes:
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


def _image_block(data: bytes, mime_type: str) -> ImageContent:
    return ImageContent(data=base64.b64encode(data).decode("ascii"), mime_type=mime_type)


async def test_returns_the_original_list_when_there_are_no_image_blocks() -> None:
    content: list[ToolResultContent] = [TextContent(text="no images here")]

    assert await normalize_tool_result_images(content) is content


async def test_returns_the_original_list_when_images_are_within_limits() -> None:
    content: list[ToolResultContent] = [
        TextContent(text="screenshot"),
        ImageContent(data=TINY_PNG_BASE64, mime_type="image/png"),
    ]

    assert await normalize_tool_result_images(content) is content


async def test_resizes_oversized_images_and_reports_original_dimensions() -> None:
    content: list[ToolResultContent] = [_image_block(_create_png(2400, 4800), "image/png")]

    normalized = await normalize_tool_result_images(content)

    assert normalized is not content
    assert len(normalized) == 2
    image = normalized[0]
    assert isinstance(image, ImageContent)
    width, height = _read_png_dimensions(image.data)
    assert width <= 2000
    assert height <= 2000
    note = normalized[1]
    assert isinstance(note, TextContent)
    assert "original 2400x4800" in note.text


async def test_leaves_oversized_images_alone_when_auto_resize_is_disabled() -> None:
    content: list[ToolResultContent] = [_image_block(_create_png(2400, 4800), "image/png")]

    normalized = await normalize_tool_result_images(content, NormalizeToolResultImagesOptions(auto_resize_images=False))

    assert normalized is content


async def test_converts_unsupported_formats_even_when_auto_resize_is_disabled() -> None:
    content: list[ToolResultContent] = [_image_block(_create_tiny_bmp_1x1_red_24bpp(), "image/bmp")]

    normalized = await normalize_tool_result_images(content, NormalizeToolResultImagesOptions(auto_resize_images=False))

    assert normalized is not content
    image = normalized[0]
    assert isinstance(image, ImageContent)
    assert image.mime_type == "image/png"
    note = normalized[1]
    assert isinstance(note, TextContent)
    assert note.text == "[Image converted from image/bmp to image/png.]"


async def test_keeps_undecodable_images_instead_of_dropping_tool_output() -> None:
    content: list[ToolResultContent] = [ImageContent(data="bm90LWFuLWltYWdl", mime_type="image/png")]

    assert await normalize_tool_result_images(content) is content


async def test_preserves_surrounding_text_blocks_and_their_order() -> None:
    content: list[ToolResultContent] = [
        TextContent(text="before"),
        _image_block(_create_png(2400, 100), "image/png"),
        TextContent(text="after"),
    ]

    normalized = await normalize_tool_result_images(content)

    assert [block.type for block in normalized] == ["text", "image", "text", "text"]
    assert normalized[0] == TextContent(text="before")
    assert normalized[3] == TextContent(text="after")
