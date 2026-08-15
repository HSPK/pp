"""Python port of `packages/coding-agent/test/image-processing.test.ts`.

The upstream pipeline runs on Photon (Rust/WASM); this port runs on Pillow, so
re-encoded byte payloads differ. Every assertion here is on observable
properties (mime type, dimensions, magic bytes, relative sizes) rather than on
exact bytes, matching the intent of the TypeScript cases.
"""

from __future__ import annotations

import base64

from pi_coding_agent.utils.image_convert import convert_to_png
from pi_coding_agent.utils.image_resize import (
    ImageResizeOptions,
    ResizedImage,
    format_dimension_note,
    resize_image,
)

# Small 2x2 red PNG image (base64) - generated with ImageMagick
TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACAQMAAABIeJ9nAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAG"
    "UExURf8AAP///0EdNBEAAAABYktHRAH/Ai3eAAAAB3RJTUUH6gEOADM5Ddoh/wAAAAxJREFUCNdjYGBgAAAABAABJzQnCgAAACV0RVh0ZGF0"
    "ZTpjcmVhdGUAMjAyNi0wMS0xNFQwMDo1MTo1NyswMDowMOnKzHgAAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjYtMDEtMTRUMDA6NTE6NTcrMDA6"
    "MDCYl3TEAAAAKHRFWHRkYXRlOnRpbWVzdGFtcAAyMDI2LTAxLTE0VDAwOjUxOjU3KzAwOjAwz4JVGwAAAABJRU5ErkJggg=="
)

# Small 2x2 blue JPEG image (base64) - generated with ImageMagick
TINY_JPEG = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8X"
    "GBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAAR"
    "CAACAAIDAREAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAVAQEBAAAAAAAAAAAAAAAAAAAG"
    "Cf/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQA/AD3VTB3/2Q=="
)

# 100x100 gray PNG
MEDIUM_PNG_100X100 = (
    "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAAAAABVicqIAAAAAmJLR0QA/4ePzL8AAAAHdElNRQfqAQ4AMzkN2iH/AAAAP0lEQVRo3u3NQQEA"
    "AAQEMASXXYrz2gqst/Lm4ZBIJBKJRCKRSCQSiUQikUgkEolEIpFIJBKJRCKRSCQSiSTsAP1cAUZeKtreAAAAJXRFWHRkYXRlOmNyZWF0ZQAy"
    "MDI2LTAxLTE0VDAwOjUxOjU3KzAwOjAw6crMeAAAACV0RVh0ZGF0ZTptb2RpZnkAMjAyNi0wMS0xNFQwMDo1MTo1NyswMDowMJiXdMQAAAAo"
    "dEVYdGRhdGU6dGltZXN0YW1wADIwMjYtMDEtMTRUMDA6NTE6NTcrMDA6MDDPglUbAAAAAElFTkSuQmCC"
)

# 200x200 colored PNG
LARGE_PNG_200X200 = (
    "iVBORw0KGgoAAAANSUhEUgAAAMgAAADIEAIAAAByquWKAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAG"
    "YktHRP///////wlY99wAAAAHdElNRQfqAQ4AMzqU03BFAAAEEElEQVR42u3dvWqUURSF4XXMRAyIhSAWQvwnWtjaWusFx9bW1kKD/wELESxE"
    "iBjDjJdwysOC57mCXW32+zGcGbtdACpsdi+T8Xz1GABzm91xMl6sHgNgblzcT/Y+rh4DYG6cv0/2H64eA2BOEgI1fHQHaoyzbXIwVo8BMCcJ"
    "gRqSEKgxft9Lrn5aPQbA3Ph1klw7Wj0GwJwkBGpYWECN8fMiub63egyAORcWUMPCAmqMH3eTG59XjwEwN76/S24+Wj0GwJwkBGpYWECN8e1f"
    "cmuzegyAORcWUMPCAmqM09vJ4dfVYwDMjS9vkzuPV48BMCcJgRqeSAZqjA/nyYP91WMAzElCoIYkBGqMk8Pk6HT1GABzm+3L5JIkBApIQqCG"
    "j+5AjfHmb/Lk8uoxAOYkIVBDEgI1LCygxmZ7nFyShEABFxZQw8ICaozXf5KnV1aPATDnwgJqWFhADQsLqOFnDUANFxZQw8ICaoxXZ8mzg9Vj"
    "AMy5sIAaFhZQw3tYQA1/QgHUkIRADUkI1JCEQA1JCNSQhEANFxZQwzcsoIYkBGpIQqCGJARqSEKghiQEalhYQA1/QgHUcGEBNSwsoIYkBGq4"
    "sIAaFhZQw8ICaviGBdRwYQE1LCyghiQEariwgBoWFlDDe1hADS+OAjUkIVBDEgI1JCFQQxICNSQhUMOFBdTwDQuoIQmBGpIQqCEJgRqSEKgh"
    "CYEaFhZQw4ujQA0XFlDDwgJqSEKghgsLqGFhATUsLKCGb1hADRcWUMPCAmpIQqCGCwuoYWEBNbyHBdTw4ihQQxICNSQhUEMSAjUkIVBDEgI1"
    "XFhADd+wgBqSEKghCYEakhCoIQmBGpIQqGFhATW8OArUcGEBNSwsoIYkBGq4sIAaFhZQw8ICaviGBdRwYQE1LCyghiQEariwgBoWFlDDe1hA"
    "DS+OAjUkIVBDEgI1JCFQQxICNSQhUMOFBdTwDQuoIQmBGpIQqCEJgRqSEKghCYEa49lB8ups9RgAc14cBWpIQqCGhQXUkIRADRcWUMPCAmqM"
    "p1eS139WjwEw5xsWUEMSAjUsLKCGJARquLCAGhYWUGM8uZy8+bt6DIA5L44CNSQhUMMTyUCNcXSanByuHgNgThICNSQhUGM82E8+nK8eA2DO"
    "zxqAGpIQqDHuPE6+vF09BsDcOPyanN5ePQbAnCQEavgdFlBj3Nok3/6tHgNgzoujQA1JCNQYNx8l39+tHgNgbtz4nPy4u3oMgDlJCNSwsIAa"
    "4/pe8vNi9RgAc37WANSQhECNce0o+XWyegyAuXH1U/L73uoxAOYkIVDDwgJqjIORnG1XjwEw508ogBqSEKgx9h8m5+9XjwEwN/Y+Jhf3V48B"
    "MCcJgRpjPE+2x6vHAJgbSbLbrR4DYO4/GqiSgXN+ksgAAAAldEVYdGRhdGU6Y3JlYXRlADIwMjYtMDEtMTRUMDA6NTE6NTcrMDA6MDDpysx4"
    "AAAAJXRFWHRkYXRlOm1vZGlmeQAyMDI2LTAxLTE0VDAwOjUxOjU3KzAwOjAwmJd0xAAAACh0RVh0ZGF0ZTp0aW1lc3RhbXAAMjAyNi0wMS0x"
    "NFQwMDo1MTo1NyswMDowMM+CVRsAAAAASUVORK5CYII="
)


def image_bytes(base64_data: str) -> bytes:
    return base64.b64decode(base64_data)


# ---------------------------------------------------------------------------
# convert_to_png
# ---------------------------------------------------------------------------


def test_convert_to_png_returns_original_data_for_png_input() -> None:
    result = convert_to_png(TINY_PNG, "image/png")

    assert result is not None
    assert result["data"] == TINY_PNG
    assert result["mimeType"] == "image/png"


def test_convert_to_png_converts_jpeg() -> None:
    result = convert_to_png(TINY_JPEG, "image/jpeg")

    assert result is not None
    assert result["mimeType"] == "image/png"
    data = base64.b64decode(result["data"])
    assert data[0] == 0x89
    assert data[1] == 0x50
    assert data[2] == 0x4E
    assert data[3] == 0x47


# ---------------------------------------------------------------------------
# resize_image
# ---------------------------------------------------------------------------


def test_resize_keeps_caller_input_bytes_intact() -> None:
    data = bytearray(image_bytes(TINY_PNG))
    original_length = len(data)
    original_first_byte = data[0]

    result = resize_image(
        bytes(data), "image/png", ImageResizeOptions(max_width=100, max_height=100, max_bytes=1 << 20)
    )

    assert result is not None
    assert len(data) == original_length
    assert data[0] == original_first_byte


def test_resize_returns_original_image_if_within_limits() -> None:
    result = resize_image(
        image_bytes(TINY_PNG), "image/png", ImageResizeOptions(max_width=100, max_height=100, max_bytes=1 << 20)
    )

    assert result is not None
    assert result.was_resized is False
    assert result.data == TINY_PNG
    assert result.original_width == 2
    assert result.original_height == 2
    assert result.width == 2
    assert result.height == 2


def test_resize_image_exceeding_dimension_limits() -> None:
    result = resize_image(
        image_bytes(MEDIUM_PNG_100X100),
        "image/png",
        ImageResizeOptions(max_width=50, max_height=50, max_bytes=1 << 20),
    )

    assert result is not None
    assert result.was_resized is True
    assert result.original_width == 100
    assert result.original_height == 100
    assert result.width <= 50
    assert result.height <= 50


def test_resize_image_exceeding_byte_limit() -> None:
    original = image_bytes(LARGE_PNG_200X200)

    result = resize_image(
        original,
        "image/png",
        ImageResizeOptions(max_width=2000, max_height=2000, max_bytes=int(len(LARGE_PNG_200X200) * 0.9)),
    )

    assert result is not None
    assert len(base64.b64decode(result.data)) < len(original)
    assert len(result.data) < len(LARGE_PNG_200X200)


def test_resize_returns_none_when_image_cannot_fit_max_bytes() -> None:
    result = resize_image(
        image_bytes(LARGE_PNG_200X200), "image/png", ImageResizeOptions(max_width=2000, max_height=2000, max_bytes=1)
    )

    assert result is None


def test_resize_handles_jpeg_input() -> None:
    result = resize_image(
        image_bytes(TINY_JPEG), "image/jpeg", ImageResizeOptions(max_width=100, max_height=100, max_bytes=1 << 20)
    )

    assert result is not None
    assert result.was_resized is False
    assert result.original_width == 2
    assert result.original_height == 2


# ---------------------------------------------------------------------------
# format_dimension_note
# ---------------------------------------------------------------------------


def test_dimension_note_is_none_for_non_resized_images() -> None:
    note = format_dimension_note(
        ResizedImage(
            data="",
            mime_type="image/png",
            original_width=100,
            original_height=100,
            width=100,
            height=100,
            was_resized=False,
        )
    )

    assert note is None


def test_dimension_note_is_formatted_for_resized_images() -> None:
    note = format_dimension_note(
        ResizedImage(
            data="",
            mime_type="image/png",
            original_width=2000,
            original_height=1000,
            width=1000,
            height=500,
            was_resized=True,
        )
    )

    assert note is not None
    assert "original 2000x1000" in note
    assert "displayed at 1000x500" in note
    assert "2.00" in note
