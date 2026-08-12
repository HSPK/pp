"""Tests for the image pipeline and `@file` CLI arguments.

The EXIF orientation vectors were produced by running the upstream
TypeScript ``getExifOrientation`` under Node against generated files; the
expected values below are that reference output, not this port's own.
"""

from __future__ import annotations

import base64
import io
import struct
from pathlib import Path

import pytest
from pi_coding_agent.cli.args import parse_args
from pi_coding_agent.cli.file_processor import (
    FileProcessingError,
    process_file_arguments,
)
from pi_coding_agent.cli.initial_message import build_initial_message
from pi_coding_agent.utils.exif_orientation import get_exif_orientation
from pi_coding_agent.utils.image_convert import convert_image_bytes_to_png, convert_to_png
from pi_coding_agent.utils.image_process import (
    base_mime_type,
    normalize_supported_image_mime_type,
    process_image,
)
from pi_coding_agent.utils.image_resize import (
    ImageResizeOptions,
    format_dimension_note,
    resize_image,
)
from PIL import Image

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _png_bytes(width: int = 8, height: int = 4, color: tuple[int, int, int] = (1, 2, 3)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_with_orientation(orientation: int) -> bytes:
    """A real JPEG carrying an APP1 EXIF segment with the given orientation."""
    buffer = io.BytesIO()
    Image.new("RGB", (8, 4), (1, 2, 3)).save(buffer, format="JPEG")
    data = buffer.getvalue()

    tiff = bytearray(b"MM\x00\x2a" + struct.pack(">I", 8))
    tiff += struct.pack(">H", 1)
    tiff += struct.pack(">HHIHH", 0x0112, 3, 1, orientation, 0)
    tiff += struct.pack(">I", 0)
    payload = b"Exif\x00\x00" + bytes(tiff)
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return data[0:2] + segment + data[2:]


def _webp_with_orientation(orientation: int, *, prefixed: bool = True) -> bytes:
    tiff = bytearray(b"II*\x00" + struct.pack("<I", 8))
    tiff += struct.pack("<H", 1)
    tiff += struct.pack("<HHIHH", 0x0112, 3, 1, orientation, 0)
    tiff += struct.pack("<I", 0)
    payload = (b"Exif\x00\x00" if prefixed else b"") + bytes(tiff)

    buffer = io.BytesIO()
    Image.new("RGB", (8, 4), (5, 5, 5)).save(buffer, format="WEBP")
    data = buffer.getvalue()
    chunk = b"EXIF" + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) % 2 else b"")
    body = data[12:] + chunk
    return b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body


# ---------------------------------------------------------------------------
# EXIF orientation (verified against the TypeScript under Node)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
def test_jpeg_orientation_round_trips(orientation: int) -> None:
    assert get_exif_orientation(_jpeg_with_orientation(orientation)) == orientation


@pytest.mark.parametrize("orientation", [0, 9, 255])
def test_out_of_range_orientation_falls_back_to_one(orientation: int) -> None:
    assert get_exif_orientation(_jpeg_with_orientation(orientation)) == 1


@pytest.mark.parametrize("orientation", [1, 2, 5, 8])
@pytest.mark.parametrize("prefixed", [True, False])
def test_webp_orientation_with_and_without_exif_prefix(orientation: int, prefixed: bool) -> None:
    data = _webp_with_orientation(orientation, prefixed=prefixed)
    assert get_exif_orientation(data) == orientation


def test_images_without_exif_report_orientation_one() -> None:
    assert get_exif_orientation(_png_bytes()) == 1
    buffer = io.BytesIO()
    Image.new("RGB", (8, 4)).save(buffer, format="JPEG")
    assert get_exif_orientation(buffer.getvalue()) == 1


@pytest.mark.parametrize(
    "data",
    [b"", b"\xff", b"\xff\xd8", b"\xff\xd8\xff\xe1", b"RIFF", b"RIFF\x00\x00\x00\x00WEBP", b"junk"],
)
def test_truncated_input_never_raises(data: bytes) -> None:
    assert get_exif_orientation(data) == 1


def test_malformed_webp_chunk_size_terminates() -> None:
    """Upstream TS reads this size as signed and loops forever; this port must not.

    ``bytes[o+7] << 24`` is signed in JavaScript, so a high-bit chunk size
    makes ``offset`` move backwards and ``findWebpTiffOffset`` never exits.
    Reading it unsigned means the scan runs off the end and reports 1.
    """
    body = b"EXIF" + struct.pack("<I", 0xFFFFFFF0) + b"\x00" * 8
    data = b"RIFF" + struct.pack("<I", len(body) + 4) + b"WEBP" + body
    assert get_exif_orientation(data) == 1


# ---------------------------------------------------------------------------
# image conversion
# ---------------------------------------------------------------------------


def test_convert_bmp_to_png() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (7, 7, 7)).save(buffer, format="BMP")
    png = convert_image_bytes_to_png(buffer.getvalue())
    assert png is not None
    assert Image.open(io.BytesIO(png)).format == "PNG"


def test_convert_rejects_non_image_bytes() -> None:
    assert convert_image_bytes_to_png(b"not an image") is None


def test_convert_to_png_passes_png_through() -> None:
    result = convert_to_png("abc", "image/png")
    assert result == {"data": "abc", "mimeType": "image/png"}


def test_convert_to_png_converts_other_types() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="BMP")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    result = convert_to_png(encoded, "image/bmp")
    assert result is not None
    assert result["mimeType"] == "image/png"


def test_convert_to_png_rejects_bad_base64() -> None:
    assert convert_to_png("!!!!not base64!!!!", "image/bmp") is None


# ---------------------------------------------------------------------------
# mime normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("image/png", "image/png"),
        ("image/jpeg", "image/jpeg"),
        ("image/jpg", "image/jpeg"),
        ("IMAGE/PNG", "image/png"),
        ("image/png; charset=binary", "image/png"),
        ("image/gif", "image/gif"),
        ("image/webp", "image/webp"),
        ("image/bmp", None),
        ("text/plain", None),
    ],
)
def test_normalize_supported_image_mime_type(value: str, expected: str | None) -> None:
    assert normalize_supported_image_mime_type(value) == expected


def test_base_mime_type_strips_parameters_and_case() -> None:
    assert base_mime_type("IMAGE/PNG; q=1") == "image/png"


# ---------------------------------------------------------------------------
# resize
# ---------------------------------------------------------------------------


def test_small_image_is_not_resized() -> None:
    result = resize_image(_png_bytes(10, 10), "image/png")
    assert result is not None
    assert result.was_resized is False
    assert (result.width, result.height) == (10, 10)


def test_oversized_image_is_scaled_to_max_dimensions() -> None:
    data = _png_bytes(400, 200)
    result = resize_image(data, "image/png", ImageResizeOptions(max_width=100, max_height=100))
    assert result is not None
    assert result.was_resized is True
    assert (result.width, result.height) == (100, 50)
    assert (result.original_width, result.original_height) == (400, 200)


def test_tall_image_is_scaled_by_height() -> None:
    result = resize_image(_png_bytes(200, 400), "image/png", ImageResizeOptions(max_width=100, max_height=100))
    assert result is not None
    assert (result.width, result.height) == (50, 100)


def test_resize_returns_none_for_undecodable_bytes() -> None:
    assert resize_image(b"nope", "image/png") is None


def test_resize_shrinks_until_under_byte_budget() -> None:
    data = _png_bytes(300, 300, (123, 45, 67))
    result = resize_image(data, "image/png", ImageResizeOptions(max_bytes=2000))
    assert result is not None
    assert len(result.data) < 2000


def test_resize_gives_up_when_budget_is_impossible() -> None:
    assert resize_image(_png_bytes(300, 300), "image/png", ImageResizeOptions(max_bytes=1)) is None


def test_dimension_note_reports_coordinate_scale() -> None:
    result = resize_image(_png_bytes(400, 200), "image/png", ImageResizeOptions(max_width=100, max_height=100))
    assert result is not None
    note = format_dimension_note(result)
    assert note == (
        "[Image: original 400x200, displayed at 100x50. Multiply coordinates by 4.00 to map to original image.]"
    )


def test_dimension_note_absent_when_not_resized() -> None:
    result = resize_image(_png_bytes(10, 10), "image/png")
    assert result is not None
    assert format_dimension_note(result) is None


# ---------------------------------------------------------------------------
# process_image
# ---------------------------------------------------------------------------


def test_process_image_passes_supported_type_through() -> None:
    result = process_image(_png_bytes(10, 10), "image/png")
    assert result.ok is True
    assert result.mime_type == "image/png"
    assert result.hints == []


def test_process_image_reports_conversion_hint() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buffer, format="BMP")
    result = process_image(buffer.getvalue(), "image/bmp", auto_resize_images=False)
    assert result.ok is True
    assert result.hints == ["[Image converted from image/bmp to image/png.]"]


def test_process_image_without_resizing_returns_original_bytes() -> None:
    data = _png_bytes(10, 10)
    result = process_image(data, "image/png", auto_resize_images=False)
    assert result.ok is True
    assert base64.b64decode(result.data) == data


def test_process_image_reports_unconvertible_input() -> None:
    result = process_image(b"not an image", "application/octet-stream")
    assert result.ok is False
    assert "could not be converted" in result.message


def test_process_image_reports_unshrinkable_input() -> None:
    result = process_image(_png_bytes(300, 300), "image/png", resize_options=ImageResizeOptions(max_bytes=1))
    assert result.ok is False
    assert "could not be resized" in result.message


# ---------------------------------------------------------------------------
# @file arguments
# ---------------------------------------------------------------------------


def test_text_file_is_inlined(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("hello\n")
    result = process_file_arguments([str(path)])
    assert result.text == f'<file name="{path}">\nhello\n\n</file>\n'
    assert result.images == []


def test_multiple_files_are_concatenated_in_order(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("one")
    second.write_text("two")
    result = process_file_arguments([str(first), str(second)])
    assert result.text.index("one") < result.text.index("two")


def test_empty_files_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("")
    assert process_file_arguments([str(path)]).text == ""


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileProcessingError, match="File not found"):
        process_file_arguments([str(tmp_path / "nope.txt")])


def test_image_file_becomes_an_attachment(tmp_path: Path) -> None:
    path = tmp_path / "shot.png"
    path.write_bytes(_png_bytes(10, 10))
    result = process_file_arguments([str(path)])
    assert len(result.images) == 1
    assert result.images[0].mime_type == "image/png"
    assert result.text == f'<file name="{path}"></file>\n'


def test_resized_image_carries_hint_in_file_marker(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    path.write_bytes(_png_bytes(3000, 2400))
    result = process_file_arguments([str(path)])
    assert len(result.images) == 1
    assert "Multiply coordinates by" in result.text


def test_relative_paths_resolve_against_cwd(tmp_path: Path) -> None:
    (tmp_path / "rel.txt").write_text("x")
    result = process_file_arguments(["rel.txt"], cwd=str(tmp_path))
    assert "rel.txt" in result.text


# ---------------------------------------------------------------------------
# initial message assembly
# ---------------------------------------------------------------------------


def test_initial_message_is_none_without_input() -> None:
    result = build_initial_message(parse_args([]))
    assert result.initial_message is None
    assert result.initial_images is None


def test_initial_message_consumes_only_the_first_message() -> None:
    parsed = parse_args(["one", "two", "three"])
    result = build_initial_message(parsed)
    assert result.initial_message == "one"
    assert parsed.messages == ["two", "three"]


def test_initial_message_orders_stdin_then_files_then_message() -> None:
    parsed = parse_args(["ask"])
    result = build_initial_message(parsed, file_text="FILE", stdin_content="STDIN")
    assert result.initial_message == "STDINFILEask"


def test_initial_message_works_without_a_cli_message() -> None:
    parsed = parse_args([])
    result = build_initial_message(parsed, file_text="FILE")
    assert result.initial_message == "FILE"


def test_initial_images_are_passed_through() -> None:
    images = [object()]
    result = build_initial_message(parse_args([]), file_images=images)
    assert result.initial_images == images


def test_empty_image_list_is_normalized_to_none() -> None:
    assert build_initial_message(parse_args([]), file_images=[]).initial_images is None
