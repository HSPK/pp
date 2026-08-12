"""Tests for the `read` tool. Ported from `tools.test.ts` (describe("read tool")).

All 13 TypeScript cases are covered, plus two extra (relative-path resolution and
EACCES) that the TypeScript suite exercises through the edit tool instead.
"""

from __future__ import annotations

import base64
import os

import pytest
from pi_coding_agent.tools.read import create_read_tool


def get_text(result) -> str:
    return "\n".join(c.text for c in result.content if c.type == "text")


def get_image(result):
    return next((c for c in result.content if c.type == "image"), None)


async def test_reads_file_within_limits(tmp_path):
    tool = create_read_tool(str(tmp_path))
    test_file = tmp_path / "test.txt"
    content = "Hello, world!\nLine 2\nLine 3"
    test_file.write_text(content)

    result = await tool.execute("call-1", {"path": str(test_file)})

    assert get_text(result) == content
    assert "Use offset=" not in get_text(result)
    assert result.details is None


async def test_nonexistent_file_raises(tmp_path):
    tool = create_read_tool(str(tmp_path))
    missing = tmp_path / "nonexistent.txt"

    with pytest.raises(FileNotFoundError, match=r"(?i)ENOENT|not found"):
        await tool.execute("call-2", {"path": str(missing)})


async def test_truncates_exceeding_line_limit(tmp_path):
    tool = create_read_tool(str(tmp_path))
    test_file = tmp_path / "large.txt"
    lines = [f"Line {i + 1}" for i in range(2500)]
    test_file.write_text("\n".join(lines))

    result = await tool.execute("call-3", {"path": str(test_file)})
    output = get_text(result)

    assert "Line 1" in output
    assert "Line 2000" in output
    assert "Line 2001" not in output
    assert "[Showing lines 1-2000 of 2500. Use offset=2001 to continue.]" in output


async def test_truncates_when_byte_limit_exceeded(tmp_path):
    tool = create_read_tool(str(tmp_path))
    test_file = tmp_path / "large-bytes.txt"
    lines = [f"Line {i + 1}: {'x' * 200}" for i in range(500)]
    test_file.write_text("\n".join(lines))

    result = await tool.execute("call-4", {"path": str(test_file)})
    output = get_text(result)

    assert "Line 1:" in output
    import re

    assert re.search(r"\[Showing lines 1-\d+ of 500 \(.* limit\)\. Use offset=\d+ to continue\.\]", output)


async def test_offset_parameter(tmp_path):
    tool = create_read_tool(str(tmp_path))
    test_file = tmp_path / "offset-test.txt"
    lines = [f"Line {i + 1}" for i in range(100)]
    test_file.write_text("\n".join(lines))

    result = await tool.execute("call-5", {"path": str(test_file), "offset": 51})
    output = get_text(result)

    assert "Line 50" not in output
    assert "Line 51" in output
    assert "Line 100" in output
    assert "Use offset=" not in output


async def test_limit_parameter(tmp_path):
    tool = create_read_tool(str(tmp_path))
    test_file = tmp_path / "limit-test.txt"
    lines = [f"Line {i + 1}" for i in range(100)]
    test_file.write_text("\n".join(lines))

    result = await tool.execute("call-6", {"path": str(test_file), "limit": 10})
    output = get_text(result)

    assert "Line 1" in output
    assert "Line 10" in output
    assert "Line 11" not in output
    assert "[90 more lines in file. Use offset=11 to continue.]" in output


async def test_offset_and_limit_together(tmp_path):
    tool = create_read_tool(str(tmp_path))
    test_file = tmp_path / "offset-limit-test.txt"
    lines = [f"Line {i + 1}" for i in range(100)]
    test_file.write_text("\n".join(lines))

    result = await tool.execute("call-7", {"path": str(test_file), "offset": 41, "limit": 20})
    output = get_text(result)

    assert "Line 40" not in output
    assert "Line 41" in output
    assert "Line 60" in output
    assert "Line 61" not in output
    assert "[40 more lines in file. Use offset=61 to continue.]" in output


async def test_offset_beyond_file_length_raises(tmp_path):
    tool = create_read_tool(str(tmp_path))
    test_file = tmp_path / "short.txt"
    test_file.write_text("Line 1\nLine 2\nLine 3")

    with pytest.raises(ValueError, match=r"Offset 100 is beyond end of file \(3 lines total\)"):
        await tool.execute("call-8", {"path": str(test_file), "offset": 100})


async def test_truncation_details_present(tmp_path):
    tool = create_read_tool(str(tmp_path))
    test_file = tmp_path / "large-file.txt"
    lines = [f"Line {i + 1}" for i in range(2500)]
    test_file.write_text("\n".join(lines))

    result = await tool.execute("call-9", {"path": str(test_file)})

    assert result.details is not None
    assert result.details.truncation is not None
    assert result.details.truncation.truncated is True
    assert result.details.truncation.truncated_by == "lines"
    assert result.details.truncation.total_lines == 2500
    assert result.details.truncation.output_lines == 2000


async def test_detects_image_mime_from_magic_bytes(tmp_path):
    tool = create_read_tool(str(tmp_path))
    png_1x1_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg=="
    png_buffer = base64.b64decode(png_1x1_base64)
    test_file = tmp_path / "image.txt"
    test_file.write_bytes(png_buffer)

    result = await tool.execute("call-img-1", {"path": str(test_file)})

    assert result.content[0].type == "text"
    assert "Read image file [image/png]" in get_text(result)
    image_block = get_image(result)
    assert image_block is not None
    assert image_block.mime_type == "image/png"
    assert isinstance(image_block.data, str)
    assert len(image_block.data) > 0


async def test_bmp_read_from_disk_as_png_image_attachment(tmp_path):
    """BMP files are converted to PNG by the read tool, matching `tools.test.ts`."""
    tool = create_read_tool(str(tmp_path))
    buffer = bytearray(58)
    buffer[0:2] = b"BM"
    buffer[2:6] = (58).to_bytes(4, "little")
    buffer[10:14] = (54).to_bytes(4, "little")
    buffer[14:18] = (40).to_bytes(4, "little")
    buffer[18:22] = (1).to_bytes(4, "little", signed=True)
    buffer[22:26] = (1).to_bytes(4, "little", signed=True)
    buffer[26:28] = (1).to_bytes(2, "little")
    buffer[28:30] = (24).to_bytes(2, "little")
    buffer[34:38] = (4).to_bytes(4, "little")
    buffer[56] = 0xFF

    test_file = tmp_path / "image.bmp"
    test_file.write_bytes(bytes(buffer))

    result = await tool.execute("call-img-bmp", {"path": str(test_file)})

    assert "Read image file [image/png]" in get_text(result)
    assert "[Image converted from image/bmp to image/png.]" in get_text(result)
    image_block = get_image(result)
    assert image_block is not None
    assert image_block.mime_type == "image/png"
    raw = base64.b64decode(image_block.data)
    assert raw[0] == 0x89


async def test_image_extension_but_non_image_content_treated_as_text(tmp_path):
    tool = create_read_tool(str(tmp_path))
    test_file = tmp_path / "not-an-image.png"
    test_file.write_text("definitely not a png")

    result = await tool.execute("call-img-2", {"path": str(test_file)})
    output = get_text(result)

    assert "definitely not a png" in output
    assert get_image(result) is None


async def test_relative_path_resolved_against_cwd(tmp_path):
    tool = create_read_tool(str(tmp_path))
    (tmp_path / "rel.txt").write_text("relative content")

    result = await tool.execute("call-rel", {"path": "rel.txt"})

    assert get_text(result) == "relative content"


async def test_permission_denied_raises(tmp_path):
    tool = create_read_tool(str(tmp_path))
    test_file = tmp_path / "noperm.txt"
    test_file.write_text("secret")
    os.chmod(test_file, 0o000)
    try:
        if os.access(test_file, os.R_OK):
            pytest.skip("running as root or filesystem ignores permission bits")
        with pytest.raises(PermissionError):
            await tool.execute("call-perm", {"path": str(test_file)})
    finally:
        os.chmod(test_file, 0o644)
