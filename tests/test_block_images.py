"""Python port of `packages/coding-agent/test/block-images.test.ts`."""

from __future__ import annotations

import base64
import struct
from pathlib import Path

from pi_coding_agent.cli.file_processor import process_file_arguments
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.tools import create_read_tool

# 1x1 red PNG image as base64 (smallest valid PNG)
TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="


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


# ---------------------------------------------------------------------------
# SettingsManager
# ---------------------------------------------------------------------------


def test_block_images_defaults_to_false() -> None:
    manager = SettingsManager.in_memory({})

    assert manager.get_block_images() is False


def test_block_images_returns_true_when_set() -> None:
    manager = SettingsManager.in_memory({"images": {"blockImages": True}})

    assert manager.get_block_images() is True


def test_set_block_images_persists_the_setting() -> None:
    manager = SettingsManager.in_memory({})
    assert manager.get_block_images() is False

    manager.set_block_images(True)
    assert manager.get_block_images() is True

    manager.set_block_images(False)
    assert manager.get_block_images() is False


def test_block_images_alongside_auto_resize() -> None:
    manager = SettingsManager.in_memory({"images": {"autoResize": True, "blockImages": True}})

    assert manager.get_image_auto_resize() is True
    assert manager.get_block_images() is True


# ---------------------------------------------------------------------------
# Read tool
# ---------------------------------------------------------------------------


async def test_read_tool_always_reads_images(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(base64.b64decode(TINY_PNG_BASE64))

    tool = create_read_tool(str(tmp_path))
    result = await tool.execute("test-1", {"path": str(image_path)})

    assert len(result.content) >= 1
    assert any(getattr(part, "type", None) == "image" for part in result.content)


async def test_read_tool_reads_text_files_normally(tmp_path: Path) -> None:
    text_path = tmp_path / "test.txt"
    text_path.write_text("Hello, world!")

    tool = create_read_tool(str(tmp_path))
    result = await tool.execute("test-2", {"path": str(text_path)})

    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert "Hello, world!" in result.content[0].text


# ---------------------------------------------------------------------------
# process_file_arguments
# ---------------------------------------------------------------------------


def test_file_arguments_always_process_images(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(base64.b64decode(TINY_PNG_BASE64))

    result = process_file_arguments([str(image_path)])

    assert len(result.images) == 1
    assert result.images[0].type == "image"


def test_bmp_images_from_disk_become_png_attachments(tmp_path: Path) -> None:
    image_path = tmp_path / "test.bmp"
    image_path.write_bytes(create_tiny_bmp_1x1_red_24bpp())

    result = process_file_arguments([str(image_path)])

    assert len(result.images) == 1
    assert result.images[0].type == "image"
    assert result.images[0].mime_type == "image/png"
    assert "[Image converted from image/bmp to image/png.]" in result.text


def test_file_arguments_process_text_files_normally(tmp_path: Path) -> None:
    text_path = tmp_path / "test.txt"
    text_path.write_text("Hello, world!")

    result = process_file_arguments([str(text_path)])

    assert len(result.images) == 0
    assert "Hello, world!" in result.text
