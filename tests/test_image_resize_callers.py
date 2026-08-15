"""Python port of `packages/coding-agent/test/image-resize-callers.test.ts`.

The TypeScript test mocks the `image-resize` module; here `resize_image` is
patched where `utils/image_process.py` imported it, which is the same seam.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from pi_coding_agent.cli.file_processor import process_file_arguments
from pi_coding_agent.tools import create_read_tool
from pi_coding_agent.utils import image_process

TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="


@pytest.fixture(autouse=True)
def _resize_always_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_process, "resize_image", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(image_process, "format_dimension_note", lambda *_args, **_kwargs: None)


async def test_read_tool_returns_text_only_when_resize_fails(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(base64.b64decode(TINY_PNG_BASE64))

    tool = create_read_tool(str(tmp_path))
    result = await tool.execute("test-read-image", {"path": str(image_path)})

    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert "Image omitted" in result.content[0].text


def test_file_processor_omits_images_when_resize_fails(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(base64.b64decode(TINY_PNG_BASE64))

    result = process_file_arguments([str(image_path)])

    assert len(result.images) == 0
    assert "Image omitted" in result.text
