"""Shared helpers for rendering tool calls and results.

Ported from ``packages/coding-agent/src/core/tools/render-utils.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pi_tui.terminal_image import get_capabilities, get_image_dimensions, hyperlink, image_fallback

from ..utils.ansi import strip_ansi
from ..utils.paths import resolve_path
from ..utils.shell import sanitize_binary_output


def shorten_path(path: Any) -> str:
    if not isinstance(path, str):
        return ""
    home = str(Path.home())
    if path.startswith(home):
        return f"~{path[len(home) :]}"
    return path


def link_path(styled_text: str, raw_path: str, cwd: str) -> str:
    if not get_capabilities().hyperlinks:
        return styled_text
    absolute_path = resolve_path(raw_path, cwd)
    return hyperlink(styled_text, Path(absolute_path).as_uri())


def str_arg(value: Any) -> str | None:
    """TS ``str``: a string passes through, nullish becomes ``""``, anything
    else is ``None`` to signal an invalid argument."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return None


def replace_tabs(text: str) -> str:
    return text.replace("\t", "   ")


def normalize_display_text(text: str) -> str:
    return text.replace("\r", "")


def get_text_output(result: Any, show_images: bool) -> str:
    """Flatten a tool result's content blocks into displayable text."""
    if not result:
        return ""

    content = result["content"] if isinstance(result, dict) else result.content
    text_blocks = [block for block in content if _block_type(block) == "text"]
    image_blocks = [block for block in content if _block_type(block) == "image"]

    output = "\n".join(
        sanitize_binary_output(strip_ansi(_block_get(block, "text") or "")).replace("\r", "") for block in text_blocks
    )

    capabilities = get_capabilities()
    if len(image_blocks) > 0 and (not capabilities.images or not show_images):
        indicators = []
        for image in image_blocks:
            mime_type = _block_get(image, "mime_type") or _block_get(image, "mimeType") or "image/unknown"
            data = _block_get(image, "data")
            dimensions = get_image_dimensions(data, mime_type) if data and mime_type else None
            indicators.append(image_fallback(mime_type, dimensions))
        image_indicators = "\n".join(indicators)
        output = f"{output}\n{image_indicators}" if output else image_indicators

    return output


def _block_type(block: Any) -> str | None:
    return block.get("type") if isinstance(block, dict) else getattr(block, "type", None)


def _block_get(block: Any, key: str) -> Any:
    return block.get(key) if isinstance(block, dict) else getattr(block, key, None)


@dataclass
class ToolRenderResultLike:
    content: list[Any]
    details: Any = None


def invalid_arg_text(theme: Any) -> str:
    return theme.fg("error", "[invalid arg]")


def render_tool_path(raw_path: str | None, theme: Any, cwd: str, empty_fallback: str | None = None) -> str:
    if raw_path is None:
        return invalid_arg_text(theme)
    value = raw_path or empty_fallback
    if not value:
        return theme.fg("toolOutput", "...")
    return link_path(theme.fg("accent", shorten_path(value)), value, cwd)


__all__ = [
    "ToolRenderResultLike",
    "get_text_output",
    "invalid_arg_text",
    "link_path",
    "normalize_display_text",
    "render_tool_path",
    "replace_tabs",
    "shorten_path",
    "str_arg",
]
