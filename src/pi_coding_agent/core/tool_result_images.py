"""Normalize image blocks returned by tool results.

Port of `packages/coding-agent/src/utils/tool-result-images.ts`.

The `read` tool and `@file` CLI attachments run their images through
`process_image`, but tools that produce images themselves (extensions, MCP
bridges, screenshot tools) hand back arbitrary base64 payloads that go straight
into session history and every subsequent provider request. Oversized images
make the provider reject the whole conversation, not just the offending turn,
so they are normalized once as they enter history.

Returns the original list object when nothing changed so callers can skip
rewriting the result.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

from pi_ai.types import ImageContent, TextContent

from pi_coding_agent.utils.image_process import process_image

ToolResultContent = TextContent | ImageContent


@dataclass
class NormalizeToolResultImagesOptions:
    auto_resize_images: bool = True
    """Whether oversized images are resized to inline provider limits."""


async def normalize_tool_result_images(
    content: list[ToolResultContent],
    options: NormalizeToolResultImagesOptions | None = None,
) -> list[ToolResultContent]:
    if not any(isinstance(block, ImageContent) for block in content):
        return content

    auto_resize_images = options.auto_resize_images if options is not None else True
    normalized: list[ToolResultContent] = []
    changed = False

    for block in content:
        if not isinstance(block, ImageContent):
            normalized.append(block)
            continue

        try:
            raw = base64.b64decode(block.data, validate=False)
        except (binascii.Error, ValueError):
            normalized.append(block)
            continue

        processed = process_image(raw, block.mime_type, auto_resize_images=auto_resize_images)
        if not processed.ok:
            # Unlike `read`, keep the original block. The tool already produced
            # this image and the failure may just be an unavailable image
            # backend, so passing it through preserves the behavior tools have
            # today instead of silently deleting their output.
            normalized.append(block)
            continue

        if processed.data == block.data and processed.mime_type == block.mime_type and not processed.hints:
            normalized.append(block)
            continue

        normalized.append(ImageContent(data=processed.data, mime_type=processed.mime_type))
        if processed.hints:
            normalized.append(TextContent(text="\n".join(processed.hints)))
        changed = True

    return normalized if changed else content


__all__ = [
    "NormalizeToolResultImagesOptions",
    "ToolResultContent",
    "normalize_tool_result_images",
]
