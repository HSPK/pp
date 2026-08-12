"""Normalize, resize and encode an image for inline attachment.

Ported from ``packages/coding-agent/src/utils/image-process.ts``.

`process_image` is the single entry point used by both ``@file`` CLI
arguments and pasted images: it converts unsupported formats to PNG, shrinks
oversized payloads, and returns base64 plus any hint strings the model should
see (format conversion, coordinate scale). On failure it returns a
``not ok`` result carrying the ``[Image omitted: ...]`` text the caller
inlines instead of the attachment.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

from pi_coding_agent.utils.image_convert import convert_image_bytes_to_png
from pi_coding_agent.utils.image_resize import (
    ImageResizeOptions,
    format_dimension_note,
    resize_image,
)

_SUPPORTED_MIME_TYPES = {
    "image/png": "image/png",
    "image/jpeg": "image/jpeg",
    "image/jpg": "image/jpeg",
    "image/gif": "image/gif",
    "image/webp": "image/webp",
}


@dataclass
class ProcessImageResult:
    ok: bool
    data: str = ""
    mime_type: str = ""
    hints: list[str] = field(default_factory=list)
    message: str = ""


@dataclass
class _NormalizedImage:
    data: bytes
    mime_type: str
    converted_from: str | None = None


def base_mime_type(mime_type: str) -> str:
    return mime_type.split(";")[0].strip().lower()


def normalize_supported_image_mime_type(mime_type: str) -> str | None:
    return _SUPPORTED_MIME_TYPES.get(base_mime_type(mime_type))


def _normalize_image(data: bytes, mime_type: str) -> _NormalizedImage | None:
    normalized_mime_type = normalize_supported_image_mime_type(mime_type)
    if normalized_mime_type:
        return _NormalizedImage(data=data, mime_type=normalized_mime_type)

    png_bytes = convert_image_bytes_to_png(data)
    if png_bytes is None:
        return None
    return _NormalizedImage(data=png_bytes, mime_type="image/png", converted_from=base_mime_type(mime_type))


def _conversion_hint(from_type: str | None, to_type: str) -> str | None:
    if not from_type or from_type == to_type:
        return None
    return f"[Image converted from {from_type} to {to_type}.]"


def process_image(
    data: bytes,
    mime_type: str,
    *,
    auto_resize_images: bool = True,
    resize_options: ImageResizeOptions | None = None,
) -> ProcessImageResult:
    normalized = _normalize_image(data, mime_type)
    if normalized is None:
        return ProcessImageResult(
            ok=False,
            message="[Image omitted: could not be converted to a supported inline image format.]",
        )

    if auto_resize_images:
        resized = resize_image(normalized.data, normalized.mime_type, resize_options)
        if resized is None:
            return ProcessImageResult(
                ok=False,
                message="[Image omitted: could not be resized below the inline image size limit.]",
            )

        hints: list[str] = []
        converted_hint = _conversion_hint(normalized.converted_from, resized.mime_type)
        if converted_hint:
            hints.append(converted_hint)
        dimension_note = format_dimension_note(resized)
        if dimension_note:
            hints.append(dimension_note)

        return ProcessImageResult(ok=True, data=resized.data, mime_type=resized.mime_type, hints=hints)

    hints = []
    converted_hint = _conversion_hint(normalized.converted_from, normalized.mime_type)
    if converted_hint:
        hints.append(converted_hint)

    return ProcessImageResult(
        ok=True,
        data=base64.b64encode(normalized.data).decode("ascii"),
        mime_type=normalized.mime_type,
        hints=hints,
    )


__all__ = [
    "ProcessImageResult",
    "base_mime_type",
    "normalize_supported_image_mime_type",
    "process_image",
]
