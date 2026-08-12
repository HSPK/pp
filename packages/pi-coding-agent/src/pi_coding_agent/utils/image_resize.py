"""Resize and re-encode images to fit inline provider limits.

Ported from ``packages/coding-agent/src/utils/image-resize-core.ts`` (the
algorithm) and ``utils/image-resize.ts`` (``formatDimensionNote``; its
worker-thread dispatch has no Python analogue and is skipped).

Providers cap inline image payloads (Anthropic at 5MB of base64), so an
oversized screenshot has to be shrunk before it can be sent. The strategy is
the TypeScript one: fit the image inside ``max_width``/``max_height``, encode
it as both PNG and JPEG at several qualities and take the first candidate
under the byte budget, then shrink by 25% and retry until 1x1.

The TS uses Photon (Rust/WASM); this port uses Pillow. When Pillow is not
installed, `resize_image` returns ``None`` — exactly what the TS does when
``loadPhoton()`` returns ``null``, so callers already handle it.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass

from pi_coding_agent.utils.exif_orientation import apply_exif_orientation
from pi_coding_agent.utils.js_number import js_round, to_fixed

# 4.5MB of base64 payload; headroom below Anthropic's 5MB limit.
DEFAULT_MAX_BYTES = int(4.5 * 1024 * 1024)
DEFAULT_MAX_WIDTH = 2000
DEFAULT_MAX_HEIGHT = 2000
DEFAULT_JPEG_QUALITY = 80


@dataclass
class ImageResizeOptions:
    max_width: int = DEFAULT_MAX_WIDTH
    max_height: int = DEFAULT_MAX_HEIGHT
    max_bytes: int = DEFAULT_MAX_BYTES
    jpeg_quality: int = DEFAULT_JPEG_QUALITY


@dataclass
class ResizedImage:
    data: str
    """base64"""
    mime_type: str
    original_width: int
    original_height: int
    width: int
    height: int
    was_resized: bool


@dataclass
class _EncodedCandidate:
    data: str
    encoded_size: int
    mime_type: str


def _encode_candidate(buffer: bytes, mime_type: str) -> _EncodedCandidate:
    data = base64.b64encode(buffer).decode("ascii")
    return _EncodedCandidate(data=data, encoded_size=len(data), mime_type=mime_type)


def _load_pillow():
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


def resize_image(input_bytes: bytes, mime_type: str, options: ImageResizeOptions | None = None) -> ResizedImage | None:
    """Fit `input_bytes` inside the dimension and byte budgets.

    Returns ``None`` when Pillow is unavailable, the bytes cannot be decoded,
    or even a 1x1 encoding stays above ``max_bytes``.
    """
    opts = options or ImageResizeOptions()
    input_base64_size = math.ceil(len(input_bytes) / 3) * 4

    pil_image = _load_pillow()
    if pil_image is None:
        return None

    try:
        image = pil_image.open(io.BytesIO(input_bytes))
        image.load()
        image = apply_exif_orientation(image, input_bytes)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA")

        original_width, original_height = image.size
        image_format = mime_type.split("/")[-1] if "/" in mime_type else "png"

        within_dimensions = original_width <= opts.max_width and original_height <= opts.max_height
        if within_dimensions and input_base64_size < opts.max_bytes:
            return ResizedImage(
                data=base64.b64encode(input_bytes).decode("ascii"),
                mime_type=mime_type or f"image/{image_format}",
                original_width=original_width,
                original_height=original_height,
                width=original_width,
                height=original_height,
                was_resized=False,
            )

        target_width = original_width
        target_height = original_height
        if target_width > opts.max_width:
            target_height = js_round(target_height * opts.max_width / target_width)
            target_width = opts.max_width
        if target_height > opts.max_height:
            target_width = js_round(target_width * opts.max_height / target_height)
            target_height = opts.max_height

        quality_steps: list[int] = []
        for quality in (opts.jpeg_quality, 85, 70, 55, 40):
            if quality not in quality_steps:
                quality_steps.append(quality)

        def try_encodings(width: int, height: int) -> list[_EncodedCandidate]:
            resized = image.resize((max(1, width), max(1, height)), pil_image.Resampling.LANCZOS)
            candidates: list[_EncodedCandidate] = []

            png_buffer = io.BytesIO()
            resized.save(png_buffer, format="PNG")
            candidates.append(_encode_candidate(png_buffer.getvalue(), "image/png"))

            jpeg_source = resized.convert("RGB") if resized.mode != "RGB" else resized
            for quality in quality_steps:
                jpeg_buffer = io.BytesIO()
                jpeg_source.save(jpeg_buffer, format="JPEG", quality=quality)
                candidates.append(_encode_candidate(jpeg_buffer.getvalue(), "image/jpeg"))
            return candidates

        current_width = target_width
        current_height = target_height
        while True:
            for candidate in try_encodings(current_width, current_height):
                if candidate.encoded_size < opts.max_bytes:
                    return ResizedImage(
                        data=candidate.data,
                        mime_type=candidate.mime_type,
                        original_width=original_width,
                        original_height=original_height,
                        width=current_width,
                        height=current_height,
                        was_resized=True,
                    )

            if current_width == 1 and current_height == 1:
                break

            next_width = 1 if current_width == 1 else max(1, math.floor(current_width * 0.75))
            next_height = 1 if current_height == 1 else max(1, math.floor(current_height * 0.75))
            if next_width == current_width and next_height == current_height:
                break

            current_width = next_width
            current_height = next_height

        return None
    except Exception:
        return None


def format_dimension_note(resized: ResizedImage) -> str | None:
    """The coordinate-mapping hint shown next to a shrunk attachment.

    The model sees the resized pixels but the user talks about the original,
    so the note carries the scale factor needed to map coordinates back.
    """
    if not resized.was_resized:
        return None
    scale = resized.original_width / resized.width
    return (
        f"[Image: original {resized.original_width}x{resized.original_height}, "
        f"displayed at {resized.width}x{resized.height}. "
        f"Multiply coordinates by {to_fixed(scale, 2)} to map to original image.]"
    )


__all__ = [
    "DEFAULT_JPEG_QUALITY",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_HEIGHT",
    "DEFAULT_MAX_WIDTH",
    "ImageResizeOptions",
    "ResizedImage",
    "format_dimension_note",
    "resize_image",
]
