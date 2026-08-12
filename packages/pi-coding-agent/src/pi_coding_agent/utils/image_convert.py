"""Convert images into a format providers and terminals accept.

Ported from ``packages/coding-agent/src/utils/image-convert.ts``.

Providers accept a small set of inline image types, and the Kitty graphics
protocol requires PNG. Anything else (BMP, TIFF, HEIC where the decoder
supports it) is decoded and re-encoded to PNG. Returns ``None`` when the
bytes cannot be decoded, matching the TS behaviour when Photon is missing or
the decode throws.
"""

from __future__ import annotations

import base64
import io

from pi_coding_agent.utils.exif_orientation import apply_exif_orientation


def convert_image_bytes_to_png(data: bytes) -> bytes | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        image = apply_exif_orientation(image, data)
        if image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
            image = image.convert("RGBA")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:
        return None


def convert_to_png(base64_data: str, mime_type: str) -> dict[str, str] | None:
    """Convert a base64 attachment to PNG for terminal display."""
    if mime_type == "image/png":
        return {"data": base64_data, "mimeType": mime_type}

    try:
        data = base64.b64decode(base64_data)
    except Exception:
        return None

    png_bytes = convert_image_bytes_to_png(data)
    if png_bytes is None:
        return None
    return {"data": base64.b64encode(png_bytes).decode("ascii"), "mimeType": "image/png"}


__all__ = ["convert_image_bytes_to_png", "convert_to_png"]
