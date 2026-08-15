"""Python port of `packages/coding-agent/test/clipboard-image-bmp-conversion.test.ts`.

Covers the WSL2/WSLg case where the clipboard offers `image/bmp` instead of
`image/png`; `read_clipboard_image` must re-encode it to PNG.
"""

from __future__ import annotations

import pytest

from pi_coding_agent.utils import clipboard_image as clipboard_image_module
from pi_coding_agent.utils.clipboard_image import CommandResult, read_clipboard_image


def create_tiny_bmp_1x1_red_24bpp() -> bytes:
    """Minimal 1x1 24bpp BMP (BGR + row padding to 4 bytes)."""
    buffer = bytearray(58)

    # BITMAPFILEHEADER
    buffer[0:2] = b"BM"
    buffer[2:6] = len(buffer).to_bytes(4, "little")
    buffer[6:8] = (0).to_bytes(2, "little")
    buffer[8:10] = (0).to_bytes(2, "little")
    buffer[10:14] = (54).to_bytes(4, "little")

    # BITMAPINFOHEADER
    buffer[14:18] = (40).to_bytes(4, "little")
    buffer[18:22] = (1).to_bytes(4, "little", signed=True)
    buffer[22:26] = (1).to_bytes(4, "little", signed=True)
    buffer[26:28] = (1).to_bytes(2, "little")
    buffer[28:30] = (24).to_bytes(2, "little")
    buffer[30:34] = (0).to_bytes(4, "little")
    buffer[34:38] = (4).to_bytes(4, "little")
    buffer[38:42] = (0).to_bytes(4, "little", signed=True)
    buffer[42:46] = (0).to_bytes(4, "little", signed=True)
    buffer[46:50] = (0).to_bytes(4, "little")
    buffer[50:54] = (0).to_bytes(4, "little")

    # Pixel data (B, G, R) + 1 byte padding
    buffer[54] = 0x00
    buffer[55] = 0x00
    buffer[56] = 0xFF
    buffer[57] = 0x00

    return bytes(buffer)


async def test_converts_bmp_to_png_on_wayland_wslg(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: str, args: list[str], **_kwargs: object) -> CommandResult:
        if command == "wl-paste" and "--list-types" in args:
            return CommandResult(ok=True, stdout=b"image/bmp\n")
        if command == "wl-paste" and "image/bmp" in args:
            return CommandResult(ok=True, stdout=create_tiny_bmp_1x1_red_24bpp())
        return CommandResult(ok=False, stdout=b"")

    monkeypatch.setattr(clipboard_image_module, "run_command", fake_run)

    image = await read_clipboard_image(env={"WAYLAND_DISPLAY": "wayland-0"}, platform="linux")

    assert image is not None
    assert image.mime_type == "image/png"

    assert image.bytes[0] == 0x89
    assert image.bytes[1] == 0x50  # P
    assert image.bytes[2] == 0x4E  # N
    assert image.bytes[3] == 0x47  # G
