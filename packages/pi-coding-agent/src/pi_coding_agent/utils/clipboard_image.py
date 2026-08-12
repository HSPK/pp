"""Read an image from the system clipboard.

Ported from ``packages/coding-agent/src/utils/clipboard-image.ts``.

The TypeScript original prefers a native Node addon (``clipboard-native.ts``,
backed by the ``clipboard-rs`` crate) on macOS, Windows and non-Wayland Linux.
There is no equivalent addon here, so `read_native_clipboard_image` is a hook
that always returns ``None`` and the command line paths (``wl-paste``,
``xclip``, and PowerShell under WSL) do all the work -- which is what actually
runs upstream on Wayland and WSL anyway.

Unsupported formats (BMP from WSLg, most notably) are re-encoded to PNG through
Pillow rather than Photon, matching the rest of this port's image pipeline.
"""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pi_coding_agent.utils.image_convert import convert_image_bytes_to_png

SUPPORTED_IMAGE_MIME_TYPES: tuple[str, ...] = ("image/png", "image/jpeg", "image/webp", "image/gif")

DEFAULT_LIST_TIMEOUT_S = 1.0
DEFAULT_READ_TIMEOUT_S = 3.0
DEFAULT_POWERSHELL_TIMEOUT_S = 5.0
DEFAULT_MAX_BUFFER_BYTES = 50 * 1024 * 1024

_WSL_RELEASE_RE = re.compile(r"microsoft|wsl", re.IGNORECASE)


@dataclass
class ClipboardImage:
    bytes: bytes
    mime_type: str


@dataclass
class CommandResult:
    ok: bool
    stdout: bytes


def is_wayland_session(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(env.get("WAYLAND_DISPLAY")) or env.get("XDG_SESSION_TYPE") == "wayland"


def base_mime_type(mime_type: str) -> str:
    return mime_type.split(";")[0].strip().lower()


def extension_for_image_mime_type(mime_type: str) -> str | None:
    return {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
    }.get(base_mime_type(mime_type))


def select_preferred_image_mime_type(mime_types: list[str]) -> str | None:
    normalized = [(raw.strip(), base_mime_type(raw.strip())) for raw in mime_types if raw.strip()]

    for preferred in SUPPORTED_IMAGE_MIME_TYPES:
        for raw, base in normalized:
            if base == preferred:
                return raw

    for raw, base in normalized:
        if base.startswith("image/"):
            return raw
    return None


def is_supported_image_mime_type(mime_type: str) -> bool:
    return base_mime_type(mime_type) in SUPPORTED_IMAGE_MIME_TYPES


def run_command(
    command: str,
    args: list[str],
    *,
    timeout_s: float = DEFAULT_READ_TIMEOUT_S,
    max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    try:
        completed = subprocess.run(
            [command, *args],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_s,
            env=dict(env) if env is not None else None,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return CommandResult(ok=False, stdout=b"")

    if completed.returncode != 0:
        return CommandResult(ok=False, stdout=b"")
    return CommandResult(ok=True, stdout=completed.stdout[:max_buffer_bytes])


def _split_types(stdout: bytes) -> list[str]:
    text = stdout.decode("utf-8", errors="replace")
    return [line.strip() for line in re.split(r"\r?\n", text) if line.strip()]


def read_clipboard_image_via_wl_paste() -> ClipboardImage | None:
    listing = run_command("wl-paste", ["--list-types"], timeout_s=DEFAULT_LIST_TIMEOUT_S)
    if not listing.ok:
        return None

    selected_type = select_preferred_image_mime_type(_split_types(listing.stdout))
    if selected_type is None:
        return None

    data = run_command("wl-paste", ["--type", selected_type, "--no-newline"])
    if not data.ok or len(data.stdout) == 0:
        return None
    return ClipboardImage(bytes=data.stdout, mime_type=base_mime_type(selected_type))


def is_wsl(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    if env.get("WSL_DISTRO_NAME") or env.get("WSLENV"):
        return True
    try:
        release = Path("/proc/version").read_text(encoding="utf-8")
    except OSError:
        return False
    return _WSL_RELEASE_RE.search(release) is not None


def read_clipboard_image_via_powershell() -> ClipboardImage | None:
    """Windows screenshots (Win+Shift+S) never reach the WSL Linux clipboard.

    PowerShell can read the Windows clipboard directly, so it is used as a
    fallback under WSL.
    """
    tmp_file = Path(tempfile.gettempdir()) / f"pi-wsl-clip-{uuid.uuid4()}.png"

    try:
        win_path_result = run_command("wslpath", ["-w", str(tmp_file)], timeout_s=DEFAULT_LIST_TIMEOUT_S)
        if not win_path_result.ok:
            return None

        win_path = win_path_result.stdout.decode("utf-8", errors="replace").strip()
        if not win_path:
            return None

        ps_quoted_win_path = win_path.replace("'", "''")
        ps_script = "; ".join(
            [
                "Add-Type -AssemblyName System.Windows.Forms",
                "Add-Type -AssemblyName System.Drawing",
                f"$path = '{ps_quoted_win_path}'",
                "$img = [System.Windows.Forms.Clipboard]::GetImage()",
                "if ($img) { $img.Save($path, [System.Drawing.Imaging.ImageFormat]::Png); "
                "Write-Output 'ok' } else { Write-Output 'empty' }",
            ]
        )

        result = run_command(
            "powershell.exe",
            ["-NoProfile", "-Command", ps_script],
            timeout_s=DEFAULT_POWERSHELL_TIMEOUT_S,
        )
        if not result.ok:
            return None
        if result.stdout.decode("utf-8", errors="replace").strip() != "ok":
            return None

        data = tmp_file.read_bytes()
        if len(data) == 0:
            return None
        return ClipboardImage(bytes=data, mime_type="image/png")
    except OSError:
        return None
    finally:
        with contextlib.suppress(OSError):
            tmp_file.unlink()


def read_clipboard_image_via_xclip() -> ClipboardImage | None:
    targets = run_command(
        "xclip",
        ["-selection", "clipboard", "-t", "TARGETS", "-o"],
        timeout_s=DEFAULT_LIST_TIMEOUT_S,
    )

    candidate_types = _split_types(targets.stdout) if targets.ok else []
    preferred = select_preferred_image_mime_type(candidate_types) if candidate_types else None
    try_types = [preferred, *SUPPORTED_IMAGE_MIME_TYPES] if preferred else list(SUPPORTED_IMAGE_MIME_TYPES)

    for mime_type in try_types:
        data = run_command("xclip", ["-selection", "clipboard", "-t", mime_type, "-o"])
        if data.ok and len(data.stdout) > 0:
            return ClipboardImage(bytes=data.stdout, mime_type=base_mime_type(mime_type))
    return None


async def read_native_clipboard_image() -> ClipboardImage | None:
    """Stand-in for the native addon path, which this port does not have."""
    return None


async def read_clipboard_image(
    *,
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
) -> ClipboardImage | None:
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform

    if env.get("TERMUX_VERSION"):
        return None

    image: ClipboardImage | None = None

    if platform == "linux":
        wsl = is_wsl(env)
        wayland = is_wayland_session(env)

        if wayland or wsl:
            image = read_clipboard_image_via_wl_paste() or read_clipboard_image_via_xclip()

        if image is None and wsl:
            image = read_clipboard_image_via_powershell()

        if image is None and not wayland:
            image = await read_native_clipboard_image() or read_clipboard_image_via_xclip()
    else:
        image = await read_native_clipboard_image()

    if image is None:
        return None

    if not is_supported_image_mime_type(image.mime_type):
        png_bytes = convert_image_bytes_to_png(image.bytes)
        if png_bytes is None:
            return None
        return ClipboardImage(bytes=png_bytes, mime_type="image/png")

    return image


__all__ = [
    "SUPPORTED_IMAGE_MIME_TYPES",
    "ClipboardImage",
    "base_mime_type",
    "extension_for_image_mime_type",
    "is_supported_image_mime_type",
    "is_wayland_session",
    "is_wsl",
    "read_clipboard_image",
    "read_clipboard_image_via_powershell",
    "read_clipboard_image_via_wl_paste",
    "read_clipboard_image_via_xclip",
    "read_native_clipboard_image",
    "select_preferred_image_mime_type",
]
