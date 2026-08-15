"""System clipboard read/write.

Ported from ``packages/coding-agent/src/utils/clipboard.ts``.

The TypeScript original prefers a native Node addon (``clipboard-native.ts``,
backed by the ``clipboard-rs`` crate) before falling back to platform command
line tools. There is no equivalent addon here, so this port implements the
fallback chain only: platform tools first, then an OSC 52 terminal escape.
That fallback chain is what actually runs on Linux upstream anyway, because the
addon is deliberately skipped there.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from collections.abc import Mapping

from pi_coding_agent.utils.clipboard_image import is_wayland_session as _is_wayland_session

MAX_OSC52_ENCODED_LENGTH = 100_000
_EXEC_TIMEOUT_SECONDS = 5.0
_READ_MAX_BYTES = 50 * 1024 * 1024


def is_wayland_session(env: Mapping[str, str] | None = None) -> bool:
    """Re-exported from `utils.clipboard_image` (TS keeps it in `clipboard-image.ts`)."""
    return _is_wayland_session(env)


def is_remote_session(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return bool(env.get("SSH_CONNECTION") or env.get("SSH_CLIENT") or env.get("MOSH_CONNECTION"))


def _run_with_input(argv: list[str], text: str) -> None:
    """Feed ``text`` to ``argv`` on stdin, raising on any failure."""
    subprocess.run(
        argv,
        input=text.encode(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=_EXEC_TIMEOUT_SECONDS,
        check=True,
    )


def _copy_to_x11_clipboard(text: str) -> None:
    try:
        _run_with_input(["xclip", "-selection", "clipboard"], text)
    except (OSError, subprocess.SubprocessError):
        _run_with_input(["xsel", "--clipboard", "--input"], text)


def emit_osc52(text: str, *, stream: object | None = None) -> bool:
    encoded = base64.b64encode(text.encode()).decode("ascii")
    if len(encoded) > MAX_OSC52_ENCODED_LENGTH:
        return False
    out = sys.stdout if stream is None else stream
    out.write(f"\x1b]52;c;{encoded}\x07")  # type: ignore[attr-defined]
    flush = getattr(out, "flush", None)
    if flush is not None:
        flush()
    return True


def _read_command(argv: list[str]) -> str | None:
    """Run ``argv`` and return its stdout, or ``None`` when it fails."""
    try:
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_EXEC_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout[:_READ_MAX_BYTES].decode("utf-8", errors="replace")


async def read_clipboard_text(env: Mapping[str, str] | None = None) -> str | None:
    """Read plain text from the system clipboard."""
    env = os.environ if env is None else env

    if sys.platform == "darwin":
        text = _read_command(["pbpaste"])
        return text or None

    if sys.platform == "win32":
        text = _read_command(["powershell", "-NoProfile", "-Command", "Get-Clipboard"])
        return text or None

    if is_wayland_session(env) and env.get("WAYLAND_DISPLAY"):
        text = _read_command(["wl-paste", "--no-newline", "--type", "text"])
        if text is not None:
            return text or None

    if env.get("DISPLAY"):
        text = _read_command(["xclip", "-selection", "clipboard", "-o"])
        if text is None:
            text = _read_command(["xsel", "--clipboard", "--output"])
        if text is not None:
            return text or None

    return None


def _copy_via_platform_tool(text: str, env: Mapping[str, str]) -> bool:
    """Try the platform-native clipboard tools. Returns whether it worked."""
    try:
        if sys.platform == "darwin":
            _run_with_input(["pbcopy"], text)
            return True
        if sys.platform == "win32":
            _run_with_input(["clip"], text)
            return True
    except (OSError, subprocess.SubprocessError):
        return False

    # Linux. Try Termux, Wayland, then X11 clipboard tools.
    if env.get("TERMUX_VERSION"):
        try:
            _run_with_input(["termux-clipboard-set"], text)
            return True
        except (OSError, subprocess.SubprocessError):
            pass

    has_x11 = bool(env.get("DISPLAY"))
    if is_wayland_session(env) and env.get("WAYLAND_DISPLAY"):
        try:
            _run_with_input(["wl-copy"], text)
            return True
        except (OSError, subprocess.SubprocessError):
            pass
        if has_x11:
            try:
                _copy_to_x11_clipboard(text)
                return True
            except (OSError, subprocess.SubprocessError):
                return False
        return False

    if has_x11:
        try:
            _copy_to_x11_clipboard(text)
            return True
        except (OSError, subprocess.SubprocessError):
            return False

    return False


async def copy_to_clipboard(text: str, env: Mapping[str, str] | None = None) -> None:
    env = os.environ if env is None else env

    copied = _copy_via_platform_tool(text, env)

    # Over SSH the local clipboard is the one the user sees, so always emit
    # OSC 52 as well even when a remote-side tool reported success.
    if is_remote_session(env) or not copied:
        copied = emit_osc52(text) or copied

    if not copied:
        raise RuntimeError("Failed to copy to clipboard")
