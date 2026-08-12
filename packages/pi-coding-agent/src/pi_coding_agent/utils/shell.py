"""Shell execution helpers shared by the bash tool.

Python port of the subset of `packages/coding-agent/src/utils/shell.ts` the
bash tool needs: shell resolution, environment preparation, output
sanitization and detached child tracking. The bundled-binary `PATH` injection
in the TypeScript version has no Python equivalent here (no bundled `fd`/`rg`
binaries are shipped) and is not ported.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import unicodedata
from dataclasses import dataclass, field

_ALLOWED_CONTROL_CODEPOINTS = {0x09, 0x0A, 0x0D}  # tab, newline, carriage return

_LEGACY_WSL_BASH_PATH_RE = re.compile(r"^[a-z]:\\windows\\(?:system32|sysnative)\\bash\.exe$")


@dataclass(frozen=True)
class ShellConfig:
    """Resolved shell, its arguments, and how the command reaches it.

    Port of TS `ShellConfig`. ``command_transport`` is ``"argv"`` when the
    command is appended to ``args`` (``bash -c '<command>'``) and ``"stdin"``
    when it must be piped in instead, which legacy WSL ``bash.exe`` requires
    because it does not accept a ``-c`` command on the Windows command line.
    """

    shell: str
    args: list[str] = field(default_factory=list)
    command_transport: str = "argv"


def _is_windows() -> bool:
    """Indirection for TS's `process.platform === "win32"`, so tests can patch it."""
    return os.name == "nt"


def _is_legacy_wsl_bash_path(path: str) -> bool:
    return bool(_LEGACY_WSL_BASH_PATH_RE.match(path.replace("/", "\\").lower()))


def _get_bash_shell_config(shell: str) -> ShellConfig:
    if _is_legacy_wsl_bash_path(shell):
        return ShellConfig(shell=shell, args=["-s"], command_transport="stdin")
    return ShellConfig(shell=shell, args=["-c"])


def _find_bash_on_path() -> str | None:
    """Locate a bash executable on `PATH`.

    Mirrors TS `findBashOnPath`: `where bash.exe` on Windows (whose output can
    name files that do not exist, hence the extra check) and `which bash`
    elsewhere, whose output is trusted so Termux and other special filesystems
    keep working.
    """
    if _is_windows():
        try:
            result = subprocess.run(["where", "bash.exe"], capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode == 0 and result.stdout:
            first_match = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
            if first_match and os.path.exists(first_match):
                return first_match
        return None

    try:
        result = subprocess.run(["which", "bash"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode == 0 and result.stdout:
        first_match = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
        if first_match:
            return first_match
    return None


def get_shell_config(custom_shell_path: str | None = None) -> ShellConfig:
    """Resolve which shell runs a command. Port of TS `getShellConfig`.

    Resolution order matches TypeScript exactly:
    1. ``custom_shell_path`` when given -- and a missing one is an error rather
       than a silent fallback, so a mistyped `shellPath` setting is visible.
    2. On Windows: Git Bash in its known install locations, then bash on PATH.
    3. Otherwise: ``/bin/bash``, then bash on PATH, then plain ``sh``.
    """
    if custom_shell_path:
        if os.path.exists(custom_shell_path):
            return _get_bash_shell_config(custom_shell_path)
        raise RuntimeError(f"Custom shell path not found: {custom_shell_path}")

    if _is_windows():
        paths: list[str] = []
        # Windows environment variable names are case-sensitive as seen by
        # Python; `PROGRAMFILES` is not the name Windows sets. Matches the
        # TypeScript, which reads `process.env.ProgramFiles`.
        program_files = os.environ.get("ProgramFiles")  # noqa: SIM112
        if program_files:
            paths.append(f"{program_files}\\Git\\bin\\bash.exe")
        program_files_x86 = os.environ.get("ProgramFiles(x86)")  # noqa: SIM112
        if program_files_x86:
            paths.append(f"{program_files_x86}\\Git\\bin\\bash.exe")

        for path in paths:
            if os.path.exists(path):
                return _get_bash_shell_config(path)

        bash_on_path = _find_bash_on_path()
        if bash_on_path:
            return _get_bash_shell_config(bash_on_path)

        searched = "\n".join(f"  {p}" for p in paths)
        raise RuntimeError(
            "No bash shell found. Options:\n"
            "  1. Install Git for Windows: https://git-scm.com/download/win\n"
            "  2. Add your bash to PATH (Cygwin, MSYS2, etc.)\n"
            "  3. Set shellPath in settings.json\n\n"
            f"Searched Git Bash in:\n{searched}"
        )

    if os.path.exists("/bin/bash"):
        return _get_bash_shell_config("/bin/bash")

    bash_on_path = _find_bash_on_path()
    if bash_on_path:
        return _get_bash_shell_config(bash_on_path)

    return ShellConfig(shell="sh", args=["-c"])


# Detached child processes must be tracked so they can be killed on parent
# shutdown signals (SIGHUP/SIGTERM). The bash tool spawns with
# `start_new_session=True`, so children do not inherit the terminal's SIGHUP
# and would otherwise survive as orphans.
_tracked_detached_child_pids: set[int] = set()


def track_detached_child_pid(pid: int) -> None:
    _tracked_detached_child_pids.add(pid)


def untrack_detached_child_pid(pid: int) -> None:
    _tracked_detached_child_pids.discard(pid)


def kill_tracked_detached_children() -> None:
    for pid in list(_tracked_detached_child_pids):
        kill_process_tree(pid)
    _tracked_detached_child_pids.clear()


def kill_process_tree(pid: int) -> None:
    """Kill a process and every process in its group.

    Falls back to killing just the child when the group kill is refused, which
    is what the TypeScript `process.kill(-pid)` / `process.kill(pid)` pair does.
    """
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, signal.SIGKILL)


def get_shell_env() -> dict[str, str]:
    """Return a copy of the current process environment for spawning a shell."""
    return dict(os.environ)


def sanitize_binary_output(text: str) -> str:
    """Strip characters that crash terminal-width calculations or corrupt display.

    Removes control characters (except tab/newline/carriage return), lone
    surrogates, and Unicode "format" category characters.
    """
    out: list[str] = []
    for char in text:
        code = ord(char)
        if code in _ALLOWED_CONTROL_CODEPOINTS:
            out.append(char)
            continue
        if code < 0x20 or code == 0x7F:
            continue
        if 0xD800 <= code <= 0xDFFF:  # lone surrogate (only possible via surrogatepass decoding)
            continue
        if unicodedata.category(char) == "Cf":
            continue
        out.append(char)
    return "".join(out)
