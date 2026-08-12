"""Path normalization and resolution helpers.

Python port of `packages/coding-agent/src/utils/paths.ts` (the subset used by
the built-in tools: ``normalize_path`` and ``resolve_path``, plus
``get_file_revision`` used by `core/auth_storage.py`, and
``normalize_windows_shell_path`` for Git Bash/MSYS/Cygwin/WSL drive
conversion). The cloud-sync/xattr helpers are TS-only concerns with no
Python caller and are not ported.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

_UNICODE_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
_INVALID_PERCENT_ESCAPE = re.compile("%(?![0-9A-Fa-f]{2})")
_REMOTE_SOURCE_PREFIXES = ("npm:", "git:", "github:", "http:", "https:", "ssh:")
_WINDOWS_SHELL_DRIVE = re.compile(r"^/(?:mnt/|cygdrive/)?([a-zA-Z])(?:/(.*))?$")


@dataclass
class PathInputOptions:
    """Mirrors the TypeScript ``PathInputOptions`` interface."""

    trim: bool = False
    expand_tilde: bool = True
    home_dir: str | None = None
    strip_at_prefix: bool = False
    normalize_unicode_spaces: bool = False


def _file_url_to_path(url: str) -> str:
    """Convert a ``file:`` URL to a filesystem path.

    Node's ``fileURLToPath`` runs the pathname through ``decodeURIComponent``,
    which throws for malformed percent escapes. `urllib`'s ``unquote`` silently
    passes them through, so the escapes are validated first to keep the same
    "invalid file URL raises" contract.
    """
    parsed = urlparse(url)
    if _INVALID_PERCENT_ESCAPE.search(parsed.path):
        raise ValueError(f"Invalid file URL: {url}")
    try:
        return unquote(parsed.path, errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"Invalid file URL: {url}") from error


def is_local_path(value: str) -> bool:
    """Whether ``value`` names a local path rather than a remote package spec.

    Port of TypeScript ``isLocalPath``. ``file:`` URLs count as local because
    `resolve_path` resolves them.
    """
    return not value.strip().startswith(_REMOTE_SOURCE_PREFIXES)


def canonicalize_path(path: str) -> str:
    """Resolve symlinks, falling back to the input when the path is missing.

    Port of TypeScript ``canonicalizePath`` (``realpathSync`` in a try/catch).
    """
    try:
        return os.path.realpath(path, strict=True)
    except OSError:
        return path


def get_file_revision(path: str) -> str | None:
    """Cheap file-identity/freshness fingerprint for cache invalidation.

    Port of TypeScript ``getFileRevision`` (device/inode/size/mtime/ctime from
    ``statSync(path, { bigint: true })``). Python's `os.stat` already exposes
    nanosecond fields, so no bigint handling is needed. Returns `None` when the
    path does not exist (mirrors the TypeScript try/catch-`undefined`).
    """
    try:
        stats = os.stat(path)
    except OSError:
        return None
    return f"{stats.st_dev}:{stats.st_ino}:{stats.st_size}:{stats.st_mtime_ns}:{stats.st_ctime_ns}"


def normalize_windows_shell_path(file_path: str) -> str:
    """Convert Git Bash, MSYS, Cygwin, and WSL drive paths to native Windows form.

    Port of TypeScript ``normalizeWindowsShellPath``.
    """
    if not file_path.startswith("/") or file_path.startswith("//") or "\\" in file_path:
        return file_path
    match = _WINDOWS_SHELL_DRIVE.match(file_path)
    if not match:
        return file_path
    suffix = match.group(2)
    suffix = suffix.replace("/", "\\") if suffix is not None else ""
    return f"{match.group(1).upper()}:\\{suffix}"


def normalize_path(input_path: str, options: PathInputOptions | None = None) -> str:
    """Apply unicode-space normalization, `@`-stripping, and `~` expansion."""
    options = options or PathInputOptions()
    normalized = input_path.strip() if options.trim else input_path
    if options.normalize_unicode_spaces:
        normalized = _UNICODE_SPACES.sub(" ", normalized)
    if options.strip_at_prefix and normalized.startswith("@"):
        normalized = normalized[1:]
    if sys.platform == "win32":
        normalized = normalize_windows_shell_path(normalized)

    if options.expand_tilde:
        home = options.home_dir if options.home_dir is not None else str(Path.home())
        if normalized == "~":
            return home
        if normalized.startswith("~/") or (sys.platform == "win32" and normalized.startswith("~\\")):
            return str(Path(home) / normalized[2:])

    if normalized.startswith("file://"):
        return _file_url_to_path(normalized)

    return normalized


def resolve_path(input_path: str, base_dir: str | None = None, options: PathInputOptions | None = None) -> str:
    """Resolve ``input_path`` to an absolute path relative to ``base_dir``.

    Mirrors Node's `path.resolve`: purely syntactic normalization (no symlink
    resolution), unlike `Path.resolve()`.
    """
    normalized = normalize_path(input_path, options)
    normalized_base = normalize_path(base_dir if base_dir is not None else str(Path.cwd()))
    if os.path.isabs(normalized):
        return os.path.normpath(normalized)
    return os.path.normpath(os.path.join(normalized_base, normalized))


def format_path_relative_to_cwd_or_absolute(file_path: str, cwd: str) -> str:
    """Display form of a path: cwd-relative when inside `cwd`, else absolute.

    Port of `formatPathRelativeToCwdOrAbsolute` (`utils/paths.ts:119`). Always
    POSIX-separated, because the result goes into rendered tool output where
    upstream normalises separators.
    """
    absolute_path = resolve_path(file_path, cwd)
    display = get_cwd_relative_path(absolute_path, cwd) or absolute_path
    return display.replace(os.sep, "/")


def get_cwd_relative_path(file_path: str, cwd: str) -> str | None:
    """``file_path`` relative to ``cwd``, or `None` when it escapes ``cwd``.

    Port of TypeScript ``getCwdRelativePath``.
    """
    resolved_cwd = resolve_path(cwd)
    resolved_path = resolve_path(file_path, resolved_cwd)
    relative_path = os.path.relpath(resolved_path, resolved_cwd)
    if relative_path == os.curdir:
        return os.curdir
    if relative_path == os.pardir or relative_path.startswith(f"{os.pardir}{os.sep}") or os.path.isabs(relative_path):
        return None
    return relative_path
