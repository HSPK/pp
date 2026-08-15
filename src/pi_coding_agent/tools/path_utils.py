"""Path resolution helpers for coding-agent tools.

Python port of `packages/coding-agent/src/core/tools/path-utils.ts`.
"""

from __future__ import annotations

import os
import re
import unicodedata

from pi_coding_agent.utils.paths import PathInputOptions, normalize_path, resolve_path

_NARROW_NO_BREAK_SPACE = "\u202f"

_RESOLVE_OPTIONS = PathInputOptions(normalize_unicode_spaces=True, strip_at_prefix=True)


def path_exists(file_path: str) -> bool:
    return os.path.exists(file_path)


def expand_path(file_path: str) -> str:
    return normalize_path(file_path, PathInputOptions(normalize_unicode_spaces=True, strip_at_prefix=True))


def resolve_to_cwd(file_path: str, cwd: str) -> str:
    """Resolve a path relative to the given cwd. Handles ``~`` expansion and absolute paths."""
    return resolve_path(file_path, cwd, _RESOLVE_OPTIONS)


def _try_macos_screenshot_path(file_path: str) -> str:
    """Replace a plain space before AM/PM with a narrow no-break space, matching macOS screenshot names."""
    return re.sub(r" (AM|PM)\.", lambda m: f"{_NARROW_NO_BREAK_SPACE}{m.group(1)}.", file_path, flags=re.IGNORECASE)


def _try_nfd_variant(file_path: str) -> str:
    """macOS stores filenames in NFD (decomposed) form; try converting user input to NFD."""
    return unicodedata.normalize("NFD", file_path)


def _try_curly_quote_variant(file_path: str) -> str:
    """macOS uses U+2019 (right single quotation mark) in names like "Capture d'écran"."""
    return file_path.replace("'", "\u2019")


def resolve_read_path(file_path: str, cwd: str) -> str:
    """Resolve a read path, trying macOS filename quirks (NFD, AM/PM spacing, curly quotes) as fallbacks."""
    resolved = resolve_to_cwd(file_path, cwd)

    if path_exists(resolved):
        return resolved

    am_pm_variant = _try_macos_screenshot_path(resolved)
    if am_pm_variant != resolved and path_exists(am_pm_variant):
        return am_pm_variant

    nfd_variant = _try_nfd_variant(resolved)
    if nfd_variant != resolved and path_exists(nfd_variant):
        return nfd_variant

    curly_variant = _try_curly_quote_variant(resolved)
    if curly_variant != resolved and path_exists(curly_variant):
        return curly_variant

    nfd_curly_variant = _try_curly_quote_variant(nfd_variant)
    if nfd_curly_variant != resolved and path_exists(nfd_curly_variant):
        return nfd_curly_variant

    return resolved


# The TypeScript source exposes both sync and async variants (`resolveReadPath`
# and `resolveReadPathAsync`) because Node's fs module has separate sync/async
# APIs. Python's `os.path.exists` is always synchronous, so a single function
# covers both call sites; `resolve_read_path_async` is kept as an alias so
# callers that awaited the TS async variant have a matching name to import.
async def resolve_read_path_async(file_path: str, cwd: str) -> str:
    return resolve_read_path(file_path, cwd)
