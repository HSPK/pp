"""CHANGELOG.md parsing and link normalization.

Ported from ``packages/coding-agent/src/utils/changelog.ts``.
"""

from __future__ import annotations

import posixpath
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from ..core.config import get_changelog_path

GITHUB_REPO = "earendil-works/pi"
CHANGELOG_LINK_BASE_PATH = "packages/coding-agent"
_LEGACY_REPO_RE = re.compile(r"^https://github\.com/(?:badlogic|earendil-works)/pi-mono(?=/|$)")
_URL_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
_INLINE_MARKDOWN_LINK_RE = re.compile(r"(!?\[[^\]\n]+\]\()([^\s)]+)((?:\s+[^)]*)?\))")
_VERSION_HEADER_RE = re.compile(r"##\s+\[?(\d+)\.(\d+)\.(\d+)\]?")


@dataclass
class ChangelogEntry:
    major: int
    minor: int
    patch: int
    content: str


def _entry_version(entry: ChangelogEntry) -> str:
    return f"{entry.major}.{entry.minor}.{entry.patch}"


def _normalize_tag(version: str | ChangelogEntry) -> str:
    version_string = version if isinstance(version, str) else _entry_version(version)
    return version_string if version_string.startswith("v") else f"v{version_string}"


def _split_local_target(target: str) -> tuple[str, str, str]:
    """Return ``(fragment, path_part, query)`` for a local link target."""
    hash_index = target.find("#")
    before_hash = target if hash_index == -1 else target[:hash_index]
    fragment = "" if hash_index == -1 else target[hash_index:]
    query_index = before_hash.find("?")

    if query_index == -1:
        return fragment, before_hash, ""
    return fragment, before_hash[:query_index], before_hash[query_index:]


def _posix_normalize(value: str) -> str:
    """Node's ``path.posix.normalize``: like ``posixpath.normpath`` but it keeps
    a trailing slash, which decides the blob-vs-tree route below."""
    normalized = posixpath.normpath(value)
    if value.endswith("/") and not normalized.endswith("/"):
        normalized += "/"
    return normalized


def _resolve_repository_path(target_path: str) -> str | None:
    normalized_target = target_path.replace("\\", "/")
    if normalized_target.startswith("/"):
        joined = _posix_normalize(normalized_target.lstrip("/"))
    else:
        joined = _posix_normalize(posixpath.join(CHANGELOG_LINK_BASE_PATH, normalized_target))

    if joined == "." or joined.startswith("../") or joined == "..":
        return None
    return joined


def _is_directory_target(original_path: str, repository_path: str) -> bool:
    if original_path.endswith("/"):
        return True
    return "." not in posixpath.basename(repository_path)


def _normalize_changelog_link_target(target: str, tag: str) -> str:
    repo_url = f"https://github.com/{GITHUB_REPO}"
    canonical_target = _LEGACY_REPO_RE.sub(repo_url, target, count=1)

    for route in ("blob", "tree"):
        for branch in ("main", "master"):
            floating_ref_prefix = f"{repo_url}/{route}/{branch}/"
            if canonical_target.startswith(floating_ref_prefix):
                canonical_target = f"{repo_url}/{route}/{tag}/{canonical_target[len(floating_ref_prefix) :]}"

    if canonical_target.startswith("#") or canonical_target.startswith("//") or _URL_SCHEME_RE.match(canonical_target):
        return canonical_target

    fragment, path_part, query = _split_local_target(canonical_target)
    if not path_part:
        return canonical_target

    repository_path = _resolve_repository_path(path_part)
    if not repository_path:
        return canonical_target

    route = "tree" if _is_directory_target(path_part, repository_path) else "blob"
    # JS `encodeURI` leaves reserved URI characters (;/?:@&=+$,#!~*'()) alone.
    encoded = quote(repository_path, safe="!#$&'()*+,-./:;=?@_~")
    return f"https://github.com/{GITHUB_REPO}/{route}/{tag}/{encoded}{query}{fragment}"


def normalize_changelog_links(markdown: str, version: str | ChangelogEntry) -> str:
    tag = _normalize_tag(version)

    def replace(match: re.Match[str]) -> str:
        prefix, target, suffix = match.group(1), match.group(2), match.group(3)
        return f"{prefix}{_normalize_changelog_link_target(target, tag)}{suffix}"

    return _INLINE_MARKDOWN_LINK_RE.sub(replace, markdown)


def parse_changelog(changelog_path: str) -> list[ChangelogEntry]:
    """Parse changelog entries from CHANGELOG.md.

    Scans for ``##`` lines and collects content until the next ``##`` or EOF.
    """
    if not Path(changelog_path).exists():
        return []

    try:
        content = Path(changelog_path).read_text(encoding="utf-8")
    except OSError as error:
        print(f"Warning: Could not parse changelog: {error}", file=sys.stderr)
        return []

    lines = content.split("\n")
    entries: list[ChangelogEntry] = []

    current_lines: list[str] = []
    current_version: tuple[int, int, int] | None = None

    for line in lines:
        if line.startswith("## "):
            if current_version is not None and len(current_lines) > 0:
                entries.append(ChangelogEntry(*current_version, content="\n".join(current_lines).strip()))

            version_match = _VERSION_HEADER_RE.search(line)
            if version_match:
                current_version = (
                    int(version_match.group(1)),
                    int(version_match.group(2)),
                    int(version_match.group(3)),
                )
                current_lines = [line]
            else:
                current_version = None
                current_lines = []
        elif current_version is not None:
            current_lines.append(line)

    if current_version is not None and len(current_lines) > 0:
        entries.append(ChangelogEntry(*current_version, content="\n".join(current_lines).strip()))

    return entries


def compare_versions(v1: ChangelogEntry, v2: ChangelogEntry) -> int:
    """-1 if v1 < v2, 0 if equal, 1 if v1 > v2 (as a sign; TS returns the delta)."""
    if v1.major != v2.major:
        return v1.major - v2.major
    if v1.minor != v2.minor:
        return v1.minor - v2.minor
    return v1.patch - v2.patch


def get_new_entries(entries: list[ChangelogEntry], last_version: str) -> list[ChangelogEntry]:
    """Entries newer than ``last_version``."""
    parts: list[int] = []
    for part in last_version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            # JS `Number("x")` is NaN and `NaN || 0` is 0.
            parts.append(0)

    last = ChangelogEntry(
        major=parts[0] if len(parts) > 0 else 0,
        minor=parts[1] if len(parts) > 1 else 0,
        patch=parts[2] if len(parts) > 2 else 0,
        content="",
    )
    return [entry for entry in entries if compare_versions(entry, last) > 0]


__all__ = [
    "CHANGELOG_LINK_BASE_PATH",
    "GITHUB_REPO",
    "ChangelogEntry",
    "compare_versions",
    "get_changelog_path",
    "get_new_entries",
    "normalize_changelog_links",
    "parse_changelog",
]
