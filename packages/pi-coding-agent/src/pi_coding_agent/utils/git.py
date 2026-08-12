"""Git source URL parsing.

Python port of `packages/coding-agent/src/utils/git.ts`.

**hosted-git-info omission.** The TypeScript implementation calls the npm
`hosted-git-info` package to recognize provider-specific shorthand forms
(`github:user/repo`, bare `user/repo` resolved against github.com, `.git`
suffix stripping via provider-specific rules, etc.) before falling back to
its own `parseGenericGitUrl`. That fallback already handles every URL shape
exercised by the ported test suite (protocol URLs for github.com/gitlab.com/
bitbucket.org/codeberg.org/any host with a dot, `git@host:path` SCP-like
syntax, and `git:host/path` shorthand), so this port implements only
`parseGenericGitUrl`'s logic and drops the `hosted-git-info` provider-alias
lookup. A consequence: bare `github:user/repo` or unqualified `user/repo`
(resolved implicitly against github.com) shorthands are not recognized here;
callers must use an explicit host (`git:github.com/user/repo`) or a full
URL, which every ported test case already does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

_SCP_LIKE_RE = re.compile(r"^git@([^:]+):(.+)$")
_PROTOCOL_RE = re.compile(r"^(https?|ssh|git)://", re.IGNORECASE)


@dataclass
class GitSource:
    """Parsed git source. Mirrors the TypeScript ``GitSource`` interface."""

    repo: str
    host: str
    path: str
    ref: str | None = None
    pinned: bool = False
    type: str = "git"


def _split_ref(url: str) -> tuple[str, str | None]:
    """Split a ``repo@ref`` suffix off ``url``. Returns ``(repo, ref)``."""
    scp_match = _SCP_LIKE_RE.match(url)
    if scp_match:
        host = scp_match.group(1)
        path_with_maybe_ref = scp_match.group(2)
        ref_separator = path_with_maybe_ref.find("@")
        if ref_separator < 0:
            return url, None
        repo_path = path_with_maybe_ref[:ref_separator]
        ref = path_with_maybe_ref[ref_separator + 1 :]
        if not repo_path or not ref:
            return url, None
        return f"git@{host}:{repo_path}", ref

    if "://" in url:
        try:
            parsed = urlparse(url)
        except ValueError:
            return url, None
        path_with_maybe_ref = parsed.path.lstrip("/")
        ref_separator = path_with_maybe_ref.find("@")
        if ref_separator < 0:
            return url, None
        repo_path = path_with_maybe_ref[:ref_separator]
        ref = path_with_maybe_ref[ref_separator + 1 :]
        if not repo_path or not ref:
            return url, None
        rebuilt = parsed._replace(path=f"/{repo_path}")
        return rebuilt.geturl().rstrip("/"), ref

    slash_index = url.find("/")
    if slash_index < 0:
        return url, None
    host = url[:slash_index]
    path_with_maybe_ref = url[slash_index + 1 :]
    ref_separator = path_with_maybe_ref.find("@")
    if ref_separator < 0:
        return url, None
    repo_path = path_with_maybe_ref[:ref_separator]
    ref = path_with_maybe_ref[ref_separator + 1 :]
    if not repo_path or not ref:
        return url, None
    return f"{host}/{repo_path}", ref


_PERCENT_ESCAPE_RE = re.compile(r"%([0-9A-Fa-f]{2})")


def _decode_for_validation(value: str) -> str | None:
    """Strict percent-decode. Unlike `urllib.parse.unquote` (which silently
    leaves malformed ``%`` escapes untouched), this mirrors JavaScript's
    ``decodeURIComponent``, which throws on a lone/malformed ``%``.
    """
    index = 0
    while True:
        index = value.find("%", index)
        if index == -1:
            break
        if not _PERCENT_ESCAPE_RE.match(value, index):
            return None
        index += 1
    try:
        return unquote(value, errors="strict")
    except (UnicodeDecodeError, ValueError):
        return None


def _has_unsafe_git_install_part(value: str, allow_slash: bool) -> bool:
    decoded = _decode_for_validation(value)
    if decoded is None:
        return True
    for candidate in (value, decoded):
        if "\0" in candidate or "\\" in candidate or candidate.startswith("/"):
            return True
        if not allow_slash and "/" in candidate:
            return True
        if ".." in candidate.split("/"):
            return True
    return False


def _build_git_source(repo: str, host: str, path: str, ref: str | None) -> GitSource | None:
    if path.startswith("/"):
        return None
    normalized_path = re.sub(r"^/+", "", re.sub(r"\.git$", "", path))
    if not host or not normalized_path or len(normalized_path.split("/")) < 2:
        return None
    if _has_unsafe_git_install_part(host, False) or _has_unsafe_git_install_part(normalized_path, True):
        return None

    return GitSource(repo=repo, host=host, path=normalized_path, ref=ref, pinned=bool(ref))


def _parse_generic_git_url(url: str) -> GitSource | None:
    repo_without_ref, ref = _split_ref(url)
    repo = repo_without_ref
    host = ""
    path = ""

    scp_match = _SCP_LIKE_RE.match(repo_without_ref)
    if scp_match:
        host, path = scp_match.group(1), scp_match.group(2)
    elif repo_without_ref.startswith(("https://", "http://", "ssh://", "git://")):
        try:
            parsed = urlparse(repo_without_ref)
        except ValueError:
            return None
        host = parsed.hostname or ""
        path = parsed.path.lstrip("/")
    else:
        slash_index = repo_without_ref.find("/")
        if slash_index < 0:
            return None
        host = repo_without_ref[:slash_index]
        path = repo_without_ref[slash_index + 1 :]
        if "." not in host and host != "localhost":
            return None
        repo = f"https://{repo_without_ref}"

    return _build_git_source(repo, host, path, ref)


def parse_git_url(source: str) -> GitSource | None:
    """Parse a git source string into a :class:`GitSource`, or ``None``.

    Rules (matching the TypeScript ``parseGitUrl``):
    - With a ``git:`` prefix, accept shorthand forms (``git@host:path``,
      ``host/path``) in addition to full URLs.
    - Without a ``git:`` prefix, only accept explicit ``http(s)://``,
      ``ssh://``, or ``git://`` URLs.
    """
    trimmed = source.strip()
    has_git_prefix = trimmed.startswith("git:")
    url = trimmed[4:].strip() if has_git_prefix else trimmed

    if not has_git_prefix and not _PROTOCOL_RE.match(url):
        return None

    return _parse_generic_git_url(url)


__all__ = ["GitSource", "parse_git_url"]
