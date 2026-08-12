"""Check GitHub Releases for a newer version.

Ported from ``packages/coding-agent/src/utils/version-check.ts``, with one
deliberate change: upstream queries ``https://pi.dev/api/latest-version``,
which only knows about the TypeScript release. This port checks a **GitHub
repository's latest release** instead, and the repository is configurable:

1. the ``repo`` argument,
2. the ``PI_VERSION_CHECK_REPO`` environment variable,
3. ``versionCheckRepo`` in settings.json,
4. :data:`DEFAULT_VERSION_CHECK_REPO`.

``PI_OFFLINE`` and ``PI_SKIP_VERSION_CHECK`` are honoured exactly as upstream.
The semver comparison and the errno-detail error formatting are direct ports.
"""

from __future__ import annotations

import os
import platform
import re
import sys
from dataclasses import dataclass
from typing import Any

import httpx

from .management_http import FetchRetryOptions, fetch_with_retry

DEFAULT_VERSION_CHECK_REPO = "earendil-works/pi"
GITHUB_API_BASE = "https://api.github.com"
DEFAULT_VERSION_CHECK_TIMEOUT_MS = 10_000

# semver "valid" subset: major.minor.patch with optional prerelease/build.
_SEMVER_RE = re.compile(
    r"^v?(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+(?P<build>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


@dataclass
class LatestPiRelease:
    version: str
    package_name: str | None = None
    note: str | None = None
    url: str | None = None


def get_pi_user_agent(version: str) -> str:
    """Ported from ``utils/pi-user-agent.ts``."""
    runtime = f"python/{platform.python_version()}"
    return f"pi/{version} ({sys.platform}; {runtime}; {platform.machine()})"


def format_version_check_error(error: BaseException) -> str:
    """Surface the errno detail that a bare "fetch failed" hides."""
    root_message = str(error) or type(error).__name__
    causes: list[BaseException] = []
    cause = error.__cause__ or error.__context__
    if isinstance(cause, BaseExceptionGroup):
        causes = list(cause.exceptions)
    elif cause is not None:
        causes = [cause]

    codes: list[str] = []
    for candidate in causes:
        code = getattr(candidate, "errno", None) or getattr(candidate, "code", None)
        if isinstance(code, str):
            codes.append(code)
        elif isinstance(code, int):
            codes.append(str(code))

    if codes:
        return f"{root_message} ({', '.join(dict.fromkeys(codes))})"
    cause_message = next((str(c) for c in causes if str(c)), None)
    return f"{root_message} (cause: {cause_message})" if cause_message else root_message


def _parse_semver(value: str) -> tuple[int, int, int, tuple[Any, ...]] | None:
    match = _SEMVER_RE.match(value.strip())
    if match is None:
        return None
    prerelease = match.group("prerelease")
    if prerelease is None:
        # No prerelease sorts *above* any prerelease, so use a sentinel.
        identifiers: tuple[Any, ...] = ()
    else:
        identifiers = tuple(int(part) if part.isdigit() else part for part in prerelease.split("."))
    return int(match.group("major")), int(match.group("minor")), int(match.group("patch")), identifiers


def _compare_prerelease(left: tuple[Any, ...], right: tuple[Any, ...]) -> int:
    if not left and not right:
        return 0
    if not left:
        return 1
    if not right:
        return -1
    for left_part, right_part in zip(left, right, strict=False):
        if left_part == right_part:
            continue
        left_numeric = isinstance(left_part, int)
        right_numeric = isinstance(right_part, int)
        if left_numeric and right_numeric:
            return -1 if left_part < right_part else 1
        if left_numeric != right_numeric:
            # Numeric identifiers always sort below alphanumeric ones.
            return -1 if left_numeric else 1
        return -1 if str(left_part) < str(right_part) else 1
    if len(left) == len(right):
        return 0
    return -1 if len(left) < len(right) else 1


def compare_package_versions(left_version: str, right_version: str) -> int | None:
    """`semver.compare` semantics; ``None`` when either side is not valid semver."""
    left = _parse_semver(left_version)
    right = _parse_semver(right_version)
    if left is None or right is None:
        return None
    for index in range(3):
        if left[index] != right[index]:
            return -1 if left[index] < right[index] else 1
    return _compare_prerelease(left[3], right[3])


def is_newer_package_version(candidate_version: str, current_version: str) -> bool:
    comparison = compare_package_versions(candidate_version, current_version)
    if comparison is not None:
        return comparison > 0
    return candidate_version.strip() != current_version.strip()


def resolve_version_check_repo(repo: str | None = None, settings_manager: Any = None, env: Any = None) -> str:
    env = os.environ if env is None else env
    if repo and repo.strip():
        return repo.strip()
    from_env = env.get("PI_VERSION_CHECK_REPO")
    if from_env and from_env.strip():
        return from_env.strip()
    if settings_manager is not None:
        getter = getattr(settings_manager, "get_version_check_repo", None)
        configured = getter() if getter is not None else None
        if configured and configured.strip():
            return configured.strip()
    return DEFAULT_VERSION_CHECK_REPO


async def get_latest_pi_release(
    current_version: str,
    *,
    repo: str | None = None,
    settings_manager: Any = None,
    timeout_ms: int | None = None,
    retry: bool = False,
    client: httpx.AsyncClient | None = None,
    env: Any = None,
) -> LatestPiRelease | None:
    env = os.environ if env is None else env
    if env.get("PI_OFFLINE"):
        return None

    resolved_repo = resolve_version_check_repo(repo, settings_manager, env)
    response = await fetch_with_retry(
        f"{GITHUB_API_BASE}/repos/{resolved_repo}/releases/latest",
        headers={
            "User-Agent": get_pi_user_agent(current_version),
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        options=FetchRetryOptions(
            max_retries=2 if retry else 0,
            timeout_ms=timeout_ms if timeout_ms is not None else DEFAULT_VERSION_CHECK_TIMEOUT_MS,
        ),
        client=client,
    )
    if response.status_code >= 400:
        return None

    try:
        data = response.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None

    # GitHub releases carry the version in `tag_name` (usually `vX.Y.Z`).
    raw_tag = data.get("tag_name") or data.get("name")
    if not isinstance(raw_tag, str) or not raw_tag.strip():
        return None
    version = raw_tag.strip()
    if version.startswith("v"):
        version = version[1:]

    body = data.get("body")
    note = body.strip() if isinstance(body, str) and body.strip() else None
    url = data.get("html_url") if isinstance(data.get("html_url"), str) else None
    return LatestPiRelease(version=version, package_name=resolved_repo, note=note, url=url)


async def get_latest_pi_version(current_version: str, **kwargs: Any) -> str | None:
    release = await get_latest_pi_release(current_version, **kwargs)
    return release.version if release is not None else None


async def check_for_new_pi_version(current_version: str, **kwargs: Any) -> LatestPiRelease | None:
    """Return the latest release only when it is newer. Never raises."""
    env = kwargs.get("env") or os.environ
    if env.get("PI_SKIP_VERSION_CHECK"):
        return None
    try:
        latest = await get_latest_pi_release(current_version, **kwargs)
    except Exception:
        return None
    if latest is not None and is_newer_package_version(latest.version, current_version):
        return latest
    return None


__all__ = [
    "DEFAULT_VERSION_CHECK_REPO",
    "DEFAULT_VERSION_CHECK_TIMEOUT_MS",
    "LatestPiRelease",
    "check_for_new_pi_version",
    "compare_package_versions",
    "format_version_check_error",
    "get_latest_pi_release",
    "get_latest_pi_version",
    "get_pi_user_agent",
    "is_newer_package_version",
    "resolve_version_check_repo",
]
