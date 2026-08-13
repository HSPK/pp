"""Check PyPI for a newer release of this port.

Ported from ``packages/coding-agent/src/utils/version-check.ts``, with one
deliberate change of source: upstream queries ``https://pi.dev/api/latest-version``,
which only knows about the TypeScript release. This port is published to PyPI,
so it asks PyPI for the latest version of its own distribution, and the
distribution is configurable:

1. the ``package`` argument,
2. the ``PI_VERSION_CHECK_PACKAGE`` environment variable,
3. ``versionCheckPackage`` in settings.json,
4. :data:`DEFAULT_VERSION_CHECK_PACKAGE` (this port's own distribution).

``PI_OFFLINE`` and ``PI_SKIP_VERSION_CHECK`` are honoured exactly as upstream.
The errno-detail error formatting is a direct port.

**Version comparison uses PEP 440, not semver.** Upstream compares with
``semver``, which cannot parse the versions PyPI actually serves: ``1.0`` has
too few components, and ``0.2.0rc1`` / ``0.2.0.post1`` / ``0.2.0.dev1`` are
spelled without semver's ``-`` separator. Every one of those parsed as "not a
version" and fell through to a string comparison that reported any difference
as an upgrade -- including a *downgrade* to a release candidate.

**Pre-releases are PyPI's call, not ours.** The JSON API's ``info.version`` is
the latest *stable* release; the ``/simple/`` listing is not filtered, so its
last entry can be a dev release that must never be offered as an update.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from typing import Any

import httpx
from packaging.version import InvalidVersion, Version

from ..core.config import PACKAGE_NAME
from .management_http import FetchRetryOptions, fetch_with_retry

DEFAULT_VERSION_CHECK_PACKAGE = PACKAGE_NAME
PYPI_API_BASE = "https://pypi.org/pypi"
DEFAULT_VERSION_CHECK_TIMEOUT_MS = 10_000


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


def _parse_version(value: str) -> Version | None:
    try:
        return Version(value.strip())
    except InvalidVersion:
        return None


def compare_package_versions(left_version: str, right_version: str) -> int | None:
    """`semver.compare` semantics over PEP 440; ``None`` when either side is unparseable."""
    left = _parse_version(left_version)
    right = _parse_version(right_version)
    if left is None or right is None:
        return None
    if left == right:
        return 0
    return -1 if left < right else 1


def is_newer_package_version(candidate_version: str, current_version: str) -> bool:
    """Whether `candidate_version` is an upgrade from `current_version`.

    An unparseable version is *not* treated as an upgrade. Upstream falls back
    to "any difference means newer", which on PyPI would offer a release
    candidate as an upgrade over the stable release it precedes.
    """
    comparison = compare_package_versions(candidate_version, current_version)
    return comparison is not None and comparison > 0


def resolve_version_check_package(package: str | None = None, settings_manager: Any = None, env: Any = None) -> str:
    env = os.environ if env is None else env
    if package and package.strip():
        return package.strip()
    from_env = env.get("PI_VERSION_CHECK_PACKAGE")
    if from_env and from_env.strip():
        return from_env.strip()
    if settings_manager is not None:
        getter = getattr(settings_manager, "get_version_check_package", None)
        configured = getter() if getter is not None else None
        if configured and configured.strip():
            return configured.strip()
    return DEFAULT_VERSION_CHECK_PACKAGE


async def get_latest_pi_release(
    current_version: str,
    *,
    package: str | None = None,
    settings_manager: Any = None,
    timeout_ms: int | None = None,
    retry: bool = False,
    client: httpx.AsyncClient | None = None,
    env: Any = None,
) -> LatestPiRelease | None:
    env = os.environ if env is None else env
    if env.get("PI_OFFLINE"):
        return None

    resolved_package = resolve_version_check_package(package, settings_manager, env)
    response = await fetch_with_retry(
        f"{PYPI_API_BASE}/{resolved_package}/json",
        headers={
            "User-Agent": get_pi_user_agent(current_version),
            "Accept": "application/json",
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

    info = data.get("info")
    if not isinstance(info, dict):
        return None

    # `info.version` is the latest *stable* release: PyPI excludes
    # pre-releases here, unlike the `/simple/` listing.
    raw_version = info.get("version")
    if not isinstance(raw_version, str) or not raw_version.strip():
        return None
    version = raw_version.strip()

    # A yanked release is one the maintainer withdrew; installers skip it, so
    # offering it as an upgrade would send the user to a dead end.
    if info.get("yanked") is True:
        return None

    url = f"https://pypi.org/project/{resolved_package}/{version}/"
    return LatestPiRelease(version=version, package_name=resolved_package, note=None, url=url)


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
    "DEFAULT_VERSION_CHECK_PACKAGE",
    "DEFAULT_VERSION_CHECK_TIMEOUT_MS",
    "LatestPiRelease",
    "check_for_new_pi_version",
    "compare_package_versions",
    "format_version_check_error",
    "get_latest_pi_release",
    "get_latest_pi_version",
    "get_pi_user_agent",
    "is_newer_package_version",
    "resolve_version_check_package",
]
