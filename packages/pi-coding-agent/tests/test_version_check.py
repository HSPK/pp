"""Python port of `packages/coding-agent/test/version-check.test.ts`.

The upstream module queries `https://pi.dev/api/latest-version`; this port
deliberately queries a GitHub repository's latest release instead (see the
module docstring of `pi_coding_agent/utils/version_check.py`). Assertions about
the request URL, the response shape and the returned package name are therefore
expressed against the GitHub releases API; every other assertion is preserved
verbatim.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Coroutine
from typing import Any, TypeVar

import httpx
import pytest
from pi_coding_agent.utils.version_check import (
    check_for_new_pi_version,
    compare_package_versions,
    format_version_check_error,
    get_latest_pi_release,
    get_latest_pi_version,
    is_newer_package_version,
)

_T = TypeVar("_T")


def _run(coro: Coroutine[Any, Any, _T]) -> _T:
    return asyncio.run(coro)


class _CodedError(Exception):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def test_compares_package_versions() -> None:
    assert (compare_package_versions("0.70.6", "0.70.5") or 0) > 0
    assert compare_package_versions("0.70.5", "0.70.5") == 0
    assert (compare_package_versions("0.70.4", "0.70.5") or 0) < 0
    assert (compare_package_versions("5.0.0-beta.20", "5.0.0-beta.9") or 0) > 0
    assert is_newer_package_version("0.70.5", "0.70.5") is False
    assert is_newer_package_version("0.70.6", "0.70.5") is True


def _release_client(payload: dict[str, object], requests: list[httpx.Request] | None = None) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return httpx.Response(200, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_returns_only_newer_versions() -> None:
    async def scenario() -> None:
        async with _release_client({"tag_name": "1.2.3"}) as client:
            assert await check_for_new_pi_version("1.2.3", client=client, env={}) is None
        async with _release_client({"tag_name": "1.2.3"}) as client:
            newer = await check_for_new_pi_version("1.2.2", client=client, env={})
        assert newer is not None
        assert newer.version == "1.2.3"

    _run(scenario())


def test_uses_the_release_api_with_a_pi_user_agent() -> None:
    requests: list[httpx.Request] = []

    async def scenario() -> None:
        async with _release_client({"tag_name": "1.2.4"}, requests) as client:
            assert await get_latest_pi_version("1.2.3", client=client, env={}) == "1.2.4"

    _run(scenario())
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://api.github.com/repos/earendil-works/pi/releases/latest"
    assert re.match(r"^pi/1\.2\.3 ", request.headers["user-agent"])
    assert request.headers["accept"] == "application/vnd.github+json"


def test_retries_a_transient_request_when_explicitly_requested() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("fetch failed")
        return httpx.Response(200, json={"tag_name": "1.2.4"})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            release = await get_latest_pi_release("1.2.3", retry=True, client=client, env={})
        assert release is not None
        assert release.version == "1.2.4"

    _run(scenario())
    assert attempts == 3


def test_keeps_automatic_version_checks_to_one_request() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("fetch failed")

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await check_for_new_pi_version("1.2.3", client=client, env={}) is None

    _run(scenario())
    assert attempts == 1


def test_formats_nested_network_error_details() -> None:
    error = RuntimeError("fetch failed")
    error.__cause__ = BaseExceptionGroup(
        "fetch failed",
        [_CodedError("connect timeout", "ETIMEDOUT"), _CodedError("network unreachable", "ENETUNREACH")],
    )
    assert format_version_check_error(error) == "fetch failed (ETIMEDOUT, ENETUNREACH)"


def test_returns_the_active_package_metadata() -> None:
    # TS "returns the active package metadata from the version check api" feeds
    # `packageName` through the *response body* (`{ packageName: "@new-scope/pi" }`)
    # because pi.dev's API returns it. GitHub's releases API has no equivalent
    # field, so this port's `package_name` is always the resolved `repo` instead
    # (see the module docstring and README's "Version checks use GitHub
    # Releases" note) -- exercised here via the `repo=` argument rather than a
    # response field.
    async def scenario() -> None:
        async with _release_client({"tag_name": "1.2.4"}) as client:
            release = await get_latest_pi_release("1.2.3", repo="new-scope/pi", client=client, env={})
        assert release is not None
        assert release.package_name == "new-scope/pi"
        assert release.version == "1.2.4"

    _run(scenario())


def test_returns_update_notes() -> None:
    async def scenario() -> None:
        async with _release_client({"tag_name": "1.2.4", "body": " **Read this** "}) as client:
            release = await get_latest_pi_release("1.2.3", client=client, env={})
        assert release is not None
        assert release.note == "**Read this**"
        assert release.version == "1.2.4"

    _run(scenario())


def test_skips_automatic_api_calls_when_version_checks_are_disabled() -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, json={"tag_name": "1.2.4"})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await check_for_new_pi_version("1.2.3", client=client, env={"PI_SKIP_VERSION_CHECK": "1"}) is None

    _run(scenario())
    assert attempts == 0


def test_allows_direct_api_calls_when_automatic_version_checks_are_disabled() -> None:
    requests: list[httpx.Request] = []

    async def scenario() -> None:
        async with _release_client({"tag_name": "1.2.4"}, requests) as client:
            version = await get_latest_pi_version("1.2.3", client=client, env={"PI_SKIP_VERSION_CHECK": "1"})
        assert version == "1.2.4"

    _run(scenario())
    assert len(requests) == 1


@pytest.mark.parametrize("current", ["1.2.3"])
def test_offline_env_short_circuits(current: str) -> None:
    async def scenario() -> None:
        assert await get_latest_pi_release(current, env={"PI_OFFLINE": "1"}) is None

    _run(scenario())
