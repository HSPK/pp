"""Python port of `packages/coding-agent/test/http-dispatcher.test.ts`.

Only the proxy-settings half is portable. The dispatcher half asserts on
undici's global dispatcher and Node's `tls.connect`
(`autoSelectFamilyAttemptTimeout`); this port has no global dispatcher -
`configure_http_dispatcher` only sets `pi_ai`'s idle timeout - so that case is
covered by asserting the observable idle-timeout effect instead.
"""

from __future__ import annotations

import os

import pytest
from pi_ai.utils import http as pi_http
from pi_coding_agent.core.http_dispatcher import (
    DEFAULT_HTTP_IDLE_TIMEOUT_MS,
    apply_http_proxy_settings,
    configure_http_dispatcher,
)

PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY")


@pytest.fixture(autouse=True)
def _clear_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in PROXY_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_applies_http_proxy_to_http_proxy_and_https_proxy() -> None:
    apply_http_proxy_settings("http://127.0.0.1:7890")

    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7890"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"


def test_does_not_override_existing_proxy_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://env-http:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://env-https:8080")

    apply_http_proxy_settings("http://settings:7890")

    assert os.environ["HTTP_PROXY"] == "http://env-http:8080"
    assert os.environ["HTTPS_PROXY"] == "http://env-https:8080"


def test_ignores_empty_values() -> None:
    apply_http_proxy_settings("   ")

    assert os.environ.get("HTTP_PROXY") is None
    assert os.environ.get("HTTPS_PROXY") is None


def test_configure_http_dispatcher_installs_the_idle_timeout() -> None:
    # Stand-in for the TypeScript `tls.connect` assertion: this port configures
    # `pi_ai`'s shared HTTP timeout rather than an undici global dispatcher.
    previous = pi_http.get_idle_timeout_ms()
    try:
        configure_http_dispatcher(2_000)
        assert pi_http.get_idle_timeout_ms() == 2_000
    finally:
        pi_http.set_idle_timeout_ms(previous)


def test_configure_http_dispatcher_defaults_without_an_argument() -> None:
    # TS calls `configureHttpDispatcher()` with no argument and asserts the
    # 2000 ms value comes from the implementation, so pin the default here
    # rather than only the value the test passes in.
    previous = pi_http.get_idle_timeout_ms()
    try:
        configure_http_dispatcher()
        assert pi_http.get_idle_timeout_ms() == DEFAULT_HTTP_IDLE_TIMEOUT_MS
    finally:
        pi_http.set_idle_timeout_ms(previous)


def test_configure_http_dispatcher_does_not_touch_the_proxy_environment() -> None:
    # Stand-in for TS's `expect(net.getDefaultAutoSelectFamilyAttemptTimeout())
    # .toBe(originalAttemptTimeoutMs)`: configuring the dispatcher must not
    # mutate unrelated process-wide state.
    previous = pi_http.get_idle_timeout_ms()
    try:
        configure_http_dispatcher(1_000)
        assert os.environ.get("HTTP_PROXY") is None
        assert os.environ.get("HTTPS_PROXY") is None
    finally:
        pi_http.set_idle_timeout_ms(previous)


def test_configure_http_dispatcher_rejects_invalid_timeouts() -> None:
    with pytest.raises(ValueError, match="Invalid HTTP idle timeout"):
        configure_http_dispatcher(-1)
