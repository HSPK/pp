"""Global HTTP idle timeout and proxy configuration.

Ported from ``packages/coding-agent/src/core/http-dispatcher.ts``.

The TypeScript version installs an undici ``EnvHttpProxyAgent`` as the global
dispatcher so every ``fetch`` inherits the configured ``bodyTimeout`` and
``headersTimeout``. There is no global dispatcher in ``httpx``; the equivalent
here is :func:`pi_ai.utils.http.set_idle_timeout_ms`, which every provider
request picks up through ``build_timeout``. Proxy handling is env-var based in
both ports, so :func:`apply_http_proxy_settings` is a direct translation.
"""

from __future__ import annotations

import math
import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from pi_ai.utils.http import set_idle_timeout_ms

DEFAULT_HTTP_IDLE_TIMEOUT_MS = 300_000


@dataclass(frozen=True)
class HttpIdleTimeoutChoice:
    label: str
    timeout_ms: int


HTTP_IDLE_TIMEOUT_CHOICES: tuple[HttpIdleTimeoutChoice, ...] = (
    HttpIdleTimeoutChoice("30 sec", 30_000),
    HttpIdleTimeoutChoice("1 min", 60_000),
    HttpIdleTimeoutChoice("2 min", 120_000),
    HttpIdleTimeoutChoice("5 min", 300_000),
    HttpIdleTimeoutChoice("disabled", 0),
)


def parse_http_idle_timeout_ms(value: Any) -> int | None:
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.lower() == "disabled":
            return 0
        if len(trimmed) == 0:
            return None
        try:
            number = float(trimmed)
        except ValueError:
            return None
        return parse_http_idle_timeout_ms(number)

    # `isinstance(True, int)` is True in Python; JS `typeof true === "number"`
    # is false, so booleans must be rejected to match.
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return math.floor(value)


def format_http_idle_timeout_ms(timeout_ms: int) -> str:
    for choice in HTTP_IDLE_TIMEOUT_CHOICES:
        if choice.timeout_ms == timeout_ms:
            return choice.label
    seconds = timeout_ms / 1000
    return f"{int(seconds) if seconds == int(seconds) else seconds} sec"


def apply_http_proxy_settings(http_proxy: str | None, env: MutableMapping[str, str] | None = None) -> None:
    """Seed ``HTTP_PROXY``/``HTTPS_PROXY`` without overwriting existing values.

    Defaults to the real environment because httpx, like Node's fetch, reads
    proxy configuration from it.
    """
    target = os.environ if env is None else env
    proxy = http_proxy.strip() if http_proxy else None
    if not proxy:
        return
    target.setdefault("HTTP_PROXY", proxy)
    target.setdefault("HTTPS_PROXY", proxy)


def configure_http_dispatcher(timeout_ms: int = DEFAULT_HTTP_IDLE_TIMEOUT_MS) -> None:
    normalized = parse_http_idle_timeout_ms(timeout_ms)
    if normalized is None:
        raise ValueError(f"Invalid HTTP idle timeout: {timeout_ms}")
    set_idle_timeout_ms(normalized)
