"""Provider attribution and session headers.

Python port of `packages/coding-agent/src/core/provider-attribution.ts`.

Some providers ask clients to identify themselves so usage can be attributed
(OpenRouter's referer/title, NVIDIA's billing origin, Cloudflare's user agent),
and opencode wants the session id so its dashboard can group a conversation's
requests. Attribution headers are sent only when install telemetry is enabled;
session headers are not telemetry and are always sent to opencode.

Caller-supplied headers are merged last, so an explicit header always wins over
an attribution default.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from pi_ai.types import Model

from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.core.telemetry import is_install_telemetry_enabled

RADIUS_PROVIDER_ID = "radius"
"""Python port of `packages/coding-agent/src/core/radius.ts`, a single constant."""

OPENROUTER_HOST = "openrouter.ai"
NVIDIA_NIM_HOST = "integrate.api.nvidia.com"
CLOUDFLARE_API_HOST = "api.cloudflare.com"
CLOUDFLARE_AI_GATEWAY_HOST = "gateway.ai.cloudflare.com"
OPENCODE_HOST = "opencode.ai"


def matches_host(base_url: str, expected_host: str) -> bool:
    """Whether `base_url`'s host is exactly `expected_host`. `False` if it will not parse."""
    try:
        return urlsplit(base_url).hostname == expected_host
    except ValueError:
        return False


def is_openrouter_model(model: Model) -> bool:
    return model.provider == "openrouter" or OPENROUTER_HOST in model.base_url


def is_nvidia_nim_model(model: Model) -> bool:
    return model.provider == "nvidia" or matches_host(model.base_url, NVIDIA_NIM_HOST)


def is_cloudflare_model(model: Model) -> bool:
    return (
        model.provider in ("cloudflare-workers-ai", "cloudflare-ai-gateway")
        or matches_host(model.base_url, CLOUDFLARE_API_HOST)
        or matches_host(model.base_url, CLOUDFLARE_AI_GATEWAY_HOST)
    )


def get_default_attribution_headers(model: Model, settings_manager: SettingsManager) -> dict[str, str] | None:
    """The provider's attribution headers, or `None` when telemetry is off or unrecognised."""
    if not is_install_telemetry_enabled(settings_manager):
        return None

    if is_openrouter_model(model):
        return {
            "HTTP-Referer": "https://pi.dev",
            "X-OpenRouter-Title": "pi",
            "X-OpenRouter-Categories": "cli-agent",
        }

    if is_nvidia_nim_model(model):
        return {"X-BILLING-INVOKE-ORIGIN": "Pi"}

    if is_cloudflare_model(model):
        return {"User-Agent": "pi-coding-agent"}

    return None


def get_session_headers(model: Model, session_id: str | None) -> dict[str, str] | None:
    """opencode's session-grouping headers. `None` for every other provider."""
    if not session_id:
        return None
    if model.provider not in ("opencode", "opencode-go") and not matches_host(model.base_url, OPENCODE_HOST):
        return None
    return {"x-opencode-session": session_id, "x-opencode-client": "pi"}


def merge_provider_attribution_headers(
    model: Model,
    settings_manager: SettingsManager,
    session_id: str | None,
    *header_sources: dict[str, str] | None,
) -> dict[str, str] | None:
    """Merge attribution, session and caller headers. `None` when the result is empty.

    Later sources overwrite earlier ones, so a caller's explicit header always
    beats an attribution default.
    """
    merged: dict[str, str] = {}
    merged.update(get_session_headers(model, session_id) or {})
    merged.update(get_default_attribution_headers(model, settings_manager) or {})

    for headers in header_sources:
        if headers:
            merged.update(headers)

    return merged or None


__all__ = [
    "CLOUDFLARE_AI_GATEWAY_HOST",
    "CLOUDFLARE_API_HOST",
    "NVIDIA_NIM_HOST",
    "OPENCODE_HOST",
    "OPENROUTER_HOST",
    "RADIUS_PROVIDER_ID",
    "get_default_attribution_headers",
    "get_session_headers",
    "is_cloudflare_model",
    "is_nvidia_nim_model",
    "is_openrouter_model",
    "matches_host",
    "merge_provider_attribution_headers",
]
