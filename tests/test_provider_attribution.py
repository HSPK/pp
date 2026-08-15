"""Tests for `pi_coding_agent.core.provider_attribution`.

Python port of `packages/coding-agent/test/sdk-openrouter-attribution.test.ts`.

The TypeScript test drives a whole `AgentSession` and captures the options a
stub provider receives. The 13 header rules are checked here against
`merge_provider_attribution_headers` directly -- the function
`sdk.create_agent_session`'s `stream_fn` calls -- which covers the same rules
without standing up a session, an auth store and a fake model registry for
every case. Because a unit test of the rules alone cannot prove `stream_fn`
actually calls the function and forwards its result,
`test_sdk.py::test_attribution_headers_reach_the_provider_through_a_real_session`
drives a real `create_agent_session` end to end the way TypeScript does.
"""

from __future__ import annotations

import pytest
from pi_ai.types import Model

from pi_coding_agent.core.provider_attribution import (
    get_default_attribution_headers,
    get_session_headers,
    matches_host,
    merge_provider_attribution_headers,
)
from pi_coding_agent.core.settings_manager import SettingsManager


@pytest.fixture(autouse=True)
def clear_telemetry_env(monkeypatch):
    """`PI_TELEMETRY`, when set at all, overrides the setting."""
    monkeypatch.delenv("PI_TELEMETRY", raising=False)


@pytest.fixture
def settings(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    return SettingsManager.create(str(cwd), str(agent_dir))


def make_model(provider: str, base_url: str, model_id: str | None = None) -> Model:
    return Model(
        id=model_id or f"{provider}-test-model",
        name=f"{provider} Test Model",
        api="openai-completions",
        provider=provider,
        base_url=base_url,
        context_window=128000,
        max_tokens=4096,
    )


def test_adds_default_attribution_headers_for_openrouter_models(settings):
    headers = merge_provider_attribution_headers(
        make_model("openrouter", "https://openrouter.ai/api/v1"), settings, None
    )

    assert headers["HTTP-Referer"] == "https://pi.dev"
    assert headers["X-OpenRouter-Title"] == "pi"
    assert headers["X-OpenRouter-Categories"] == "cli-agent"


def test_does_not_add_attribution_headers_when_telemetry_is_disabled(settings):
    settings.set_enable_install_telemetry(False)

    headers = merge_provider_attribution_headers(
        make_model("openrouter", "https://openrouter.ai/api/v1"), settings, None
    )

    assert headers is None


def test_adds_attribution_headers_for_custom_providers_routed_through_openrouter(settings):
    headers = merge_provider_attribution_headers(
        make_model("custom-openrouter", "https://openrouter.ai/api/v1"), settings, None
    )

    assert headers["HTTP-Referer"] == "https://pi.dev"
    assert headers["X-OpenRouter-Title"] == "pi"
    assert headers["X-OpenRouter-Categories"] == "cli-agent"


def test_preserves_legacy_openrouter_base_url_substring_matching(settings):
    # OpenRouter deliberately matches on a substring, not a parsed host, so a
    # base URL that is not even a valid URL still attributes.
    headers = merge_provider_attribution_headers(
        make_model("custom-openrouter", "not-a-url-openrouter.ai"), settings, None
    )

    assert headers["HTTP-Referer"] == "https://pi.dev"
    assert headers["X-OpenRouter-Title"] == "pi"
    assert headers["X-OpenRouter-Categories"] == "cli-agent"


def test_provider_and_request_headers_override_the_defaults(settings):
    headers = merge_provider_attribution_headers(
        make_model("openrouter", "https://openrouter.ai/api/v1"),
        settings,
        None,
        {"HTTP-Referer": "https://provider.example", "X-OpenRouter-Categories": "provider-category"},
        {"X-OpenRouter-Title": "request-title"},
    )

    assert headers["HTTP-Referer"] == "https://provider.example"
    assert headers["X-OpenRouter-Title"] == "request-title"
    assert headers["X-OpenRouter-Categories"] == "provider-category"


def test_adds_attribution_headers_for_direct_nvidia_nim_endpoints(settings):
    headers = merge_provider_attribution_headers(
        make_model("custom-nim", "https://integrate.api.nvidia.com/v1"), settings, None
    )

    assert headers["X-BILLING-INVOKE-ORIGIN"] == "Pi"


def test_adds_attribution_headers_for_the_nvidia_provider(settings):
    headers = merge_provider_attribution_headers(make_model("nvidia", "https://example.test/v1"), settings, None)

    assert headers["X-BILLING-INVOKE-ORIGIN"] == "Pi"


def test_does_not_add_nvidia_attribution_headers_when_telemetry_is_disabled(settings):
    settings.set_enable_install_telemetry(False)

    headers = merge_provider_attribution_headers(
        make_model("nvidia", "https://integrate.api.nvidia.com/v1"), settings, None
    )

    assert headers is None


def test_provider_and_request_headers_override_nvidia_defaults(settings):
    headers = merge_provider_attribution_headers(
        make_model("nvidia", "https://integrate.api.nvidia.com/v1"),
        settings,
        None,
        {"X-BILLING-INVOKE-ORIGIN": "Provider"},
        {"X-BILLING-INVOKE-ORIGIN": "Request"},
    )

    assert headers["X-BILLING-INVOKE-ORIGIN"] == "Request"


def test_nvidia_models_routed_through_openrouter_get_openrouter_attribution_only(settings):
    headers = merge_provider_attribution_headers(
        make_model("openrouter", "https://openrouter.ai/api/v1", "nvidia/nemotron-3-super-120b-a12b"),
        settings,
        None,
    )

    assert headers["HTTP-Referer"] == "https://pi.dev"
    assert "X-BILLING-INVOKE-ORIGIN" not in headers


def test_nvidia_models_routed_through_vercel_gateway_get_no_nvidia_attribution(settings):
    headers = merge_provider_attribution_headers(
        make_model("vercel-ai-gateway", "https://ai-gateway.vercel.sh/v1", "nvidia/nemotron-3-super-120b-a12b"),
        settings,
        None,
    )

    assert headers is None


def test_adds_cloudflare_attribution_headers(settings):
    for provider, base_url in (
        ("cloudflare-workers-ai", "https://api.cloudflare.com/client/v4"),
        ("cloudflare-ai-gateway", "https://gateway.ai.cloudflare.com/v1"),
        ("custom", "https://api.cloudflare.com/client/v4"),
    ):
        headers = merge_provider_attribution_headers(make_model(provider, base_url), settings, None)
        assert headers["User-Agent"] == "pi-coding-agent"


def test_adds_opencode_session_headers(settings):
    headers = merge_provider_attribution_headers(
        make_model("opencode", "https://opencode.ai/zen/v1"), settings, "opencode-session"
    )

    assert headers["x-opencode-session"] == "opencode-session"
    assert headers["x-opencode-client"] == "pi"


def test_configured_opencode_headers_override_the_defaults(settings):
    headers = merge_provider_attribution_headers(
        make_model("opencode", "https://opencode.ai/zen/v1"),
        settings,
        "opencode-session",
        {"x-opencode-session": "configured-session", "x-opencode-client": "configured-client"},
    )

    assert headers["x-opencode-session"] == "configured-session"
    assert headers["x-opencode-client"] == "configured-client"


def test_session_headers_are_not_telemetry_gated(settings):
    settings.set_enable_install_telemetry(False)

    headers = merge_provider_attribution_headers(
        make_model("opencode", "https://opencode.ai/zen/v1"), settings, "opencode-session"
    )

    assert headers["x-opencode-session"] == "opencode-session"


def test_session_headers_need_a_session_id(settings):
    assert get_session_headers(make_model("opencode", "https://opencode.ai/zen/v1"), None) is None
    assert get_session_headers(make_model("opencode", "https://opencode.ai/zen/v1"), "") is None


def test_session_headers_are_not_sent_to_other_providers(settings):
    assert get_session_headers(make_model("openai", "https://api.openai.com/v1"), "s1") is None


def test_unrecognised_providers_get_no_attribution_headers(settings):
    assert get_default_attribution_headers(make_model("openai", "https://api.openai.com/v1"), settings) is None


def test_merge_returns_none_when_nothing_applies(settings):
    assert merge_provider_attribution_headers(make_model("openai", "https://api.openai.com/v1"), settings, None) is None


def test_caller_headers_alone_are_returned(settings):
    headers = merge_provider_attribution_headers(
        make_model("openai", "https://api.openai.com/v1"), settings, None, {"X-Custom": "value"}
    )

    assert headers == {"X-Custom": "value"}


@pytest.mark.parametrize(
    ("base_url", "host", "expected"),
    [
        ("https://openrouter.ai/api/v1", "openrouter.ai", True),
        ("https://not-openrouter.ai/api/v1", "openrouter.ai", False),
        ("not-a-url", "openrouter.ai", False),
        ("", "openrouter.ai", False),
    ],
)
def test_matches_host_compares_the_parsed_host(base_url, host, expected):
    assert matches_host(base_url, host) is expected
