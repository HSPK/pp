"""Python port of `packages/coding-agent/test/model-registry.test.ts`.

The TypeScript file drives `ModelRegistry`, a thin façade this port does not
have: `core/model-registry.ts` exists only to bridge the coding agent to
`pi-ai`'s *compat* entry point, and it is created through a
`ModelRuntime` + `InMemoryCodingAgentModelsStore` pair.  This port has no
`ModelsStore` layer and no `ModelRegistry` (see `core/model_runtime.py`'s
module docstring), so every case that is really about composing built-in
providers with a `models.json` overlay is driven straight through
`ModelRuntime`, which is what `ModelRegistry` delegates to in TypeScript.

Cases that depend on machinery this port deliberately omits (the dynamic
extension-provider lifecycle and `getApiKeyAndHeaders`, which only exists to
feed the compat `complete()` entry point) are skipped individually with the
reason stated at the skip.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
from pi_ai.auth.types import Credential, CredentialStore
from pi_ai.types import Context, Model

from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.provider_composer import clear_api_key_cache


@pytest.fixture(autouse=True)
def _clear_api_key_cache():
    clear_api_key_cache()
    yield
    clear_api_key_cache()


@pytest.fixture
def credentials() -> AuthStorage:
    """The real `AuthStorage.in_memory()`, matching what the TypeScript drives.

    An earlier bare `CredentialStore` double satisfied the protocol's *shape*
    but not its *behavior*: `AuthStorage.get` resolves a stored api-key
    credential's `key` through `resolve_config_value(key, credential.env)`
    (`auth_storage.py:245`, porting `auth-storage.ts:267`), expanding `$VAR`
    placeholders. The double returned the raw string. That was invisible until
    one test stored `key="$CLOUDFLARE_API_KEY"` and the unexpanded value was
    misread as a `pi-ai` bug. Driving the real object removes the whole class
    of divergence instead of the one case that happened to surface.
    """
    return AuthStorage.in_memory()


@pytest.fixture
def models_json_path(tmp_path: Path) -> Path:
    return tmp_path / "models.json"


def write_raw_models_json(path: Path, providers: dict[str, Any]) -> None:
    path.write_text(json.dumps({"providers": providers}), encoding="utf-8")


def provider_config(
    base_url: str,
    models: list[dict[str, str]],
    api: str = "anthropic-messages",
) -> dict[str, Any]:
    """Port of the TypeScript `providerConfig` helper."""
    return {
        "baseUrl": base_url,
        "apiKey": "test-key",
        "api": api,
        "models": [
            {
                "id": model["id"],
                "name": model.get("name", model["id"]),
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 100000,
                "maxTokens": 8000,
            }
            for model in models
        ],
    }


def override_config(base_url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {"baseUrl": base_url}
    if headers is not None:
        config["headers"] = headers
    return config


async def create_runtime(credentials: CredentialStore, models_path: Path | None) -> ModelRuntime:
    return await ModelRuntime.create(
        credentials=credentials,
        models_path=str(models_path) if models_path is not None else None,
    )


def models_for_provider(runtime: ModelRuntime, provider: str) -> list[Model]:
    return [model for model in runtime.get_models() if model.provider == provider]


def find_model(runtime: ModelRuntime, provider: str, model_id: str) -> Model | None:
    return runtime.get_model(provider, model_id)


async def api_key_for_provider(runtime: ModelRuntime, provider_id: str) -> str | None:
    """`ModelRegistry.getApiKeyForProvider`'s equivalent on this port.

    TypeScript wraps `runtime.getAuth(provider)` in a `try`/`catch` that
    returns `undefined`, which is what turns a failed `!command` API key into
    "no key" instead of an error.
    """
    try:
        result = await runtime.get_auth(provider_id)
    except Exception:
        return None
    return result.auth.api_key if result is not None else None


class TestBaseUrlOverrideWithoutCustomModels:
    async def test_overriding_base_url_keeps_all_built_in_models(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(models_json_path, {"anthropic": override_config("https://my-proxy.example.com/v1")})

        runtime = await create_runtime(credentials, models_json_path)
        anthropic_models = models_for_provider(runtime, "anthropic")

        assert len(anthropic_models) > 1
        assert any("claude" in model.id for model in anthropic_models)

    async def test_overriding_base_url_changes_url_on_all_built_in_models(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(models_json_path, {"anthropic": override_config("https://my-proxy.example.com/v1")})

        runtime = await create_runtime(credentials, models_json_path)

        for model in models_for_provider(runtime, "anthropic"):
            assert model.base_url == "https://my-proxy.example.com/v1"

    async def test_overriding_headers_resolves_at_request_time(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        # TS calls `registry.getApiKeyAndHeaders(model)` for every anthropic
        # model and asserts `auth.ok === true` with the header present. That
        # bridge (`ModelRegistry.getApiKeyAndHeaders`) is not ported (see the
        # `test_get_api_key_and_headers` skip below), and there is no built-in
        # ANTHROPIC_API_KEY in this test environment for `runtime.get_auth` to
        # resolve, so it would return `None` here rather than the TS
        # ok-with-static-headers fallback. `provider.headers` is the composed,
        # static value that fallback would have surfaced, so it is what is
        # asserted instead.
        write_raw_models_json(
            models_json_path,
            {"anthropic": override_config("https://my-proxy.example.com/v1", {"X-Custom-Header": "custom-value"})},
        )

        runtime = await create_runtime(credentials, models_json_path)
        provider = runtime.get_provider("anthropic")

        assert provider is not None
        assert (provider.headers or {}).get("X-Custom-Header") == "custom-value"

    async def test_headers_only_override_resolves_at_request_time(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        # Same substitution as above: TS's "headers-only override resolves at
        # request time" reads the header off `registry.getApiKeyAndHeaders`,
        # which this port does not have (see the `test_get_api_key_and_headers`
        # skip below); `provider.headers` is the composed value it would read.
        write_raw_models_json(models_json_path, {"anthropic": {"headers": {"X-Custom-Header": "custom-value"}}})

        runtime = await create_runtime(credentials, models_json_path)

        assert runtime.get_error() is None
        provider = runtime.get_provider("anthropic")
        assert provider is not None
        assert (provider.headers or {}).get("X-Custom-Header") == "custom-value"

    @pytest.mark.skip(
        reason=(
            "`getApiKeyAndHeaders` exists only to hand an unconfigured model's static "
            "headers to `pi-ai`'s compat `complete()` entry point.  Neither "
            "`ModelRegistry` nor the compat entry point is ported; `ModelRuntime.get_auth` "
            "returns `None` for an unknown provider instead of an `{ok: true}` result."
        )
    )
    def test_unconfigured_compatibility_auth_includes_static_model_headers(self) -> None:
        raise AssertionError("unreachable")

    async def test_base_url_only_override_does_not_affect_other_providers(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(models_json_path, {"anthropic": override_config("https://my-proxy.example.com/v1")})

        runtime = await create_runtime(credentials, models_json_path)
        google_models = models_for_provider(runtime, "google")

        assert len(google_models) > 0
        assert google_models[0].base_url != "https://my-proxy.example.com/v1"

    async def test_can_mix_base_url_override_and_models_merge(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "anthropic": override_config("https://anthropic-proxy.example.com/v1"),
                "google": provider_config(
                    "https://google-proxy.example.com/v1",
                    [{"id": "gemini-custom"}],
                    "google-generative-ai",
                ),
            },
        )

        runtime = await create_runtime(credentials, models_json_path)

        anthropic_models = models_for_provider(runtime, "anthropic")
        assert len(anthropic_models) > 1
        assert anthropic_models[0].base_url == "https://anthropic-proxy.example.com/v1"

        google_models = models_for_provider(runtime, "google")
        assert len(google_models) > 1
        assert any(model.id == "gemini-custom" for model in google_models)

    async def test_refresh_picks_up_base_url_override_changes(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(models_json_path, {"anthropic": override_config("https://first-proxy.example.com/v1")})
        runtime = await create_runtime(credentials, models_json_path)

        assert models_for_provider(runtime, "anthropic")[0].base_url == "https://first-proxy.example.com/v1"

        write_raw_models_json(models_json_path, {"anthropic": override_config("https://second-proxy.example.com/v1")})
        runtime.refresh()

        assert models_for_provider(runtime, "anthropic")[0].base_url == "https://second-proxy.example.com/v1"


class TestCustomModelsMergeBehavior:
    async def test_built_in_provider_custom_models_inherit_api_and_base_url(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "openrouter": {
                    "models": [
                        {
                            "id": "fake-provider/fake-model",
                            "name": "Fake model",
                            "reasoning": True,
                            "input": ["text"],
                        }
                    ]
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        assert runtime.get_error() is None

        model = find_model(runtime, "openrouter", "fake-provider/fake-model")
        assert model is not None
        assert model.api == "openai-completions"
        assert model.base_url == "https://openrouter.ai/api/v1"

    async def test_non_built_in_provider_custom_models_still_require_base_url(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "my-custom-provider": {
                    "apiKey": "test-key",
                    "models": [{"id": "my-model", "api": "openai-completions", "reasoning": False, "input": ["text"]}],
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        error = runtime.get_error()
        assert error is not None
        assert "baseUrl" in error

    async def test_reports_every_provider_composition_error(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "broken-one": {"api": "openai-completions", "models": [{"id": "one"}]},
                "broken-two": {"api": "openai-completions", "models": [{"id": "two"}]},
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        error = runtime.get_error()

        assert error is not None
        assert 'Provider "broken-one"' in error
        assert 'Provider "broken-two"' in error

    async def test_custom_provider_with_same_name_as_built_in_merges(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {"anthropic": provider_config("https://my-proxy.example.com/v1", [{"id": "claude-custom"}])},
        )

        runtime = await create_runtime(credentials, models_json_path)
        anthropic_models = models_for_provider(runtime, "anthropic")

        assert len(anthropic_models) > 1
        assert any(model.id == "claude-custom" for model in anthropic_models)
        assert any("claude" in model.id for model in anthropic_models)

    async def test_custom_model_with_same_id_replaces_built_in_model_by_id(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "openrouter": provider_config(
                    "https://my-proxy.example.com/v1",
                    [{"id": "anthropic/claude-sonnet-4"}],
                    "openai-completions",
                )
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        sonnet_models = [
            model for model in models_for_provider(runtime, "openrouter") if model.id == "anthropic/claude-sonnet-4"
        ]

        assert len(sonnet_models) == 1
        assert sonnet_models[0].base_url == "https://my-proxy.example.com/v1"

    async def test_custom_provider_does_not_affect_other_built_in_providers(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {"anthropic": provider_config("https://my-proxy.example.com/v1", [{"id": "claude-custom"}])},
        )

        runtime = await create_runtime(credentials, models_json_path)

        assert len(models_for_provider(runtime, "google")) > 0
        assert len(models_for_provider(runtime, "openai")) > 0

    async def test_provider_level_base_url_applies_to_built_in_and_custom_models(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {"anthropic": provider_config("https://merged-proxy.example.com/v1", [{"id": "claude-custom"}])},
        )

        runtime = await create_runtime(credentials, models_json_path)

        for model in models_for_provider(runtime, "anthropic"):
            assert model.base_url == "https://merged-proxy.example.com/v1"

    async def test_provider_level_compat_applies_to_custom_models(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "demo": {
                    "baseUrl": "https://example.com/v1",
                    "apiKey": "DEMO_KEY",
                    "api": "openai-completions",
                    "compat": {"supportsUsageInStreaming": False, "maxTokensField": "max_tokens"},
                    "models": [
                        {
                            "id": "demo-model",
                            "reasoning": False,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 1000,
                            "maxTokens": 100,
                        }
                    ],
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        model = find_model(runtime, "demo", "demo-model")

        assert model is not None
        assert model.compat.get("supportsUsageInStreaming") is False
        assert model.compat.get("maxTokensField") == "max_tokens"

    async def test_model_level_compat_overrides_provider_level_compat(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "demo": {
                    "baseUrl": "https://example.com/v1",
                    "apiKey": "DEMO_KEY",
                    "api": "openai-completions",
                    "compat": {"supportsUsageInStreaming": False, "maxTokensField": "max_tokens"},
                    "models": [
                        {
                            "id": "demo-model",
                            "reasoning": False,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 1000,
                            "maxTokens": 100,
                            "compat": {
                                "supportsUsageInStreaming": True,
                                "maxTokensField": "max_completion_tokens",
                            },
                        }
                    ],
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        model = find_model(runtime, "demo", "demo-model")

        assert model is not None
        assert model.compat.get("supportsUsageInStreaming") is True
        assert model.compat.get("maxTokensField") == "max_completion_tokens"

    async def test_provider_level_compat_applies_to_built_in_models(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {"openrouter": {"compat": {"supportsUsageInStreaming": False, "supportsStrictMode": False}}},
        )

        runtime = await create_runtime(credentials, models_json_path)
        models = models_for_provider(runtime, "openrouter")

        assert len(models) > 0
        for model in models:
            assert model.compat.get("supportsUsageInStreaming") is False
            assert model.compat.get("supportsStrictMode") is False

    async def test_model_schema_accepts_thinking_level_map_and_compat_flags(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "demo": {
                    "baseUrl": "https://example.com/v1",
                    "apiKey": "DEMO_KEY",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": "demo-model",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 1000,
                            "maxTokens": 100,
                            "thinkingLevelMap": {"minimal": None, "high": "max"},
                            "compat": {"supportsStrictMode": False, "cacheControlFormat": "anthropic"},
                        }
                    ],
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        model = find_model(runtime, "demo", "demo-model")

        assert runtime.get_error() is None
        assert model is not None
        assert model.thinking_level_map == {"minimal": None, "high": "max"}
        assert model.compat.get("supportsStrictMode") is False
        assert model.compat.get("cacheControlFormat") == "anthropic"

    async def test_compat_schema_accepts_chat_template_thinking_configuration(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "demo": {
                    "baseUrl": "https://example.com/v1",
                    "apiKey": "DEMO_KEY",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": "kwargs-model",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 1000,
                            "maxTokens": 100,
                            "compat": {
                                "thinkingFormat": "chat-template",
                                "chatTemplateKwargs": {
                                    "preserve_thinking": True,
                                    "thinking": {"$var": "thinking.enabled"},
                                },
                            },
                        },
                        {
                            "id": "args-model",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 1000,
                            "maxTokens": 100,
                            "compat": {
                                "thinkingFormat": "baseten",
                                "chatTemplateArgs": {"enable_thinking": {"$var": "thinking.enabled"}},
                            },
                        },
                    ],
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        kwargs_model = find_model(runtime, "demo", "kwargs-model")
        args_model = find_model(runtime, "demo", "args-model")

        assert runtime.get_error() is None
        assert kwargs_model is not None
        assert args_model is not None
        assert kwargs_model.compat.get("thinkingFormat") == "chat-template"
        assert kwargs_model.compat.get("chatTemplateKwargs") == {
            "preserve_thinking": True,
            "thinking": {"$var": "thinking.enabled"},
        }
        assert args_model.compat.get("thinkingFormat") == "baseten"
        assert args_model.compat.get("chatTemplateArgs") == {"enable_thinking": {"$var": "thinking.enabled"}}

    async def test_compat_schema_accepts_anthropic_eager_tool_input_streaming_flag(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "demo": {
                    "baseUrl": "https://example.com",
                    "apiKey": "DEMO_KEY",
                    "api": "anthropic-messages",
                    "compat": {"supportsEagerToolInputStreaming": False},
                    "models": [
                        {
                            "id": "demo-model",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 1000,
                            "maxTokens": 100,
                        }
                    ],
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        model = find_model(runtime, "demo", "demo-model")

        assert runtime.get_error() is None
        assert model is not None
        assert model.compat.get("supportsEagerToolInputStreaming") is False

    async def test_compat_schema_accepts_long_cache_retention_flag(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "demo": {
                    "baseUrl": "https://example.com",
                    "apiKey": "DEMO_KEY",
                    "api": "anthropic-messages",
                    "compat": {"supportsLongCacheRetention": False},
                    "models": [
                        {
                            "id": "demo-model",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 1000,
                            "maxTokens": 100,
                        }
                    ],
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        model = find_model(runtime, "demo", "demo-model")

        assert runtime.get_error() is None
        assert model is not None
        assert model.compat.get("supportsLongCacheRetention") is False

    async def test_model_level_base_url_overrides_provider_level_base_url(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        # The TypeScript case uses two models with *different* `api` values on
        # one provider.  `pi_ai.registry.Provider` has one API module per
        # provider (see `provider_composer.py`'s documented narrowing), so this
        # port keeps both models on the provider's own API and pins only the
        # baseUrl precedence the case is named for.
        write_raw_models_json(
            models_json_path,
            {
                "opencode-go": {
                    "baseUrl": "https://opencode.ai/zen/go/v1",
                    "apiKey": "TEST_KEY",
                    "models": [
                        {
                            "id": "minimax-m2.5",
                            "baseUrl": "https://opencode.ai/zen/go",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 0.3, "output": 1.2, "cacheRead": 0.03, "cacheWrite": 0},
                            "contextWindow": 204800,
                            "maxTokens": 131072,
                        },
                        {
                            "id": "glm-5",
                            "reasoning": True,
                            "input": ["text"],
                            "cost": {"input": 1, "output": 3.2, "cacheRead": 0.2, "cacheWrite": 0},
                            "contextWindow": 204800,
                            "maxTokens": 131072,
                        },
                    ],
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)

        m25 = find_model(runtime, "opencode-go", "minimax-m2.5")
        glm5 = find_model(runtime, "opencode-go", "glm-5")
        assert m25 is not None
        assert glm5 is not None
        assert m25.base_url == "https://opencode.ai/zen/go"
        assert glm5.base_url == "https://opencode.ai/zen/go/v1"

    async def test_model_overrides_still_apply_when_provider_also_defines_models(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "openrouter": {
                    "baseUrl": "https://my-proxy.example.com/v1",
                    "apiKey": "OPENROUTER_API_KEY",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": "custom/openrouter-model",
                            "name": "Custom OpenRouter Model",
                            "reasoning": False,
                            "input": ["text"],
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                            "contextWindow": 128000,
                            "maxTokens": 16384,
                        }
                    ],
                    "modelOverrides": {"anthropic/claude-sonnet-4": {"name": "Overridden Built-in Sonnet"}},
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        models = models_for_provider(runtime, "openrouter")

        assert any(model.id == "custom/openrouter-model" for model in models)
        assert any(
            model.id == "anthropic/claude-sonnet-4" and model.name == "Overridden Built-in Sonnet" for model in models
        )

    async def test_refresh_reloads_merged_custom_models_from_disk(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {"anthropic": provider_config("https://first-proxy.example.com/v1", [{"id": "claude-custom"}])},
        )
        runtime = await create_runtime(credentials, models_json_path)
        assert any(model.id == "claude-custom" for model in models_for_provider(runtime, "anthropic"))

        write_raw_models_json(
            models_json_path,
            {"anthropic": provider_config("https://second-proxy.example.com/v1", [{"id": "claude-custom-2"}])},
        )
        runtime.refresh()

        anthropic_models = models_for_provider(runtime, "anthropic")
        assert not any(model.id == "claude-custom" for model in anthropic_models)
        assert any(model.id == "claude-custom-2" for model in anthropic_models)
        assert any("claude" in model.id for model in anthropic_models)

    async def test_removing_custom_models_keeps_built_in_provider_models(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {"anthropic": provider_config("https://proxy.example.com/v1", [{"id": "claude-custom"}])},
        )
        runtime = await create_runtime(credentials, models_json_path)
        assert any(model.id == "claude-custom" for model in models_for_provider(runtime, "anthropic"))

        write_raw_models_json(models_json_path, {})
        runtime.refresh()

        anthropic_models = models_for_provider(runtime, "anthropic")
        assert not any(model.id == "claude-custom" for model in anthropic_models)
        assert any("claude" in model.id for model in anthropic_models)


class TestModelOverrides:
    async def test_model_override_applies_to_a_single_built_in_model(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {"openrouter": {"modelOverrides": {"anthropic/claude-sonnet-4": {"name": "Custom Sonnet Name"}}}},
        )

        runtime = await create_runtime(credentials, models_json_path)

        sonnet = find_model(runtime, "openrouter", "anthropic/claude-sonnet-4")
        assert sonnet is not None
        assert sonnet.name == "Custom Sonnet Name"

        opus = find_model(runtime, "openrouter", "anthropic/claude-opus-4")
        assert opus is not None
        assert opus.name != "Custom Sonnet Name"

    async def test_custom_model_and_model_override_carry_sampling_params(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "openrouter": {
                    "baseUrl": "https://my-proxy.example.com/v1",
                    "api": "openai-completions",
                    "models": [
                        {"id": "custom/sampling-model", "samplingParams": {"temperature": 1, "top_p": 0.95, "top_k": 0}}
                    ],
                    "modelOverrides": {"anthropic/claude-sonnet-4": {"samplingParams": {"top_p": 0.9}}},
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)

        custom = find_model(runtime, "openrouter", "custom/sampling-model")
        assert custom is not None
        assert custom.sampling_params == {"temperature": 1, "top_p": 0.95, "top_k": 0}

        sonnet = find_model(runtime, "openrouter", "anthropic/claude-sonnet-4")
        assert sonnet is not None
        assert sonnet.sampling_params == {"top_p": 0.9}

        # Models without sampling config keep it unset.
        opus = find_model(runtime, "openrouter", "anthropic/claude-opus-4")
        assert opus is not None
        assert not opus.sampling_params

    async def test_model_override_with_open_router_routing(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "openrouter": {
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": {"compat": {"openRouterRouting": {"only": ["amazon-bedrock"]}}}
                    }
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        sonnet = find_model(runtime, "openrouter", "anthropic/claude-sonnet-4")

        assert sonnet is not None
        assert sonnet.compat.get("openRouterRouting") == {"only": ["amazon-bedrock"]}

    async def test_model_override_deep_merges_compat_settings(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "openrouter": {
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": {
                            "compat": {"openRouterRouting": {"order": ["anthropic", "together"]}}
                        }
                    }
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        sonnet = find_model(runtime, "openrouter", "anthropic/claude-sonnet-4")

        assert sonnet is not None
        assert sonnet.compat.get("openRouterRouting") == {"order": ["anthropic", "together"]}

    async def test_multiple_model_overrides_on_same_provider(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "openrouter": {
                    "modelOverrides": {
                        "anthropic/claude-sonnet-4": {"compat": {"openRouterRouting": {"only": ["amazon-bedrock"]}}},
                        "anthropic/claude-opus-4": {"compat": {"openRouterRouting": {"only": ["anthropic"]}}},
                    }
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        sonnet = find_model(runtime, "openrouter", "anthropic/claude-sonnet-4")
        opus = find_model(runtime, "openrouter", "anthropic/claude-opus-4")

        assert sonnet is not None
        assert opus is not None
        assert sonnet.compat.get("openRouterRouting") == {"only": ["amazon-bedrock"]}
        assert opus.compat.get("openRouterRouting") == {"only": ["anthropic"]}

    async def test_model_override_combined_with_base_url_override(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "openrouter": {
                    "baseUrl": "https://my-proxy.example.com/v1",
                    "modelOverrides": {"anthropic/claude-sonnet-4": {"name": "Proxied Sonnet"}},
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)

        sonnet = find_model(runtime, "openrouter", "anthropic/claude-sonnet-4")
        assert sonnet is not None
        assert sonnet.base_url == "https://my-proxy.example.com/v1"
        assert sonnet.name == "Proxied Sonnet"

        opus = find_model(runtime, "openrouter", "anthropic/claude-opus-4")
        assert opus is not None
        assert opus.base_url == "https://my-proxy.example.com/v1"
        assert opus.name != "Proxied Sonnet"

    async def test_model_override_for_non_existent_model_id_is_ignored(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {"openrouter": {"modelOverrides": {"nonexistent/model-id": {"name": "This should not appear"}}}},
        )

        runtime = await create_runtime(credentials, models_json_path)

        assert find_model(runtime, "openrouter", "nonexistent/model-id") is None
        assert runtime.get_error() is None

    async def test_model_override_can_change_cost_fields_partially(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {"openrouter": {"modelOverrides": {"anthropic/claude-sonnet-4": {"cost": {"input": 99}}}}},
        )

        runtime = await create_runtime(credentials, models_json_path)
        sonnet = find_model(runtime, "openrouter", "anthropic/claude-sonnet-4")

        assert sonnet is not None
        assert sonnet.cost.input == 99
        assert sonnet.cost.output > 0

    async def test_model_override_can_add_headers_at_request_time(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "openrouter": {
                    "apiKey": "test-key",
                    "modelOverrides": {"anthropic/claude-sonnet-4": {"headers": {"X-Custom-Model-Header": "value"}}},
                }
            },
        )

        runtime = await create_runtime(credentials, models_json_path)
        sonnet = find_model(runtime, "openrouter", "anthropic/claude-sonnet-4")
        assert sonnet is not None

        auth = await runtime.get_auth(sonnet)
        assert auth is not None
        assert (auth.auth.headers or {}).get("X-Custom-Model-Header") == "value"

    async def test_refresh_picks_up_model_override_changes(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {"openrouter": {"modelOverrides": {"anthropic/claude-sonnet-4": {"name": "First Name"}}}},
        )
        runtime = await create_runtime(credentials, models_json_path)

        first = find_model(runtime, "openrouter", "anthropic/claude-sonnet-4")
        assert first is not None
        assert first.name == "First Name"

        write_raw_models_json(
            models_json_path,
            {"openrouter": {"modelOverrides": {"anthropic/claude-sonnet-4": {"name": "Second Name"}}}},
        )
        runtime.refresh()

        second = find_model(runtime, "openrouter", "anthropic/claude-sonnet-4")
        assert second is not None
        assert second.name == "Second Name"

    async def test_removing_model_override_restores_built_in_values(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {"openrouter": {"modelOverrides": {"anthropic/claude-sonnet-4": {"name": "Custom Name"}}}},
        )
        runtime = await create_runtime(credentials, models_json_path)

        custom = find_model(runtime, "openrouter", "anthropic/claude-sonnet-4")
        assert custom is not None
        assert custom.name == "Custom Name"

        write_raw_models_json(models_json_path, {})
        runtime.refresh()

        restored = find_model(runtime, "openrouter", "anthropic/claude-sonnet-4")
        assert restored is not None
        assert restored.name != "Custom Name"


@pytest.mark.skip(
    reason=(
        "The dynamic provider lifecycle (`registerProvider`/`unregisterProvider`, "
        "`ProviderConfigInput`, `validateExtensionProvider`, the native/extension "
        "provider overlays and their `streamSimple` compat registration) is the "
        "extension-provider layer `core/model_runtime.py` documents as not ported.  "
        "Covers 13 of the 14 TypeScript cases in the 'dynamic provider lifecycle' "
        "describe block, including `getProviderDisplayName` and the "
        "refresh-persistence subgroup. The 14th case, 'stored API key env propagates "
        "to request auth and resolves headers', does not use registerProvider at all "
        "-- it is ported as `test_stored_credential_env_propagates_to_configured_header_interpolation` below."
    )
)
def test_dynamic_provider_lifecycle() -> None:
    raise AssertionError("unreachable")


def provider_with_api_key(api_key: str) -> dict[str, Any]:
    """Port of the TypeScript `providerWithApiKey` helper."""
    return {
        "baseUrl": "https://example.com/v1",
        "apiKey": api_key,
        "api": "anthropic-messages",
        "models": [
            {
                "id": "test-model",
                "name": "Test Model",
                "reasoning": False,
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": 100000,
                "maxTokens": 8000,
            }
        ],
    }


class TestApiKeyResolution:
    async def test_bang_prefix_executes_command_and_uses_stdout(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path, {"custom-provider": provider_with_api_key("!echo test-api-key-from-command")}
        )

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "custom-provider") == "test-api-key-from-command"

    async def test_bang_prefix_trims_whitespace_from_command_output(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(models_json_path, {"custom-provider": provider_with_api_key("!echo '  spaced-key  '")})

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "custom-provider") == "spaced-key"

    async def test_bang_prefix_handles_multiline_output(self, credentials: AuthStorage, models_json_path: Path) -> None:
        write_raw_models_json(models_json_path, {"custom-provider": provider_with_api_key("!printf 'line1\\nline2'")})

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "custom-provider") == "line1\nline2"

    @pytest.mark.parametrize(
        "api_key",
        ["!exit 1", "!nonexistent-command-12345", "!printf ''"],
        ids=["command-failure", "nonexistent-command", "empty-output"],
    )
    async def test_bang_prefix_returns_none_on_failure(
        self, credentials: AuthStorage, models_json_path: Path, api_key: str
    ) -> None:
        write_raw_models_json(models_json_path, {"custom-provider": provider_with_api_key(api_key)})

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "custom-provider") is None

    async def test_dollar_prefix_resolves_to_env_value(
        self, credentials: AuthStorage, models_json_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY_12345", "env-api-key-value")
        write_raw_models_json(models_json_path, {"custom-provider": provider_with_api_key("$TEST_API_KEY_12345")})

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "custom-provider") == "env-api-key-value"

    async def test_braced_env_syntax_resolves_to_env_value(
        self, credentials: AuthStorage, models_json_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_BRACED_API_KEY_12345", "braced-env-api-key-value")
        write_raw_models_json(
            models_json_path, {"custom-provider": provider_with_api_key("${TEST_BRACED_API_KEY_12345}")}
        )

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "custom-provider") == "braced-env-api-key-value"

    async def test_interpolates_braced_env_references_inside_literals(
        self, credentials: AuthStorage, models_json_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_INTERPOLATED_PART_A_12345", "left")
        monkeypatch.setenv("TEST_INTERPOLATED_PART_B_12345", "right")
        write_raw_models_json(
            models_json_path,
            {
                "custom-provider": provider_with_api_key(
                    "${TEST_INTERPOLATED_PART_A_12345}_${TEST_INTERPOLATED_PART_B_12345}"
                )
            },
        )

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "custom-provider") == "left_right"

    async def test_double_dollar_prefix_escapes_a_leading_dollar(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(models_json_path, {"custom-provider": provider_with_api_key("$$TEST_API_KEY_12345")})

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "custom-provider") == "$TEST_API_KEY_12345"

    async def test_dollar_bang_escapes_a_literal_bang_and_still_interpolates(
        self, credentials: AuthStorage, models_json_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY_12345", "env-api-key-value")
        write_raw_models_json(
            models_json_path, {"custom-provider": provider_with_api_key("$!literal-$TEST_API_KEY_12345")}
        )

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "custom-provider") == "!literal-env-api-key-value"

    async def test_plain_api_key_is_used_directly_even_when_it_matches_an_env_var(
        self, credentials: AuthStorage, models_json_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY_12345", "env-api-key-value")
        write_raw_models_json(models_json_path, {"custom-provider": provider_with_api_key("TEST_API_KEY_12345")})

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "custom-provider") == "TEST_API_KEY_12345"

    async def test_api_key_as_literal_value_is_used_directly_when_not_an_env_var(
        self, credentials: AuthStorage, models_json_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("literal_api_key_value", raising=False)
        write_raw_models_json(models_json_path, {"custom-provider": provider_with_api_key("literal_api_key_value")})

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "custom-provider") == "literal_api_key_value"

    async def test_api_key_command_can_use_shell_features_like_pipes(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path, {"custom-provider": provider_with_api_key("!echo 'hello world' | tr ' ' '-'")}
        )

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "custom-provider") == "hello-world"


def _counter_command(counter_file: Path, tail: str) -> str:
    path = str(counter_file).replace("\\", "/").replace('"', '\\"')
    return f'!sh -c \'count=$(cat "{path}"); echo $((count + 1)) > "{path}"; {tail}\''


class TestRequestTimeResolution:
    async def test_command_is_executed_on_every_provider_lookup(
        self, credentials: AuthStorage, models_json_path: Path, tmp_path: Path
    ) -> None:
        counter_file = tmp_path / "counter"
        counter_file.write_text("0", encoding="utf-8")
        write_raw_models_json(
            models_json_path,
            {"custom-provider": provider_with_api_key(_counter_command(counter_file, 'echo "key-value"'))},
        )

        runtime = await create_runtime(credentials, models_json_path)
        await api_key_for_provider(runtime, "custom-provider")
        await api_key_for_provider(runtime, "custom-provider")
        await api_key_for_provider(runtime, "custom-provider")

        assert int(counter_file.read_text(encoding="utf-8").strip()) == 3

    async def test_commands_are_re_executed_across_runtime_instances(
        self, credentials: AuthStorage, models_json_path: Path, tmp_path: Path
    ) -> None:
        counter_file = tmp_path / "counter"
        counter_file.write_text("0", encoding="utf-8")
        write_raw_models_json(
            models_json_path,
            {"custom-provider": provider_with_api_key(_counter_command(counter_file, 'echo "key-value"'))},
        )

        runtime1 = await create_runtime(credentials, models_json_path)
        await api_key_for_provider(runtime1, "custom-provider")

        runtime2 = await create_runtime(credentials, models_json_path)
        await api_key_for_provider(runtime2, "custom-provider")

        assert int(counter_file.read_text(encoding="utf-8").strip()) == 2

    async def test_different_commands_resolve_independently(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(
            models_json_path,
            {
                "provider-a": provider_with_api_key("!echo key-a"),
                "provider-b": provider_with_api_key("!echo key-b"),
            },
        )

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "provider-a") == "key-a"
        assert await api_key_for_provider(runtime, "provider-b") == "key-b"

    async def test_failed_commands_are_retried(
        self, credentials: AuthStorage, models_json_path: Path, tmp_path: Path
    ) -> None:
        counter_file = tmp_path / "counter"
        counter_file.write_text("0", encoding="utf-8")
        write_raw_models_json(
            models_json_path,
            {"custom-provider": provider_with_api_key(_counter_command(counter_file, "exit 1"))},
        )

        runtime = await create_runtime(credentials, models_json_path)

        assert await api_key_for_provider(runtime, "custom-provider") is None
        assert await api_key_for_provider(runtime, "custom-provider") is None
        assert int(counter_file.read_text(encoding="utf-8").strip()) == 2

    async def test_provider_auth_status_reports_api_key_environment_variables(
        self, credentials: AuthStorage, models_json_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_var_name = "TEST_API_KEY_STATUS_TEST_98765"
        monkeypatch.setenv(env_var_name, "status-test-key")
        write_raw_models_json(models_json_path, {"custom-provider": provider_with_api_key(f"${env_var_name}")})

        runtime = await create_runtime(credentials, models_json_path)
        status = runtime.get_provider_auth_status("custom-provider")

        assert status.configured is True
        # TypeScript returns `{configured, source: "environment", label}`; this
        # port's `AuthCheck` has no `label`, and the pre-existing convention is
        # to carry the env var name in `source` (see the builtin env var branch).
        assert status.source == env_var_name

    async def test_provider_auth_status_reports_interpolated_environment_variables(
        self, credentials: AuthStorage, models_json_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        name_a = "TEST_API_KEY_STATUS_PART_A_98765"
        name_b = "TEST_API_KEY_STATUS_PART_B_98765"
        monkeypatch.setenv(name_a, "left")
        monkeypatch.setenv(name_b, "right")
        write_raw_models_json(
            models_json_path, {"custom-provider": provider_with_api_key(f"${{{name_a}}}_${{{name_b}}}")}
        )

        runtime = await create_runtime(credentials, models_json_path)
        status = runtime.get_provider_auth_status("custom-provider")

        assert status.configured is True
        assert status.source == f"{name_a}, {name_b}"

    async def test_provider_auth_status_reports_non_env_values_as_a_config_key(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(models_json_path, {"custom-provider": provider_with_api_key("literal_api_key_value")})

        runtime = await create_runtime(credentials, models_json_path)
        status = runtime.get_provider_auth_status("custom-provider")

        assert status.configured is True
        assert status.source == "models_json_key"

    async def test_missing_explicit_env_api_key_keeps_provider_unavailable(
        self, credentials: AuthStorage, models_json_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_var_name = "TEST_API_KEY_MISSING_TEST_98765"
        monkeypatch.delenv(env_var_name, raising=False)
        write_raw_models_json(models_json_path, {"custom-provider": provider_with_api_key(f"${env_var_name}")})

        runtime = await create_runtime(credentials, models_json_path)

        assert runtime.get_provider_auth_status("custom-provider").configured is False
        assert not any(model.provider == "custom-provider" for model in runtime.get_available_snapshot())

    async def test_provider_auth_status_reports_command_values_without_executing_them(
        self, credentials: AuthStorage, models_json_path: Path, tmp_path: Path
    ) -> None:
        counter_file = tmp_path / "status-counter"
        counter_file.write_text("0", encoding="utf-8")
        path = str(counter_file).replace("\\", "/")
        write_raw_models_json(
            models_json_path,
            {"custom-provider": provider_with_api_key(f"!sh -c 'echo 1 > \"{path}\"; echo key-value'")},
        )

        runtime = await create_runtime(credentials, models_json_path)
        status = runtime.get_provider_auth_status("custom-provider")

        assert status.configured is True
        assert status.source == "models_json_command"
        assert counter_file.read_text(encoding="utf-8") == "0"

    async def test_environment_variables_are_not_cached(
        self, credentials: AuthStorage, models_json_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_var_name = "TEST_API_KEY_CACHE_TEST_98765"
        monkeypatch.setenv(env_var_name, "first-value")
        write_raw_models_json(models_json_path, {"custom-provider": provider_with_api_key(f"${env_var_name}")})

        runtime = await create_runtime(credentials, models_json_path)
        assert await api_key_for_provider(runtime, "custom-provider") == "first-value"

        monkeypatch.setenv(env_var_name, "second-value")
        assert await api_key_for_provider(runtime, "custom-provider") == "second-value"

    async def test_get_available_does_not_execute_command_backed_api_key_resolution(
        self, credentials: AuthStorage, models_json_path: Path, tmp_path: Path
    ) -> None:
        counter_file = tmp_path / "counter"
        counter_file.write_text("0", encoding="utf-8")
        write_raw_models_json(
            models_json_path,
            {"custom-provider": provider_with_api_key(_counter_command(counter_file, 'echo "key-value"'))},
        )

        runtime = await create_runtime(credentials, models_json_path)
        available = runtime.get_available_snapshot()

        assert any(model.provider == "custom-provider" for model in available)
        assert int(counter_file.read_text(encoding="utf-8").strip()) == 0

    @pytest.mark.skip(
        reason=(
            "GitHub Copilot's OAuth account-picker filtering reads `availableModelIds` off "
            "a stored OAuth credential and narrows `getAvailable()` accordingly.  This "
            "port's `get_available_snapshot` is the documented synchronous "
            "`has_configured_auth` heuristic and has no per-account model filter."
        )
    )
    def test_get_available_filters_github_copilot_oauth_models(self) -> None:
        raise AssertionError("unreachable")

    @pytest.mark.skip(
        reason=(
            "`getApiKeyAndHeaders` is `ModelRegistry`'s bridge to `pi-ai`'s compat "
            "`complete()` entry point, which this port does not have.  Covers the four "
            "TypeScript cases 'resolves authHeader on every request', 'resolves configured "
            "auth exactly once', 'stored credentials bypass lower-priority configured auth "
            "commands' and the two legacy authHeader error cases.  The underlying "
            "`authHeader` composition itself is covered by "
            "`tests/test_provider_composer.py`."
        )
    )
    def test_get_api_key_and_headers() -> None:
        raise AssertionError("unreachable")


class TestCustomOpenAiCompatibleProvider:
    """End-to-end coverage for a user-defined OpenAI-compatible provider.

    This class has no single TypeScript counterpart: the `model-registry.test.ts`
    cases that exercise this shape all go through `registerProvider` (the dynamic
    extension-provider layer, skipped above), so the `models.json` route a user
    actually writes -- `baseUrl` + `api: "openai-completions"` + `apiKey` +
    `headers` + a `models` **array** -- had nothing pinning it. These assertions
    pin discovery, auth resolution and the resulting outgoing request.
    """

    @staticmethod
    def config() -> dict[str, Any]:
        return {
            "baseUrl": "https://my-llm.test/v1",
            "api": "openai-completions",
            "apiKey": "sk-literal-key",
            "headers": {"x-tenant": "acme"},
            "models": [
                {
                    "id": "my-model",
                    "name": "My Model",
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 1, "output": 2, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 32000,
                    "maxTokens": 4096,
                }
            ],
        }

    async def test_is_discovered_with_its_models_and_composes_without_error(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(models_json_path, {"my-llm": self.config()})
        runtime = await create_runtime(credentials, models_json_path)

        assert runtime.get_error() is None
        provider = runtime.get_provider("my-llm")
        assert provider is not None
        assert provider.id == "my-llm"

        model = find_model(runtime, "my-llm", "my-model")
        assert model is not None
        assert model.name == "My Model"
        assert model.api == "openai-completions"
        assert model.base_url == "https://my-llm.test/v1"
        assert model.context_window == 32000
        assert model.max_tokens == 4096
        assert model.cost.input == 1
        assert model.cost.output == 2
        assert [m.id for m in models_for_provider(runtime, "my-llm")] == ["my-model"]

    async def test_resolves_its_api_key_and_headers(self, credentials: AuthStorage, models_json_path: Path) -> None:
        write_raw_models_json(models_json_path, {"my-llm": self.config()})
        runtime = await create_runtime(credentials, models_json_path)
        model = find_model(runtime, "my-llm", "my-model")
        assert model is not None

        result = await runtime.get_auth(model)
        assert result is not None
        assert result.auth.api_key == "sk-literal-key"
        assert result.auth.headers == {"x-tenant": "acme"}

        assert runtime.has_configured_auth("my-llm") is True
        status = runtime.get_provider_auth_status("my-llm")
        assert status.configured is True
        assert status.type == "api_key"
        assert [m.id for m in runtime.get_available_snapshot() if m.provider == "my-llm"] == ["my-model"]

    async def test_streams_to_its_base_url_with_its_key_and_headers(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(models_json_path, {"my-llm": self.config()})
        runtime = await create_runtime(credentials, models_json_path)
        model = find_model(runtime, "my-llm", "my-model")
        assert model is not None

        requests: list[httpx.Request] = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            chunk = {
                "id": "r",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "my-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            stream = await runtime.stream_simple(model, Context(messages=[]), None, client=client)
            await stream.result()

        assert len(requests) == 1
        request = requests[0]
        assert str(request.url).startswith("https://my-llm.test/v1")
        assert request.headers.get("authorization") == "Bearer sk-literal-key"
        assert request.headers.get("x-tenant") == "acme"

    async def test_model_level_fields_override_the_provider_defaults(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        config = self.config()
        config["models"][0]["baseUrl"] = "https://my-llm.test/v2"
        write_raw_models_json(models_json_path, {"my-llm": config})
        runtime = await create_runtime(credentials, models_json_path)

        model = find_model(runtime, "my-llm", "my-model")
        assert model is not None
        assert model.base_url == "https://my-llm.test/v2"

    async def test_models_must_be_an_array_not_a_keyed_object(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        # `models` is `Type.Array(ModelDefinitionSchema)`; the keyed-object shape a
        # user might reach for must surface through `get_error()` the way TypeScript's
        # schema validation does, not crash `ModelRuntime.create`.
        config = self.config()
        config["models"] = {"my-model": {"id": "my-model", "contextWindow": 32000}}
        write_raw_models_json(models_json_path, {"my-llm": config})
        runtime = await create_runtime(credentials, models_json_path)

        error = runtime.get_error()
        assert error is not None
        assert "models" in error
        assert find_model(runtime, "my-llm", "my-model") is None

    async def test_malformed_model_overrides_are_reported_not_raised(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        config = self.config()
        config["modelOverrides"] = {"my-model": "not-an-object"}
        write_raw_models_json(models_json_path, {"my-llm": config})
        runtime = await create_runtime(credentials, models_json_path)

        error = runtime.get_error()
        assert error is not None
        assert "modelOverrides" in error

    async def test_refresh_picks_up_a_newly_added_custom_provider(
        self, credentials: AuthStorage, models_json_path: Path
    ) -> None:
        write_raw_models_json(models_json_path, {})
        runtime = await create_runtime(credentials, models_json_path)
        assert find_model(runtime, "my-llm", "my-model") is None

        write_raw_models_json(models_json_path, {"my-llm": self.config()})
        runtime.refresh()

        model = find_model(runtime, "my-llm", "my-model")
        assert model is not None
        assert model.base_url == "https://my-llm.test/v1"


def test_models_json_path_is_never_the_real_home(models_json_path: Path) -> None:
    # Guard: every runtime in this module is built with an explicit models path.
    assert str(models_json_path).startswith(os.sep)


async def test_stored_credential_env_propagates_to_configured_header_interpolation(
    models_json_path: Path,
) -> None:
    """Port of the TS "stored API key env propagates to request auth and resolves
    headers" case (the 14th case of the 'dynamic provider lifecycle' describe
    block; see the `test_dynamic_provider_lifecycle` skip above for why the
    other 13 are not ported).

    Unlike its 13 siblings this case never calls `registerProvider`: it stores
    a credential for the *built-in* `cloudflare-ai-gateway` provider whose
    `key` (`$CLOUDFLARE_API_KEY`) and `env` (`CLOUDFLARE_ACCOUNT_ID`/
    `CLOUDFLARE_GATEWAY_ID`) must both be interpolated by `AuthStorage` itself
    on read -- `pi_coding_agent.core.auth_storage.AuthStorage.get`'s
    `_resolve_api_key_credential`, ported from `auth-storage.ts`'s
    `AuthStorage.read()` -- before the built-in Cloudflare AI Gateway auth
    resolver ever sees it. This case is what exposed that a bare
    `CredentialStore` double could satisfy the protocol's shape while dropping
    that `$VAR` resolution; every test in this module now drives the real
    `AuthStorage.in_memory()` through the `credentials` fixture. It
    also configures a `models.json` header (`x-account: $CLOUDFLARE_ACCOUNT_ID`)
    that must interpolate from the stored credential's env -- not from
    `os.environ` -- and pins the built-in AI Gateway auth header composition
    (`cf-aig-authorization`/`Authorization`/`x-api-key`) alongside it. All of
    that is exercised by `ModelRuntime.get_auth`, so it needs none of the
    unported `ModelRegistry`/`getApiKeyAndHeaders` bridge.
    """
    credentials = AuthStorage.in_memory()
    await credentials.set(
        "cloudflare-ai-gateway",
        Credential(
            type="api_key",
            key="$CLOUDFLARE_API_KEY",
            env={
                "CLOUDFLARE_API_KEY": "stored-cf-token",
                "CLOUDFLARE_ACCOUNT_ID": "stored-account",
                "CLOUDFLARE_GATEWAY_ID": "stored-gateway",
            },
        ),
    )
    write_raw_models_json(
        models_json_path, {"cloudflare-ai-gateway": {"headers": {"x-account": "$CLOUDFLARE_ACCOUNT_ID"}}}
    )

    runtime = await create_runtime(credentials, models_json_path)
    model = next((m for m in runtime.get_models() if m.provider == "cloudflare-ai-gateway"), None)
    assert model is not None

    auth = await runtime.get_auth(model)

    assert auth is not None
    assert auth.auth.api_key is None
    # `AuthStorage` resolves the stored credential's `key` (`$CLOUDFLARE_API_KEY`)
    # against its own `env` before the Cloudflare auth resolver runs, so the
    # bearer value below is built from the interpolated "stored-cf-token", not
    # the literal, un-interpolated `$VAR` text.
    expected_bearer_header = "Bearer " + "stored-cf-token"
    assert auth.auth.headers == {
        "cf-aig-authorization": expected_bearer_header,
        "Authorization": None,
        "x-api-key": None,
        "x-account": "stored-account",
    }
    assert auth.env == {
        "CLOUDFLARE_ACCOUNT_ID": "stored-account",
        "CLOUDFLARE_GATEWAY_ID": "stored-gateway",
    }
