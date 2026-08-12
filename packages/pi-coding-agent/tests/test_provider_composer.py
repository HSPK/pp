"""Tests for `core/resolve_config_value.py`, `core/model_config.py`, and `core/provider_composer.py`.

Ported from `packages/coding-agent/test/resolve-config-value.test.ts` (literal
resolution, env templates, command execution/caching) -- the full
assertion-by-assertion port of that file lives in
`test_resolve_config_value_coverage.py`; what is here is the provider-facing
use of it -- plus targeted coverage
of `compose_model_provider`'s built-in + `models.json` overlay and
API-key-only auth composition (`provider-composer.ts`'s equivalent cases,
narrowed to the API-key/no-extension subset this port implements -- see
`provider_composer.py`'s module docstring for the documented OAuth/extension
boundary).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pi_ai.api import openai_completions
from pi_ai.auth.types import ApiKeyAuth, AuthResult, Credential, OAuthAuth, ProviderAuth, ResolvedAuth
from pi_ai.providers import openai_compatible_provider
from pi_ai.registry import Provider, create_provider
from pi_ai.types import Model, ModelCost
from pi_coding_agent.core.model_config import (
    ModelConfig,
    ModelsJsonModel,
    ModelsJsonModelOverride,
    ModelsJsonProvider,
)
from pi_coding_agent.core.provider_composer import (
    apply_model_override,
    apply_models_json,
    compose_api_key_auth,
    compose_model_provider,
    configured_request_auth_status,
    get_api_module,
    merge_compat,
    resolve_compatibility_request_config,
    resolve_configured_model_headers,
)
from pi_coding_agent.core.resolve_config_value import (
    clear_config_value_cache,
    resolve_config_value,
    resolve_config_value_uncached,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_config_value_cache()
    yield
    clear_config_value_cache()


def test_resolves_literals_environment_templates_and_escapes(monkeypatch):
    monkeypatch.setenv("TEST_CONFIG_LEFT", "left")
    monkeypatch.setenv("TEST_CONFIG_RIGHT", "right")

    assert resolve_config_value("literal-key") == "literal-key"
    assert resolve_config_value("$TEST_CONFIG_LEFT") == "left"
    assert resolve_config_value("${TEST_CONFIG_LEFT}_$TEST_CONFIG_RIGHT") == "left_right"
    assert resolve_config_value("$$TEST_CONFIG_LEFT") == "$TEST_CONFIG_LEFT"
    assert resolve_config_value("$!literal-$TEST_CONFIG_RIGHT") == "!literal-right"


def test_uses_credential_scoped_environment_before_process_env(monkeypatch):
    monkeypatch.setenv("TEST_CONFIG_SCOPED", "process")
    assert resolve_config_value("$TEST_CONFIG_SCOPED", {"TEST_CONFIG_SCOPED": "credential"}) == "credential"


def test_executes_shell_commands_and_trims_output():
    assert resolve_config_value("!echo '  spaced-key  '") == "spaced-key"
    assert resolve_config_value("!printf 'line1\\nline2'") == "line1\nline2"
    assert resolve_config_value("!echo 'hello world' | tr ' ' '-'") == "hello-world"


@pytest.mark.parametrize("command", ["!exit 1", "!nonexistent-command-12345", "!printf ''"])
def test_returns_none_when_command_resolution_fails(command):
    assert resolve_config_value(command) is None


def test_caches_successful_and_failed_commands_until_cleared(tmp_path: Path):
    counter_file = tmp_path / "counter"
    counter_file.write_text("0")
    success = f'!sh -c \'count=$(cat "{counter_file}"); echo $((count + 1)) > "{counter_file}"; echo value\''

    assert resolve_config_value(success) == "value"
    assert resolve_config_value(success) == "value"
    assert counter_file.read_text().strip() == "1"

    clear_config_value_cache()
    assert resolve_config_value(success) == "value"
    assert counter_file.read_text().strip() == "2"

    failure = f'!sh -c \'count=$(cat "{counter_file}"); echo $((count + 1)) > "{counter_file}"; exit 1\''
    assert resolve_config_value(failure) is None
    assert resolve_config_value(failure) is None
    # A failed command is cached too: the second call must not re-run it, so
    # the counter reaches 3 rather than 4.
    assert counter_file.read_text().strip() == "3"


def test_does_not_cache_environment_values(monkeypatch):
    monkeypatch.setenv("TEST_CONFIG_DYNAMIC", "first")
    assert resolve_config_value("$TEST_CONFIG_DYNAMIC") == "first"
    monkeypatch.setenv("TEST_CONFIG_DYNAMIC", "second")
    assert resolve_config_value("$TEST_CONFIG_DYNAMIC") == "second"


def test_uncached_resolution_executes_a_command_on_every_call(tmp_path: Path):
    counter_file = tmp_path / "uncached-counter"
    counter_file.write_text("0")
    command = f'!sh -c \'count=$(cat "{counter_file}"); echo $((count + 1)) > "{counter_file}"; echo value\''
    assert resolve_config_value_uncached(command) == "value"
    assert resolve_config_value_uncached(command) == "value"
    assert counter_file.read_text().strip() == "2"


@pytest.mark.skip(
    reason="TS `uses stdin when the configured Windows shell requires it` fakes `process.platform = 'win32'` "
    "and mocks `getShellConfig()` to return `{shell: '/bin/bash', args: ['-s'], commandTransport: 'stdin'}`. "
    "That whole branch is Windows-only: `executeCommandUncached` only consults `getShellConfig` on win32, "
    "and `core/resolve_config_value.py` has no `getShellConfig` equivalent -- it always runs "
    "`/bin/sh -c <command>`, which is exactly TypeScript's non-win32 `executeWithDefaultShell` path."
)
def test_uses_stdin_when_the_configured_windows_shell_requires_it() -> None:
    raise AssertionError("unreachable")


def _base_provider():
    return openai_compatible_provider(
        provider_id="test-provider",
        name="Test Provider",
        base_url="https://api.example.com",
        env_vars=["TEST_PROVIDER_API_KEY"],
        models=[],
    )


def test_model_config_load_missing_file_returns_empty(tmp_path: Path):
    config = ModelConfig.load(tmp_path / "does-not-exist.json")
    assert config.get_error() is None
    assert config.get_provider("anything") is None


def test_model_config_load_invalid_json_reports_error(tmp_path: Path):
    path = tmp_path / "models.json"
    path.write_text("{not valid json")
    config = ModelConfig.load(path)
    assert config.get_error() is not None


def test_model_config_load_strips_comments_and_parses_providers(tmp_path: Path):
    path = tmp_path / "models.json"
    path.write_text(
        """
        {
          // a comment
          "providers": {
            "test-provider": {
              "baseUrl": "https://override.example.com",
              /* block comment */
              "models": [
                {"id": "custom-model", "contextWindow": 32000, "maxTokens": 4096}
              ]
            }
          }
        }
        """
    )
    config = ModelConfig.load(path)
    assert config.get_error() is None
    provider = config.get_provider("test-provider")
    assert provider is not None
    assert provider.base_url == "https://override.example.com"
    assert provider.models[0].id == "custom-model"


def test_compose_model_provider_without_config_returns_equivalent_provider():
    base = _base_provider()
    config = ModelConfig()
    composed = compose_model_provider("test-provider", base, config)
    assert composed.id == "test-provider"
    assert composed.base_url == "https://api.example.com"
    assert composed.get_models() == []


def test_compose_model_provider_overlays_base_url_and_models(tmp_path: Path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "test-provider": {
                        "baseUrl": "https://override.example.com",
                        "models": [
                            {
                                "id": "custom-model",
                                "name": "Custom Model",
                                "contextWindow": 64000,
                                "maxTokens": 8192,
                            }
                        ],
                    }
                }
            }
        )
    )
    config = ModelConfig.load(path)
    composed = compose_model_provider("test-provider", _base_provider(), config)
    assert composed.base_url == "https://override.example.com"
    model = composed.get_model("custom-model")
    assert model is not None
    assert model.name == "Custom Model"
    assert model.context_window == 64000
    assert model.provider == "test-provider"
    assert model.base_url == "https://override.example.com"


def test_compose_model_provider_applies_model_overrides(tmp_path: Path):
    base = openai_compatible_provider(
        provider_id="test-provider",
        name="Test Provider",
        base_url="https://api.example.com",
        env_vars=["TEST_PROVIDER_API_KEY"],
        models=[],
    )
    from pi_ai.types import Model, ModelCost

    base.models.append(
        Model(
            id="base-model",
            name="Base Model",
            api="openai-completions",
            provider="test-provider",
            base_url="https://api.example.com",
            context_window=8000,
            max_tokens=1024,
            cost=ModelCost(input=1, output=2),
        )
    )

    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "test-provider": {
                        "modelOverrides": {"base-model": {"contextWindow": 99000, "name": "Renamed"}},
                    }
                }
            }
        )
    )
    config = ModelConfig.load(path)
    composed = compose_model_provider("test-provider", base, config)
    model = composed.get_model("base-model")
    assert model is not None
    assert model.context_window == 99000
    assert model.name == "Renamed"
    # Unset fields fall back to the base model's values.
    assert model.cost.input == 1


def test_compose_model_provider_configured_api_key_resolves_lazily(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CONFIGURED_KEY_ENV", raising=False)
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": {"test-provider": {"apiKey": "$CONFIGURED_KEY_ENV"}}}))
    config = ModelConfig.load(path)
    composed = compose_model_provider("test-provider", _base_provider(), config)

    async def run():
        # No env var set yet: resolution should raise via resolve_config_value_or_throw.
        with pytest.raises(ValueError):
            await composed.auth.api_key.resolve()

    asyncio.run(asyncio.wait_for(run(), timeout=5))

    monkeypatch.setenv("CONFIGURED_KEY_ENV", "sk-configured")

    async def run_again():
        result = await composed.auth.api_key.resolve()
        assert result is not None
        assert result.auth.api_key == "sk-configured"

    asyncio.run(asyncio.wait_for(run_again(), timeout=5))


def test_configured_request_auth_status_reports_environment_source(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STATUS_ENV_KEY", "value")
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": {"test-provider": {"apiKey": "$STATUS_ENV_KEY"}}}))
    config = ModelConfig.load(path)
    status = configured_request_auth_status(config.get_provider("test-provider"))
    assert status is not None
    assert status.configured is True
    assert status.source == "environment"


def test_compose_model_provider_allows_config_only_provider_without_api(tmp_path: Path):
    """TypeScript resolves the api per model inside `modelFromJson`, so a config-only
    provider that declares no models composes with an empty model list rather than
    failing. Verified against `composeModelProvider` in the TypeScript original."""
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": {"only-config": {"baseUrl": "https://x.example.com", "apiKey": "k"}}}))
    config = ModelConfig.load(path)
    provider = compose_model_provider("only-config", None, config)
    assert provider.name == "only-config"
    assert list(provider.get_models()) == []


def test_compose_model_provider_requires_api_per_model_when_no_base(tmp_path: Path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "only-config": {
                        "baseUrl": "https://x.example.com",
                        "apiKey": "k",
                        "models": [{"id": "m1", "name": "M1", "contextWindow": 1000, "maxTokens": 100}],
                    }
                }
            }
        )
    )
    config = ModelConfig.load(path)
    with pytest.raises(ValueError, match='no "api" specified'):
        compose_model_provider("only-config", None, config)


def _base_model(model_id: str = "base-model", **overrides) -> Model:
    fields: dict = {
        "id": model_id,
        "name": "Base Model",
        "api": "openai-completions",
        "provider": "test-provider",
        "base_url": "https://api.example.com",
        "context_window": 8000,
        "max_tokens": 1024,
        "cost": ModelCost(input=1, output=2),
    }
    fields.update(overrides)
    return Model(**fields)


def _provider_with_models(*models: Model) -> Provider:
    return openai_compatible_provider(
        provider_id="test-provider",
        name="Test Provider",
        base_url="https://api.example.com",
        env_vars=["TEST_PROVIDER_API_KEY"],
        models=list(models),
    )


def _custom_auth_provider(resolve) -> Provider:
    """A base provider whose API-key auth has an explicit resolver, so the
    composed provider inherits that resolver instead of the default env one."""
    return create_provider(
        id="custom",
        name="Custom",
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Custom key", resolve=resolve)),
        api=openai_completions,
        models=[],
        base_url="https://custom.example.com",
    )


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=5))


def test_get_api_module_returns_none_for_an_unknown_api():
    assert get_api_module("openai-completions") is openai_completions
    assert get_api_module("not-a-real-api") is None


def test_merge_compat_keeps_the_base_when_there_is_no_override():
    base = {"forceLegacy": True}
    assert merge_compat(base, None) is base
    assert merge_compat(None, None) is None


def test_merge_compat_deep_merges_the_known_nested_option_bags():
    merged = merge_compat(
        {"openRouterRouting": {"order": ["a"]}, "chatTemplateKwargs": {"x": 1}, "forceLegacy": True},
        {"openRouterRouting": {"sort": "price"}, "chatTemplateKwargs": {"y": 2}, "forceLegacy": False},
    )
    assert merged == {
        "openRouterRouting": {"order": ["a"], "sort": "price"},
        "chatTemplateKwargs": {"x": 1, "y": 2},
        "forceLegacy": False,
    }


def test_merge_compat_promotes_a_nested_bag_that_only_one_side_defines():
    assert merge_compat({"vercelGatewayRouting": {"only": "base"}}, {"forceLegacy": True}) == {
        "vercelGatewayRouting": {"only": "base"},
        "forceLegacy": True,
    }


def test_apply_model_override_merges_cost_thinking_and_sampling_field_by_field():
    model = _base_model(
        thinking_level_map={"low": "1024"},
        sampling_params={"temperature": 0.2},
        compat={"openRouterRouting": {"order": ["a"]}},
    )
    override = ModelsJsonModelOverride(
        reasoning=True,
        thinking_level_map={"high": "8192"},
        input=["text", "image"],
        cost={"input": 5, "cacheRead": 0.5},
        max_tokens=4096,
        sampling_params={"top_p": 0.9},
        compat={"openRouterRouting": {"sort": "price"}},
    )

    result = apply_model_override(model, override)

    assert result.reasoning is True
    assert result.thinking_level_map == {"low": "1024", "high": "8192"}
    assert result.input == ["text", "image"]
    # Unset cost fields fall back to the base cost.
    assert (result.cost.input, result.cost.output, result.cost.cache_read) == (5, 2, 0.5)
    assert result.context_window == 8000
    assert result.max_tokens == 4096
    assert result.sampling_params == {"temperature": 0.2, "top_p": 0.9}
    assert result.compat == {"openRouterRouting": {"order": ["a"], "sort": "price"}}
    # The original model is untouched.
    assert model.max_tokens == 1024


def test_apply_models_json_without_a_config_returns_the_base_models():
    models = [_base_model()]
    assert apply_models_json("test-provider", models, None, "openai-completions") == models


def test_apply_models_json_rejects_an_empty_provider_entry():
    with pytest.raises(ValueError, match="must specify"):
        apply_models_json("test-provider", [], ModelsJsonProvider(), "openai-completions")


def test_apply_models_json_requires_a_base_url_for_oauth_providers():
    with pytest.raises(ValueError, match='"baseUrl" is required when "oauth" is set'):
        apply_models_json("test-provider", [], ModelsJsonProvider(oauth="radius"), "openai-completions")


def test_apply_models_json_keeps_base_urls_for_radius_oauth_providers():
    config = ModelsJsonProvider(oauth="radius", base_url="https://radius.example.com")
    [model] = apply_models_json("test-provider", [_base_model()], config, "openai-completions")
    # A radius provider's configured base URL routes auth, it does not rewrite models.
    assert model.base_url == "https://api.example.com"


def test_apply_models_json_rewrites_base_urls_and_merges_compat_for_other_providers():
    config = ModelsJsonProvider(base_url="https://override.example.com", compat={"forceLegacy": True})
    [model] = apply_models_json("test-provider", [_base_model(compat={"other": 1})], config, "openai-completions")
    assert model.base_url == "https://override.example.com"
    assert model.compat == {"other": 1, "forceLegacy": True}


def test_apply_models_json_replaces_a_model_that_shadows_a_base_model():
    config = ModelsJsonProvider(
        base_url="https://override.example.com",
        models=[ModelsJsonModel(id="base-model", name="Replaced", context_window=4242)],
    )
    models = apply_models_json("test-provider", [_base_model()], config, "openai-completions")
    assert len(models) == 1
    assert models[0].name == "Replaced"
    assert models[0].context_window == 4242


def test_apply_models_json_appends_a_new_model_using_the_first_base_model_as_defaults():
    config = ModelsJsonProvider(models=[ModelsJsonModel(id="extra-model")])
    models = apply_models_json("test-provider", [_base_model()], config, "openai-completions")
    assert [model.id for model in models] == ["base-model", "extra-model"]
    extra = models[1]
    # `baseUrl` and `api` are inherited from the defaults model.
    assert extra.base_url == "https://api.example.com"
    assert extra.api == "openai-completions"
    assert extra.name == "extra-model"
    assert (extra.context_window, extra.max_tokens) == (128_000, 16_384)
    assert extra.input == ["text"]


def test_model_definition_overrides_the_provider_level_api():
    # TypeScript's `modelFromJson` reads `definition.api ?? providerConfig.api`,
    # so a per-model `api` wins over the provider default. `compose_model_provider`
    # then carries a `{api name: module}` mapping for every wire format in play.
    config = ModelsJsonProvider(
        base_url="https://x.example.com",
        models=[ModelsJsonModel(id="m", api="anthropic-messages")],
    )
    models = apply_models_json("test-provider", [], config, "openai-completions")

    assert [model.api for model in models] == ["anthropic-messages"]


def test_model_definition_requires_a_base_url():
    config = ModelsJsonProvider(models=[ModelsJsonModel(id="m")])
    with pytest.raises(ValueError, match='"baseUrl" is required when defining custom models'):
        apply_models_json("test-provider", [], config, "openai-completions")


@pytest.mark.parametrize(
    ("definition", "message"),
    [
        (ModelsJsonModel(id="m", context_window=0), "invalid contextWindow"),
        (ModelsJsonModel(id="m", max_tokens=-1), "invalid maxTokens"),
    ],
)
def test_model_definition_rejects_non_positive_limits(definition: ModelsJsonModel, message: str):
    config = ModelsJsonProvider(base_url="https://x.example.com", models=[definition])
    with pytest.raises(ValueError, match=message):
        apply_models_json("test-provider", [], config, "openai-completions")


def test_model_definition_carries_its_own_fields_and_merges_provider_compat():
    config = ModelsJsonProvider(
        base_url="https://x.example.com",
        compat={"forceLegacy": True},
        models=[
            ModelsJsonModel(
                id="m",
                name="M",
                base_url="https://model.example.com",
                reasoning=True,
                thinking_level_map={"low": "1"},
                input=["text", "image"],
                cost={"input": 3, "output": 4, "cacheRead": 1, "cacheWrite": 2},
                context_window=1000,
                max_tokens=100,
                sampling_params={"temperature": 0},
                compat={"chatTemplateKwargs": {"a": 1}},
            )
        ],
    )
    [model] = apply_models_json("test-provider", [], config, "openai-completions")
    assert model.base_url == "https://model.example.com"
    assert model.reasoning is True
    assert model.thinking_level_map == {"low": "1"}
    assert model.input == ["text", "image"]
    assert (model.cost.input, model.cost.cache_write) == (3, 2)
    assert model.sampling_params == {"temperature": 0}
    assert model.compat == {"forceLegacy": True, "chatTemplateKwargs": {"a": 1}}


def test_composed_provider_keeps_the_env_var_auth_of_its_base(monkeypatch):
    """Regression: a `models.json` overlay that only sets `baseUrl` must not
    drop the base provider's environment-variable API key resolution."""
    monkeypatch.setenv("TEST_PROVIDER_API_KEY", "sk-from-env")
    config = ModelConfig({"test-provider": ModelsJsonProvider(base_url="https://override.example.com")})
    composed = compose_model_provider("test-provider", _base_provider(), config)

    result = _run(composed.auth.api_key.resolve())

    assert result is not None
    assert result.auth.api_key == "sk-from-env"
    assert result.source == "TEST_PROVIDER_API_KEY"


def test_composed_provider_without_any_auth_source_resolves_to_none():
    config = ModelConfig({"test-provider": ModelsJsonProvider(base_url="https://override.example.com")})
    composed = compose_model_provider("test-provider", _base_provider(), config)

    assert _run(composed.auth.api_key.resolve(env=lambda _name: None)) is None


def test_compose_api_key_auth_without_a_base_or_a_configured_key_resolves_to_none():
    auth = compose_api_key_auth("solo", None, ModelsJsonProvider(base_url="https://x.example.com"))
    assert auth.name == "API key"
    assert _run(auth.resolve()) is None


def test_compose_api_key_auth_uses_a_stored_credential_when_there_is_no_base():
    auth = compose_api_key_auth("solo", None, ModelsJsonProvider(base_url="https://x.example.com"))
    result = _run(auth.resolve(credential=Credential(type="api_key", key="sk-stored", env={"E": "v"})))
    assert result is not None
    assert result.auth.api_key == "sk-stored"
    assert result.source == "stored credential"
    assert result.env == {"E": "v"}


def test_compose_api_key_auth_ignores_a_keyless_credential_when_there_is_no_base():
    auth = compose_api_key_auth("solo", None, ModelsJsonProvider(base_url="https://x.example.com"))
    assert _run(auth.resolve(credential=Credential(type="oauth", access="token"))) is None


def test_compose_api_key_auth_prefers_a_stored_credential_over_the_environment(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_API_KEY", "sk-from-env")
    auth = compose_api_key_auth("test-provider", _base_provider(), None)
    result = _run(auth.resolve(credential=Credential(type="api_key", key="sk-stored")))
    assert result is not None
    assert result.auth.api_key == "sk-stored"
    assert result.source == "stored credential"
    assert auth.name == "Test Provider API key"


def test_compose_api_key_auth_routes_a_configured_key_through_the_inherited_resolver():
    config = ModelsJsonProvider(api_key="sk-configured")
    auth = compose_api_key_auth("test-provider", _base_provider(), config)
    result = _run(auth.resolve())
    assert result is not None
    assert result.auth.api_key == "sk-configured"
    # The base provider treats the injected key as a stored credential.
    assert result.source == "stored credential"


def test_compose_api_key_auth_reports_a_configured_key_without_a_base():
    auth = compose_api_key_auth("solo", None, ModelsJsonProvider(api_key="sk-configured"))
    result = _run(auth.resolve())
    assert result is not None
    assert result.auth.api_key == "sk-configured"
    assert result.source == "configured API key"


def test_compose_api_key_auth_resolves_a_configured_key_from_an_async_env_lookup():
    async def env(name: str) -> str | None:
        return "sk-async" if name == "ASYNC_COMPOSER_KEY" else None

    auth = compose_api_key_auth("solo", None, ModelsJsonProvider(api_key="$ASYNC_COMPOSER_KEY"))
    result = _run(auth.resolve(env=env))
    assert result is not None
    assert result.auth.api_key == "sk-async"


def test_compose_api_key_auth_resolves_configured_headers_and_the_auth_header():
    async def resolve(credential=None, env=None):
        return AuthResult(auth=ResolvedAuth(api_key="sk-base", headers={"X-Base": "base"}), source="custom")

    config = ModelsJsonProvider(headers={"X-Configured": "$COMPOSER_HEADER_ENV"}, auth_header=True)
    auth = compose_api_key_auth("custom", _custom_auth_provider(resolve), config)

    result = _run(auth.resolve(env=lambda name: "header-value" if name == "COMPOSER_HEADER_ENV" else None))

    assert result is not None
    assert result.auth.headers["X-Base"] == "base"
    assert result.auth.headers["X-Configured"] == "header-value"
    assert "sk-base" in result.auth.headers["Authorization"]


def test_compose_api_key_auth_rejects_an_auth_header_without_a_resolved_key():
    async def resolve(credential=None, env=None):
        return AuthResult(auth=ResolvedAuth(api_key=None, headers={"X-Base": "base"}), source="custom")

    config = ModelsJsonProvider(auth_header=True)
    auth = compose_api_key_auth("custom", _custom_auth_provider(resolve), config)

    with pytest.raises(ValueError, match="authHeader requires a resolved API key"):
        _run(auth.resolve())


def test_compose_api_key_auth_resolves_headers_from_the_credential_environment():
    config = ModelsJsonProvider(headers={"X-Token": "$SHARED_COMPOSER_ENV"})
    auth = compose_api_key_auth("solo", None, config)

    def env(name: str) -> str | None:
        raise AssertionError(f"env lookup should not run for {name}")

    result = _run(
        auth.resolve(
            credential=Credential(type="api_key", key="sk", env={"SHARED_COMPOSER_ENV": "from-credential"}),
            env=env,
        )
    )

    assert result is not None
    assert result.auth.headers == {"X-Token": "from-credential"}


def test_compose_api_key_auth_fails_when_a_configured_header_cannot_be_resolved():
    auth = compose_api_key_auth("solo", None, ModelsJsonProvider(api_key="sk", headers={"X": "$UNSET_COMPOSER_ENV"}))
    with pytest.raises(ValueError, match="X"):
        _run(auth.resolve())


def test_resolve_configured_model_headers_merges_definition_over_override():
    model = _base_model("m")
    config = ModelsJsonProvider(
        models=[ModelsJsonModel(id="m", headers={"X-Def": "$MODEL_HEADER_ENV"})],
        model_overrides={"m": ModelsJsonModelOverride(headers={"X-Ovr": "ovr", "X-Def": "loses"})},
    )
    headers = resolve_configured_model_headers(model, config, {"MODEL_HEADER_ENV": "resolved"})
    assert headers == {"X-Ovr": "ovr", "X-Def": "resolved"}


def test_resolve_configured_model_headers_returns_none_without_configured_headers():
    model = _base_model("m")
    assert resolve_configured_model_headers(model, None) is None
    assert resolve_configured_model_headers(model, ModelsJsonProvider(base_url="https://x.example.com")) is None


def test_resolve_compatibility_request_config_merges_provider_and_model_headers():
    model = _base_model("m", headers={"X-Builtin": "builtin", "X-Provider": "loses"})
    config = ModelsJsonProvider(
        headers={"X-Provider": "provider"},
        auth_header=True,
        models=[ModelsJsonModel(id="m", headers={"X-Model": "model"})],
    )
    result = resolve_compatibility_request_config(model, config)
    assert result.headers == {"X-Builtin": "builtin", "X-Provider": "provider", "X-Model": "model"}
    assert result.auth_header is True


def test_resolve_compatibility_request_config_without_any_headers():
    result = resolve_compatibility_request_config(_base_model("m"), None)
    assert result.headers is None
    assert result.auth_header is False


@pytest.mark.parametrize(
    ("api_key", "configured", "source"),
    [
        ("!echo sk-from-command", True, "models_json_command"),
        ("sk-literal", True, "models_json_key"),
    ],
)
def test_configured_request_auth_status_classifies_key_shapes(api_key: str, configured: bool, source: str):
    status = configured_request_auth_status(ModelsJsonProvider(api_key=api_key))
    assert status is not None
    assert status.configured is configured
    assert status.source == source


def test_configured_request_auth_status_reports_an_unset_environment_variable(monkeypatch):
    monkeypatch.delenv("UNSET_STATUS_ENV", raising=False)
    status = configured_request_auth_status(ModelsJsonProvider(api_key="$UNSET_STATUS_ENV"))
    assert status is not None
    assert status.configured is False
    assert status.source is None


def test_configured_request_auth_status_is_none_without_a_configured_key():
    assert configured_request_auth_status(None) is None
    assert configured_request_auth_status(ModelsJsonProvider(base_url="https://x.example.com")) is None


def test_compose_model_provider_builds_a_config_only_provider():
    config = ModelConfig(
        {
            "only-config": ModelsJsonProvider(
                api="openai-completions",
                base_url="https://only.example.com",
                api_key="sk-only",
                headers={"X-Provider": "provider"},
                models=[ModelsJsonModel(id="only-model", context_window=1000, max_tokens=100)],
            )
        }
    )
    composed = compose_model_provider("only-config", None, config)

    assert composed.name == "only-config"
    assert composed.api is openai_completions
    assert composed.base_url == "https://only.example.com"
    assert composed.headers == {"X-Provider": "provider"}
    assert [model.id for model in composed.get_models()] == ["only-model"]
    assert _run(composed.auth.api_key.resolve()).auth.api_key == "sk-only"


def test_compose_model_provider_rejects_an_unknown_api_for_a_config_only_provider():
    config = ModelConfig(
        {"only-config": ModelsJsonProvider(api="made-up-api", base_url="https://x.example.com", api_key="k")}
    )
    with pytest.raises(ValueError, match="no API module registered"):
        compose_model_provider("only-config", None, config)


def test_compose_model_provider_prefers_the_configured_name_and_merges_headers():
    base = _provider_with_models(_base_model())
    base.headers["X-Base"] = "base"
    config = ModelConfig(
        {
            "test-provider": ModelsJsonProvider(
                name="Renamed Provider",
                headers={"X-Configured": "configured"},
            )
        }
    )
    composed = compose_model_provider("test-provider", base, config)
    assert composed.name == "Renamed Provider"
    assert composed.headers == {"X-Base": "base", "X-Configured": "configured"}
    assert composed.base_url == "https://api.example.com"


def test_compose_model_provider_applies_overrides_to_models_added_by_models_json():
    config = ModelConfig(
        {
            "test-provider": ModelsJsonProvider(
                base_url="https://override.example.com",
                models=[ModelsJsonModel(id="added", context_window=1000, max_tokens=100)],
                model_overrides={"added": ModelsJsonModelOverride(name="Overridden")},
            )
        }
    )
    composed = compose_model_provider("test-provider", _provider_with_models(), config)
    model = composed.get_model("added")
    assert model is not None
    assert model.name == "Overridden"


def test_compose_api_key_auth_keeps_inherited_headers_and_base_url_without_configured_headers():
    async def resolve(credential=None, env=None):
        return AuthResult(
            auth=ResolvedAuth(api_key="sk-base", headers={"X-Base": "base"}, base_url="https://per-token.example.com"),
            source="custom",
        )

    auth = compose_api_key_auth("custom", _custom_auth_provider(resolve), None)
    result = _run(auth.resolve())

    assert result is not None
    assert result.auth.headers == {"X-Base": "base"}
    assert result.auth.base_url == "https://per-token.example.com"


def test_compose_model_provider_overrides_only_the_matching_model():
    config = ModelConfig(
        {"test-provider": ModelsJsonProvider(model_overrides={"second": ModelsJsonModelOverride(name="Second!")})}
    )
    base = _provider_with_models(_base_model("first"), _base_model("second"))
    composed = compose_model_provider("test-provider", base, config)
    assert composed.get_model("first").name == "Base Model"
    assert composed.get_model("second").name == "Second!"


async def _unused_login(_interaction: object) -> Credential:  # pragma: no cover - never invoked
    raise AssertionError("login must not be called")


async def _unused_refresh(credential: Credential, _signal: object) -> Credential:  # pragma: no cover
    return credential


async def _unused_to_auth(_credential: Credential) -> ResolvedAuth:  # pragma: no cover
    return ResolvedAuth(api_key="unused")


def _oauth_only_provider() -> Provider:
    return Provider(
        id="oauth-only",
        name="OAuth Only",
        auth=ProviderAuth(
            api_key=None,
            oauth=OAuthAuth(
                name="OAuth Only",
                login=_unused_login,
                refresh=_unused_refresh,
                to_auth=_unused_to_auth,
            ),
        ),
        api=openai_completions,
        base_url="https://example.invalid/v1",
        models=[],
    )


def test_oauth_only_provider_gets_no_fabricated_api_key_method() -> None:
    """Port of `composeApiKeyAuth`'s early return (`provider-composer.ts:310`).

    TypeScript: `if (!inherited && rawKey === undefined && oauth) return undefined;`
    A provider that inherits no API-key method and configures no key, but does
    have OAuth, must not be handed a synthesized "enter API key" login -- the
    UI would offer a credential path the provider cannot honor.
    """
    assert compose_api_key_auth("oauth-only", _oauth_only_provider(), None) is None


def test_configured_api_key_still_composes_for_an_oauth_provider() -> None:
    """The early return is conditional on there being no configured key.

    Guards against over-applying the rule: `models.json` supplying an explicit
    key must still produce an API-key method even when the base has OAuth,
    because `rawKey !== undefined` fails TypeScript's condition.
    """
    composed = compose_api_key_auth("oauth-only", _oauth_only_provider(), ModelsJsonProvider(api_key="sk-configured"))
    assert composed is not None


def _config_with(tmp_path: Path, provider_id: str, entry: dict[str, object]) -> ModelConfig:
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"providers": {provider_id: entry}}))
    config = ModelConfig.load(path)
    assert config.get_error() is None
    return config


def test_composition_preserves_the_base_providers_oauth_method(tmp_path: Path) -> None:
    """Port of `composeOAuthAuth` (`provider-composer.ts:367`).

    `const oauth = extension?.oauth ? adaptOAuth(...) : base?.auth.oauth` -- a
    provider that appears in `models.json` keeps the OAuth login it had before,
    so `/login` still offers the subscription flow. Dropping it silently
    downgraded every listed provider to API-key-only.
    """
    config = _config_with(tmp_path, "oauth-only", {"name": "Renamed", "baseUrl": "https://gw.invalid/v1"})
    composed = compose_model_provider("oauth-only", _oauth_only_provider(), config)

    assert composed.name == "Renamed"
    assert composed.auth.oauth is not None
    assert composed.auth.oauth.name == "OAuth Only"
    # The method must stay awaitable: the login flow awaits `login`/`to_auth`,
    # and re-wrapping `to_auth` is the one thing composition changes about it.
    assert asyncio.iscoroutinefunction(composed.auth.oauth.login)
    assert asyncio.iscoroutinefunction(composed.auth.oauth.to_auth)


async def test_composed_oauth_applies_configured_headers_to_resolved_auth(tmp_path: Path) -> None:
    """`composeOAuthAuth`'s `toAuth` wrapper runs `withConfiguredAuth`.

    Configured `headers` from `models.json` must reach OAuth-authenticated
    requests, not only API-key ones.
    """
    config = _config_with(tmp_path, "oauth-only", {"headers": {"X-Tenant": "acme"}})
    composed = compose_model_provider("oauth-only", _oauth_only_provider(), config)
    assert composed.auth.oauth is not None

    resolved = await composed.auth.oauth.to_auth(Credential(type="oauth"))
    assert resolved.headers["X-Tenant"] == "acme"


def test_the_no_authentication_method_guard_is_unreachable_for_a_base_without_auth() -> None:
    """`provider-composer.ts:451` -- `if (!apiKey && !oauth) throw ...`.

    The guard is ported literally but, as in TypeScript, cannot fire:
    `composeApiKeyAuth` only returns `undefined` when `oauth` is truthy
    (`if (!inherited && rawKey === undefined && oauth) return undefined`), so
    `!apiKey` implies `oauth`. A base with neither method therefore still gets
    an API-key method whose `resolve` yields nothing, rather than an error.
    """
    base = Provider(
        id="no-auth",
        name="No Auth",
        auth=ProviderAuth(api_key=None, oauth=None),
        api=openai_completions,
        base_url="https://example.invalid/v1",
        models=[],
    )

    composed = compose_model_provider("no-auth", base, ModelConfig())
    assert composed.auth.oauth is None
    assert composed.auth.api_key is not None
    assert asyncio.run(composed.auth.api_key.resolve()) is None
