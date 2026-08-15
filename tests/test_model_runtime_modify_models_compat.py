"""Python port of `packages/coding-agent/test/model-runtime-modify-models-compat.test.ts`.

The TypeScript file exercises the *extension provider* layer of `ModelRuntime`:
`registerNativeProvider`, `registerProvider`, `ModelRegistry`, `ModelsStore`
persistence, deferred fetch/cancel and the legacy OAuth `modifyModels` hook.
This port deliberately omits that layer (see `core/model_runtime.py`'s module
docstring: "No extension providers", "No locked, revision-tracked ModelsStore",
"No remote model catalog refresh"), so only the two behaviours that survive get
real assertions here:

- a natively-supplied `Provider` is reachable through the runtime and its own
  auth implementation is what resolves credentials
  (`ModelRuntime.create(providers=...)` is this port's stand-in for
  `registerNativeProvider`), and
- `models.json` `modelOverrides` win over the provider's own model definitions.

Everything else is skipped at the exact assertion, with the reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pi_ai.utils.http as pi_ai_http
import pytest
from pi_ai.auth.types import (
    ApiKeyAuth,
    AuthResult,
    Credential,
    EnvLookup,
    InMemoryCredentialStore,
    ProviderAuth,
    ResolvedAuth,
)
from pi_ai.registry import Provider
from pi_ai.types import AssistantMessage, Context, Model, ModelCost, TextContent, UserMessage

from pi_coding_agent.core.model_runtime import ModelRuntime


def make_model(model_id: str, *, provider: str, base_url: str) -> Model:
    """Port of the TS `model(id)` helper (provider/baseUrl are per-test there)."""
    return Model(
        id=model_id,
        name=model_id,
        api="openai-completions",
        provider=provider,
        base_url=base_url,
        reasoning=False,
        input=["text"],
        cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
        context_window=1000,
        max_tokens=100,
    )


def _unused(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("unused")


async def test_registers_native_providers_with_their_auth_implementation(tmp_path: Path) -> None:
    native_model = make_model("native", provider="extension-native", base_url="https://fallback.test/v1")

    async def resolve(*, credential: Credential | None = None, env: EnvLookup | None = None) -> AuthResult | None:
        if credential is not None and credential.key:
            return AuthResult(
                auth=ResolvedAuth(api_key=credential.key, base_url="https://resolved.test/v1"),
                source="stored native key",
            )
        return None

    provider = Provider(
        id="extension-native",
        name="Extension Native",
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Native setup", resolve=resolve)),
        api=_unused,
        base_url="https://fallback.test/v1",
        models=[native_model],
    )

    runtime = await ModelRuntime.create(
        agent_dir=tmp_path / "agent",
        models_path=tmp_path / "missing-models.json",
        credentials=InMemoryCredentialStore(),
        providers=[provider],
    )

    # `expect(registry.getProvider("extension-native")).toBe(provider)`. With no
    # `models.json` entry for the id, `_compose_all` passes the provider through
    # untouched, so identity holds here exactly as in TypeScript.
    assert runtime.get_provider("extension-native") is provider
    # Skipped: `registry.getRegisteredNativeProvider(...)`,
    # `registry.getRegisteredProviderIds()` and `registry.unregisterProvider(...)`
    # belong to `ModelRegistry`, which this port does not have (see
    # `core/model_runtime.py`: "No extension providers").
    assert runtime.get_model("extension-native", "native") is not None

    # TS logs in through the provider's own `apiKey.login(interaction)` prompt;
    # this port's `ApiKeyAuth` has no `login`/`check` hooks, so the credential is
    # stored directly. What is asserted -- that the provider's own `resolve` is
    # what produces the auth -- is unchanged.
    await runtime.login("extension-native", "secret")

    result = await runtime.get_auth("extension-native")
    assert result is not None
    assert result.auth.api_key == "secret"
    assert result.auth.base_url == "https://resolved.test/v1"
    assert result.source == "stored native key"


@pytest.mark.skip(
    reason="`fetchDeferred`/`cancelDeferred` and the provider-overlay plumbing around them have no ported "
    "counterpart (`core/model_runtime.py`: no extension providers, no deferred request layer)."
)
async def test_preserves_native_deferred_methods_through_provider_overlays() -> None: ...


async def test_applies_models_json_overrides_above_native_providers(tmp_path: Path) -> None:
    models_path = tmp_path / "models.json"
    models_path.write_text(
        json.dumps({"providers": {"extension-native": {"modelOverrides": {"native": {"contextWindow": 4242}}}}}),
        encoding="utf-8",
    )

    native_model = make_model("native", provider="extension-native", base_url="https://native.test/v1")

    async def resolve(*, credential: Credential | None = None, env: EnvLookup | None = None) -> AuthResult:
        return AuthResult(auth=ResolvedAuth(api_key="key"), source="native")

    runtime = await ModelRuntime.create(
        agent_dir=tmp_path / "agent",
        models_path=models_path,
        credentials=InMemoryCredentialStore(),
        providers=[
            Provider(
                id="extension-native",
                name="Extension Native",
                auth=ProviderAuth(api_key=ApiKeyAuth(name="Native key", resolve=resolve)),
                api=_unused,
                base_url="https://native.test/v1",
                models=[native_model],
            )
        ],
    )

    model = runtime.get_model("extension-native", "native")
    assert model is not None
    assert model.context_window == 4242
    # Composition must not mutate the provider's own model definition.
    assert native_model.context_window == 1000


@pytest.mark.skip(
    reason="`registerProvider(..., {refreshModels})` and `ModelsStore` persistence have no ported counterpart "
    "(`core/model_runtime.py`: no locked, revision-tracked ModelsStore; `refresh()` never hits the network)."
)
async def test_publishes_refresh_models_results_without_forcing_models_store_persistence() -> None: ...


@pytest.mark.skip(
    reason="The legacy OAuth `modifyModels` hook lives on `registerProvider`'s extension-provider config, "
    "which has no ported counterpart (`core/model_runtime.py`: no extension providers)."
)
async def test_applies_legacy_oauth_modify_models_after_async_credential_initialization() -> None: ...


# --------------------------------------------------------------------------
# Custom OpenAI-compatible provider from models.json, all the way to the wire.
#
# `models.json` is this port's stand-in for `registerProvider`, and it is the
# only way to add a provider the built-ins do not ship. The cases above stop at
# the composed `Model`; these carry the same configuration through to the
# outbound HTTP request, which is where a wrong `apiKey`, header or `compat`
# flag actually bites. `httpx.MockTransport` stands in for the endpoint -- no
# socket is opened and no provider is contacted.
# --------------------------------------------------------------------------


def custom_provider_config(
    *,
    api_key: str = "$MY_LLM_KEY",
    headers: dict[str, str] | None = None,
    provider_compat: dict[str, Any] | None = None,
    model_compat: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_definition: dict[str, Any] = {
        "id": "my-model",
        "name": "My Model",
        "contextWindow": 32_000,
        "maxTokens": 4_096,
    }
    if model_compat is not None:
        model_definition["compat"] = model_compat
    provider: dict[str, Any] = {
        "name": "My LLM",
        "baseUrl": "https://my-llm.test/v1",
        "api": "openai-completions",
        "apiKey": api_key,
        "models": [model_definition],
    }
    if headers is not None:
        provider["headers"] = headers
    if provider_compat is not None:
        provider["compat"] = provider_compat
    return {"providers": {"my-llm": provider}}


@dataclass
class _CapturedRequest:
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)


def install_mock_endpoint(monkeypatch: pytest.MonkeyPatch) -> _CapturedRequest:
    """Route every provider request into an in-process `httpx.MockTransport`.

    The real `httpx.AsyncClient` is still what issues the request -- only its
    transport is swapped -- so header casing, JSON encoding and the SSE decoder
    all behave exactly as they do against a live endpoint.
    """
    captured = _CapturedRequest()

    def handler(request: httpx.Request) -> httpx.Response:
        captured.url = str(request.url)
        captured.headers = dict(request.headers)
        captured.body = json.loads(request.content)
        return httpx.Response(
            200,
            content=(
                'data: {"choices":[{"delta":{"content":"hi"},"index":0}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"stop","index":0}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        pi_ai_http,
        "build_client",
        lambda _request: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return captured


async def stream_once(runtime: ModelRuntime, model: Model) -> AssistantMessage:
    context = Context(messages=[UserMessage(content=[TextContent(text="hello")])])
    stream = await runtime.stream_simple(model, context)
    async for _event in stream:
        pass
    return await stream.result()


async def runtime_for(tmp_path: Path, config: dict[str, Any]) -> ModelRuntime:
    models_path = tmp_path / "models.json"
    models_path.write_text(json.dumps(config), encoding="utf-8")
    return await ModelRuntime.create(
        agent_dir=tmp_path / "agent",
        models_path=models_path,
        credentials=InMemoryCredentialStore(),
    )


async def test_configured_api_key_and_headers_reach_the_outbound_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_LLM_KEY", "custom-secret")
    captured = install_mock_endpoint(monkeypatch)
    runtime = await runtime_for(tmp_path, custom_provider_config(headers={"X-Org": "acme"}))

    model = runtime.get_model("my-llm", "my-model")
    assert model is not None
    result = await stream_once(runtime, model)

    assert result.stop_reason == "stop", result.error_message
    assert captured.url == "https://my-llm.test/v1/chat/completions"
    assert captured.headers["authorization"] == "Bearer custom-secret"
    assert captured.headers["x-org"] == "acme"
    assert captured.body["model"] == "my-model"


async def test_env_references_are_interpolated_in_api_key_and_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_LLM_KEY", "env-secret")
    captured = install_mock_endpoint(monkeypatch)
    runtime = await runtime_for(
        tmp_path,
        custom_provider_config(api_key="$MY_LLM_KEY", headers={"X-Token": "prefix-${MY_LLM_KEY}"}),
    )

    # The provider keeps the *raw* reference; interpolation happens when auth
    # resolves, so rotating the variable does not need a recompose.
    provider = runtime.get_provider("my-llm")
    assert provider is not None
    assert (provider.headers or {}).get("X-Token") == "prefix-${MY_LLM_KEY}"

    model = runtime.get_model("my-llm", "my-model")
    assert model is not None
    await stream_once(runtime, model)

    assert captured.headers["authorization"] == "Bearer env-secret"
    assert captured.headers["x-token"] == "prefix-env-secret"


async def test_missing_env_api_key_leaves_the_custom_provider_unconfigured(tmp_path: Path) -> None:
    runtime = await runtime_for(tmp_path, custom_provider_config(api_key="$MY_LLM_MISSING_KEY"))

    assert runtime.get_model("my-llm", "my-model") is not None
    assert runtime.has_configured_auth("my-llm") is False
    assert not any(model.provider == "my-llm" for model in runtime.get_available_snapshot())


async def test_provider_level_compat_flags_reach_the_outbound_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_LLM_KEY", "custom-secret")
    captured = install_mock_endpoint(monkeypatch)
    runtime = await runtime_for(tmp_path, custom_provider_config(provider_compat={"maxTokensField": "max_tokens"}))

    model = runtime.get_model("my-llm", "my-model")
    assert model is not None
    assert model.compat == {"maxTokensField": "max_tokens"}

    await stream_once(runtime, model)

    assert "max_tokens" in captured.body
    assert "max_completion_tokens" not in captured.body


async def test_model_level_compat_overrides_provider_level_compat_on_the_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_LLM_KEY", "custom-secret")
    captured = install_mock_endpoint(monkeypatch)
    runtime = await runtime_for(
        tmp_path,
        custom_provider_config(
            provider_compat={"maxTokensField": "max_tokens", "supportsStore": False},
            model_compat={"maxTokensField": "max_completion_tokens"},
        ),
    )

    model = runtime.get_model("my-llm", "my-model")
    assert model is not None
    # The model block wins on the key it sets and inherits the rest.
    assert model.compat == {"maxTokensField": "max_completion_tokens", "supportsStore": False}

    await stream_once(runtime, model)

    assert "max_completion_tokens" in captured.body
    assert "max_tokens" not in captured.body


async def test_a_custom_model_without_compat_can_still_be_streamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: composing a custom model must not leave `compat` as `None`.

    `pi_ai.types.Model` declares `compat`/`sampling_params` as dicts with `{}`
    defaults and `pi_ai.api.openai_completions.build_params` reads
    `model.compat.get("openRouterRouting")` with no `None` guard, so a
    `models.json` model definition that omits `compat` used to fail every
    request with `'NoneType' object has no attribute 'get'`.
    """
    monkeypatch.setenv("MY_LLM_KEY", "custom-secret")
    install_mock_endpoint(monkeypatch)
    runtime = await runtime_for(tmp_path, custom_provider_config())

    model = runtime.get_model("my-llm", "my-model")
    assert model is not None
    assert model.compat == {}
    assert model.sampling_params == {}

    result = await stream_once(runtime, model)
    assert result.stop_reason == "stop", result.error_message


async def test_model_overrides_on_a_built_in_model_reach_the_outbound_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_DEEPSEEK_KEY", "deepseek-secret")
    captured = install_mock_endpoint(monkeypatch)
    runtime = await runtime_for(
        tmp_path,
        {
            "providers": {
                "deepseek": {
                    "baseUrl": "https://deepseek-proxy.test/v1",
                    "apiKey": "$MY_DEEPSEEK_KEY",
                    "headers": {"X-Proxy": "on"},
                    "modelOverrides": {
                        "deepseek-v4-pro": {
                            "contextWindow": 4242,
                            "maxTokens": 321,
                            "compat": {"maxTokensField": "max_tokens"},
                        }
                    },
                }
            }
        },
    )

    model = runtime.get_model("deepseek", "deepseek-v4-pro")
    assert model is not None
    assert model.context_window == 4242
    assert model.max_tokens == 321
    assert model.base_url == "https://deepseek-proxy.test/v1"
    assert model.compat["maxTokensField"] == "max_tokens"
    # The override must not disturb the provider's other built-in models, and
    # the provider-level baseUrl must still apply to them.
    other = runtime.get_model("deepseek", "deepseek-v4-flash")
    assert other is not None
    assert other.context_window != 4242
    assert other.base_url == "https://deepseek-proxy.test/v1"

    await stream_once(runtime, model)

    assert captured.url == "https://deepseek-proxy.test/v1/chat/completions"
    assert captured.headers["authorization"] == "Bearer deepseek-secret"
    assert captured.headers["x-proxy"] == "on"
    assert captured.body["model"] == "deepseek-v4-pro"
    assert "max_tokens" in captured.body
    assert "max_completion_tokens" not in captured.body
