"""Compose a runtime provider from a built-in base and a `models.json` overlay.

Python port of `packages/coding-agent/src/core/provider-composer.ts`, narrowed
to the API-key-only, no-extension subset that `pi_ai`'s ported registry can
support today:

- **OAuth composition is partial.** `compose_oauth_auth` below carries the base
  provider's `OAuthAuth` through composition and re-wraps `to_auth` with the
  configured headers/`authHeader`, matching `composeOAuthAuth`. What is *not*
  ported is `adaptOAuth`/`ExtensionOAuthConfig`: the extension layer can define
  a *new* OAuth method, and there is no extension layer here.
- **No extension layer** (`ProviderConfigInput`/`applyExtension`/
  `validateExtensionProvider`/`refreshModels`). The extension system itself is
  out of scope for this port; `compose_model_provider` here only composes the
  built-in provider with the `models.json` layer.
- **One API module per provider, not per model.** `pi_ai.registry.Provider`
  accepts either a single `api` module or a `{api name: module}` mapping, so
  per-model dispatch *is* supported: a `models.json` model may name any wire
  format `get_api_module` knows, and the composed provider carries a mapping
  whenever more than one is in play. What is still missing versus TypeScript
  is the pluggable `getApiProvider` registry itself -- `_API_MODULES` below is
  a fixed table of the api names this port's catalog uses.
- **Configured headers are resolved once, eagerly, at compose time**, not
  per-request. `pi_ai.registry.Provider.headers` is a plain static `dict`,
  not a per-request resolver function, so there is no seam to re-resolve
  templated header values (e.g. `$ROTATING_TOKEN`) on every call the way
  `resolveConfiguredModelHeaders`/`resolveCompatibilityRequestConfig` do in
  TypeScript. `resolve_compatibility_request_config` below is still provided,
  for parity and for any future direct caller, but nothing in this port wires
  it into a live request path yet, since `pi_ai`'s API modules do not expose
  an injectable per-request header/compat seam.
- API-key *resolution* (as opposed to headers) stays lazy: the composed
  `ApiKeyAuth.resolve` callable re-resolves the configured key (including
  `!command` execution and env look-ups) on every auth check/request, exactly
  as `composeApiKeyAuth` does in TypeScript.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import Any, Literal

from pi_ai.api import anthropic_messages, google_generative_ai, openai_completions, openai_responses
from pi_ai.auth.helpers import resolve_api_key_auth
from pi_ai.auth.types import (
    ApiKeyAuth,
    AuthResult,
    Credential,
    EnvLookup,
    OAuthAuth,
    ProviderAuth,
    ResolvedAuth,
)
from pi_ai.registry import Provider
from pi_ai.types import Model, ModelCost

from .model_config import ModelConfig, ModelsJsonModel, ModelsJsonModelOverride, ModelsJsonProvider
from .resolve_config_value import (
    clear_config_value_cache,
    get_config_value_env_var_names,
    is_command_config_value,
    is_config_value_configured,
    resolve_config_value_or_throw,
    resolve_headers_or_throw,
)

clear_api_key_cache = clear_config_value_cache

# Maps `Model.api` values used by this port's built-in provider catalog
# (`pi_ai.providers`) to the module implementing that API. There is no
# ported equivalent of `@earendil-works/pi-ai/compat`'s `getApiProvider`
# registry; this local mapping plays that narrow role for the api names the
# built-in catalog actually uses.
_API_MODULES: dict[str, Any] = {
    "openai-completions": openai_completions,
    "openai-responses": openai_responses,
    "anthropic-messages": anthropic_messages,
    "google-generative-ai": google_generative_ai,
}


def get_api_module(api_name: str) -> Any | None:
    return _API_MODULES.get(api_name)


AuthSource = Literal["stored", "runtime", "environment", "fallback", "models_json_key", "models_json_command"]


@dataclass
class AuthStatus:
    configured: bool
    source: AuthSource | None = None
    label: str | None = None


def merge_compat(base: dict[str, Any] | None, override: dict[str, Any] | None) -> dict[str, Any] | None:
    """Port of TS `mergeCompat`: returns `base` unchanged when there is no override.

    Callers that build a `Model` must normalize the result with `or {}`:
    TypeScript's `Model.compat` is optional and every read site uses optional
    chaining (`model.compat?.openRouterRouting`), while `pi_ai.types.Model`
    declares `compat: dict` with a `{}` default and its request builders read
    `model.compat.get(...)` with no `None` guard.
    """
    if not override:
        return base
    merged: dict[str, Any] = {**(base or {}), **override}
    for key in ("openRouterRouting", "vercelGatewayRouting", "chatTemplateKwargs", "chatTemplateArgs"):
        base_value = (base or {}).get(key)
        override_value = override.get(key)
        if isinstance(base_value, dict) or isinstance(override_value, dict):
            merged[key] = {**(base_value or {}), **(override_value or {})}
    return merged


def _merge_cost(base: ModelCost, override: dict[str, Any]) -> ModelCost:
    return ModelCost(
        input=override.get("input", base.input),
        output=override.get("output", base.output),
        cache_read=override.get("cacheRead", base.cache_read),
        cache_write=override.get("cacheWrite", base.cache_write),
        tiers=override.get("tiers", base.tiers),
    )


def apply_model_override(model: Model, override: ModelsJsonModelOverride) -> Model:
    return replace(
        model,
        name=override.name if override.name is not None else model.name,
        reasoning=override.reasoning if override.reasoning is not None else model.reasoning,
        thinking_level_map=(
            {**model.thinking_level_map, **override.thinking_level_map}
            if override.thinking_level_map
            else model.thinking_level_map
        ),
        input=override.input if override.input is not None else model.input,
        cost=_merge_cost(model.cost, override.cost) if override.cost else model.cost,
        context_window=override.context_window if override.context_window is not None else model.context_window,
        max_tokens=override.max_tokens if override.max_tokens is not None else model.max_tokens,
        sampling_params=(
            {**(model.sampling_params or {}), **override.sampling_params}
            if override.sampling_params
            else (model.sampling_params or {})
        ),
        compat=merge_compat(model.compat, override.compat) or {},
    )


def _model_from_json(
    provider_id: str,
    definition: ModelsJsonModel,
    provider_config: ModelsJsonProvider,
    defaults: Model | None,
    provider_api: str,
) -> Model:
    api = definition.api or provider_config.api or (defaults.api if defaults else None) or provider_api
    if not api:
        raise ValueError(
            f'Provider {provider_id}, model {definition.id}: no "api" specified. Set at provider or model level.'
        )
    base_url = definition.base_url or provider_config.base_url or (defaults.base_url if defaults else None)
    if not base_url:
        raise ValueError(f'Provider {provider_id}: "baseUrl" is required when defining custom models.')
    if definition.context_window is not None and definition.context_window <= 0:
        raise ValueError(f"Provider {provider_id}, model {definition.id}: invalid contextWindow")
    if definition.max_tokens is not None and definition.max_tokens <= 0:
        raise ValueError(f"Provider {provider_id}, model {definition.id}: invalid maxTokens")
    cost = definition.cost or {}
    return Model(
        id=definition.id,
        name=definition.name or definition.id,
        api=api,
        provider=provider_id,
        base_url=base_url,
        reasoning=bool(definition.reasoning),
        thinking_level_map=definition.thinking_level_map or {},
        input=definition.input or ["text"],
        cost=ModelCost(
            input=cost.get("input", 0),
            output=cost.get("output", 0),
            cache_read=cost.get("cacheRead", 0),
            cache_write=cost.get("cacheWrite", 0),
        ),
        context_window=definition.context_window or 128_000,
        max_tokens=definition.max_tokens or 16_384,
        sampling_params=definition.sampling_params or {},
        compat=merge_compat(provider_config.compat, definition.compat) or {},
    )


def apply_models_json(
    provider_id: str,
    base_models: list[Model],
    config: ModelsJsonProvider | None,
    provider_api: str,
) -> list[Model]:
    if config is None:
        return list(base_models)
    if config.oauth and not config.base_url:
        raise ValueError(f'Provider {provider_id}: "baseUrl" is required when "oauth" is set.')
    has_overrides = bool(config.model_overrides)
    if (
        not config.models
        and not config.base_url
        and not config.headers
        and not config.compat
        and not has_overrides
        and not config.api_key
        and not config.oauth
        and config.auth_header is None
    ):
        raise ValueError(
            f'Provider {provider_id}: must specify "baseUrl", "headers", "compat", "modelOverrides", or "models".'
        )

    models: list[Model] = [
        replace(
            model,
            base_url=model.base_url if config.oauth == "radius" else (config.base_url or model.base_url),
            compat=merge_compat(model.compat, config.compat) or {},
        )
        for model in base_models
    ]
    for definition in config.models:
        existing_index = next((i for i, m in enumerate(models) if m.id == definition.id), -1)
        defaults = models[existing_index] if existing_index >= 0 else (models[0] if models else None)
        model = _model_from_json(provider_id, definition, config, defaults, provider_api)
        if existing_index >= 0:
            models[existing_index] = model
        else:
            models.append(model)
    return models


def _configured_api_key(config: ModelsJsonProvider | None) -> str | None:
    return config.api_key if config else None


def _configured_headers(config: ModelsJsonProvider | None) -> dict[str, str] | None:
    return dict(config.headers) if config and config.headers else None


async def _config_context_env(
    values: list[str], env: EnvLookup | None, explicit: dict[str, str] | None = None
) -> dict[str, str] | None:
    resolved = dict(explicit or {})
    names: set[str] = set()
    for value in values:
        names.update(get_config_value_env_var_names(value))
    lookup = env if env is not None else (lambda _name: None)
    for name in names:
        if name in resolved:
            continue
        value = lookup(name)
        if inspect.isawaitable(value):
            value = await value
        if value is not None:
            resolved[name] = value
    return resolved if resolved else None


def _with_configured_auth(auth: ResolvedAuth, headers: dict[str, str] | None, auth_header: bool) -> ResolvedAuth:
    merged_headers: dict[str, str] | None = {**auth.headers, **(headers or {})} if (auth.headers or headers) else None
    if auth_header:
        if not auth.api_key:
            raise ValueError("authHeader requires a resolved API key")
        merged_headers = {**(merged_headers or {}), "Authorization": f"Bearer {auth.api_key}"}
    return replace(auth, headers=merged_headers or {})


def compose_api_key_auth(
    provider_id: str,
    base: Provider | None,
    config: ModelsJsonProvider | None,
) -> ApiKeyAuth | None:
    """Compose the API-key auth method for a provider (no OAuth/extension layer).

    Returns `None` for an OAuth-only provider, matching TypeScript's
    `composeApiKeyAuth` (`provider-composer.ts:310`): a provider that inherits
    no API-key method and configures no key, but does have OAuth, must not be
    given a fabricated "enter API key" login it cannot honor.
    """
    inherited = base.auth.api_key if base else None
    raw_key = _configured_api_key(config)
    if inherited is None and raw_key is None and compose_oauth_auth(provider_id, base, config) is not None:
        return None
    raw_headers = _configured_headers(config)
    auth_header = bool(config.auth_header) if config and config.auth_header is not None else False

    async def resolve(credential: Credential | None = None, env: EnvLookup | None = None) -> AuthResult | None:
        result: AuthResult | None
        if credential is not None:
            if inherited is not None:
                result = await resolve_api_key_auth(inherited, credential, env)
            elif credential.key:
                result = AuthResult(
                    auth=ResolvedAuth(api_key=credential.key), source="stored credential", env=dict(credential.env)
                )
            else:
                result = None
        elif raw_key is not None:
            key_env = await _config_context_env([raw_key], env)
            key = resolve_config_value_or_throw(raw_key, f'API key for provider "{provider_id}"', key_env)
            if inherited is not None:
                result = await resolve_api_key_auth(inherited, Credential(type="api_key", key=key), env)
            else:
                result = AuthResult(auth=ResolvedAuth(api_key=key), source="configured API key")
        elif inherited is not None:
            result = await resolve_api_key_auth(inherited, credential, env)
        else:
            result = None

        if result is None:
            return None

        explicit_env = {**(dict(credential.env) if credential is not None else {}), **(result.env or {})}
        header_env = await _config_context_env(list((raw_headers or {}).values()), env, explicit_env)
        headers = resolve_headers_or_throw(raw_headers, f'provider "{provider_id}"', header_env)
        return AuthResult(
            auth=_with_configured_auth(result.auth, headers, auth_header),
            source=result.source,
            env=result.env,
        )

    return ApiKeyAuth(name=(inherited.name if inherited else "API key"), resolve=resolve)


def compose_oauth_auth(
    provider_id: str,
    base: Provider | None,
    config: ModelsJsonProvider | None,
) -> OAuthAuth | None:
    """Carry the base provider's OAuth method through composition.

    Port of `composeOAuthAuth`. Without the extension layer the method itself
    always comes from the base provider; composition only re-wraps `to_auth`
    so configured headers and `authHeader` apply to OAuth requests too.
    """
    oauth = base.auth.oauth if base else None
    if oauth is None:
        return None
    raw_headers = _configured_headers(config)
    auth_header = bool(config.auth_header) if config and config.auth_header is not None else False

    async def to_auth(credential: Credential) -> ResolvedAuth:
        auth = await oauth.to_auth(credential)
        headers = resolve_headers_or_throw(raw_headers, f'provider "{provider_id}"', dict(credential.env))
        return _with_configured_auth(auth, headers, auth_header)

    return replace(oauth, to_auth=to_auth)


def _raw_model_headers(model: Model, config: ModelsJsonProvider | None) -> dict[str, str] | None:
    definition = next((m for m in (config.models if config else []) if m.id == model.id), None)
    override = config.model_overrides.get(model.id) if config else None
    headers: dict[str, str] = {
        **(override.headers if override and override.headers else {}),
        **(definition.headers if definition and definition.headers else {}),
    }
    return headers or None


def resolve_configured_model_headers(
    model: Model, config: ModelsJsonProvider | None, env: dict[str, str] | None = None
) -> dict[str, str] | None:
    return resolve_headers_or_throw(_raw_model_headers(model, config), f'model "{model.provider}/{model.id}"', env)


@dataclass
class CompatibilityRequestConfig:
    headers: dict[str, str] | None
    auth_header: bool


def resolve_compatibility_request_config(model: Model, config: ModelsJsonProvider | None) -> CompatibilityRequestConfig:
    configured = resolve_headers_or_throw(
        {**(_configured_headers(config) or {}), **(_raw_model_headers(model, config) or {})},
        f'model "{model.provider}/{model.id}"',
    )
    headers = {**(model.headers or {}), **configured} if (model.headers or configured) else None
    auth_header = bool(config.auth_header) if config and config.auth_header is not None else False
    return CompatibilityRequestConfig(headers=headers, auth_header=auth_header)


def configured_request_auth_status(config: ModelsJsonProvider | None) -> AuthStatus | None:
    value = _configured_api_key(config)
    if value is None:
        return None
    if is_command_config_value(value):
        return AuthStatus(configured=True, source="models_json_command")
    names = get_config_value_env_var_names(value)
    if names:
        if is_config_value_configured(value):
            return AuthStatus(configured=True, source="environment", label=", ".join(names))
        return AuthStatus(configured=False)
    return AuthStatus(configured=True, source="models_json_key")


def _base_api_modules(base: Provider) -> dict[str, Any]:
    """The `{api name: module}` map a base provider implements.

    `pi_ai.registry.Provider.api` is either a single module or, for providers
    that speak several wire formats (OpenCode, Cloudflare AI Gateway, GitHub
    Copilot, ...), a `{api name: module}` mapping. A single module is keyed by
    its name in `_API_MODULES`; a wrapped module that is not in that table
    (Cloudflare's streaming decorator, say) is keyed by every api name the
    provider's own models declare.
    """
    if isinstance(base.api, dict):
        return dict(base.api)
    names = {name for name, module in _API_MODULES.items() if module is base.api}
    if not names:
        names = {model.api for model in base.get_models() if model.api}
    return {name: base.api for name in names}


def compose_model_provider(provider_id: str, base: Provider | None, model_config: ModelConfig) -> Provider:
    """Compose a runtime `Provider` from a built-in base and its `models.json` overlay.

    Either `base` or a `models.json` entry for `provider_id` (or both) must be
    present; a `models.json`-only provider must supply enough to build a
    fully custom provider (`api`, `baseUrl`, and either `apiKey` or a base to
    inherit auth from).
    """
    config = model_config.get_provider(provider_id)

    # TypeScript validates the `models.json` overlay (via `applyModelsJson`)
    # before it needs an API module, so an `oauth` provider without `baseUrl`
    # reports the baseUrl error rather than the missing-api one.
    if config is not None and config.oauth and not config.base_url:
        raise ValueError(f'Provider {provider_id}: "baseUrl" is required when "oauth" is set.')

    api_modules: dict[str, Any] = _base_api_modules(base) if base is not None else {}
    default_api = (config.api if config else None) or (next(iter(api_modules)) if len(api_modules) == 1 else None)
    if base is None and not default_api and config is not None:
        # TypeScript's `modelFromJson` reads `definition.api ?? providerConfig.api`,
        # so a config-only provider may declare its api per model. When it declares
        # no models at all there is nothing to resolve an api for: TypeScript
        # composes such a provider with an empty model list rather than failing
        # (its `stream`/`streamSimple` look the api up per model at call time), and
        # `apply_models_json`/`_model_from_json` below still raise TypeScript's own
        # messages for a config that does declare models without an api.
        default_api = next((definition.api for definition in config.models if definition.api), None)
    if default_api and default_api not in api_modules:
        module = get_api_module(default_api)
        if module is None:
            raise ValueError(f"Provider {provider_id}: no API module registered for api: {default_api}")
        api_modules[default_api] = module

    models = apply_models_json(provider_id, base.get_models() if base else [], config, default_api or "")
    for model_id, override in (config.model_overrides if config else {}).items():
        for i, model in enumerate(models):
            if model.id == model_id:
                models[i] = apply_model_override(model, override)

    # A model may name a wire format the base provider does not implement;
    # TypeScript falls back to the global `getApiProvider` registry for it.
    for model in models:
        if model.api in api_modules:
            continue
        module = get_api_module(model.api)
        if module is None:
            raise ValueError(f"Provider {provider_id}: no API module registered for api: {model.api}")
        api_modules[model.api] = module

    if not api_modules:
        provider_api: Any = base.api if base is not None else None
    elif len(api_modules) == 1:
        provider_api = next(iter(api_modules.values()))
    else:
        provider_api = api_modules

    api_key = compose_api_key_auth(provider_id, base, config)
    oauth = compose_oauth_auth(provider_id, base, config)
    if api_key is None and oauth is None:
        raise ValueError(f"Provider {provider_id}: no authentication method configured.")
    auth = ProviderAuth(api_key=api_key, oauth=oauth)

    name = (config.name if config else None) or (base.name if base else None) or provider_id
    base_url = (config.base_url if config else None) or (base.base_url if base else "")
    headers = {**(base.headers if base else {}), **(_configured_headers(config) or {})}

    return Provider(
        id=provider_id,
        name=name,
        auth=auth,
        api=provider_api,
        base_url=base_url,
        headers=headers,
        models=models,
    )


__all__ = [
    "AuthStatus",
    "CompatibilityRequestConfig",
    "apply_model_override",
    "apply_models_json",
    "clear_api_key_cache",
    "compose_api_key_auth",
    "compose_model_provider",
    "compose_oauth_auth",
    "configured_request_auth_status",
    "get_api_module",
    "merge_compat",
    "resolve_compatibility_request_config",
    "resolve_configured_model_headers",
]
