"""A configured `pi_ai` `Models` collection for the coding agent and its tests.

Python port of the credential-blind, file-backed subset of
`packages/coding-agent/src/core/model-runtime.ts`'s `ModelRuntime` class. The
full TypeScript `ModelRuntime` composes four layers this port cannot follow
faithfully without inventing large amounts of currently-unported `pi_ai`
infrastructure (owned by a concurrent porting session):

- **Partial OAuth.** `login_oauth` runs a provider's ported OAuth flow and
  persists the credential, and `is_using_oauth`/`get_provider_auth_status`
  report it. What is *not* ported is `runtime-credentials.ts`'s
  OAuth-vs-api-key request dispatch and `composeOAuthAuth`, so a stored OAuth
  token is not yet used to sign outgoing provider requests.
- **No remote model catalog refresh** (`remote-catalog-provider.ts`,
  `withRemoteCatalog`, the `radius` gateway provider, `PI_OFFLINE`/network
  timeouts). `refresh()` only rebuilds the *local* composition (builtin
  providers + `models.json` overlay); it never makes a network call. Static,
  hand-written catalogs from `pi_ai.providers.all_providers()` are the only
  model source.
- **No locked, revision-tracked `ModelsStore`** (`models-store.ts`'s
  `FileModelsStore`/`InMemoryCodingAgentModelsStore`, cross-process file
  locking, `getFileRevision`). Custom/extension-added models are not
  persisted between processes in this port; only `models.json` (loaded once,
  read-only) contributes model definitions beyond the builtin catalog.
- **No extension providers** (`ProviderConfigInput`, `validateExtensionProvider`,
  `nativeExtensionProviders`/`extensionProviders` maps). Only builtin +
  `models.json` composition, per `provider_composer.py`'s own documented
  boundary.
- **No credential-synchronization retry/error class**
  (`CredentialSynchronizationError`, `credentialOperations` de-duplication
  map, provider-scoped `refresh({allowNetwork, providers, signal})` and its
  `{aborted}` result). `login`/`logout` do serialize same-provider calls
  against each other (`_credential_lock`, a plain per-provider `asyncio.Lock`)
  so a login and a logout fired close together can't interleave their store
  writes, but there is no network refresh for a credential operation to queue
  behind, no cancellation, and failures propagate as whatever the credential
  store raises rather than a typed `CredentialSynchronizationError`.

What *is* ported faithfully: composing every builtin provider (and any
`models.json`-only custom provider) through `compose_model_provider`,
API-key credential persistence to a JSON file (`FileCredentialStore`, a
narrow stand-in for the full `auth-storage.ts`), a per-provider ordering lock
for `login`/`logout`, and a synchronous "is this provider configured"
snapshot (`has_configured_auth`/`get_available_snapshot`) satisfying
`pi_coding_agent.core.model_resolver`'s `ModelSource` protocol, so a
`ModelRuntime` can be used anywhere a `ModelSource` is expected.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pi_ai.auth.resolve import resolve_provider_auth
from pi_ai.auth.types import AuthInteraction, Credential, CredentialInfo, CredentialStore, EnvLookup
from pi_ai.providers import all_providers
from pi_ai.providers.radius import radius_provider
from pi_ai.registry import AuthCheck, AuthResult, Context, Model, Models, Provider, SimpleStreamOptions
from pi_ai.utils.event_stream import AssistantMessageEventStream

from .config import get_agent_dir, get_auth_path, get_models_path
from .model_config import ModelConfig
from .provider_attribution import RADIUS_PROVIDER_ID
from .provider_composer import (
    compose_model_provider,
    configured_request_auth_status,
    resolve_configured_model_headers,
)
from .runtime_credentials import RuntimeCredentials


def _merge_headers(base: dict[str, str] | None, override: dict[str, str] | None) -> dict[str, str]:
    """Case-insensitive header merge; port of `model-runtime.ts`'s `mergeHeaders`."""
    merged = dict(base or {})
    for name, value in (override or {}).items():
        lower = name.lower()
        for existing in [key for key in merged if key.lower() == lower]:
            del merged[existing]
        merged[name] = value
    return merged


async def list_stored_credentials(store: CredentialStore) -> list[CredentialInfo]:
    """Enumerate a credential store's providers and their credential types.

    TS `ModelRuntime` calls `this.credentials.list(...)` unconditionally, so
    this does too: every `CredentialStore` implementation must provide it.
    """
    return list(await store.list())


@dataclass
class ModelRuntimeSnapshot:
    all: list[Model] = field(default_factory=list)
    available: list[Model] = field(default_factory=list)
    configured_providers: set[str] = field(default_factory=set)


class FileCredentialStore(CredentialStore):
    """A narrow, file-backed `CredentialStore`.

    Stand-in for `packages/coding-agent/src/core/auth-storage.ts`'s
    `FileAuthStorageBackend`: persists
    `{providerId: {"type": ..., "key": ..., "env": ..., "data": ...}}` to a
    single JSON file, so both API keys and OAuth tokens round-trip. Unlike the
    TypeScript original there is no cross-process file locking or revision
    tracking, and all I/O is synchronous (acceptable for a single-process
    CLI/test use, but a documented narrowing versus `auth-storage.ts`).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def _read_all(self) -> dict[str, dict[str, Any]]:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_all(self, data: dict[str, dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    async def get(self, provider_id: str) -> Credential | None:
        entry = self._read_all().get(provider_id)
        if entry is None:
            return None
        return Credential(
            type=entry.get("type", "api_key"),
            key=entry.get("key"),
            env=entry.get("env") or {},
            data=entry.get("data") or {},
        )

    async def set(self, provider_id: str, credential: Credential) -> None:
        data = self._read_all()
        data[provider_id] = {
            "type": credential.type,
            "key": credential.key,
            "env": credential.env,
            "data": credential.data,
        }
        self._write_all(data)

    async def delete(self, provider_id: str) -> None:
        data = self._read_all()
        if provider_id in data:
            del data[provider_id]
            self._write_all(data)

    def has_sync(self, provider_id: str) -> bool:
        """Synchronous existence check, used by `has_configured_auth`."""
        return provider_id in self._read_all()

    async def list(self) -> list[CredentialInfo]:
        return [
            CredentialInfo(provider_id=pid, type=entry.get("type", "api_key"))
            for pid, entry in self._read_all().items()
        ]

    def get_sync(self, provider_id: str) -> Credential | None:
        """Synchronous read, used by the login/logout selectors on every render."""
        entry = self._read_all().get(provider_id)
        if entry is None:
            return None
        return Credential(
            type=entry.get("type", "api_key"),
            key=entry.get("key"),
            env=entry.get("env") or {},
            data=entry.get("data") or {},
        )


class ModelRuntime:
    """A configured `pi_ai.registry.Models`, composed from builtins + `models.json`."""

    def __init__(
        self,
        models: Models,
        config: ModelConfig,
        credentials: CredentialStore,
        builtins: list[Provider],
        models_path: str | None = None,
    ) -> None:
        self.models = models
        self._config = config
        self.credentials = credentials
        self._builtins = list(builtins)
        # Path `refresh()` re-reads so an edited `models.json` takes effect
        # without restarting (TS `refresh()` starts with `ModelConfig.load`).
        self._models_path = models_path
        # Env var names each builtin provider accepts, captured before
        # `models.json` composition replaces `ApiKeyAuth.resolve` with a
        # closure that no longer exposes `env_vars` -- needed for the
        # synchronous `has_configured_auth` fallback below.
        self._builtin_env_vars: dict[str, tuple[str, ...]] = {
            provider.id: tuple(provider.auth.api_key.env_vars) for provider in builtins if provider.auth.api_key
        }
        self._composition_errors: dict[str, str] = {}
        self._availability_error: str | None = None
        # Monotonic counter behind `get_available`'s staleness guard, the port
        # of TS `availabilityErrorSeq`: overlapping availability passes must
        # not let a superseded one write over newer error state.
        self._availability_seq = 0
        # Env lookup used for auth resolutions this class performs itself
        # (see `get_auth`'s `min_oauth_validity_ms` path); `Models` keeps its
        # own copy but does not expose it.
        self._env_lookup: EnvLookup = os.environ.get
        # Per-provider ordering for `login`/`logout`, the local (non-network)
        # slice of TS `model-runtime.ts`'s credential-operation queue: two
        # calls for the *same* provider run one after the other in call
        # order, while different providers stay fully concurrent. This port
        # has no per-provider network refresh or `CredentialSynchronizationError`
        # (see the module docstring), but ordering same-provider credential
        # writes is cheap and closes a real race (a login and a logout fired
        # close together could otherwise interleave their store writes).
        self._credential_locks: dict[str, asyncio.Lock] = {}

    def _credential_lock(self, provider_id: str) -> asyncio.Lock:
        lock = self._credential_locks.get(provider_id)
        if lock is None:
            lock = asyncio.Lock()
            self._credential_locks[provider_id] = lock
        return lock

    def get_error(self) -> str | None:
        """Startup configuration errors, or `None`. Port of TS ``getError``.

        TypeScript also appends the last availability-refresh failure. This
        port has no background refresh queue, so `get_available` records the
        failure directly, but it carries TypeScript's sequence counter so a
        superseded pass cannot overwrite newer state; the observable contract
        (an availability failure shows up here, and a later successful refresh
        clears it) is the same.
        """
        errors: list[str] = []
        config_error = self._config.get_error()
        if config_error:
            errors.append(config_error)
        for provider_id, error in self._composition_errors.items():
            errors.append(f'Provider "{provider_id}": {error}')
        if self._availability_error:
            errors.append(f"Availability refresh: {self._availability_error}")
        return "\n\n".join(errors) if errors else None

    @classmethod
    async def create(
        cls,
        *,
        agent_dir: str | Path | None = None,
        auth_path: str | Path | None = None,
        models_path: str | Path | None = None,
        credentials: CredentialStore | None = None,
        providers: list[Provider] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ModelRuntime:
        """Build a `ModelRuntime`. All paths accept overrides so tests never touch `$HOME`."""
        resolved_agent_dir = str(agent_dir) if agent_dir is not None else get_agent_dir()
        resolved_models_path = str(models_path) if models_path is not None else get_models_path(resolved_agent_dir)
        resolved_auth_path = str(auth_path) if auth_path is not None else get_auth_path(resolved_agent_dir)

        config = ModelConfig.load(resolved_models_path)
        base_providers = providers if providers is not None else all_providers()

        # Wrapped so `--api-key` and unsaved logins authenticate this process
        # without ever writing to auth.json.
        credential_store = RuntimeCredentials(credentials or FileCredentialStore(resolved_auth_path))

        if providers is None:
            # Port of TS `model-runtime.ts`'s `refreshModels` "legacy credential
            # catalog" path for the built-in Radius provider (`radius.ts`): a
            # stored OAuth credential caches the gateway's last `/v1/config`
            # response (`credential.data["gatewayConfig"]`), so the models it
            # published are available immediately, with no network call. This
            # is the only Radius catalog behavior this port implements --
            # `refresh_radius_models`'s network fetch and `ModelsStore`
            # persistence are out of scope (see the module docstring).
            try:
                radius_credential = await credential_store.get(RADIUS_PROVIDER_ID)
            except Exception:
                # A malformed or unreadable auth.json must not stop the runtime
                # from being constructed. TypeScript never reads auth.json in
                # `create()` -- `configureRadiusProviders` builds from
                # models.json alone -- so a broken credential file surfaces
                # later, through `check_auth`/`get_auth`, which callers already
                # wrap (see `cli/auth_command.py`'s `invalid_state` path).
                radius_credential = None
            if radius_credential is not None:
                base_providers = [
                    radius_provider(credential=radius_credential) if provider.id == RADIUS_PROVIDER_ID else provider
                    for provider in base_providers
                ]
            base_providers = await cls._configure_radius_providers(base_providers, config, credential_store)

        composed, composition_errors = cls._compose_all(base_providers, config)
        env_lookup = (lambda name: env.get(name)) if env is not None else os.environ.get
        models = Models(providers=composed, credential_store=credential_store, env=env_lookup)
        runtime = cls(models, config, credential_store, base_providers, resolved_models_path)
        runtime._composition_errors = composition_errors
        runtime._env_lookup = env_lookup
        return runtime

    @staticmethod
    async def _configure_radius_providers(
        base_providers: list[Provider], config: ModelConfig, credentials: CredentialStore
    ) -> list[Provider]:
        """Port of TS `model-runtime.ts`'s `configureRadiusProviders`.

        A `models.json` provider declaring `"oauth": "radius"` with a `baseUrl`
        is a Radius *gateway*, not a plain config provider: TypeScript promotes
        it to a full `radiusProvider({id, name, gateway})` base, so it gets the
        `pi-messages` api, the Radius OAuth flow, and the gateway's catalog.
        Without this a custom gateway composes as an api-less config provider
        with no models and no login.
        """
        gateways: list[Provider] = []
        for provider_id in config.get_provider_ids():
            provider_config = config.get_provider(provider_id)
            if provider_config is None or provider_config.oauth != "radius" or not provider_config.base_url:
                continue
            try:
                credential = await credentials.get(provider_id)
            except Exception:
                credential = None
            gateways.append(
                radius_provider(
                    id=provider_id,
                    name=provider_config.name or provider_id,
                    gateway=re.sub(r"/v1/?$", "", provider_config.base_url),
                    credential=credential,
                )
            )
        if not gateways:
            return base_providers
        replaced = {provider.id for provider in gateways}
        return [provider for provider in base_providers if provider.id not in replaced] + gateways

    @staticmethod
    def _compose_all(base_providers: list[Provider], config: ModelConfig) -> tuple[list[Provider], dict[str, str]]:
        composed: list[Provider] = []
        seen: set[str] = set()
        errors: dict[str, str] = {}
        for provider in base_providers:
            seen.add(provider.id)
            if config.get_provider(provider.id) is not None:
                try:
                    composed.append(compose_model_provider(provider.id, provider, config))
                except ValueError as error:
                    # TypeScript records composition failures per provider and
                    # keeps the builtin, surfacing the message through
                    # `getError()`; it never fails runtime construction.
                    errors[provider.id] = str(error)
                    composed.append(provider)
            else:
                composed.append(provider)
        for provider_id in config.get_provider_ids():
            if provider_id in seen:
                continue
            try:
                composed.append(compose_model_provider(provider_id, None, config))
            except ValueError as error:
                errors[provider_id] = str(error)
        return composed, errors

    # -- Models delegation -------------------------------------------------

    def get_providers(self) -> list[Provider]:
        return self.models.get_providers()

    def get_provider(self, provider_id: str) -> Provider | None:
        return self.models.get_provider(provider_id)

    def get_models(self, provider_id: str | None = None) -> list[Model]:
        return self.models.get_models(provider_id)

    def get_model(self, provider_id: str, model_id: str) -> Model | None:
        return self.models.get_model(provider_id, model_id)

    def find_model(self, reference: str) -> Model | None:
        return self.models.find_model(reference)

    async def get_auth(self, target: str | Model, *, min_oauth_validity_ms: int | None = None) -> AuthResult | None:
        if min_oauth_validity_ms is None:
            result = await self.models.get_auth(target)
        else:
            result = await self._resolve_auth_with_min_validity(target, min_oauth_validity_ms)
        if isinstance(target, str) or result is None:
            return result
        # Port of TS `ModelRuntime.getAuth`: models.json `models[].headers` and
        # `modelOverrides[id].headers` are per-model, so they are not part of the
        # provider-level auth composition and must be merged in here.
        configured = resolve_configured_model_headers(
            target, self._config.get_provider(target.provider), dict(result.env or {})
        )
        if not configured:
            return result
        return replace(result, auth=replace(result.auth, headers=_merge_headers(result.auth.headers, configured)))

    async def _resolve_auth_with_min_validity(
        self, target: str | Model, min_oauth_validity_ms: int
    ) -> AuthResult | None:
        """`Models.get_auth` has no `minOAuthValidityMs` override, so resolve directly."""
        provider_id = target if isinstance(target, str) else target.provider
        provider = self.models.get_provider(provider_id)
        if provider is None:
            return None
        result = await resolve_provider_auth(
            provider_id,
            provider.auth,
            self.credentials,
            self._env_lookup,
            min_oauth_validity_ms=min_oauth_validity_ms,
        )
        if result is None or isinstance(target, str) or not target.headers:
            return result
        return replace(result, auth=replace(result.auth, headers=_merge_headers(result.auth.headers, target.headers)))

    async def list_credentials(self) -> list[CredentialInfo]:
        """Provider ids with a stored (or runtime) credential, with their type."""
        return await list_stored_credentials(self.credentials)

    async def check_auth(self, provider_id: str) -> AuthCheck | None:
        return await self.models.check_auth(provider_id)

    async def get_available(self, provider_id: str | None = None) -> list[Model]:
        # Every pass takes a sequence number and only writes availability error
        # state while it is still the newest one, so a slow refresh that is
        # superseded mid-flight cannot clobber a newer result (TS
        # `availabilityErrorSeq`, issue #7301).
        self._availability_seq += 1
        error_seq = self._availability_seq
        try:
            available = await self.models.get_available(provider_id)
        except Exception as error:
            if error_seq == self._availability_seq:
                self._availability_error = str(error)
            raise
        if error_seq == self._availability_seq:
            self._availability_error = None
        return available

    async def login(
        self, provider_id: str, api_key: str | None = None, *, interaction: AuthInteraction | None = None
    ) -> Credential:
        """Store an api-key credential, or run the provider's own login flow.

        With `interaction` the provider's `auth.api_key.login(interaction)`
        runs instead of storing `api_key` directly, mirroring TS
        `runtime.login(id, "api_key", {prompt, notify})` for providers whose
        credential is not a single secret. Serialized per `provider_id`
        against concurrent `login`/`logout` calls for the same provider (see
        `_credential_lock`).
        """
        async with self._credential_lock(provider_id):
            return await self.models.login(provider_id, api_key, interaction=interaction)

    async def login_oauth(self, provider_id: str, interaction: AuthInteraction) -> Credential:
        """Run a provider's OAuth flow and persist the resulting credential."""
        async with self._credential_lock(provider_id):
            return await self.models.login_oauth(provider_id, interaction)

    async def logout(self, provider_id: str) -> None:
        async with self._credential_lock(provider_id):
            await self.models.logout(provider_id)

    async def set_runtime_api_key(self, provider_id: str, api_key: str) -> None:
        """Authenticate `provider_id` for this process only.

        Backs ``--api-key``: the key must work immediately but must never
        reach ``auth.json``, so it goes into the `RuntimeCredentials` overlay
        rather than through `login`.

        TypeScript enqueues this on the same per-provider credential queue as
        `login`/`logout` (`enqueueCredentialOperation`), so it takes the same
        lock here.
        """
        async with self._credential_lock(provider_id):
            self.credentials.set_runtime_api_key(provider_id, api_key)
            await self.get_available(provider_id)

    async def remove_runtime_api_key(self, provider_id: str) -> None:
        async with self._credential_lock(provider_id):
            self.credentials.remove_runtime_api_key(provider_id)
            await self.get_available(provider_id)

    async def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream:
        return await self.models.stream_simple(model, context, options, **kwargs)

    # -- Additional, coding-agent-specific surface --------------------------

    def is_using_oauth(self, provider_id: str) -> bool:
        """Whether the stored credential for `provider_id` is an OAuth token."""
        get_sync = getattr(self.credentials, "get_sync", None)
        if get_sync is None:
            return False
        stored = get_sync(provider_id)
        return stored is not None and stored.type == "oauth"

    def get_provider_auth_status(self, provider_id: str) -> AuthCheck:
        """Synchronous auth summary for the login/logout selectors.

        Port of TS `getProviderAuthStatus`. The async `check_auth` resolves
        command-based credentials; this is the cheap variant the selectors use
        on every render.
        """
        get_sync = getattr(self.credentials, "get_sync", None)
        stored = get_sync(provider_id) if get_sync is not None else None
        if stored is not None:
            return AuthCheck(
                configured=True,
                type=stored.type,
                source="OAuth" if stored.type == "oauth" else "stored credential",
            )
        # A `models.json` `apiKey` short-circuits the builtin env-var fallback,
        # exactly as TypeScript's `configuredRequestAuthStatus` step does: an
        # explicit `$MISSING_VAR` reference keeps the provider unconfigured even
        # when the builtin provider's own env var happens to be set.
        configured = configured_request_auth_status(self._config.get_provider(provider_id))
        if configured is not None:
            if not configured.configured:
                return AuthCheck(configured=False)
            return AuthCheck(configured=True, type="api_key", source=configured.label or configured.source)
        for name in self._builtin_env_vars.get(provider_id, ()):
            if os.environ.get(name):
                return AuthCheck(configured=True, type="api_key", source=name)
        return AuthCheck(configured=False)

    def is_using_subscription(self, provider_id: str) -> bool:
        """Port of TS `isUsingSubscription`.

        TypeScript requires both an active OAuth credential and a provider
        whose OAuth method is flagged as a subscription; the footer calls this
        on every render, so it stays synchronous and reads the credential
        store's cached snapshot.
        """
        if not self.is_using_oauth(provider_id):
            return False
        provider = self.models.get_provider(provider_id)
        oauth = getattr(getattr(provider, "auth", None), "oauth", None)
        return getattr(oauth, "is_subscription", False) is True

    def has_configured_auth(self, provider_id: str) -> bool:
        """Synchronous best-effort "is this provider configured" check.

        Checks, in order: a `models.json`-configured API key (env var or
        literal/command reference), a stored credential file entry, and the
        provider's original (pre-composition) env vars. This mirrors
        `pi_coding_agent.core.model_resolver.ModelsAuthSource`'s synchronous
        heuristic (an async `Models.check_auth` credential-store lookup is
        also available via `check_auth` for callers that can await).
        """
        provider_config = self._config.get_provider(provider_id)
        status = configured_request_auth_status(provider_config)
        if status is not None and status.configured:
            return True
        # A configured key that is *not* satisfied (say `"apiKey": "${MY_KEY}"`
        # with `MY_KEY` unset) must not veto the checks below: TypeScript's
        # `hasConfiguredAuth` reads `snapshot.configuredProviders`, which is
        # built from the async `getAvailable()` path and so counts a stored
        # credential. Returning `status.configured` here instead hid a
        # logged-in provider from `get_available_snapshot()` even though
        # `get_auth()`/`get_provider_auth_status()` both resolved it.
        # Capability check, not `isinstance`: the store is wrapped in a
        # `RuntimeCredentials` overlay, so any wrapper exposing `has_sync`
        # must be honoured or `--api-key` providers look unconfigured.
        has_sync = getattr(self.credentials, "has_sync", None)
        if has_sync is not None and has_sync(provider_id):
            return True
        return any(os.environ.get(name) for name in self._builtin_env_vars.get(provider_id, ()))

    def get_available_snapshot(self) -> list[Model]:
        """Synchronous snapshot of models whose providers look configured."""
        return [model for model in self.get_models() if self.has_configured_auth(model.provider)]

    def get_snapshot(self) -> ModelRuntimeSnapshot:
        available = self.get_available_snapshot()
        return ModelRuntimeSnapshot(
            all=self.get_models(),
            available=available,
            configured_providers={model.provider for model in available},
        )

    def refresh(self) -> None:
        """Reload `models.json` and rebuild the provider composition from it.

        Local-only: no network catalog fetch (see module docstring). TypeScript's
        `refresh()` starts with `this.config = await ModelConfig.load(this.modelsPath)`,
        so opening `/model` picks up an edited `models.json` (issue #6999); the
        rebuild also has to *drop* providers the edited file no longer declares,
        which is why the whole `Models` collection is rebuilt rather than
        re-adding over the old one.
        """
        self._config = ModelConfig.load(self._models_path) if self._models_path else self._config
        composed, self._composition_errors = self._compose_all(self._builtins, self._config)
        self.models = Models(providers=composed, credential_store=self.credentials, env=self._env_lookup)


__all__ = ["FileCredentialStore", "ModelRuntime", "ModelRuntimeSnapshot"]
