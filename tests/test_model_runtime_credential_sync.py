"""Python port of `packages/coding-agent/test/model-runtime-credential-sync.test.ts`.

The TypeScript file pins ten cases.

- Case 1 ("publishes locally consistent availability before login and logout
  resolve") ports directly.
- Cases 2 and 3 (same-provider ordering, cross-provider concurrency) port on
  top of two additions made while auditing this file: `ModelRuntime.login`
  now accepts an `interaction: AuthInteraction` (mirroring TS
  `runtime.login(id, "api_key", {prompt, notify})`, which runs the provider's
  own api-key login flow instead of storing a literal key), and
  `ModelRuntime` now serializes `login`/`logout` per provider id through a
  plain `asyncio.Lock` (`_credential_lock` in `core/model_runtime.py`) so
  same-provider calls run in call order while different providers stay
  concurrent. Neither addition builds the network refresh queue TS actually
  serializes against; see `core/model_runtime.py`'s docstring.
- Cases 4 through 10 pin `ModelRuntime`'s *network* credential-synchronization
  layer, which this port does not have: there is no provider-scoped
  `refresh({allowNetwork, providers, signal})`, no network catalog refresh,
  and no `CredentialSynchronizationError`. Each is skipped individually below
  with the specific missing machinery named.

`registerNativeProvider` is likewise unported; `ModelRuntime.create(providers=
[...])` is this port's equivalent and is what the ported cases use.

`TestCustomOpenAiCompatibleProviderCredentials` at the bottom is *extra*
coverage, not a port of a TypeScript case: neither this file nor any other
TypeScript test exercises the credential path of a provider declared in
`models.json` with `baseUrl` + `api: "openai-completions"` + `apiKey` +
`headers`, even though that is the shape users reach for when pointing the
agent at a self-hosted OpenAI-compatible endpoint. The behaviour it pins was
verified against `src/core/provider-composer.ts` (`composeApiKeyAuth` and
`configuredRequestAuthStatus`) rather than against a TypeScript test.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pi_ai.auth.types import AuthInteraction, AuthPrompt, Credential
from pi_ai.models import Provider
from pi_ai.providers.openai_compatible import openai_compatible_provider
from pi_ai.types import Model, ModelCost
from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.model_runtime import ModelRuntime


def _provider(provider_id: str) -> Provider:
    """Port of the TS `provider()` helper: one model, api-key auth only."""
    return openai_compatible_provider(
        provider_id=provider_id,
        name=provider_id,
        base_url="https://example.test/v1",
        env_vars=[],
        models=[
            Model(
                id="dynamic",
                name="Dynamic",
                api="openai-completions",
                reasoning=False,
                input=["text"],
                cost=ModelCost(input=0, output=0, cache_read=0, cache_write=0),
                context_window=1000,
                max_tokens=100,
            )
        ],
    )


async def _runtime_with_provider(provider: Provider, credentials: object) -> ModelRuntime:
    """Port of the TS `runtimeWithProvider` helper.

    TypeScript registers the provider with `registerNativeProvider` and then
    awaits a provider-scoped `refresh({ allowNetwork: false })`; this port
    composes providers at construction time, so passing `providers=[...]` to
    `create` is the whole equivalent.
    """
    return await ModelRuntime.create(credentials=credentials, models_path=None, providers=[provider])


async def test_publishes_locally_consistent_availability_before_login_and_logout_resolve() -> None:
    credentials = AuthStorage.in_memory()
    runtime = await _runtime_with_provider(_provider("dynamic"), credentials)

    await runtime.login("dynamic", "dynamic-key")
    assert runtime.has_configured_auth("dynamic") is True
    assert "dynamic" in [entry.id for entry in runtime.get_available_snapshot()]
    assert await credentials.get("dynamic") == Credential(type="api_key", key="dynamic-key")

    await runtime.logout("dynamic")
    assert runtime.has_configured_auth("dynamic") is False
    assert not any(entry.provider == "dynamic" for entry in runtime.get_available_snapshot())
    assert await credentials.get("dynamic") is None


@dataclass
class _BlockingInteraction(AuthInteraction):
    """An `AuthInteraction` whose `prompt()` blocks until released.

    Stands in for the TS helper's custom provider `login: async () => { ...;
    await blockedLogin; return credential; }` callbacks in cases 2 and 3:
    this port's `openai_compatible_provider` api-key auth always logs in
    through `interaction.prompt()` (see `pi_ai.auth.helpers.env_api_key_auth`),
    so blocking `prompt()` blocks the login the same way.
    """

    started: asyncio.Event
    finish: asyncio.Event
    key: str
    signal: AbortSignal = field(default_factory=AbortSignal)

    async def prompt(self, prompt: AuthPrompt) -> str:
        self.started.set()
        await self.finish.wait()
        return self.key

    def notify(self, event: object) -> None:
        pass


async def test_orders_same_provider_credential_operations_through_local_synchronization() -> None:
    """Port of TS "orders same-provider credential operations through local
    synchronization".

    Only the local half of TS's per-provider operation chain is ported (see
    the module docstring): a `login` blocked inside its interaction `prompt()`
    holds `_credential_lock("ordered")`, so a concurrent `logout` for the same
    provider must not run -- and must not touch the store -- until the login
    finishes.
    """
    login_started = asyncio.Event()
    finish_login = asyncio.Event()
    interaction = _BlockingInteraction(login_started, finish_login, "ordered-key")

    credentials = AuthStorage.in_memory()
    runtime = await _runtime_with_provider(_provider("ordered"), credentials)

    login = asyncio.ensure_future(runtime.login("ordered", interaction=interaction))
    await login_started.wait()
    logout = asyncio.ensure_future(runtime.logout("ordered"))
    # TS yields one macrotask (`setTimeout(resolve, 0)`), which drains every
    # pending microtask; a single `asyncio.sleep(0)` only yields once, so the
    # loop below is what actually gives the queued logout every chance to run.
    for _ in range(10):
        await asyncio.sleep(0)
    # The queued logout has not run yet: the store is untouched by either call.
    assert await credentials.get("ordered") is None
    assert not logout.done()

    finish_login.set()
    await asyncio.gather(login, logout)
    assert await credentials.get("ordered") is None
    assert runtime.has_configured_auth("ordered") is False


async def test_allows_different_providers_to_run_credential_flows_concurrently() -> None:
    """Port of TS "allows different providers to run credential flows
    concurrently": two different providers' blocked logins must both reach
    their `prompt()` before either is released, proving `_credential_lock`
    only serializes same-provider calls.
    """
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    finish = asyncio.Event()
    interaction_one = _BlockingInteraction(first_started, finish, "one-key")
    interaction_two = _BlockingInteraction(second_started, finish, "two-key")

    runtime = await ModelRuntime.create(
        credentials=AuthStorage.in_memory(),
        models_path=None,
        providers=[_provider("one"), _provider("two")],
    )

    login_one = asyncio.ensure_future(runtime.login("one", interaction=interaction_one))
    login_two = asyncio.ensure_future(runtime.login("two", interaction=interaction_two))

    # Both providers' logins must reach `prompt()` -- if `_credential_lock`
    # were a single lock shared across providers instead of per-provider,
    # the second login would never start and this would hang.
    await asyncio.wait_for(asyncio.gather(first_started.wait(), second_started.wait()), timeout=5)
    finish.set()
    await asyncio.gather(login_one, login_two)


@pytest.mark.skip(
    reason=(
        "Needs provider-scoped availability publication -- `refresh({providers: [...]})`. "
        "This port's `refresh()` takes no arguments and recomposes every provider "
        "synchronously, so a stalled unrelated provider cannot be observed (TS case 4)."
    )
)
def test_does_not_wait_for_unrelated_provider_availability() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "Needs `refresh({signal})` and the `{aborted: boolean}` result it resolves to. "
        "This port's `refresh()` returns None and is not cancellable (TS cases 5 and 7)."
    )
)
def test_reports_cancellation_during_provider_scoped_availability() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "Needs `Provider.refreshModels` and the network catalog refresh, neither of "
        "which is ported: this port never fetches model catalogs over the network, so "
        "there is no network refresh to keep out of the credential path (TS case 6)."
    )
)
def test_does_not_run_network_refresh_inside_the_credential_operation_chain() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "Needs `CredentialSynchronizationError` (the typed error carrying the committed "
        "credential when post-commit synchronization is cancelled or fails) and the "
        "commit/settle split it reports on. Neither is ported (TS cases 8, 9 and 10)."
    )
)
def test_reports_committed_credentials_when_local_synchronization_fails() -> None:
    raise AssertionError("unreachable")


class TestCustomOpenAiCompatibleProviderCredentials:
    """Credential handling for a `models.json` OpenAI-compatible provider.

    Not a port of a TypeScript case (see the module docstring). `models` is an
    array here because `ModelsJsonProviderSchema.models` is
    `Type.Array(ModelDefinitionSchema)`; the keyed-object shape a user might
    reach for is rejected, which `test_model_registry.py` pins separately.
    """

    def config(self, api_key: str) -> dict[str, Any]:
        return {
            "baseUrl": "https://my-llm.test/v1",
            "api": "openai-completions",
            "apiKey": api_key,
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

    async def _runtime(
        self, tmp_path: Path, api_key: str, credentials: object | None = None
    ) -> tuple[ModelRuntime, object]:
        models_path = tmp_path / "models.json"
        models_path.write_text(json.dumps({"providers": {"my-llm": self.config(api_key)}}), encoding="utf-8")
        store = credentials if credentials is not None else AuthStorage.in_memory()
        runtime = await ModelRuntime.create(agent_dir=str(tmp_path), models_path=str(models_path), credentials=store)
        assert runtime.get_error() is None
        return runtime, store

    async def test_a_literal_api_key_resolves_as_configured_api_key(self, tmp_path: Path) -> None:
        runtime, _ = await self._runtime(tmp_path, "sk-literal-key")

        model = runtime.get_model("my-llm", "my-model")
        assert model is not None
        assert model.base_url == "https://my-llm.test/v1"

        result = await runtime.get_auth(model)
        assert result is not None
        assert result.source == "configured API key"
        assert result.auth.api_key == "sk-literal-key"
        # The provider's configured headers ride along on the resolved auth, so
        # every request carries them without the caller re-reading models.json.
        assert result.auth.headers == {"x-tenant": "acme"}

    async def test_a_literal_api_key_makes_the_provider_configured_without_a_login(self, tmp_path: Path) -> None:
        runtime, store = await self._runtime(tmp_path, "sk-literal-key")

        assert await store.get("my-llm") is None
        assert runtime.has_configured_auth("my-llm") is True
        status = runtime.get_provider_auth_status("my-llm")
        assert status.configured is True
        assert status.type == "api_key"
        assert status.source == "models_json_key"
        assert [m.id for m in runtime.get_available_snapshot() if m.provider == "my-llm"] == ["my-model"]

    async def test_an_env_var_api_key_is_unconfigured_until_the_variable_is_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MY_LLM_KEY", raising=False)
        runtime, _ = await self._runtime(tmp_path, "${MY_LLM_KEY}")

        assert runtime.has_configured_auth("my-llm") is False
        assert runtime.get_provider_auth_status("my-llm").configured is False
        assert not any(m.provider == "my-llm" for m in runtime.get_available_snapshot())

        monkeypatch.setenv("MY_LLM_KEY", "env-key")
        assert runtime.has_configured_auth("my-llm") is True
        status = runtime.get_provider_auth_status("my-llm")
        assert status.configured is True
        assert status.type == "api_key"
        # TypeScript reports `{ source: "environment", label: "MY_LLM_KEY" }`;
        # this port's `AuthCheck` has no `label` field and folds it into
        # `source` (`model_runtime.get_provider_auth_status`).
        assert status.source == "MY_LLM_KEY"
        assert [m.id for m in runtime.get_available_snapshot() if m.provider == "my-llm"] == ["my-model"]

        result = await runtime.get_auth(runtime.get_model("my-llm", "my-model"))
        assert result is not None
        assert result.source == "configured API key"
        assert result.auth.api_key == "env-key"

    async def test_a_stored_credential_takes_precedence_over_the_configured_key(self, tmp_path: Path) -> None:
        runtime, store = await self._runtime(tmp_path, "sk-literal-key")

        await runtime.login("my-llm", "stored-key")
        assert await store.get("my-llm") == Credential(type="api_key", key="stored-key")

        result = await runtime.get_auth(runtime.get_model("my-llm", "my-model"))
        assert result is not None
        assert result.source == "stored credential"
        assert result.auth.api_key == "stored-key"
        # Configured headers are applied to the stored-credential branch too.
        assert result.auth.headers == {"x-tenant": "acme"}

    async def test_logout_falls_back_to_the_configured_key_instead_of_unconfiguring(self, tmp_path: Path) -> None:
        runtime, store = await self._runtime(tmp_path, "sk-literal-key")
        await runtime.login("my-llm", "stored-key")

        await runtime.logout("my-llm")
        assert await store.get("my-llm") is None
        # The provider stays usable: `models.json` still supplies a key, so
        # logging out must not remove it from the available snapshot.
        assert runtime.has_configured_auth("my-llm") is True
        assert [m.id for m in runtime.get_available_snapshot() if m.provider == "my-llm"] == ["my-model"]

        result = await runtime.get_auth(runtime.get_model("my-llm", "my-model"))
        assert result is not None
        assert result.source == "configured API key"
        assert result.auth.api_key == "sk-literal-key"

    async def test_login_configures_a_provider_whose_env_var_key_is_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`login` must make an otherwise-unconfigured custom provider available.

        This is the realistic flow for a `models.json` provider written as
        `"apiKey": "${MY_LLM_KEY}"`: the user never exports the variable and
        runs a login instead. TypeScript's `hasConfiguredAuth` reads
        `snapshot.configuredProviders`, which the async `getAvailable()` path
        fills from the credential store, so the model shows up.
        """
        monkeypatch.delenv("MY_LLM_KEY", raising=False)
        runtime, store = await self._runtime(tmp_path, "${MY_LLM_KEY}")
        assert runtime.has_configured_auth("my-llm") is False

        await runtime.login("my-llm", "stored-key")

        assert await store.get("my-llm") == Credential(type="api_key", key="stored-key")
        assert runtime.has_configured_auth("my-llm") is True
        status = runtime.get_provider_auth_status("my-llm")
        assert status.configured is True
        assert status.type == "api_key"
        assert status.source == "stored credential"
        assert [m.id for m in runtime.get_available_snapshot() if m.provider == "my-llm"] == ["my-model"]

        result = await runtime.get_auth(runtime.get_model("my-llm", "my-model"))
        assert result is not None
        assert result.source == "stored credential"
        assert result.auth.api_key == "stored-key"
        assert result.auth.headers == {"x-tenant": "acme"}
