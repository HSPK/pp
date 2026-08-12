"""Python port of `packages/coding-agent/test/agent-session-dynamic-provider.test.ts`.

The TypeScript file drives every case through `pi.registerProvider(...)` from
an extension factory. This port has **no extension provider registration**
(`registerProvider`/`registerNativeProvider`/`unregisterProvider`); see the
module docstring of `core/extensions/types.py` and the "not ported" note in
`core/provider_composer.py`. There is therefore no way to reach the *trigger*
the TypeScript uses.

What the trigger ultimately does is portable, though: it overlays a provider
config onto the built-in provider so that the active model -- the one already
selected by `create_agent_session` -- and the model the agent hands to its
stream function both carry the overridden `base_url`. That path exists here
via the `models.json` overlay, and the first test below pins it end to end,
which is the behavior every TypeScript case asserts. The four cases that only
differ in *when* the extension registers are skipped individually with the
reason at each spot.

`models.json` is also the *only* way this port can add a provider that the
built-ins do not ship, so the second half of this file pins that path end to
end through `create_agent_session`: a wholly custom OpenAI-compatible provider
is discovered, its model is selectable, and its `baseUrl`/`contextWindow`/
`maxTokens`/`headers`/`apiKey` all resolve. The outbound-request half of the
same feature lives in `test_model_runtime_modify_models_compat.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pi_ai.auth.types import Credential
from pi_ai.providers.all import get_builtin_model
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager

CUSTOM_PROVIDER_CONFIG: dict[str, Any] = {
    "providers": {
        "my-llm": {
            "name": "My LLM",
            "baseUrl": "https://my-llm.test/v1",
            "api": "openai-completions",
            "apiKey": "$MY_LLM_KEY",
            "headers": {"X-Org": "acme"},
            "models": [
                {
                    "id": "my-model",
                    "name": "My Model",
                    "contextWindow": 32_000,
                    "maxTokens": 4_096,
                }
            ],
        }
    }
}


async def create_session(tmp_path: Path, base_url: str | None):
    """Mirrors the TypeScript `createSession` helper.

    `extensionFactories` is replaced by the `models.json` overlay; see the
    module docstring.
    """
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    models_path = agent_dir / "models.json"
    if base_url is not None:
        models_path.write_text(json.dumps({"providers": {"anthropic": {"baseUrl": base_url}}}), encoding="utf-8")

    settings_manager = SettingsManager.create(str(tmp_path), str(agent_dir))
    session_manager = SessionManager.in_memory()
    auth_storage = AuthStorage.create(str(agent_dir / "auth.json"))
    await auth_storage.set("anthropic", Credential(type="api_key", key="test-key"))
    model_runtime = await ModelRuntime.create(credentials=auth_storage, models_path=str(models_path))

    resource_loader = ResourceLoader(
        ResourceLoaderOptions(
            cwd=str(tmp_path),
            agent_dir=str(agent_dir),
            no_skills=True,
            no_prompt_templates=True,
        )
    )
    resource_loader.reload()

    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(tmp_path),
            agent_dir=str(agent_dir),
            model=model_runtime.get_model("anthropic", "claude-sonnet-4-5"),
            settings_manager=settings_manager,
            session_manager=session_manager,
            model_runtime=model_runtime,
            resource_loader=resource_loader,
        )
    )
    return result.session


async def capture_prompt_base_url(session: Any) -> str | None:
    """Mirrors the TypeScript `capturePromptBaseUrl` helper."""
    captured: dict[str, str | None] = {}

    async def stream_function(model, context, options=None, **kwargs):
        captured["base_url"] = model.base_url
        raise RuntimeError("stop")

    session.agent.stream_function = stream_function
    await session.prompt("hello")
    return captured.get("base_url")


async def test_applies_provider_overrides_to_the_active_model(tmp_path: Path) -> None:
    session = await create_session(tmp_path, "http://localhost:8080/top-level")

    assert session.model is not None
    assert session.model.base_url == "http://localhost:8080/top-level"
    assert await capture_prompt_base_url(session) == "http://localhost:8080/top-level"

    session.dispose()


async def test_leaves_the_built_in_base_url_alone_without_an_override(tmp_path: Path) -> None:
    """Control for the case above: the overlay, not the port, is what moves the URL."""
    session = await create_session(tmp_path, None)

    builtin = get_builtin_model("anthropic", "claude-sonnet-4-5")
    assert builtin is not None
    assert session.model is not None
    assert session.model.base_url == builtin.base_url
    assert await capture_prompt_base_url(session) == builtin.base_url

    session.dispose()


_NO_REGISTER_PROVIDER = (
    "`pi.registerProvider` is not ported: this port has no extension provider "
    "registration (see core/extensions/types.py and core/provider_composer.py). "
    "The override behavior it triggers is covered by "
    "test_applies_provider_overrides_to_the_active_model above."
)


@pytest.mark.skip(reason=_NO_REGISTER_PROVIDER)
def test_applies_session_start_register_provider_overrides_to_the_active_model() -> None:
    """`it("applies session_start registerProvider overrides to the active model")`.

    Registers the override from a `session_start` handler, calls
    `session.bindExtensions({})`, then asserts `session.model.baseUrl` and the
    base URL seen by the stream function are both
    `http://localhost:8080/session-start`.
    """


@pytest.mark.skip(reason=_NO_REGISTER_PROVIDER)
def test_registers_native_providers_during_extension_loading() -> None:
    """`it("registers native pi-ai providers during extension loading")`.

    Passes a whole `Provider` object (not a config overlay) to
    `pi.registerProvider`, then asserts `session.model.baseUrl` and the
    streamed model's base URL are `http://localhost:8080/native-top-level`.
    """


@pytest.mark.skip(reason=_NO_REGISTER_PROVIDER)
def test_applies_command_time_register_provider_overrides_without_reload() -> None:
    """`it("applies command-time registerProvider overrides without reload")`.

    Registers a `/use-proxy` command whose handler calls `pi.registerProvider`,
    runs it via `session.prompt("/use-proxy")`, then asserts the active model
    and the streamed model both carry `http://localhost:8080/command` without
    any resource-loader reload.
    """


@pytest.mark.skip(reason=_NO_REGISTER_PROVIDER)
def test_registers_native_providers_at_command_time() -> None:
    """`it("registers native pi-ai providers at command time")`.

    Same as above but the command handler registers a whole `Provider`;
    asserts `http://localhost:8080/native-command` on both.
    """


# --------------------------------------------------------------------------
# Custom OpenAI-compatible provider declared entirely in models.json.
#
# `models.json` is this port's only way to add a provider the built-ins do not
# ship, so it is what a user reaches for instead of `pi.registerProvider`. The
# cases below pin the whole discovery path through `create_agent_session`.
# --------------------------------------------------------------------------


async def create_custom_provider_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`create_session`, but the provider itself comes from `models.json`."""
    monkeypatch.setenv("MY_LLM_KEY", "custom-secret")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    models_path = agent_dir / "models.json"
    models_path.write_text(json.dumps(CUSTOM_PROVIDER_CONFIG), encoding="utf-8")

    settings_manager = SettingsManager.create(str(tmp_path), str(agent_dir))
    session_manager = SessionManager.in_memory()
    auth_storage = AuthStorage.create(str(agent_dir / "auth.json"))
    model_runtime = await ModelRuntime.create(credentials=auth_storage, models_path=str(models_path))

    resource_loader = ResourceLoader(
        ResourceLoaderOptions(
            cwd=str(tmp_path),
            agent_dir=str(agent_dir),
            no_skills=True,
            no_prompt_templates=True,
        )
    )
    resource_loader.reload()

    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(tmp_path),
            agent_dir=str(agent_dir),
            model=model_runtime.get_model("my-llm", "my-model"),
            settings_manager=settings_manager,
            session_manager=session_manager,
            model_runtime=model_runtime,
            resource_loader=resource_loader,
        )
    )
    return result.session, model_runtime


async def test_discovers_a_custom_openai_compatible_provider_from_models_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session, runtime = await create_custom_provider_session(tmp_path, monkeypatch)

    assert runtime.get_error() is None
    assert "my-llm" in [provider.id for provider in runtime.get_providers()]

    provider = runtime.get_provider("my-llm")
    assert provider is not None
    assert provider.name == "My LLM"
    # Configured provider headers are resolved once at compose time and live on
    # the provider, not the model (see `core/provider_composer.py`).
    assert (provider.headers or {}).get("X-Org") == "acme"


async def test_custom_provider_model_resolves_its_configured_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session, runtime = await create_custom_provider_session(tmp_path, monkeypatch)

    model = runtime.get_model("my-llm", "my-model")
    assert model is not None
    assert model.provider == "my-llm"
    assert model.name == "My Model"
    assert model.api == "openai-completions"
    assert model.base_url == "https://my-llm.test/v1"
    assert model.context_window == 32_000
    assert model.max_tokens == 4_096
    # A model definition that omits `compat`/`samplingParams` must still carry
    # dicts: `pi_ai.types.Model` declares both as `dict` and the request
    # builders read `model.compat.get(...)` without a `None` guard.
    assert model.compat == {}
    assert model.sampling_params == {}

    assert runtime.find_model("my-llm/my-model") is not None


async def test_custom_provider_api_key_resolves_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session, runtime = await create_custom_provider_session(tmp_path, monkeypatch)

    assert runtime.has_configured_auth("my-llm") is True
    assert "my-model" in [model.id for model in runtime.get_available_snapshot()]

    result = await runtime.get_auth("my-llm")
    assert result is not None
    assert result.auth.api_key == "custom-secret"
    assert result.source == "configured API key"


async def test_custom_provider_model_is_the_one_handed_to_the_stream_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _runtime = await create_custom_provider_session(tmp_path, monkeypatch)

    assert session.model is not None
    assert session.model.provider == "my-llm"
    assert session.model.id == "my-model"
    assert await capture_prompt_base_url(session) == "https://my-llm.test/v1"

    session.dispose()
