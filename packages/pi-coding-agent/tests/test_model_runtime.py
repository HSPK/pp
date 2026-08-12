"""Tests for `core/model_runtime.py`.

The full `ModelRuntime` in `packages/coding-agent/src/core/model-runtime.ts`
composes OAuth, a locked/versioned `ModelsStore`, a remote catalog refresher,
and an extension-provider layer. This port's `ModelRuntime` deliberately
narrows all of that away (see the module's docstring for the documented
boundary), so these tests cover what is actually implemented: composing
builtin providers with a `models.json` overlay, persisting API-key
credentials to a sandboxed file (never the real `$HOME`), the synchronous
`has_configured_auth`/`get_available_snapshot` heuristics, and a local-only
`refresh()`. Loosely inspired by the setup in
`packages/coding-agent/test/model-runtime-auth-options.test.ts`, adapted to
this narrower surface.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pi_ai.auth.types import Credential, InMemoryCredentialStore
from pi_ai.providers import openai_compatible_provider
from pi_ai.types import Model, ModelCost, now_ms
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.model_runtime import FileCredentialStore, ModelRuntime


def _fake_provider(provider_id: str = "fake-provider", env_vars: list[str] | None = None) -> object:
    return openai_compatible_provider(
        provider_id=provider_id,
        name="Fake Provider",
        base_url="https://fake.example.com",
        env_vars=env_vars or ["FAKE_PROVIDER_API_KEY"],
        models=[
            Model(
                id="fake-model",
                name="Fake Model",
                api="openai-completions",
                context_window=8000,
                max_tokens=1024,
                cost=ModelCost(input=0, output=0),
            )
        ],
    )


async def _create_runtime(tmp_path: Path, **kwargs) -> ModelRuntime:
    return await asyncio.wait_for(
        ModelRuntime.create(
            agent_dir=tmp_path / "agent",
            providers=kwargs.pop("providers", [_fake_provider()]),
            **kwargs,
        ),
        timeout=5,
    )


def test_create_uses_sandboxed_paths_never_real_home(tmp_path: Path):
    runtime = asyncio.run(_create_runtime(tmp_path))
    assert runtime.get_models()[0].id == "fake-model"
    # No real config file should have been touched: the sandboxed auth path
    # only exists once something is written to it.
    assert not (tmp_path / "agent" / "auth.json").exists()


def test_login_persists_to_sandboxed_auth_file_and_get_auth_resolves(tmp_path: Path):
    async def run():
        runtime = await _create_runtime(tmp_path)
        await runtime.login("fake-provider", "sk-test-123")
        auth = await runtime.get_auth("fake-provider")
        assert auth is not None
        assert auth.auth.api_key == "sk-test-123"

    asyncio.run(asyncio.wait_for(run(), timeout=5))
    auth_path = tmp_path / "agent" / "auth.json"
    assert auth_path.exists()
    data = json.loads(auth_path.read_text())
    assert data["fake-provider"]["key"] == "sk-test-123"


def test_logout_removes_stored_credential(tmp_path: Path):
    async def run():
        runtime = await _create_runtime(tmp_path)
        await runtime.login("fake-provider", "sk-test-123")
        await runtime.logout("fake-provider")
        auth = await runtime.get_auth("fake-provider")
        assert auth is None

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_has_configured_auth_checks_env_vars(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FAKE_PROVIDER_API_KEY", raising=False)
    runtime = asyncio.run(_create_runtime(tmp_path))
    assert runtime.has_configured_auth("fake-provider") is False
    monkeypatch.setenv("FAKE_PROVIDER_API_KEY", "sk-env")
    assert runtime.has_configured_auth("fake-provider") is True


def test_has_configured_auth_checks_stored_credential(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FAKE_PROVIDER_API_KEY", raising=False)

    async def run():
        runtime = await _create_runtime(tmp_path)
        assert runtime.has_configured_auth("fake-provider") is False
        await runtime.login("fake-provider", "sk-stored")
        assert runtime.has_configured_auth("fake-provider") is True

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_get_available_snapshot_filters_by_configured_auth(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FAKE_PROVIDER_API_KEY", raising=False)
    runtime = asyncio.run(_create_runtime(tmp_path))
    assert runtime.get_available_snapshot() == []
    monkeypatch.setenv("FAKE_PROVIDER_API_KEY", "sk-env")
    snapshot = runtime.get_available_snapshot()
    assert len(snapshot) == 1
    assert snapshot[0].id == "fake-model"


def test_models_json_overlay_adds_custom_model_and_api_key(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FAKE_PROVIDER_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM_KEY_ENV", "sk-custom")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "fake-provider": {
                        "apiKey": "$CUSTOM_KEY_ENV",
                        "models": [{"id": "extra-model", "contextWindow": 4000, "maxTokens": 512}],
                    }
                }
            }
        )
    )

    async def run():
        runtime = await ModelRuntime.create(agent_dir=agent_dir, providers=[_fake_provider()])
        model_ids = {m.id for m in runtime.get_models("fake-provider")}
        assert "extra-model" in model_ids
        assert runtime.has_configured_auth("fake-provider") is True
        auth = await runtime.get_auth("fake-provider")
        assert auth is not None
        assert auth.auth.api_key == "sk-custom"

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_is_using_oauth_reflects_the_stored_credential_type(tmp_path: Path):
    async def run():
        runtime = await _create_runtime(tmp_path, credentials=AuthStorage.in_memory())
        assert runtime.is_using_oauth("fake-provider") is False
        assert runtime.is_using_oauth("anthropic") is False

        oauth_runtime = await asyncio.wait_for(
            ModelRuntime.create(
                agent_dir=tmp_path / "oauth",
                models_path=tmp_path / "oauth" / "models.json",
                credentials=AuthStorage.in_memory(
                    {
                        "anthropic": Credential(
                            type="oauth", data={"access": "a", "refresh": "r", "expires": now_ms() + 60_000}
                        )
                    }
                ),
            ),
            timeout=5,
        )
        assert oauth_runtime.is_using_oauth("anthropic") is True
        assert oauth_runtime.is_using_subscription("anthropic") is True

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_refresh_rebuilds_from_current_config_without_network(tmp_path: Path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True)
    models_json = agent_dir / "models.json"

    async def run():
        runtime = await ModelRuntime.create(agent_dir=agent_dir, providers=[_fake_provider()])
        assert "extra-model" not in {m.id for m in runtime.get_models("fake-provider")}

        models_json.write_text(
            json.dumps(
                {
                    "providers": {
                        "fake-provider": {
                            "models": [{"id": "extra-model", "contextWindow": 4000, "maxTokens": 512}],
                        }
                    }
                }
            )
        )
        runtime._config = runtime._config.load(models_json)  # simulate an external config edit
        runtime.refresh()
        assert "extra-model" in {m.id for m in runtime.get_models("fake-provider")}

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_accepts_an_in_memory_credential_store(tmp_path: Path):
    from pi_ai.auth.types import Credential

    async def run():
        credentials = InMemoryCredentialStore()
        await credentials.set("fake-provider", Credential(key="stored-key"))
        runtime = await ModelRuntime.create(
            agent_dir=tmp_path / "agent", providers=[_fake_provider()], credentials=credentials
        )
        auth = await runtime.get_auth("fake-provider")
        assert auth is not None
        assert auth.auth.api_key == "stored-key"

    asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_file_credential_store_roundtrip(tmp_path: Path):
    from pi_ai.auth.types import Credential

    async def run():
        store = FileCredentialStore(tmp_path / "auth.json")
        assert await store.get("p") is None
        await store.set("p", Credential(key="abc"))
        cred = await store.get("p")
        assert cred is not None
        assert cred.key == "abc"
        await store.delete("p")
        assert await store.get("p") is None

    asyncio.run(asyncio.wait_for(run(), timeout=5))
