"""Python port of `packages/coding-agent/test/auth-check.test.ts`.

The TypeScript test drives the OAuth cases through `openai-codex`; this port
uses `anthropic` instead because the OpenAI Codex provider in `pi_ai` has no
OAuth flow (it needs the OAuth/WebSocket transport the port omits).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pi_ai.auth.types import Credential, CredentialStore
from pi_ai.types import now_ms

from pi_coding_agent.cli.args import parse_args
from pi_coding_agent.cli.auth_command import (
    AuthCheckResult,
    AuthCommand,
    check_provider_auth,
    create_auth_check_model_runtime,
    get_provider_credential,
    parse_auth_command,
)
from pi_coding_agent.core.auth_storage import AuthStorage, ReadOnlyAuthStorage
from pi_coding_agent.core.model_runtime import ModelRuntime

OAUTH_PROVIDER = "anthropic"


async def create_runtime(credentials: CredentialStore, models_path: Path) -> ModelRuntime:
    return await ModelRuntime.create(credentials=credentials, models_path=str(models_path))


@pytest.fixture
def models_path(tmp_path: Path) -> Path:
    return tmp_path / "models.json"


async def test_reports_a_configured_provider_as_ready(models_path: Path) -> None:
    runtime = await create_runtime(
        AuthStorage.in_memory({"openai": Credential(type="api_key", key="test-key")}), models_path
    )

    result = await check_provider_auth(parse_args(["--provider", "openai"]), runtime)
    assert result == AuthCheckResult(status="ready", provider="openai", auth_type="api_key")


async def test_resolves_the_provider_from_model(models_path: Path) -> None:
    runtime = await create_runtime(
        AuthStorage.in_memory({"openai": Credential(type="api_key", key="test-key")}), models_path
    )

    assert await check_provider_auth(parse_args(["--model", "openai/gpt-5.5"]), runtime) == AuthCheckResult(
        status="ready", provider="openai", auth_type="api_key"
    )

    scoped = await check_provider_auth(parse_args(["--provider", "openai", "--model", "gpt-5.5"]), runtime)
    assert scoped.status == "ready"
    assert scoped.provider == "openai"


async def test_reads_credentials_without_refreshing_oauth_when_requested(models_path: Path) -> None:
    api_credentials = AuthStorage.in_memory({"openai": Credential(type="api_key", key="test-key")})
    api_runtime = await create_runtime(api_credentials, models_path)
    assert await get_provider_credential("openai", api_runtime, api_credentials, refresh=False) == "test-key"

    credentials = AuthStorage.in_memory(
        {OAUTH_PROVIDER: Credential(type="oauth", access="old-token", refresh="refresh-token", expires=0)}
    )
    oauth_runtime = await create_runtime(credentials, models_path)
    provider = oauth_runtime.get_provider(OAUTH_PROVIDER)
    assert provider is not None
    oauth = provider.auth.oauth
    assert oauth is not None, f"{OAUTH_PROVIDER} OAuth provider is not registered"

    calls: list[Credential] = []

    async def refresh(credential: Credential, signal: object = None) -> Credential:
        calls.append(credential)
        raise AssertionError("refresh must not be called")

    oauth.refresh = refresh

    assert await get_provider_credential(OAUTH_PROVIDER, oauth_runtime, credentials, refresh=False) == "old-token"
    assert calls == []


async def test_refreshes_oauth_by_default(models_path: Path) -> None:
    credentials = AuthStorage.in_memory(
        {OAUTH_PROVIDER: Credential(type="oauth", access="old-token", refresh="refresh-token", expires=0)}
    )
    runtime = await create_runtime(credentials, models_path)
    provider = runtime.get_provider(OAUTH_PROVIDER)
    assert provider is not None
    oauth = provider.auth.oauth
    assert oauth is not None, f"{OAUTH_PROVIDER} OAuth provider is not registered"

    calls: list[Credential] = []

    async def refresh(credential: Credential, signal: object = None) -> Credential:
        calls.append(credential)
        return Credential(
            type="oauth",
            access="fresh-token",
            refresh="refresh-token",
            expires=now_ms() + 60 * 60 * 1000,
        )

    oauth.refresh = refresh

    result = await check_provider_auth(parse_args(["--provider", OAUTH_PROVIDER]), runtime, refresh=True)
    assert result.status == "ready"
    assert len(calls) == 1


async def test_reports_an_unknown_provider_as_not_ready(models_path: Path) -> None:
    runtime = await create_runtime(AuthStorage.in_memory(), models_path)

    assert await check_provider_auth(parse_args(["--provider", "not-installed"]), runtime) == AuthCheckResult(
        status="not_ready", provider="not-installed", reason="provider_not_found"
    )


async def test_unresolved_stored_environment_reference_is_not_configured(tmp_path: Path, models_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({"openai": {"type": "api_key", "key": "$MISSING_AUTH_CHECK_KEY"}}), "utf-8")
    runtime = await create_runtime(ReadOnlyAuthStorage(str(auth_path)), models_path)

    assert await check_provider_auth(parse_args(["--provider", "openai"]), runtime) == AuthCheckResult(
        status="not_ready", provider="openai", reason="credentials_not_configured"
    )


async def test_reports_malformed_auth_state_as_invalid(tmp_path: Path, models_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text("{invalid-json", "utf-8")
    runtime = await create_runtime(ReadOnlyAuthStorage(str(auth_path)), models_path)

    assert await check_provider_auth(parse_args(["--provider", "openai"]), runtime) == AuthCheckResult(
        status="invalid", provider="openai", reason="invalid_state"
    )


async def test_does_not_create_an_auth_file_or_its_parent_directory(tmp_path: Path, models_path: Path) -> None:
    auth_path = tmp_path / "agent" / "auth.json"
    runtime = await create_runtime(ReadOnlyAuthStorage(str(auth_path)), models_path)

    result = await check_provider_auth(parse_args(["--provider", "openai"]), runtime)
    assert result.status == "not_ready"
    assert result.reason == "credentials_not_configured"
    assert not auth_path.exists()
    assert not (tmp_path / "agent").exists()


def test_accepts_optional_json_credential_output_and_no_refresh() -> None:
    assert parse_auth_command(["auth", "check", "--provider", "openai"]) == AuthCommand(
        kind="check",
        args=["--provider", "openai"],
        json=False,
        credentials=False,
        no_refresh=False,
    )
    assert parse_auth_command(
        ["auth", "check", "--json", "--credentials", "--no-refresh", "--provider", "openai"]
    ) == AuthCommand(
        kind="check",
        args=["--provider", "openai"],
        json=True,
        credentials=True,
        no_refresh=True,
    )


async def test_creates_an_auth_check_runtime_without_catalog_storage() -> None:
    runtime: Any = await create_auth_check_model_runtime(AuthStorage.in_memory())
    assert runtime.get_provider("openai") is not None
