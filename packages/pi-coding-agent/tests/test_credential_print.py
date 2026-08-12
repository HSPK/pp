"""Python port of `packages/coding-agent/test/credential-print.test.ts`.

Covers `pi auth print-api-key` / `print-bearer-token`: which stored credential
kind each command is willing to print, OAuth refresh on the way out, and the
argument grammar (`parse_auth_command` / `is_auth_command_help`).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from pi_ai.auth.types import Credential
from pi_coding_agent.cli.args import parse_args
from pi_coding_agent.cli.auth_command import (
    AuthCommand,
    AuthCommandError,
    is_auth_command_help,
    parse_auth_command,
)
from pi_coding_agent.cli.credential_print import resolve_credential_for_print
from pi_coding_agent.cli.entry import main
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.config import ENV_AGENT_DIR
from pi_coding_agent.core.model_runtime import ModelRuntime

_HOUR_MS = 60 * 60 * 1000


async def _create_runtime(credentials: AuthStorage, tmp_path: Path) -> ModelRuntime:
    return await ModelRuntime.create(
        credentials=credentials,
        agent_dir=str(tmp_path),
        auth_path=str(tmp_path / "auth.json"),
        models_path=str(tmp_path / "models.json"),
    )


async def test_prints_a_resolved_api_key(tmp_path: Path) -> None:
    runtime = await _create_runtime(
        AuthStorage.in_memory({"openai": Credential(type="api_key", key="test-api-key")}), tmp_path
    )
    args = parse_args(["--provider", "openai"])

    assert await resolve_credential_for_print(args, runtime, "api_key") == "test-api-key"


async def test_prints_bearer_tokens_resolved_from_an_authorization_header(tmp_path: Path) -> None:
    runtime = await _create_runtime(
        AuthStorage.in_memory(
            {
                "kimi-coding": Credential(
                    type="oauth",
                    access="header-test-token",
                    refresh="test-refresh-token",
                    expires=time.time() * 1000 + _HOUR_MS,
                )
            }
        ),
        tmp_path,
    )
    args = parse_args(["--provider", "kimi-coding"])

    assert await resolve_credential_for_print(args, runtime, "bearer_token") == "header-test-token"


async def test_refreshes_an_expired_oauth_token_before_printing_it(tmp_path: Path) -> None:
    # TypeScript uses `openai-codex`; this port's `pi_ai` has no OAuth flow for
    # that provider, so `anthropic` stands in as an OAuth-capable provider.
    storage = AuthStorage.in_memory(
        {"anthropic": Credential(type="oauth", access="old-test-token", refresh="test-refresh-token", expires=0)}
    )
    runtime = await _create_runtime(storage, tmp_path)
    calls: list[Credential] = []

    async def refresh(credential: Credential, signal: Any = None) -> Credential:
        calls.append(credential)
        return Credential(
            type="oauth",
            access="fresh-test-token",
            refresh="test-refresh-token",
            expires=time.time() * 1000 + _HOUR_MS,
        )

    provider = runtime.get_provider("anthropic")
    assert provider is not None and provider.auth.oauth is not None, "Anthropic OAuth provider is not registered"
    provider.auth.oauth.refresh = refresh
    args = parse_args(["--provider", "anthropic"])

    assert await resolve_credential_for_print(args, runtime, "bearer_token") == "fresh-test-token"
    assert len(calls) == 1
    stored = await storage.get("anthropic")
    assert stored is not None and stored.access == "fresh-test-token"


def test_reports_unknown_auth_options_like_package_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Driven through the real `main([...])`, as TypeScript does, rather than
    # `handle_auth_command` directly: that also covers the pre-parse dispatch
    # that decides `auth` is a subcommand and returns its exit code instead of
    # falling through to the agent argument scanner.
    monkeypatch.setenv(ENV_AGENT_DIR, str(tmp_path))
    code = main(["auth", "check", "--provider", "openai-codex", "--credentails"])
    stderr = capsys.readouterr().err
    assert 'Unknown option --credentails for "auth check".' in stderr
    assert 'Use "pi --help" or "pi auth check --provider <provider> [--json] [--credentials] [--no-refresh]".' in stderr
    assert code == 1


async def test_parses_credential_commands_and_rejects_invalid_arguments(tmp_path: Path) -> None:
    runtime = await _create_runtime(
        AuthStorage.in_memory(
            {
                "anthropic": Credential(
                    type="oauth",
                    access="test-token-not-to-be-printed",
                    refresh="test-refresh-token",
                    expires=time.time() * 1000 + _HOUR_MS,
                )
            }
        ),
        tmp_path,
    )

    assert parse_auth_command(["auth", "print-api-key", "--provider", "openai"]) == AuthCommand(
        kind="api_key",
        args=["--provider", "openai"],
        json=False,
        credentials=False,
        no_refresh=False,
    )
    assert parse_auth_command(["auth", "print-bearer-token"]).kind == "bearer_token"
    assert parse_auth_command(["auth", "print-bearer-token", "--min-expiry", "30m"]) == AuthCommand(
        kind="bearer_token",
        args=[],
        json=False,
        credentials=False,
        no_refresh=False,
        min_expiry_ms=30 * 60_000,
    )
    with pytest.raises(AuthCommandError, match="only supported by print-bearer-token"):
        parse_auth_command(["auth", "print-api-key", "--min-expiry", "30m"])
    assert is_auth_command_help(["auth", "--help"]) is True
    assert is_auth_command_help(["auth", "print-api-key", "--help"]) is True
    assert is_auth_command_help(["auth", "print-bearer-token", "-h"]) is True
    assert is_auth_command_help(["auth", "check", "--help"]) is True
    with pytest.raises(AuthCommandError):
        parse_auth_command(["auth", "unknown"])
    with pytest.raises(AuthCommandError, match="requires --provider <provider> or --model <model>"):
        await resolve_credential_for_print(parse_args([]), runtime, "api_key")
    with pytest.raises(AuthCommandError, match="configured with OAuth"):
        await resolve_credential_for_print(parse_args(["--provider", "anthropic"]), runtime, "api_key")
