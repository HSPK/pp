"""The ``pi auth`` subcommands.

Ported from ``packages/coding-agent/src/cli/auth-command.ts``,
``cli/auth-check.ts`` and ``cli/credential-print.ts``.

``pi auth check`` reports whether a provider is ready, and
``pi auth print-api-key`` / ``print-bearer-token`` emit the resolved
credential for scripting.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pi_ai.auth.types import CredentialStore

from pi_coding_agent.cli.args import Args, parse_args
from pi_coding_agent.core.config import APP_NAME
from pi_coding_agent.core.model_resolver import ModelsAuthSource, resolve_cli_model

AuthCommandKind = Literal["check", "api_key", "bearer_token"]
AuthCheckStatus = Literal["ready", "not_ready", "invalid"]
AuthCheckReason = Literal[
    "provider_not_found",
    "credentials_not_configured",
    "credential_not_available",
    "invalid_state",
]

_DURATION_RE = re.compile(r"^(\d+)(ms|s|m|h)$", re.IGNORECASE)
_BEARER_RE = re.compile(r"^Bearer\s+(.+)$", re.IGNORECASE)

AUTH_COMMAND_USAGE: dict[str, str] = {
    "check": "pi auth check --provider <provider> [--json] [--credentials] [--no-refresh]",
    "api_key": "pi auth print-api-key --provider <provider> [--model <model>]",
    "bearer_token": ("pi auth print-bearer-token --provider <provider> [--model <model>] [--min-expiry <duration>]"),
}

AUTH_HELP_TEXT = """Usage:
  pi auth print-api-key [--provider <provider>] [--model <model>]
  pi auth print-bearer-token [--provider <provider>] [--model <model>] [--min-expiry <duration>]
  pi auth check [--provider <provider>] [--model <model>] [--json] [--credentials] [--no-refresh]

Auth commands require at least one of --provider or --model. Checks refresh \
expired OAuth credentials by default; --no-refresh prevents this. --credentials \
emits the credential, or includes it in JSON output."""


class AuthCommandError(Exception):
    """A user-facing problem with an ``auth`` invocation."""


@dataclass
class AuthCommand:
    kind: AuthCommandKind
    args: list[str] = field(default_factory=list)
    json: bool = False
    credentials: bool = False
    no_refresh: bool = False
    min_expiry_ms: int | None = None


@dataclass
class AuthCheckResult:
    status: AuthCheckStatus
    provider: str
    reason: AuthCheckReason | None = None
    auth_type: Literal["api_key", "oauth"] | None = None


def get_auth_command_name(kind: AuthCommandKind) -> str:
    if kind == "check":
        return "auth check"
    return "auth print-api-key" if kind == "api_key" else "auth print-bearer-token"


def get_auth_command_usage(kind: AuthCommandKind) -> str:
    return AUTH_COMMAND_USAGE[kind]


def is_auth_command_help(args: list[str]) -> bool:
    if not args or args[0] != "auth":
        return False
    return len(args) < 2 or args[1] == "help" or "--help" in args or "-h" in args


def print_auth_command_help(write: Callable[[str], None] | None = None) -> None:
    (write or print)(AUTH_HELP_TEXT)


def parse_auth_command(args: list[str]) -> AuthCommand | None:
    if not args or args[0] != "auth":
        return None

    sub = args[1] if len(args) > 1 else None
    kind: AuthCommandKind | None = (
        "check"
        if sub == "check"
        else "api_key"
        if sub == "print-api-key"
        else "bearer_token"
        if sub == "print-bearer-token"
        else None
    )
    if kind is None:
        raise AuthCommandError(
            f'Unknown auth command "{sub or ""}". Use "pi auth print-api-key", '
            '"pi auth print-bearer-token", or "pi auth check".'
        )

    command = AuthCommand(kind=kind)
    index = 2
    while index < len(args):
        arg = args[index]
        if arg == "--min-expiry":
            if kind != "bearer_token":
                raise AuthCommandError("--min-expiry is only supported by print-bearer-token")
            index += 1
            value = args[index] if index < len(args) else None
            match = _DURATION_RE.match(value) if value else None
            if match is None:
                raise AuthCommandError("--min-expiry must use a duration such as 30m or 1h")
            amount = int(match.group(1))
            unit = match.group(2).lower()
            multiplier = {"ms": 1, "s": 1_000, "m": 60_000, "h": 3_600_000}[unit]
            command.min_expiry_ms = amount * multiplier
        elif arg in ("--json", "--credentials", "--no-refresh"):
            if kind != "check":
                raise AuthCommandError(f"{arg} is only supported by auth check")
            if arg == "--json":
                command.json = True
            elif arg == "--credentials":
                command.credentials = True
            else:
                command.no_refresh = True
        else:
            command.args.append(arg)
        index += 1

    return command


def validate_auth_command_args(args: Args, kind: AuthCommandKind) -> tuple[str | None, str | None]:
    provider = (args.provider or "").strip() or None
    model = (args.model or "").strip() or None

    if args.unknown_flags:
        option = next(iter(args.unknown_flags))
        raise AuthCommandError(f'Unknown option --{option} for "{get_auth_command_name(kind)}".')
    if args.api_key is not None or args.messages or args.file_args:
        raise AuthCommandError("Auth commands only accept --provider and --model")

    if not provider and not model:
        if kind == "check":
            raise AuthCommandError("Auth checks require --provider <provider> or --model <model>")
        raise AuthCommandError("Credential printing requires --provider <provider> or --model <model>")
    return provider, model


def get_auth_credential(auth: Any) -> str | None:
    """Pull the API key or bearer token out of a resolved ``AuthResult``."""
    if auth is None:
        return None
    inner = getattr(auth, "auth", None)
    if inner is None:
        return None
    api_key = getattr(inner, "api_key", None)
    if api_key:
        return api_key
    headers = getattr(inner, "headers", None) or {}
    for name, value in headers.items():
        if name.lower() == "authorization" and isinstance(value, str):
            match = _BEARER_RE.match(value)
            if match:
                return match.group(1)
    return None


async def _resolve_provider(args: Args, model_runtime: Any, kind: AuthCommandKind) -> str:
    provider, model_reference = validate_auth_command_args(args, kind)
    if model_reference:
        # `find_model` ignores `--provider`, so `--provider openai --model gpt-5.5`
        # could resolve to a different provider's model. TS uses `resolveCliModel`,
        # which scopes the pattern to the requested provider.
        resolved = resolve_cli_model(
            ModelsAuthSource(model_runtime.models),
            cli_provider=provider,
            cli_model=model_reference,
        )
        if resolved.error or resolved.model is None:
            raise AuthCommandError(resolved.error or f'Unable to resolve model "{model_reference}"')
        provider = resolved.model.provider
    if not provider:
        raise AuthCommandError("Unable to resolve an auth provider")
    return provider


async def check_provider_auth(args: Args, model_runtime: Any, *, refresh: bool = False) -> AuthCheckResult:
    provider = await _resolve_provider(args, model_runtime, "check")

    get_error = getattr(model_runtime, "get_error", None)
    if get_error is not None and get_error():
        return AuthCheckResult(status="invalid", provider=provider, reason="invalid_state")
    if model_runtime.get_provider(provider) is None:
        return AuthCheckResult(status="not_ready", provider=provider, reason="provider_not_found")

    try:
        # The async check resolves stored credentials and command-based API keys;
        # the synchronous `get_provider_auth_status` only sees env vars.
        status = await model_runtime.check_auth(provider)
        if status is None or not status.configured:
            return AuthCheckResult(status="not_ready", provider=provider, reason="credentials_not_configured")
        if refresh and await model_runtime.get_auth(provider) is None:
            return AuthCheckResult(status="not_ready", provider=provider, reason="credentials_not_configured")
        return AuthCheckResult(status="ready", provider=provider, auth_type=status.type or "api_key")
    except Exception:
        # A malformed auth.json (or an OAuth refresh failure) surfaces here.
        return AuthCheckResult(status="invalid", provider=provider, reason="invalid_state")


async def get_provider_credential(
    provider_id: str,
    model_runtime: Any,
    credentials: CredentialStore,
    *,
    refresh: bool,
) -> str | None:
    """Resolve one provider credential, optionally without refreshing OAuth.

    Port of TS ``getProviderCredential``. When ``refresh`` is false a stored
    OAuth access token is returned as-is, so ``auth check --no-refresh`` never
    triggers a token exchange.
    """
    credential = await credentials.get(provider_id)
    if not refresh and credential is not None and credential.type == "oauth":
        return credential.access
    return get_auth_credential(await model_runtime.get_auth(provider_id))


async def create_auth_check_model_runtime(credentials: CredentialStore) -> Any:
    """Build a `ModelRuntime` for ``auth check`` (no catalog network access)."""
    from pi_coding_agent.core.model_runtime import ModelRuntime

    return await ModelRuntime.create(credentials=credentials)


async def print_credential(
    args: Args,
    model_runtime: Any,
    kind: AuthCommandKind,
    write: Callable[[str], None] | None = None,
    min_expiry_ms: int | None = None,
) -> int:
    """``pi auth print-api-key`` / ``print-bearer-token``."""
    from pi_coding_agent.cli.credential_print import resolve_credential_for_print

    emit = write or print
    try:
        credential = await resolve_credential_for_print(args, model_runtime, kind, min_expiry_ms)
    except AuthCommandError:
        raise
    except Exception as error:
        print(f"Failed to resolve credentials: {error}", file=sys.stderr)
        return 1
    emit(credential)
    return 0


async def handle_auth_command(
    raw_args: list[str],
    *,
    agent_dir: str | None = None,
    model_runtime: Any = None,
    write: Callable[[str], None] | None = None,
) -> int | None:
    """Run an ``auth`` subcommand, or return ``None`` when ``raw_args`` is not one."""
    if not raw_args or raw_args[0] != "auth":
        return None

    emit = write or print
    if is_auth_command_help(raw_args):
        print_auth_command_help(emit)
        return 0

    try:
        command = parse_auth_command(raw_args)
    except AuthCommandError as error:
        print(str(error), file=sys.stderr)
        return 1
    if command is None:
        return None

    parsed = parse_args(command.args)

    # Port of TS `runAuthCommand`: unknown flags are reported before any model
    # runtime is built, with the "Use ..." hint rather than the usage line.
    if parsed.unknown_flags:
        option = next(iter(parsed.unknown_flags))
        print(f'Unknown option --{option} for "{get_auth_command_name(command.kind)}".', file=sys.stderr)
        print(f'Use "{APP_NAME} --help" or "{get_auth_command_usage(command.kind)}".', file=sys.stderr)
        return 1

    if model_runtime is None:
        from pi_coding_agent.core.model_runtime import ModelRuntime

        model_runtime = await ModelRuntime.create(agent_dir=agent_dir)

    try:
        if command.kind == "check":
            result = await check_provider_auth(parsed, model_runtime, refresh=not command.no_refresh)
            if command.json:
                payload: dict[str, Any] = {"status": result.status, "provider": result.provider}
                if result.reason:
                    payload["reason"] = result.reason
                if result.auth_type:
                    payload["authType"] = result.auth_type
                if command.credentials:
                    auth = await model_runtime.get_auth(result.provider)
                    payload["credential"] = get_auth_credential(auth)
                emit(json.dumps(payload))
            else:
                suffix = f" ({result.reason})" if result.reason else ""
                emit(f"{result.provider}: {result.status}{suffix}")
                if command.credentials and result.status == "ready":
                    auth = await model_runtime.get_auth(result.provider)
                    credential = get_auth_credential(auth)
                    if credential:
                        emit(credential)
            return 0 if result.status == "ready" else 1

        return await print_credential(parsed, model_runtime, command.kind, emit, command.min_expiry_ms)
    except AuthCommandError as error:
        print(str(error), file=sys.stderr)
        print(f"Usage: {get_auth_command_usage(command.kind)}", file=sys.stderr)
        return 1


__all__ = [
    "AUTH_COMMAND_USAGE",
    "AUTH_HELP_TEXT",
    "AuthCheckResult",
    "AuthCommand",
    "AuthCommandError",
    "AuthCommandKind",
    "check_provider_auth",
    "create_auth_check_model_runtime",
    "get_auth_command_name",
    "get_auth_command_usage",
    "get_auth_credential",
    "get_provider_credential",
    "handle_auth_command",
    "is_auth_command_help",
    "parse_auth_command",
    "print_auth_command_help",
    "print_credential",
    "validate_auth_command_args",
]
