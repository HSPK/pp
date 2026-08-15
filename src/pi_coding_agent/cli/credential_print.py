"""``pi auth print-api-key`` / ``print-bearer-token`` credential resolution.

Python port of ``packages/coding-agent/src/cli/credential-print.ts``.

Resolving a credential for printing is deliberately *not* the same as
"resolve auth for a request": the kind asked for (API key vs OAuth bearer
token) filters which configured providers are eligible, and an ambiguous match
is an error rather than an arbitrary pick.
"""

from __future__ import annotations

from dataclasses import dataclass

from pi_ai.auth.types import AuthType
from pi_ai.registry import Model

from pi_coding_agent.cli.args import Args
from pi_coding_agent.cli.auth_command import (
    AuthCommandError,
    AuthCommandKind,
    get_auth_credential,
    validate_auth_command_args,
)
from pi_coding_agent.core.model_resolver import resolve_cli_model
from pi_coding_agent.core.model_runtime import ModelRuntime

DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS = 30 * 60_000


@dataclass
class _Candidate:
    id: str
    model: Model | None = None


async def resolve_credential_for_print(
    args: Args,
    model_runtime: ModelRuntime,
    kind: AuthCommandKind,
    min_expiry_ms: int | None = None,
) -> str:
    """Resolve one configured provider credential for printing.

    Calls `ModelRuntime.get_auth`, which refreshes and persists OAuth
    credentials that are close to expiry through the normal request-auth path.
    """
    cli_provider, cli_model = validate_auth_command_args(args, kind)
    credential_types: dict[str, AuthType] = {
        info.provider_id: info.type for info in await model_runtime.list_credentials()
    }

    candidates: list[_Candidate] = []
    if cli_provider:
        provider = model_runtime.get_provider(cli_provider)
        if provider is None:
            raise AuthCommandError(f'Unknown provider "{cli_provider}". Use --list-models to see available providers.')
        if cli_model:
            resolved = resolve_cli_model(model_runtime, cli_provider=provider.id, cli_model=cli_model)
            if resolved.error or resolved.model is None:
                raise AuthCommandError(resolved.error or "Unable to resolve the requested provider/model")
            candidates.append(_Candidate(id=provider.id, model=resolved.model))
        else:
            candidates.append(_Candidate(id=provider.id))
    else:
        for provider in model_runtime.get_providers():
            if provider.id not in credential_types:
                continue
            resolved = resolve_cli_model(model_runtime, cli_provider=provider.id, cli_model=cli_model)
            if (
                resolved.model is not None
                and not resolved.error
                and "Using custom model id" not in (resolved.warning or "")
            ):
                candidates.append(_Candidate(id=provider.id, model=resolved.model))
        if not candidates:
            raise AuthCommandError(f'Model "{cli_model}" not found. Use --list-models to see available models.')

    credentials: list[tuple[str, str]] = []
    for candidate in candidates:
        credential_type = credential_types.get(candidate.id)
        if kind == "api_key" and credential_type == "oauth":
            continue
        if kind == "bearer_token" and credential_type != "oauth":
            continue
        min_validity = (
            (min_expiry_ms if min_expiry_ms is not None else DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS)
            if kind == "bearer_token"
            else None
        )
        target: str | Model = candidate.model if candidate.model is not None else candidate.id
        auth = await model_runtime.get_auth(target, min_oauth_validity_ms=min_validity)
        value = get_auth_credential(auth)
        if value:
            credentials.append((candidate.id, value))

    if len(credentials) == 1:
        return credentials[0][1]
    if not credentials:
        provider_id = candidates[0].id if candidates else None
        credential_type = credential_types.get(provider_id) if provider_id else None
        if cli_provider and kind == "api_key" and credential_type == "oauth":
            raise AuthCommandError(f'Provider "{provider_id}" is configured with OAuth, not an API key')
        if cli_provider and kind == "bearer_token" and credential_type != "oauth":
            raise AuthCommandError(f'Provider "{provider_id}" is not configured with an OAuth bearer token')
        label = "API key" if kind == "api_key" else "OAuth bearer token"
        raise AuthCommandError(f"No usable {label} is configured")
    matched = ", ".join(provider_id for provider_id, _ in credentials)
    raise AuthCommandError(f"Multiple configured providers matched ({matched}). Specify --provider.")


__all__ = ["DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS", "resolve_credential_for_print"]
