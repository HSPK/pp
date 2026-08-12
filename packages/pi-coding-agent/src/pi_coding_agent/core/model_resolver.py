"""Model resolution, scoping, and initial selection.

Python port of `packages/coding-agent/src/core/model-resolver.ts`. Resolution
runs against the ported `pi_ai.registry.Models` collection: `find_exact_model_reference_match`
and `parse_model_pattern` fuzzy-match `provider/model-id` references and bare
ids against `Model` objects returned by `Models.get_models()`, preferring an
alias (e.g. `claude-sonnet-4-5`) over a dated snapshot
(`claude-sonnet-4-5-20250929`), and `resolve_cli_model` layers `--provider`/
`--model` CLI parsing, glob patterns, and per-provider fallback models on top.

**ModelRuntime deviation.** The TypeScript `ModelRuntime` wraps a network
model-catalog cache, an on-disk credential store, and async auth resolution
(`hasConfiguredAuth`, `getAvailableSnapshot`). This port has no equivalent
runtime; `ModelSource` is a small Protocol (`get_models`, `get_model`,
`has_configured_auth`, `get_available_snapshot`) satisfied either by a test
double (as the upstream tests do) or by `ModelsAuthSource`, a synchronous
adapter over `pi_ai.registry.Models` that treats a provider as "configured"
when one of its auth env vars is set in the process environment (mirroring
`pi_coding_agent.cli.resolve_model`'s existing heuristic) rather than
`Models.check_auth`'s async credential-store lookup.
"""

from __future__ import annotations

import fnmatch
import os
import re
import sys
from dataclasses import dataclass, replace
from typing import Protocol

from pi_ai.models import models_are_equal
from pi_ai.registry import Models
from pi_ai.types import Model

ThinkingLevel = str
"""One of "off", "minimal", "low", "medium", "high", "xhigh", "max"."""

DEFAULT_THINKING_LEVEL: ThinkingLevel = "medium"
"""Port of `packages/coding-agent/src/core/defaults.ts`."""
VALID_THINKING_LEVELS: tuple[str, ...] = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")

# Default model per known provider, used both to seed a fallback model's cost/
# metadata (`build_fallback_model`) and to pick a starting model when nothing
# else applies (`find_initial_model`).
DEFAULT_MODEL_PER_PROVIDER: dict[str, str] = {
    "amazon-bedrock": "us.anthropic.claude-opus-4-6-v1",
    "ant-ling": "Ring-2.6-1T",
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.5",
    "azure-openai-responses": "gpt-5.4",
    "openai-codex": "gpt-5.5",
    "radius": "auto",
    "nvidia": "nvidia/nemotron-3-super-120b-a12b",
    "deepseek": "deepseek-v4-pro",
    "google": "gemini-3.1-pro-preview",
    "google-vertex": "gemini-3.1-pro-preview",
    "github-copilot": "gpt-5.4",
    "openrouter": "moonshotai/kimi-k2.6",
    "vercel-ai-gateway": "zai/glm-5.1",
    "xai": "grok-4.5",
    "groq": "openai/gpt-oss-120b",
    "cerebras": "zai-glm-4.7",
    "zai": "glm-5.1",
    "zai-coding-cn": "glm-5.1",
    "mistral": "devstral-medium-latest",
    "minimax": "MiniMax-M2.7",
    "minimax-cn": "MiniMax-M2.7",
    "moonshotai": "kimi-k2.6",
    "moonshotai-cn": "kimi-k2.6",
    "huggingface": "moonshotai/Kimi-K2.6",
    "fireworks": "accounts/fireworks/models/kimi-k2p6",
    "together": "moonshotai/Kimi-K2.6",
    "baseten": "zai-org/GLM-5.2",
    "opencode": "kimi-k2.6",
    "opencode-go": "kimi-k2.6",
    "kimi-coding": "kimi-for-coding",
    "cloudflare-workers-ai": "@cf/moonshotai/kimi-k2.6",
    "cloudflare-ai-gateway": "workers-ai/@cf/moonshotai/kimi-k2.6",
    "qwen-token-plan": "qwen3.7-max",
    "qwen-token-plan-cn": "qwen3.7-max",
    "qwen-token-plan-individual": "qwen3.8-max",
    "xiaomi": "mimo-v2.5-pro",
    "xiaomi-token-plan-cn": "mimo-v2.5-pro",
    "xiaomi-token-plan-ams": "mimo-v2.5-pro",
    "xiaomi-token-plan-sgp": "mimo-v2.5-pro",
}


def is_valid_thinking_level(level: str) -> bool:
    return level in VALID_THINKING_LEVELS


@dataclass
class ScopedModel:
    model: Model
    thinking_level: ThinkingLevel | None = None
    """Thinking level if explicitly specified in the pattern (e.g. "model:high")."""


def _is_alias(model_id: str) -> bool:
    """A model id "looks like" an alias when it has no date suffix (``-YYYYMMDD``)."""
    if model_id.endswith("-latest"):
        return True
    return not _DATE_SUFFIX_RE.search(model_id)


def find_exact_model_reference_match(model_reference: str, available_models: list[Model]) -> Model | None:
    """Look up an exact ``provider/model-id`` or bare model id.

    Ambiguous bare-id matches across multiple providers are rejected (returns
    ``None``) rather than picking one arbitrarily.
    """
    trimmed_reference = model_reference.strip()
    if not trimmed_reference:
        return None

    normalized_reference = trimmed_reference.lower()

    canonical_matches = [
        model for model in available_models if f"{model.provider}/{model.id}".lower() == normalized_reference
    ]
    if len(canonical_matches) == 1:
        return canonical_matches[0]
    if len(canonical_matches) > 1:
        return None

    if "/" in trimmed_reference:
        provider, _, model_id = trimmed_reference.partition("/")
        provider = provider.strip()
        model_id = model_id.strip()
        if provider and model_id:
            provider_matches = [
                model
                for model in available_models
                if model.provider.lower() == provider.lower() and model.id.lower() == model_id.lower()
            ]
            if len(provider_matches) == 1:
                return provider_matches[0]
            if len(provider_matches) > 1:
                return None

    id_matches = [model for model in available_models if model.id.lower() == normalized_reference]
    return id_matches[0] if len(id_matches) == 1 else None


def _try_match_model(model_pattern: str, available_models: list[Model]) -> Model | None:
    """Exact match first, else partial id/name matching preferring an alias."""
    exact_match = find_exact_model_reference_match(model_pattern, available_models)
    if exact_match:
        return exact_match

    lowered_pattern = model_pattern.lower()
    matches = [
        model
        for model in available_models
        if lowered_pattern in model.id.lower() or (model.name and lowered_pattern in model.name.lower())
    ]
    if not matches:
        return None

    aliases = [model for model in matches if _is_alias(model.id)]
    dated_versions = [model for model in matches if not _is_alias(model.id)]

    if aliases:
        aliases.sort(key=lambda m: m.id, reverse=True)
        return aliases[0]
    dated_versions.sort(key=lambda m: m.id, reverse=True)
    return dated_versions[0]


@dataclass
class ParsedModelResult:
    model: Model | None
    thinking_level: ThinkingLevel | None = None
    warning: str | None = None


def _build_fallback_model(provider: str, model_id: str, available_models: list[Model]) -> Model | None:
    provider_models = [m for m in available_models if m.provider == provider]
    if not provider_models:
        return None

    default_id = DEFAULT_MODEL_PER_PROVIDER.get(provider)
    base_model = next((m for m in provider_models if m.id == default_id), None) if default_id else None
    base_model = base_model or provider_models[0]

    return replace(base_model, id=model_id, name=model_id)


def parse_model_pattern(
    pattern: str,
    available_models: list[Model],
    *,
    allow_invalid_thinking_level_fallback: bool = True,
) -> ParsedModelResult:
    """Parse a pattern to extract a model and an optional trailing ``:level``.

    Handles model ids that themselves contain colons (e.g. OpenRouter's
    ``:exacto`` suffix) by trying the full pattern first, then progressively
    stripping colon-suffixes.
    """
    exact_match = _try_match_model(pattern, available_models)
    if exact_match:
        return ParsedModelResult(model=exact_match)

    last_colon_index = pattern.rfind(":")
    if last_colon_index == -1:
        return ParsedModelResult(model=None)

    prefix = pattern[:last_colon_index]
    suffix = pattern[last_colon_index + 1 :]

    if is_valid_thinking_level(suffix):
        result = parse_model_pattern(
            prefix, available_models, allow_invalid_thinking_level_fallback=allow_invalid_thinking_level_fallback
        )
        if result.model:
            return ParsedModelResult(
                model=result.model,
                thinking_level=None if result.warning else suffix,
                warning=result.warning,
            )
        return result

    if not allow_invalid_thinking_level_fallback:
        # Strict mode (CLI --model parsing): treat the suffix as part of the model id.
        return ParsedModelResult(model=None)

    result = parse_model_pattern(
        prefix, available_models, allow_invalid_thinking_level_fallback=allow_invalid_thinking_level_fallback
    )
    if result.model:
        return ParsedModelResult(
            model=result.model,
            thinking_level=None,
            warning=f'Invalid thinking level "{suffix}" in pattern "{pattern}". Using default instead.',
        )
    return result


@dataclass
class ModelScopeDiagnostic:
    code: str
    """"no-match" or "invalid-thinking-level"."""
    message: str
    pattern: str
    type: str = "warning"


@dataclass
class ResolveModelScopeResult:
    scoped_models: list[ScopedModel]
    diagnostics: list[ModelScopeDiagnostic]


def _matches_glob(text: str, pattern: str) -> bool:
    return fnmatch.fnmatch(text.lower(), pattern.lower())


def resolve_model_scope_from_models(patterns: list[str], models: list[Model]) -> ResolveModelScopeResult:
    """Resolve `--models`-style patterns (globs and `pattern:level` suffixes) to models."""
    available_models = list(models)
    scoped_models: list[ScopedModel] = []
    diagnostics: list[ModelScopeDiagnostic] = []

    def already_scoped(model: Model) -> bool:
        return any(models_are_equal(sm.model, model) for sm in scoped_models)

    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            colon_idx = pattern.rfind(":")
            glob_pattern = pattern
            thinking_level: str | None = None
            if colon_idx != -1:
                suffix = pattern[colon_idx + 1 :]
                if is_valid_thinking_level(suffix):
                    thinking_level = suffix
                    glob_pattern = pattern[:colon_idx]

            exact_match = find_exact_model_reference_match(glob_pattern, available_models)
            if exact_match:
                if not already_scoped(exact_match):
                    scoped_models.append(ScopedModel(model=exact_match, thinking_level=thinking_level))
                continue

            matching_models = [
                m
                for m in available_models
                if _matches_glob(f"{m.provider}/{m.id}", glob_pattern) or _matches_glob(m.id, glob_pattern)
            ]
            if not matching_models:
                diagnostics.append(
                    ModelScopeDiagnostic(
                        code="no-match", message=f'No models match pattern "{pattern}"', pattern=pattern
                    )
                )
                continue

            for model in matching_models:
                if not already_scoped(model):
                    scoped_models.append(ScopedModel(model=model, thinking_level=thinking_level))
            continue

        result = parse_model_pattern(pattern, available_models)

        if result.warning:
            diagnostics.append(
                ModelScopeDiagnostic(code="invalid-thinking-level", message=result.warning, pattern=pattern)
            )

        if not result.model:
            diagnostics.append(
                ModelScopeDiagnostic(code="no-match", message=f'No models match pattern "{pattern}"', pattern=pattern)
            )
            continue

        if not already_scoped(result.model):
            scoped_models.append(ScopedModel(model=result.model, thinking_level=result.thinking_level))

    return ResolveModelScopeResult(scoped_models=scoped_models, diagnostics=diagnostics)


async def resolve_model_scope_with_diagnostics(
    patterns: list[str], model_runtime: ModelAvailabilitySource
) -> ResolveModelScopeResult:
    """Port of `resolveModelScopeWithDiagnostics`.

    Scoping matches only *available* models -- ones whose provider has
    configured auth -- not the whole catalog, so `--models sonnet` reports
    "No models match pattern" when Anthropic is not authenticated.
    """
    return resolve_model_scope_from_models(patterns, await model_runtime.get_available())


async def resolve_model_scope(patterns: list[str], model_runtime: ModelAvailabilitySource) -> list[ScopedModel]:
    result = await resolve_model_scope_with_diagnostics(patterns, model_runtime)
    for diagnostic in result.diagnostics:
        print(f"Warning: {diagnostic.message}", file=sys.stderr)
    return result.scoped_models


class ModelAvailabilitySource(Protocol):
    """The one method `resolve_model_scope*` needs from a `ModelRuntime`."""

    async def get_available(self, provider_id: str | None = None) -> list[Model]: ...


class ModelSource(Protocol):
    """Minimal surface `resolve_cli_model`/`find_initial_model` need from a model catalog."""

    def get_models(self) -> list[Model]: ...

    def get_model(self, provider: str, model_id: str) -> Model | None: ...

    def has_configured_auth(self, provider: str) -> bool: ...

    def get_available_snapshot(self) -> list[Model]: ...


@dataclass
class ModelsAuthSource:
    """Adapts `pi_ai.registry.Models` to `ModelSource` via a synchronous env-var auth check."""

    models: Models

    def get_models(self) -> list[Model]:
        return self.models.get_models()

    def get_model(self, provider: str, model_id: str) -> Model | None:
        return self.models.get_model(provider, model_id)

    def has_configured_auth(self, provider: str) -> bool:
        provider_obj = self.models.get_provider(provider)
        if provider_obj is None:
            return False
        api_key = provider_obj.auth.api_key
        if api_key is None:
            return False
        return any(os.environ.get(env_var) for env_var in api_key.env_vars)

    def get_available_snapshot(self) -> list[Model]:
        return [m for m in self.models.get_models() if self.has_configured_auth(m.provider)]


@dataclass
class ResolveCliModelResult:
    model: Model | None
    thinking_level: ThinkingLevel | None = None
    warning: str | None = None
    error: str | None = None
    """Set (with model=None) when the CLI reference could not be resolved."""


def resolve_cli_model(
    model_source: ModelSource,
    *,
    cli_provider: str | None = None,
    cli_model: str | None = None,
    cli_thinking: ThinkingLevel | None = None,
) -> ResolveCliModelResult:
    """Resolve a single model from CLI flags.

    Supports ``--provider <provider> --model <pattern>``, ``--model
    <provider>/<pattern>``, and fuzzy matching (same rules as model scoping).
    Does not itself apply the thinking level, but may parse and return one
    from ``<pattern>:<thinking>`` for the caller to apply.
    """
    if not cli_model:
        return ResolveCliModelResult(model=None)

    # Use *all* models here, not just ones with pre-configured auth: this allows
    # "--api-key" to be used for first-time setup.
    available_models = model_source.get_models()
    if not available_models:
        return ResolveCliModelResult(
            model=None, error="No models available. Check your installation or add models to models.json."
        )

    provider_map: dict[str, str] = {}
    for m in available_models:
        provider_map[m.provider.lower()] = m.provider

    provider = provider_map.get(cli_provider.lower()) if cli_provider else None
    if cli_provider and not provider:
        return ResolveCliModelResult(
            model=None,
            error=f'Unknown provider "{cli_provider}". Use --list-models to see available providers/models.',
        )

    pattern = cli_model
    inferred_provider = False

    if not provider and "/" in cli_model:
        maybe_provider, _, rest = cli_model.partition("/")
        canonical = provider_map.get(maybe_provider.lower())
        if canonical:
            provider = canonical
            pattern = rest
            inferred_provider = True

    if not provider:
        lower = cli_model.lower()
        exact_matches = [
            m for m in available_models if m.id.lower() == lower or f"{m.provider}/{m.id}".lower() == lower
        ]
        if len(exact_matches) == 1:
            return ResolveCliModelResult(model=exact_matches[0])
        if len(exact_matches) > 1:
            authenticated_exact_matches = [m for m in exact_matches if model_source.has_configured_auth(m.provider)]
            if len(authenticated_exact_matches) == 1:
                return ResolveCliModelResult(model=authenticated_exact_matches[0])

            matches = ", ".join(sorted(f"{m.provider}/{m.id}" for m in exact_matches))
            auth_hint = (
                "No matching provider is authenticated."
                if not authenticated_exact_matches
                else "More than one matching provider is authenticated."
            )
            return ResolveCliModelResult(
                model=None,
                error=(
                    f'Model "{cli_model}" is ambiguous across providers: {matches}. {auth_hint} '
                    "Use --provider or provider/model."
                ),
            )

    if cli_provider and provider:
        prefix = f"{provider}/"
        if cli_model.lower().startswith(prefix.lower()):
            pattern = cli_model[len(prefix) :]

    candidates = [m for m in available_models if m.provider == provider] if provider else available_models
    result = parse_model_pattern(pattern, candidates, allow_invalid_thinking_level_fallback=False)

    if result.model:
        if inferred_provider:
            raw_exact_matches = [
                m
                for m in available_models
                if m.id.lower() == cli_model.lower() and not models_are_equal(m, result.model)
            ]
            if raw_exact_matches and not model_source.has_configured_auth(result.model.provider):
                authenticated_raw_matches = [
                    m for m in raw_exact_matches if model_source.has_configured_auth(m.provider)
                ]
                if len(authenticated_raw_matches) == 1:
                    return ResolveCliModelResult(model=authenticated_raw_matches[0])
        return ResolveCliModelResult(model=result.model, thinking_level=result.thinking_level, warning=result.warning)

    if inferred_provider:
        lower = cli_model.lower()
        exact = next(
            (m for m in available_models if m.id.lower() == lower or f"{m.provider}/{m.id}".lower() == lower), None
        )
        if exact:
            return ResolveCliModelResult(model=exact)
        fallback = parse_model_pattern(cli_model, available_models, allow_invalid_thinking_level_fallback=False)
        if fallback.model:
            return ResolveCliModelResult(
                model=fallback.model, thinking_level=fallback.thinking_level, warning=fallback.warning
            )

    if provider:
        fallback_pattern = pattern
        fallback_thinking: str | None = None
        if not cli_thinking:
            last_colon = pattern.rfind(":")
            if last_colon != -1:
                suffix = pattern[last_colon + 1 :]
                if is_valid_thinking_level(suffix):
                    fallback_pattern = pattern[:last_colon]
                    fallback_thinking = suffix

        fallback_model = _build_fallback_model(provider, fallback_pattern, available_models)
        if fallback_model:
            requested_thinking = cli_thinking or fallback_thinking
            model = (
                replace(fallback_model, reasoning=True) if requested_thinking not in (None, "off") else fallback_model
            )
            fallback_warning = (
                f'{result.warning} Model "{fallback_pattern}" not found for provider "{provider}". '
                "Using custom model id."
                if result.warning
                else f'Model "{fallback_pattern}" not found for provider "{provider}". Using custom model id.'
            )
            return ResolveCliModelResult(model=model, thinking_level=fallback_thinking, warning=fallback_warning)

    display = f"{provider}/{pattern}" if provider else cli_model
    return ResolveCliModelResult(
        model=None,
        warning=result.warning,
        error=f'Model "{display}" not found. Use --list-models to see available models.',
    )


@dataclass
class InitialModelResult:
    model: Model | None
    thinking_level: ThinkingLevel = DEFAULT_THINKING_LEVEL
    fallback_message: str | None = None


def find_initial_model(
    model_source: ModelSource,
    *,
    cli_provider: str | None = None,
    cli_model: str | None = None,
    scoped_models: list[ScopedModel] | None = None,
    is_continuing: bool = False,
    default_provider: str | None = None,
    default_model_id: str | None = None,
    default_thinking_level: ThinkingLevel | None = None,
) -> InitialModelResult:
    """Find the initial model, in priority order:

    1. CLI args (provider + model)
    2. First model from scoped models (unless continuing/resuming a session)
    3. Saved default from settings, if still authenticated
    4. First available model with valid auth (preferring the known-provider default)
    5. Nothing found
    """
    scoped_models = scoped_models or []

    if cli_provider and cli_model:
        resolved = resolve_cli_model(model_source, cli_provider=cli_provider, cli_model=cli_model)
        if resolved.error:
            raise SystemExit(resolved.error)
        if resolved.model:
            return InitialModelResult(model=resolved.model, thinking_level=DEFAULT_THINKING_LEVEL)

    if scoped_models and not is_continuing:
        first = scoped_models[0]
        return InitialModelResult(
            model=first.model, thinking_level=first.thinking_level or default_thinking_level or DEFAULT_THINKING_LEVEL
        )

    if default_provider and default_model_id:
        found = model_source.get_model(default_provider, default_model_id)
        if found and model_source.has_configured_auth(found.provider):
            return InitialModelResult(model=found, thinking_level=default_thinking_level or DEFAULT_THINKING_LEVEL)

    available_models = model_source.get_available_snapshot()
    if available_models:
        for provider, default_id in DEFAULT_MODEL_PER_PROVIDER.items():
            match = next((m for m in available_models if m.provider == provider and m.id == default_id), None)
            if match:
                return InitialModelResult(model=match, thinking_level=DEFAULT_THINKING_LEVEL)
        return InitialModelResult(model=available_models[0], thinking_level=DEFAULT_THINKING_LEVEL)

    return InitialModelResult(model=None, thinking_level=DEFAULT_THINKING_LEVEL)


def restore_model_from_session(
    model_source: ModelSource,
    saved_provider: str,
    saved_model_id: str,
    current_model: Model | None,
    *,
    print_messages: bool = False,
) -> tuple[Model | None, str | None]:
    """Restore a model saved in a session, falling back when it's gone or unauthenticated.

    Returns ``(model, fallback_message)``.
    """
    restored_model = model_source.get_model(saved_provider, saved_model_id)
    has_configured_auth = model_source.has_configured_auth(restored_model.provider) if restored_model else False

    if restored_model and has_configured_auth:
        if print_messages:
            print(f"Restored model: {saved_provider}/{saved_model_id}")
        return restored_model, None

    reason = "model no longer exists" if not restored_model else "no auth configured"
    if print_messages:
        print(f"Warning: Could not restore model {saved_provider}/{saved_model_id} ({reason}).", file=sys.stderr)

    if current_model:
        if print_messages:
            print(f"Falling back to: {current_model.provider}/{current_model.id}")
        return current_model, (
            f"Could not restore model {saved_provider}/{saved_model_id} ({reason}). "
            f"Using {current_model.provider}/{current_model.id}."
        )

    available_models = model_source.get_available_snapshot()
    if available_models:
        fallback_model: Model | None = None
        for provider, default_id in DEFAULT_MODEL_PER_PROVIDER.items():
            match = next((m for m in available_models if m.provider == provider and m.id == default_id), None)
            if match:
                fallback_model = match
                break
        fallback_model = fallback_model or available_models[0]

        if print_messages:
            print(f"Falling back to: {fallback_model.provider}/{fallback_model.id}")
        return fallback_model, (
            f"Could not restore model {saved_provider}/{saved_model_id} ({reason}). "
            f"Using {fallback_model.provider}/{fallback_model.id}."
        )

    return None, None


__all__ = [
    "DEFAULT_MODEL_PER_PROVIDER",
    "DEFAULT_THINKING_LEVEL",
    "VALID_THINKING_LEVELS",
    "InitialModelResult",
    "ModelAvailabilitySource",
    "ModelScopeDiagnostic",
    "ModelSource",
    "ModelsAuthSource",
    "ParsedModelResult",
    "ResolveCliModelResult",
    "ResolveModelScopeResult",
    "ScopedModel",
    "ThinkingLevel",
    "find_exact_model_reference_match",
    "find_initial_model",
    "is_valid_thinking_level",
    "parse_model_pattern",
    "resolve_cli_model",
    "resolve_model_scope",
    "resolve_model_scope_from_models",
    "resolve_model_scope_with_diagnostics",
    "restore_model_from_session",
]
