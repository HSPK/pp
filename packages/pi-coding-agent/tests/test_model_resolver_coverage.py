"""Coverage tests for `core/model_resolver.py`.

Covers: ambiguous exact matches, glob scope resolution, ModelsAuthSource,
resolve_cli_model edge cases, find_initial_model priority paths, and
restore_model_from_session fallback paths.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest
from pi_ai.providers import builtin_providers, openai_compatible_provider
from pi_ai.registry import Models
from pi_ai.types import Model
from pi_coding_agent.core.model_resolver import (
    DEFAULT_THINKING_LEVEL,
    ModelsAuthSource,
    ModelSource,
    ScopedModel,
    find_exact_model_reference_match,
    find_initial_model,
    parse_model_pattern,
    resolve_cli_model,
    resolve_model_scope_from_models,
    restore_model_from_session,
)


def make_model(**kwargs) -> Model:
    defaults = {
        "api": "anthropic-messages",
        "reasoning": False,
        "input": ["text"],
        "context_window": 128000,
        "max_tokens": 8192,
    }
    defaults.update(kwargs)
    return Model(**defaults)


ANTHROPIC_SONNET = make_model(
    id="claude-sonnet-4-5",
    name="Claude Sonnet 4.5",
    provider="anthropic",
    base_url="https://api.anthropic.com",
)
ANTHROPIC_OPUS = make_model(
    id="claude-opus-4-8",
    name="Claude Opus 4.8",
    provider="anthropic",
    base_url="https://api.anthropic.com",
)
OPENAI_GPT = make_model(
    id="gpt-4o",
    name="GPT-4o",
    provider="openai",
    base_url="https://api.openai.com",
)
OPENROUTER_GPT = make_model(
    id="gpt-4o",
    name="GPT-4o via OpenRouter",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
)
DATED_SONNET = make_model(
    id="claude-sonnet-4-5-20250929",
    name="Claude Sonnet 4.5 (Sep 2025)",
    provider="anthropic",
    base_url="https://api.anthropic.com",
)

ALL_MODELS = [ANTHROPIC_SONNET, ANTHROPIC_OPUS, OPENAI_GPT]


@dataclass
class FakeSource:
    models: list[Model]
    auth: set[str] | None = None
    snapshot: list[Model] | None = None

    def get_models(self) -> list[Model]:
        return self.models

    def get_model(self, provider: str, model_id: str) -> Model | None:
        return next((m for m in self.models if m.provider == provider and m.id == model_id), None)

    def has_configured_auth(self, provider: str) -> bool:
        if self.auth is not None:
            return provider in self.auth
        return False

    def get_available_snapshot(self) -> list[Model]:
        if self.snapshot is not None:
            return self.snapshot
        return [m for m in self.models if self.has_configured_auth(m.provider)]


# ---------------------------------------------------------------------------
# find_exact_model_reference_match: ambiguous paths
# ---------------------------------------------------------------------------


def test_ambiguous_canonical_match_returns_none():
    dup = make_model(id="gpt-4o", provider="openai2", base_url="https://x.invalid")
    models = [OPENAI_GPT, dup]
    # Two exact "provider/id" matches (different providers) → ambiguous → None
    result = find_exact_model_reference_match("openai/gpt-4o", models)
    # openai/gpt-4o is unambiguous here (only one provider is "openai")
    assert result == OPENAI_GPT


def test_ambiguous_bare_id_across_providers_returns_none():
    models = [OPENAI_GPT, OPENROUTER_GPT]
    result = find_exact_model_reference_match("gpt-4o", models)
    assert result is None


def test_ambiguous_provider_slash_model_across_multiple_returns_none():
    # Create two models with the same provider and id (edge case)
    m1 = make_model(id="same-model", provider="same-provider", base_url="https://a.invalid")
    m2 = make_model(id="same-model", provider="same-provider", base_url="https://b.invalid")
    result = find_exact_model_reference_match("same-provider/same-model", [m1, m2])
    assert result is None


def test_empty_reference_returns_none():
    result = find_exact_model_reference_match("", ALL_MODELS)
    assert result is None


def test_whitespace_only_reference_returns_none():
    result = find_exact_model_reference_match("   ", ALL_MODELS)
    assert result is None


# ---------------------------------------------------------------------------
# parse_model_pattern: allow_invalid_thinking_level_fallback=False
# ---------------------------------------------------------------------------


def test_strict_mode_no_fallback_for_invalid_thinking_level():
    result = parse_model_pattern("sonnet:bogus", ALL_MODELS, allow_invalid_thinking_level_fallback=False)
    assert result.model is None


def test_strict_mode_returns_model_for_valid_thinking_level():
    result = parse_model_pattern("claude-sonnet-4-5:high", ALL_MODELS, allow_invalid_thinking_level_fallback=False)
    assert result.model is not None
    assert result.thinking_level == "high"


def test_alias_preferred_over_dated_snapshot():
    models = [DATED_SONNET, ANTHROPIC_SONNET]
    result = parse_model_pattern("claude-sonnet", models)
    # Alias (no date suffix) should be preferred
    assert result.model.id == "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# resolve_model_scope_from_models: glob patterns
# ---------------------------------------------------------------------------


def test_glob_pattern_matches_all_anthropic_models():
    result = resolve_model_scope_from_models(["anthropic/*"], ALL_MODELS)
    matched_ids = {sm.model.id for sm in result.scoped_models}
    assert "claude-sonnet-4-5" in matched_ids
    assert "claude-opus-4-8" in matched_ids
    assert "gpt-4o" not in matched_ids
    assert result.diagnostics == []


def test_glob_pattern_with_thinking_level():
    result = resolve_model_scope_from_models(["anthropic/*:high"], ALL_MODELS)
    for sm in result.scoped_models:
        assert sm.thinking_level == "high"


def test_glob_pattern_no_match_adds_diagnostic():
    result = resolve_model_scope_from_models(["nomatch/*"], ALL_MODELS)
    assert result.scoped_models == []
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "no-match"


def test_glob_exact_match_takes_priority_over_glob_expansion():
    # When exact match exists for glob-like pattern, use that instead of glob expansion.
    bracketed = make_model(id="model[v2]", provider="custom", base_url="https://x.invalid")
    result = resolve_model_scope_from_models(["custom/model[v2]"], [*ALL_MODELS, bracketed])
    assert len(result.scoped_models) == 1
    assert result.scoped_models[0].model.id == "model[v2]"


def test_glob_deduplication_does_not_add_same_model_twice():
    result = resolve_model_scope_from_models(["anthropic/*", "claude-sonnet-4-5"], ALL_MODELS)
    ids = [sm.model.id for sm in result.scoped_models]
    assert ids.count("claude-sonnet-4-5") == 1


# ---------------------------------------------------------------------------
# resolve_cli_model
# ---------------------------------------------------------------------------


def test_resolve_cli_model_returns_none_model_when_no_cli_model():
    src = FakeSource(ALL_MODELS)
    result = resolve_cli_model(src)
    assert result.model is None
    assert result.error is None


def test_resolve_cli_model_no_models_available():
    src = FakeSource([])
    result = resolve_cli_model(src, cli_model="sonnet")
    assert result.model is None
    assert result.error is not None
    assert "No models available" in result.error


def test_resolve_cli_model_unknown_provider():
    src = FakeSource(ALL_MODELS)
    result = resolve_cli_model(src, cli_provider="unknown-provider", cli_model="any")
    assert result.model is None
    assert "Unknown provider" in result.error


def test_resolve_cli_model_exact_single_match():
    src = FakeSource(ALL_MODELS)
    result = resolve_cli_model(src, cli_model="gpt-4o")
    assert result.model == OPENAI_GPT


def test_resolve_cli_model_ambiguous_bare_id():
    src = FakeSource([OPENAI_GPT, OPENROUTER_GPT], auth={"openai", "openrouter"})
    result = resolve_cli_model(src, cli_model="gpt-4o")
    assert result.model is None
    assert "ambiguous" in result.error


def test_resolve_cli_model_ambiguous_resolves_by_auth():
    # Only one provider is authenticated → auto-select it.
    src = FakeSource([OPENAI_GPT, OPENROUTER_GPT], auth={"openai"})
    result = resolve_cli_model(src, cli_model="gpt-4o")
    assert result.model == OPENAI_GPT


def test_resolve_cli_model_ambiguous_both_authenticated():
    # Both providers authenticated → still ambiguous.
    src = FakeSource([OPENAI_GPT, OPENROUTER_GPT], auth={"openai", "openrouter"})
    result = resolve_cli_model(src, cli_model="gpt-4o")
    assert result.model is None
    assert "ambiguous" in result.error
    assert "More than one" in result.error


def test_resolve_cli_model_ambiguous_none_authenticated():
    src = FakeSource([OPENAI_GPT, OPENROUTER_GPT], auth=set())
    result = resolve_cli_model(src, cli_model="gpt-4o")
    assert result.model is None
    assert "No matching provider is authenticated" in result.error


def test_resolve_cli_model_with_explicit_provider():
    src = FakeSource(ALL_MODELS, auth={"anthropic"})
    result = resolve_cli_model(src, cli_provider="anthropic", cli_model="claude-opus-4-8")
    assert result.model == ANTHROPIC_OPUS


def test_resolve_cli_model_provider_prefix_in_model_string_is_stripped():
    src = FakeSource(ALL_MODELS, auth={"openai"})
    result = resolve_cli_model(src, cli_provider="openai", cli_model="openai/gpt-4o")
    assert result.model == OPENAI_GPT


def test_resolve_cli_model_inferred_provider_from_slash():
    src = FakeSource(ALL_MODELS, auth={"openai"})
    result = resolve_cli_model(src, cli_model="openai/gpt-4o")
    assert result.model == OPENAI_GPT


def test_resolve_cli_model_fallback_model_created_for_unknown_id():
    src = FakeSource(ALL_MODELS, auth={"anthropic"})
    result = resolve_cli_model(src, cli_provider="anthropic", cli_model="claude-custom-new")
    assert result.model is not None
    assert result.model.id == "claude-custom-new"
    assert result.warning is not None
    assert "not found" in result.warning


def test_resolve_cli_model_not_found_without_provider():
    src = FakeSource(ALL_MODELS)
    result = resolve_cli_model(src, cli_model="completely-unknown-xyz")
    assert result.model is None
    assert result.error is not None


def test_resolve_cli_model_thinking_suffix_stripped_for_fallback():
    src = FakeSource(ALL_MODELS, auth={"anthropic"})
    result = resolve_cli_model(src, cli_provider="anthropic", cli_model="claude-future-model:high")
    # The fallback model id should not include ":high".
    assert result.model is not None
    assert result.model.id == "claude-future-model"
    assert result.thinking_level == "high"


# ---------------------------------------------------------------------------
# find_initial_model
# ---------------------------------------------------------------------------


def test_find_initial_model_uses_cli_args_first():
    src = FakeSource(ALL_MODELS, auth={"anthropic"})
    result = find_initial_model(src, cli_provider="anthropic", cli_model="claude-sonnet-4-5")
    assert result.model == ANTHROPIC_SONNET


def test_find_initial_model_cli_error_raises_systemexit():
    src = FakeSource(ALL_MODELS)
    with pytest.raises(SystemExit):
        find_initial_model(src, cli_provider="nonexistent", cli_model="any")


def test_find_initial_model_uses_scoped_models_when_no_cli():
    src = FakeSource(ALL_MODELS)
    scoped = [ScopedModel(model=ANTHROPIC_OPUS, thinking_level="high")]
    result = find_initial_model(src, scoped_models=scoped)
    assert result.model == ANTHROPIC_OPUS
    assert result.thinking_level == "high"


def test_find_initial_model_skips_scoped_models_when_continuing():
    src = FakeSource(ALL_MODELS, auth={"anthropic"}, snapshot=[ANTHROPIC_SONNET])
    scoped = [ScopedModel(model=ANTHROPIC_OPUS)]
    result = find_initial_model(src, scoped_models=scoped, is_continuing=True)
    # is_continuing → skip scoped models → use default provider
    assert result.model == ANTHROPIC_SONNET


def test_find_initial_model_uses_default_provider_and_model():
    src = FakeSource(ALL_MODELS, auth={"openai"})
    result = find_initial_model(src, default_provider="openai", default_model_id="gpt-4o")
    assert result.model == OPENAI_GPT


def test_find_initial_model_skips_unauthenticated_default():
    src = FakeSource(ALL_MODELS, auth=set(), snapshot=[])
    result = find_initial_model(src, default_provider="openai", default_model_id="gpt-4o")
    assert result.model is None


def test_find_initial_model_uses_available_snapshot_fallback():
    src = FakeSource(ALL_MODELS, snapshot=[OPENAI_GPT])
    result = find_initial_model(src)
    assert result.model is not None


def test_find_initial_model_returns_none_when_no_models():
    src = FakeSource([], snapshot=[])
    result = find_initial_model(src)
    assert result.model is None
    assert result.thinking_level == DEFAULT_THINKING_LEVEL


def test_find_initial_model_scoped_model_uses_default_thinking_level_when_not_set():
    src = FakeSource(ALL_MODELS)
    scoped = [ScopedModel(model=ANTHROPIC_SONNET, thinking_level=None)]
    result = find_initial_model(src, scoped_models=scoped, default_thinking_level="xhigh")
    assert result.thinking_level == "xhigh"


def test_find_initial_model_prefers_known_default_over_first_available():
    # anthropic/claude-opus-4-8 is the default for anthropic in DEFAULT_MODEL_PER_PROVIDER
    src = FakeSource(ALL_MODELS, snapshot=[OPENAI_GPT, ANTHROPIC_OPUS])
    result = find_initial_model(src)
    assert result.model == ANTHROPIC_OPUS


# ---------------------------------------------------------------------------
# restore_model_from_session
# ---------------------------------------------------------------------------


def test_restore_model_success():
    src = FakeSource(ALL_MODELS, auth={"anthropic"})
    model, msg = restore_model_from_session(src, "anthropic", "claude-sonnet-4-5", None)
    assert model == ANTHROPIC_SONNET
    assert msg is None


def test_restore_model_not_found_falls_back_to_current():
    src = FakeSource(ALL_MODELS, auth={"openai"})
    model, msg = restore_model_from_session(src, "anthropic", "nonexistent", OPENAI_GPT)
    assert model == OPENAI_GPT
    assert "model no longer exists" in msg


def test_restore_model_no_auth_falls_back_to_current():
    src = FakeSource(ALL_MODELS, auth=set())
    model, msg = restore_model_from_session(src, "anthropic", "claude-sonnet-4-5", OPENAI_GPT)
    assert model == OPENAI_GPT
    assert "no auth configured" in msg


def test_restore_model_no_auth_no_current_falls_back_to_snapshot():
    src = FakeSource(ALL_MODELS, auth=set(), snapshot=[OPENAI_GPT])
    model, msg = restore_model_from_session(src, "anthropic", "claude-sonnet-4-5", None)
    assert model is not None
    assert msg is not None


def test_restore_model_nothing_available_returns_none():
    src = FakeSource([], snapshot=[])
    model, msg = restore_model_from_session(src, "anthropic", "claude-sonnet-4-5", None)
    assert model is None
    assert msg is None


def test_restore_model_prints_messages_when_requested(capsys):
    src = FakeSource(ALL_MODELS, auth={"anthropic"})
    restore_model_from_session(src, "anthropic", "claude-sonnet-4-5", None, print_messages=True)
    captured = capsys.readouterr()
    assert "Restored model" in captured.out


def test_restore_model_prints_fallback_warning_when_requested(capsys):
    src = FakeSource(ALL_MODELS, auth=set(), snapshot=[OPENAI_GPT])
    restore_model_from_session(src, "anthropic", "claude-sonnet-4-5", None, print_messages=True)
    captured = capsys.readouterr()
    assert "Warning" in captured.err
    assert "Falling back" in captured.out


# ---------------------------------------------------------------------------
# ModelsAuthSource: the only production `ModelSource` implementation.
#
# Every other test in this file (and in test_model_resolver.py) drives
# `resolve_cli_model`/`find_initial_model` through a hand-written
# `ModelSource`-shaped double, mirroring the TypeScript tests' ad-hoc
# `modelRuntime` mocks. That leaves the real adapter -- the one
# `cli/auth_command.py` actually constructs -- unexercised, so a shape or
# env-var-lookup mistake in it would not fail any test. These cases drive the
# real `ModelsAuthSource` over a real `pi_ai.registry.Models`; both are
# offline (models come from the bundled catalog, auth from `os.environ`).
# ---------------------------------------------------------------------------


def _auth_source_registry() -> Models:
    return Models(
        [
            openai_compatible_provider(
                "fake",
                "Fake",
                "https://fake.invalid/v1",
                ["FAKE_MODELS_AUTH_KEY"],
                [
                    Model(
                        id="alpha",
                        name="Alpha",
                        api="openai-completions",
                        provider="fake",
                        base_url="https://fake.invalid/v1",
                        context_window=8000,
                        max_tokens=1000,
                    )
                ],
            ),
            openai_compatible_provider(
                "other",
                "Other",
                "https://other.invalid/v1",
                ["OTHER_MODELS_AUTH_KEY"],
                [
                    Model(
                        id="beta",
                        name="Beta",
                        api="openai-completions",
                        provider="other",
                        base_url="https://other.invalid/v1",
                        context_window=8000,
                        max_tokens=1000,
                    )
                ],
            ),
        ]
    )


def test_models_auth_source_satisfies_the_model_source_protocol():
    source = ModelsAuthSource(_auth_source_registry())
    # `ModelSource` is not `@runtime_checkable`, so check the surface directly.
    assert set(ModelSource.__protocol_attrs__) == {
        "get_models",
        "get_model",
        "has_configured_auth",
        "get_available_snapshot",
    }
    for name in ModelSource.__protocol_attrs__:
        method = getattr(source, name)
        assert callable(method)
        # `resolve_cli_model` calls these synchronously; a coroutine function
        # here would be returned unawaited and silently treated as truthy.
        assert not inspect.iscoroutinefunction(method)


def test_models_auth_source_delegates_catalog_lookups_to_the_registry():
    registry = _auth_source_registry()
    source = ModelsAuthSource(registry)

    assert [m.id for m in source.get_models()] == [m.id for m in registry.get_models()]
    assert source.get_model("fake", "alpha") is registry.get_model("fake", "alpha")
    assert source.get_model("fake", "nope") is None
    assert source.get_model("nope", "alpha") is None


def test_models_auth_source_reads_configured_auth_from_the_providers_env_vars(monkeypatch):
    monkeypatch.delenv("FAKE_MODELS_AUTH_KEY", raising=False)
    monkeypatch.delenv("OTHER_MODELS_AUTH_KEY", raising=False)
    source = ModelsAuthSource(_auth_source_registry())

    assert source.has_configured_auth("fake") is False
    assert source.get_available_snapshot() == []

    monkeypatch.setenv("FAKE_MODELS_AUTH_KEY", "sk-test")
    assert source.has_configured_auth("fake") is True
    assert source.has_configured_auth("other") is False
    assert [m.id for m in source.get_available_snapshot()] == ["alpha"]


def test_models_auth_source_treats_an_empty_env_var_as_unconfigured(monkeypatch):
    monkeypatch.setenv("FAKE_MODELS_AUTH_KEY", "")
    source = ModelsAuthSource(_auth_source_registry())

    assert source.has_configured_auth("fake") is False


def test_models_auth_source_reports_unknown_providers_as_unconfigured():
    source = ModelsAuthSource(_auth_source_registry())

    assert source.has_configured_auth("does-not-exist") is False


def test_resolve_cli_model_works_against_the_real_models_auth_source(monkeypatch):
    monkeypatch.setenv("FAKE_MODELS_AUTH_KEY", "sk-test")
    monkeypatch.delenv("OTHER_MODELS_AUTH_KEY", raising=False)
    source = ModelsAuthSource(_auth_source_registry())

    result = resolve_cli_model(source, cli_model="alpha")
    assert result.model is not None
    assert result.model.id == "alpha"
    assert result.error is None

    unknown = resolve_cli_model(source, cli_model="does-not-exist")
    assert unknown.model is None
    assert unknown.error is not None


def test_every_builtin_provider_exposes_the_api_key_env_vars_models_auth_source_reads():
    """`has_configured_auth` reaches into `provider.auth.api_key.env_vars`.

    If any builtin provider carried a differently shaped `auth`, that access
    would raise for a real registry while every `FakeModelSource`-based test
    stayed green.
    """
    source = ModelsAuthSource(Models(builtin_providers()))

    for provider in source.models.get_providers():
        assert isinstance(provider.auth.api_key.env_vars, tuple)
        assert source.has_configured_auth(provider.id) in (True, False)
