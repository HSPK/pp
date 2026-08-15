"""Tests for `core/model_resolver.py`.

Ported from `test/model-resolver.test.ts` in the TypeScript coding-agent
package: `parseModelPattern`, `resolveModelScopeWithDiagnostics`,
`resolveCliModel` (including the `#5552` custom-model `:thinking`-suffix
regression cases), and `findInitialModel`/default-model-selection. Fake
`ModelSource`-shaped test doubles replace the TypeScript tests' ad-hoc
`modelRuntime` mocks (see `model_resolver.py`'s module docstring for the
`ModelRuntime` -> `ModelSource` deviation) -- no real network or registry
lookups are used.
"""

from __future__ import annotations

from dataclasses import dataclass

from pi_ai.types import Model

from pi_coding_agent.core.model_resolver import (
    DEFAULT_MODEL_PER_PROVIDER,
    find_initial_model,
    parse_model_pattern,
    resolve_cli_model,
    resolve_model_scope,
    resolve_model_scope_with_diagnostics,
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


CLAUDE_SONNET = make_model(
    id="claude-sonnet-4-5",
    name="Claude Sonnet 4.5",
    provider="anthropic",
    base_url="https://api.anthropic.com",
    reasoning=True,
    input=["text", "image"],
    context_window=200000,
)
GPT_4O = make_model(
    id="gpt-4o",
    name="GPT-4o",
    provider="openai",
    base_url="https://api.openai.com",
    reasoning=False,
    input=["text", "image"],
    max_tokens=4096,
)
QWEN_CODER = make_model(
    id="qwen/qwen3-coder:exacto",
    name="Qwen3 Coder Exacto",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    reasoning=True,
)
GPT_4O_EXTENDED = make_model(
    id="openai/gpt-4o:extended",
    name="GPT-4o Extended",
    provider="openrouter",
    base_url="https://openrouter.ai/api/v1",
    reasoning=False,
    input=["text", "image"],
    max_tokens=4096,
)

MOCK_MODELS = [CLAUDE_SONNET, GPT_4O]
MOCK_OPENROUTER_MODELS = [QWEN_CODER, GPT_4O_EXTENDED]
ALL_MODELS = [*MOCK_MODELS, *MOCK_OPENROUTER_MODELS]


@dataclass
class FakeModelSource:
    """Minimal `ModelSource`-shaped test double (mirrors the TS tests' ad-hoc mocks)."""

    models: list[Model]
    auth_provider: str | None = None
    auth_predicate: object = None
    available_snapshot: list[Model] | None = None
    model_lookup: object = None

    def get_models(self) -> list[Model]:
        return self.models

    def get_model(self, provider: str, model_id: str) -> Model | None:
        if self.model_lookup:
            return self.model_lookup(provider, model_id)
        return next((m for m in self.models if m.provider == provider and m.id == model_id), None)

    def has_configured_auth(self, provider: str) -> bool:
        if self.auth_predicate:
            return self.auth_predicate(provider)
        return provider == self.auth_provider

    def get_available_snapshot(self) -> list[Model]:
        if self.available_snapshot is not None:
            return self.available_snapshot
        return [m for m in self.models if self.has_configured_auth(m.provider)]


class FakeRegistry:
    """Test double satisfying `resolve_model_scope_with_diagnostics`'s runtime param.

    Mirrors the TypeScript tests' `{ getAvailable: () => allModels }` mock: scoping
    matches *available* models (provider has configured auth), not the raw catalog.
    """

    def __init__(self, models: list[Model]) -> None:
        self._models = models

    async def get_available(self, provider_id: str | None = None) -> list[Model]:
        return self._models


# ---------------------------------------------------------------------------
# parseModelPattern
# ---------------------------------------------------------------------------


def test_exact_match_returns_model_with_no_thinking_level():
    result = parse_model_pattern("claude-sonnet-4-5", ALL_MODELS)
    assert result.model.id == "claude-sonnet-4-5"
    assert result.thinking_level is None
    assert result.warning is None


def test_partial_match_returns_best_model():
    result = parse_model_pattern("sonnet", ALL_MODELS)
    assert result.model.id == "claude-sonnet-4-5"
    assert result.thinking_level is None
    assert result.warning is None


def test_no_match_returns_none_model():
    result = parse_model_pattern("nonexistent", ALL_MODELS)
    assert result.model is None
    assert result.thinking_level is None
    assert result.warning is None


def test_sonnet_high_returns_sonnet_with_high_thinking_level():
    result = parse_model_pattern("sonnet:high", ALL_MODELS)
    assert result.model.id == "claude-sonnet-4-5"
    assert result.thinking_level == "high"
    assert result.warning is None


def test_gpt_4o_medium_returns_gpt_4o_with_medium_thinking_level():
    result = parse_model_pattern("gpt-4o:medium", ALL_MODELS)
    assert result.model.id == "gpt-4o"
    assert result.thinking_level == "medium"
    assert result.warning is None


def test_all_valid_thinking_levels_work():
    for level in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
        result = parse_model_pattern(f"sonnet:{level}", ALL_MODELS)
        assert result.model.id == "claude-sonnet-4-5"
        assert result.thinking_level == level
        assert result.warning is None


def test_sonnet_random_returns_sonnet_with_warning():
    result = parse_model_pattern("sonnet:random", ALL_MODELS)
    assert result.model.id == "claude-sonnet-4-5"
    assert result.thinking_level is None
    assert "Invalid thinking level" in result.warning
    assert "random" in result.warning


def test_gpt_4o_invalid_returns_gpt_4o_with_warning():
    result = parse_model_pattern("gpt-4o:invalid", ALL_MODELS)
    assert result.model.id == "gpt-4o"
    assert result.thinking_level is None
    assert "Invalid thinking level" in result.warning


def test_qwen_coder_exacto_matches_model_with_colon_in_id():
    result = parse_model_pattern("qwen/qwen3-coder:exacto", ALL_MODELS)
    assert result.model.id == "qwen/qwen3-coder:exacto"
    assert result.thinking_level is None
    assert result.warning is None


def test_openrouter_prefixed_qwen_matches_with_provider_prefix():
    result = parse_model_pattern("openrouter/qwen/qwen3-coder:exacto", ALL_MODELS)
    assert result.model.id == "qwen/qwen3-coder:exacto"
    assert result.model.provider == "openrouter"
    assert result.thinking_level is None
    assert result.warning is None


def test_qwen_coder_exacto_high_matches_with_thinking_level():
    result = parse_model_pattern("qwen/qwen3-coder:exacto:high", ALL_MODELS)
    assert result.model.id == "qwen/qwen3-coder:exacto"
    assert result.thinking_level == "high"
    assert result.warning is None


def test_openrouter_prefixed_qwen_high_matches_with_provider_and_thinking():
    result = parse_model_pattern("openrouter/qwen/qwen3-coder:exacto:high", ALL_MODELS)
    assert result.model.id == "qwen/qwen3-coder:exacto"
    assert result.model.provider == "openrouter"
    assert result.thinking_level == "high"
    assert result.warning is None


def test_gpt_4o_extended_matches_the_extended_model():
    result = parse_model_pattern("openai/gpt-4o:extended", ALL_MODELS)
    assert result.model.id == "openai/gpt-4o:extended"
    assert result.thinking_level is None
    assert result.warning is None


def test_qwen_coder_exacto_random_returns_model_with_warning():
    result = parse_model_pattern("qwen/qwen3-coder:exacto:random", ALL_MODELS)
    assert result.model.id == "qwen/qwen3-coder:exacto"
    assert result.thinking_level is None
    assert "Invalid thinking level" in result.warning
    assert "random" in result.warning


def test_qwen_coder_exacto_high_random_discards_thinking_level():
    result = parse_model_pattern("qwen/qwen3-coder:exacto:high:random", ALL_MODELS)
    assert result.model.id == "qwen/qwen3-coder:exacto"
    assert result.thinking_level is None
    assert "Invalid thinking level" in result.warning
    assert "random" in result.warning


def test_empty_pattern_matches_via_partial_matching():
    result = parse_model_pattern("", ALL_MODELS)
    assert result.model is not None
    assert result.thinking_level is None


def test_pattern_ending_with_colon_treats_empty_suffix_as_invalid():
    result = parse_model_pattern("sonnet:", ALL_MODELS)
    assert result.model.id == "claude-sonnet-4-5"
    assert "Invalid thinking level" in result.warning


# ---------------------------------------------------------------------------
# resolveModelScopeWithDiagnostics / resolveModelScope
# ---------------------------------------------------------------------------


async def test_resolve_model_scope_with_diagnostics_returns_structured_diagnostics(capsys):
    registry = FakeRegistry(ALL_MODELS)
    result = await resolve_model_scope_with_diagnostics(["sonnet:high", "gpt-4o:invalid", "missing"], registry)

    assert [sm.model.id for sm in result.scoped_models] == ["claude-sonnet-4-5", "gpt-4o"]
    assert result.scoped_models[0].thinking_level == "high"
    assert result.scoped_models[1].thinking_level is None
    assert [(d.type, d.code, d.pattern) for d in result.diagnostics] == [
        ("warning", "invalid-thinking-level", "gpt-4o:invalid"),
        ("warning", "no-match", "missing"),
    ]
    assert (
        result.diagnostics[0].message
        == 'Invalid thinking level "invalid" in pattern "gpt-4o:invalid". Using default instead.'
    )
    assert result.diagnostics[1].message == 'No models match pattern "missing"'
    # TS: `expect(warn).not.toHaveBeenCalled()` -- the diagnostics variant must
    # stay silent so callers own the presentation.
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


async def test_resolve_model_scope_prints_warning_to_stderr(capsys):
    registry = FakeRegistry(ALL_MODELS)
    scoped_models = await resolve_model_scope(["missing"], registry)

    assert scoped_models == []
    captured = capsys.readouterr()
    assert 'Warning: No models match pattern "missing"' in captured.err
    # TS: `expect(warn).toHaveBeenCalledOnce()`.
    assert len([line for line in captured.err.splitlines() if line.strip()]) == 1


async def test_scoping_matches_only_available_models():
    """TS passes `modelRuntime` and resolves against `await modelRuntime.getAvailable()`,
    so a pattern matching only unauthenticated providers reports no match."""
    registry = FakeRegistry([GPT_4O])
    result = await resolve_model_scope_with_diagnostics(["sonnet", "gpt-4o"], registry)

    assert [sm.model.id for sm in result.scoped_models] == ["gpt-4o"]
    assert [(d.code, d.pattern) for d in result.diagnostics] == [("no-match", "sonnet")]


async def test_resolves_bracketed_model_ids_as_exact_references():
    bracketed = make_model(
        id="bracketed-model[1m]",
        name="Bracketed Model",
        provider="custom",
        base_url="https://example.invalid",
        reasoning=True,
    )
    registry = FakeRegistry([*ALL_MODELS, bracketed])
    result = await resolve_model_scope_with_diagnostics(["custom/bracketed-model[1m]"], registry)

    assert [sm.model.id for sm in result.scoped_models] == ["bracketed-model[1m]"]
    assert result.diagnostics == []


async def test_resolves_bracketed_model_ids_with_thinking_level():
    bracketed = make_model(
        id="bracketed-model[1m]",
        name="Bracketed Model",
        provider="custom",
        base_url="https://example.invalid",
        reasoning=True,
    )
    registry = FakeRegistry([*ALL_MODELS, bracketed])
    result = await resolve_model_scope_with_diagnostics(["custom/bracketed-model[1m]:high"], registry)

    assert [sm.model.id for sm in result.scoped_models] == ["bracketed-model[1m]"]
    assert result.scoped_models[0].thinking_level == "high"
    assert result.diagnostics == []


# ---------------------------------------------------------------------------
# resolveCliModel
# ---------------------------------------------------------------------------


def test_resolves_provider_id_without_explicit_provider():
    source = FakeModelSource(ALL_MODELS)
    result = resolve_cli_model(source, cli_model="openai/gpt-4o")
    assert result.error is None
    assert result.model.provider == "openai"
    assert result.model.id == "gpt-4o"


def test_resolves_fuzzy_patterns_within_explicit_provider():
    source = FakeModelSource(ALL_MODELS)
    result = resolve_cli_model(source, cli_provider="openai", cli_model="4o")
    assert result.error is None
    assert result.model.provider == "openai"
    assert result.model.id == "gpt-4o"


def test_supports_model_pattern_thinking_suffix_without_explicit_thinking():
    source = FakeModelSource(ALL_MODELS)
    result = resolve_cli_model(source, cli_model="sonnet:high")
    assert result.error is None
    assert result.model.id == "claude-sonnet-4-5"
    assert result.thinking_level == "high"


def test_prefers_exact_model_id_match_over_provider_inference():
    source = FakeModelSource(ALL_MODELS)
    result = resolve_cli_model(source, cli_model="openai/gpt-4o:extended")
    assert result.error is None
    assert result.model.provider == "openrouter"
    assert result.model.id == "openai/gpt-4o:extended"


def test_does_not_strip_invalid_suffix_as_thinking_level():
    source = FakeModelSource(ALL_MODELS)
    result = resolve_cli_model(source, cli_provider="openai", cli_model="gpt-4o:extended")
    assert result.error is None
    assert result.model.provider == "openai"
    assert result.model.id == "gpt-4o:extended"


def test_allows_custom_model_ids_for_explicit_providers_without_double_prefixing():
    source = FakeModelSource(ALL_MODELS)
    result = resolve_cli_model(source, cli_provider="openrouter", cli_model="openrouter/openai/ghost-model")
    assert result.error is None
    assert result.model.provider == "openrouter"
    assert result.model.id == "openai/ghost-model"


def test_returns_clear_error_when_no_models():
    source = FakeModelSource([])
    result = resolve_cli_model(source, cli_provider="openai", cli_model="gpt-4o")
    assert result.model is None
    assert "No models available" in result.error


def test_prefers_sole_authenticated_provider_for_ambiguous_bare_id():
    azure = make_model(id="gpt-5.6-sol", name="GPT 5.6 Sol", provider="azure-openai-responses")
    codex = make_model(id="gpt-5.6-sol", name="GPT 5.6 Sol", provider="openai-codex")
    source = FakeModelSource([azure, codex], auth_predicate=lambda p: p == "openai-codex")
    result = resolve_cli_model(source, cli_model="gpt-5.6-sol")
    assert result.error is None
    assert result.model.provider == "openai-codex"
    assert result.model.id == "gpt-5.6-sol"


def test_requires_explicit_provider_for_ambiguous_bare_id_without_unique_auth():
    azure = make_model(id="gpt-5.6-sol", name="GPT 5.6 Sol", provider="azure-openai-responses")
    codex = make_model(id="gpt-5.6-sol", name="GPT 5.6 Sol", provider="openai-codex")
    source = FakeModelSource([azure, codex], auth_predicate=lambda p: False)
    result = resolve_cli_model(source, cli_model="gpt-5.6-sol")
    assert result.model is None
    assert 'Model "gpt-5.6-sol" is ambiguous across providers' in result.error
    assert "azure-openai-responses/gpt-5.6-sol" in result.error
    assert "openai-codex/gpt-5.6-sol" in result.error
    assert "Use --provider or provider/model" in result.error


def test_prefers_provider_model_split_over_gateway_model_with_matching_id():
    zai_model = make_model(id="glm-5", name="GLM-5", provider="zai", base_url="https://open.bigmodel.cn/api/paas/v4")
    gateway_model = make_model(
        id="zai/glm-5", name="GLM-5", provider="vercel-ai-gateway", base_url="https://ai-gateway.vercel.sh"
    )
    source = FakeModelSource([*ALL_MODELS, zai_model, gateway_model], auth_predicate=lambda p: True)
    result = resolve_cli_model(source, cli_model="zai/glm-5")
    assert result.error is None
    assert result.model.provider == "zai"
    assert result.model.id == "glm-5"


def test_prefers_authenticated_exact_raw_model_id_over_unauthenticated_inferred_provider():
    commandcode_model = make_model(
        id="xiaomi/mimo-v2.5-pro", name="Xiaomi MiMo via Commandcode", provider="commandcode"
    )
    xiaomi_model = make_model(id="mimo-v2.5-pro", name="Xiaomi MiMo", provider="xiaomi")
    source = FakeModelSource(
        [*ALL_MODELS, commandcode_model, xiaomi_model], auth_predicate=lambda p: p == "commandcode"
    )
    result = resolve_cli_model(source, cli_model="xiaomi/mimo-v2.5-pro")
    assert result.error is None
    assert result.model.provider == "commandcode"
    assert result.model.id == "xiaomi/mimo-v2.5-pro"


def test_resolves_provider_prefixed_fuzzy_patterns():
    source = FakeModelSource(ALL_MODELS)
    result = resolve_cli_model(source, cli_model="openrouter/qwen")
    assert result.error is None
    assert result.model.provider == "openrouter"
    assert result.model.id == "qwen/qwen3-coder:exacto"


# -- custom model fallback with :thinking suffix (#5552) --------------------


NEURALWATT_MODEL = make_model(id="some-base-model", name="Some Base Model", provider="neuralwatt")
MODELS_WITH_NEURALWATT = [*ALL_MODELS, NEURALWATT_MODEL]


def test_strips_thinking_suffix_from_custom_model_id_in_fallback_path():
    source = FakeModelSource(MODELS_WITH_NEURALWATT)
    result = resolve_cli_model(source, cli_model="neuralwatt/zai-org/GLM-5.1-FP8:high")
    assert result.error is None
    assert result.model.provider == "neuralwatt"
    assert result.model.id == "zai-org/GLM-5.1-FP8"
    assert result.model.reasoning is True
    assert result.thinking_level == "high"


def test_custom_model_without_thinking_suffix_works_normally():
    source = FakeModelSource(MODELS_WITH_NEURALWATT)
    result = resolve_cli_model(source, cli_model="neuralwatt/zai-org/GLM-5.1-FP8")
    assert result.error is None
    assert result.model.provider == "neuralwatt"
    assert result.model.id == "zai-org/GLM-5.1-FP8"
    assert result.thinking_level is None


def test_all_valid_thinking_levels_work_in_fallback_path():
    source = FakeModelSource(MODELS_WITH_NEURALWATT)
    for level in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
        result = resolve_cli_model(source, cli_model=f"neuralwatt/zai-org/GLM-5.1-FP8:{level}")
        assert result.error is None
        assert result.model.id == "zai-org/GLM-5.1-FP8"
        assert result.thinking_level == level


def test_invalid_thinking_suffix_on_custom_model_treated_as_part_of_id():
    source = FakeModelSource(MODELS_WITH_NEURALWATT)
    result = resolve_cli_model(source, cli_model="neuralwatt/zai-org/GLM-5.1-FP8:banana")
    assert result.error is None
    assert result.model.provider == "neuralwatt"
    assert result.model.id == "zai-org/GLM-5.1-FP8:banana"
    assert result.thinking_level is None


def test_explicit_provider_with_custom_model_thinking_strips_suffix():
    source = FakeModelSource(MODELS_WITH_NEURALWATT)
    result = resolve_cli_model(source, cli_provider="neuralwatt", cli_model="zai-org/GLM-5.1-FP8:high")
    assert result.error is None
    assert result.model.provider == "neuralwatt"
    assert result.model.id == "zai-org/GLM-5.1-FP8"
    assert result.thinking_level == "high"


def test_explicit_thinking_keeps_suffix_as_part_of_model_id():
    source = FakeModelSource(MODELS_WITH_NEURALWATT)
    result = resolve_cli_model(source, cli_model="neuralwatt/zai-org/GLM-5.1-FP8:high", cli_thinking="medium")
    assert result.error is None
    assert result.model.provider == "neuralwatt"
    assert result.model.id == "zai-org/GLM-5.1-FP8:high"
    assert result.thinking_level is None


# ---------------------------------------------------------------------------
# default model selection / findInitialModel
# ---------------------------------------------------------------------------


def test_openai_defaults_track_current_models():
    assert DEFAULT_MODEL_PER_PROVIDER["openai"] == "gpt-5.5"
    assert DEFAULT_MODEL_PER_PROVIDER["openai-codex"] == "gpt-5.5"


def test_zai_minimax_cerebras_antling_defaults_track_current_models():
    assert DEFAULT_MODEL_PER_PROVIDER["zai"] == "glm-5.1"
    assert DEFAULT_MODEL_PER_PROVIDER["minimax"] == "MiniMax-M2.7"
    assert DEFAULT_MODEL_PER_PROVIDER["minimax-cn"] == "MiniMax-M2.7"
    assert DEFAULT_MODEL_PER_PROVIDER["cerebras"] == "zai-glm-4.7"
    assert DEFAULT_MODEL_PER_PROVIDER["ant-ling"] == "Ring-2.6-1T"


def test_ai_gateway_default_tracks_current_model():
    assert DEFAULT_MODEL_PER_PROVIDER["vercel-ai-gateway"] == "zai/glm-5.1"


def test_qwen_token_plan_individual_default_tracks_current_model():
    assert DEFAULT_MODEL_PER_PROVIDER["qwen-token-plan-individual"] == "qwen3.8-max"


def test_find_initial_model_accepts_explicit_provider_custom_model_ids():
    source = FakeModelSource(ALL_MODELS)
    result = find_initial_model(
        source,
        cli_provider="openrouter",
        cli_model="openrouter/openai/ghost-model",
        scoped_models=[],
        is_continuing=False,
    )
    assert result.model.provider == "openrouter"
    assert result.model.id == "openai/ghost-model"


def test_find_initial_model_selects_ai_gateway_default_when_available():
    ai_gateway_model = make_model(
        id="anthropic/claude-opus-4-6",
        name="Claude Opus 4.6",
        provider="vercel-ai-gateway",
        base_url="https://ai-gateway.vercel.sh",
        reasoning=True,
        input=["text", "image"],
        context_window=200000,
    )
    source = FakeModelSource([], available_snapshot=[ai_gateway_model])
    result = find_initial_model(source, scoped_models=[], is_continuing=False)
    assert result.model.provider == "vercel-ai-gateway"
    assert result.model.id == "anthropic/claude-opus-4-6"


def test_find_initial_model_ignores_an_unauthenticated_saved_default():
    saved_deepseek = make_model(id="deepseek-v4-flash", name="DeepSeek V4 Flash", provider="deepseek", reasoning=True)
    local_deepseek = make_model(
        id="deepseek-v4-flash",
        name="DeepSeek V4 Flash",
        provider="spark-two",
        base_url="http://spark-two:8000/v1",
        reasoning=True,
    )

    def get_model(provider: str, model_id: str):
        if provider == saved_deepseek.provider and model_id == saved_deepseek.id:
            return saved_deepseek
        return None

    source = FakeModelSource(
        [],
        auth_predicate=lambda p: p == "spark-two",
        available_snapshot=[local_deepseek],
        model_lookup=get_model,
    )
    result = find_initial_model(
        source,
        scoped_models=[],
        is_continuing=False,
        default_provider="deepseek",
        default_model_id="deepseek-v4-flash",
    )
    assert result.model.provider == "spark-two"
    assert result.model.id == "deepseek-v4-flash"
