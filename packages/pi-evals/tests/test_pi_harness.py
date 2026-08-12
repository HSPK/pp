"""Tests for the Pi harness's model resolution.

Python port of `packages/evals/test/pi-harness.test.ts`.
"""

from __future__ import annotations

import re

import pytest
from pi_evals.pi_harness import (
    PiCodingAgentModelSelection,
    PromptStep,
    ReloadStep,
    resolve_model_selection,
)
from pi_evals.vitest_evals.harness_table import derive_eval_group_key


class TestResolveModelSelection:
    def test_prefers_an_explicit_harness_model_over_environment_defaults(self) -> None:
        assert resolve_model_selection(
            PiCodingAgentModelSelection(provider="anthropic", id="claude-opus-4-6"),
            {"PI_PROVIDER": "openai-codex", "PI_MODEL": "gpt-5.6-sol"},
        ) == PiCodingAgentModelSelection(provider="anthropic", id="claude-opus-4-6")

    def test_uses_trimmed_environment_defaults_when_the_harness_has_no_explicit_model(self) -> None:
        assert resolve_model_selection(
            None, {"PI_PROVIDER": " openai-codex ", "PI_MODEL": " gpt-5.6-sol "}
        ) == PiCodingAgentModelSelection(provider="openai-codex", id="gpt-5.6-sol")

    @pytest.mark.parametrize(
        ("explicit_model", "environment"),
        [
            (None, {}),
            (None, {"PI_PROVIDER": "openai-codex"}),
            (None, {"PI_MODEL": "gpt-5.6-sol"}),
            (
                PiCodingAgentModelSelection(provider="", id="gpt-5.6-sol"),
                {"PI_PROVIDER": "openai-codex", "PI_MODEL": "gpt-5.6-sol"},
            ),
        ],
    )
    def test_rejects_an_incomplete_model_selection(
        self,
        explicit_model: PiCodingAgentModelSelection | None,
        environment: dict[str, str],
    ) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape("Select a harness model explicitly or set both PI_PROVIDER and PI_MODEL as defaults."),
        ):
            resolve_model_selection(explicit_model, environment)


class TestPiStepGroupKeys:
    """The real step dataclasses must group like TypeScript's plain step objects.

    `extensions_eval` passes `[PromptStep(...), ReloadStep(), PromptStep(...)]`
    as its input, and the comparative reporter pairs baseline with candidate by
    the group key derived from it. TypeScript passes the plain objects
    `[{type:"prompt",content},{type:"reload"},...]`, so both must hash alike.
    """

    def test_match_the_equivalent_plain_objects(self) -> None:
        steps = [
            PromptStep(content="Create a Pi extension."),
            ReloadStep(),
            PromptStep(content="Use the extension."),
        ]
        plain = [
            {"type": "prompt", "content": "Create a Pi extension."},
            {"type": "reload"},
            {"type": "prompt", "content": "Use the extension."},
        ]

        assert derive_eval_group_key(steps, 3) == derive_eval_group_key(plain, 3)

    def test_group_the_same_input_across_harnesses_and_split_repetitions(self) -> None:
        steps = [PromptStep(content="Create a Pi extension."), ReloadStep()]

        assert derive_eval_group_key(steps, 1) == derive_eval_group_key(list(steps), 1)
        assert derive_eval_group_key(steps, 1) != derive_eval_group_key(steps, 2)
