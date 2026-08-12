"""Tests for the `pp-evals` runner CLI.

Covers `pi_evals.run_evals`, this port's replacement for
`packages/evals/scripts/run-evals.mjs`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pi_evals.run_evals import (
    EVALS_PATH,
    RunnerError,
    build_command,
    parse_arguments,
    resolve_default_model,
)


class TestParseArguments:
    def test_reads_separated_and_joined_flags_and_forwards_the_rest(self) -> None:
        parsed = parse_arguments(["--provider", "openai", "--model=gpt-5.6-sol", "-k", "smoke"])

        assert (parsed.provider, parsed.model) == ("openai", "gpt-5.6-sol")
        assert parsed.has_cli_model_selection is True
        assert parsed.forwarded == ["-k", "smoke"]

    def test_rejects_a_flag_without_a_value(self) -> None:
        with pytest.raises(RunnerError, match="Missing value for --model"):
            parse_arguments(["--model", "--provider"])
        with pytest.raises(RunnerError, match="Missing value for --provider"):
            parse_arguments(["--provider"])


class TestResolveDefaultModel:
    def test_cli_selection_wins_over_the_environment(self) -> None:
        parsed = parse_arguments(["--provider", "anthropic", "--model", "claude-opus-4-6"])

        assert resolve_default_model(parsed, {"PI_PROVIDER": "openai", "PI_MODEL": "gpt-5.6-sol"}) == (
            "anthropic",
            "claude-opus-4-6",
        )

    def test_falls_back_to_the_environment(self) -> None:
        assert resolve_default_model(parse_arguments([]), {"PI_PROVIDER": " openai ", "PI_MODEL": "gpt-5.6-sol"}) == (
            "openai",
            "gpt-5.6-sol",
        )

    def test_allows_no_default_at_all(self) -> None:
        assert resolve_default_model(parse_arguments([]), {}) == (None, None)

    def test_requires_both_halves(self) -> None:
        with pytest.raises(RunnerError, match="CLI model selection requires both"):
            resolve_default_model(parse_arguments(["--provider", "openai"]), {})
        with pytest.raises(RunnerError, match="Default model selection requires both"):
            resolve_default_model(parse_arguments([]), {"PI_PROVIDER": "openai"})


class TestBuildCommand:
    def test_defaults_to_the_eval_modules(self) -> None:
        command = build_command([])

        assert command[1:3] == ["-m", "pytest"]
        assert command[3:5] == ["-o", "python_files=*_eval.py test_*.py"]
        assert command[5].endswith("pi_evals/evals")

    def test_keeps_the_eval_modules_when_only_filters_are_forwarded(self) -> None:
        assert build_command(["-k", "smoke"])[5:] == ["-k", "smoke", str(EVALS_PATH)]

    def test_an_explicit_path_replaces_the_default_target(self, tmp_path: Path) -> None:
        target = tmp_path / "custom_eval.py"
        target.write_text("", encoding="utf-8")

        assert build_command([str(target)])[5:] == [str(target)]
