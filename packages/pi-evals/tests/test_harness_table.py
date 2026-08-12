"""Tests for the baseline/candidate/repetition harness table.

Python port of `packages/evals/test/vitest-evals/harness-table.test.ts`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from pi_evals.harness import (
    Harness,
    HarnessContext,
    SimpleHarnessResult,
    TranscriptMessageEvent,
    create_harness,
)
from pi_evals.vitest_evals.harness_table import (
    EVAL_HARNESS_ITERATION_ARTIFACT,
    EvalHarnessIterationArtifact,
    derive_eval_group_key,
    eval_harness_table,
    parse_eval_harness_iteration_artifact,
)


@dataclass
class PromptStep:
    """Mirrors `pi_evals.pi_harness.PromptStep`.

    Redeclared locally so this stays a pure-logic test: the canonicalizer must
    accept any dataclass, and `pi_harness` pulls in the whole coding agent.
    `tests/test_pi_harness.py` checks the real step classes.
    """

    content: str
    type: str = "prompt"


@dataclass
class ReloadStep:
    """Mirrors `pi_evals.pi_harness.ReloadStep`."""

    type: str = "reload"


@dataclass
class _IdentifiedInput:
    id: str


@dataclass
class _OpaqueInput:
    payload: object


class TestDeriveEvalGroupKey:
    def test_combines_a_trimmed_string_input_id_with_repetition(self) -> None:
        assert derive_eval_group_key({"id": " input-1 ", "prompt": "hello"}, 2) == json.dumps(
            ["input-1", 2], separators=(",", ":")
        )

    def test_hashes_canonical_json_independently_of_object_key_order(self) -> None:
        assert derive_eval_group_key({"first": 1, "second": [True, "value"]}, 1) == derive_eval_group_key(
            {"second": [True, "value"], "first": 1}, 1
        )
        assert derive_eval_group_key({"first": 1}, 1) != derive_eval_group_key({"first": 2}, 1)
        assert derive_eval_group_key({"first": 1}, 1) != derive_eval_group_key({"first": 1}, 2)
        assert derive_eval_group_key(["first", "second"], 1) != derive_eval_group_key(["second", "first"], 1)

    def test_rejects_non_json_and_circular_input(self) -> None:
        circular: dict[str, object] = {}
        circular["self"] = circular
        with pytest.raises(TypeError, match="only plain objects and arrays"):
            derive_eval_group_key(object(), 1)
        with pytest.raises(TypeError, match="only finite numbers"):
            derive_eval_group_key([float("nan")], 1)
        with pytest.raises(TypeError, match="must not contain circular references"):
            derive_eval_group_key(circular, 1)


class TestDeriveEvalGroupKeyForDataclassInput:
    """This port models TypeScript's plain step objects as dataclasses.

    `PiCodingAgentInput` is `Array<{type:"prompt";content:string}|{type:"reload"}>`
    in TypeScript and `list[PromptStep | ReloadStep]` here, so a dataclass
    instance must canonicalize exactly like the plain object it stands for --
    otherwise the two implementations derive different group keys for the same
    logical input (and `extensions_eval` cannot derive one at all).
    """

    def test_matches_the_equivalent_plain_dict_input(self) -> None:
        steps = [PromptStep(content="Create a Pi extension."), ReloadStep(), PromptStep(content="Use it.")]
        plain = [
            {"type": "prompt", "content": "Create a Pi extension."},
            {"type": "reload"},
            {"type": "prompt", "content": "Use it."},
        ]

        assert derive_eval_group_key(steps, 1) == derive_eval_group_key(plain, 1)

    def test_distinguishes_different_step_sequences(self) -> None:
        first = [PromptStep(content="Create a Pi extension."), ReloadStep()]
        second = [PromptStep(content="Create a Pi extension."), ReloadStep(), PromptStep(content="Use it.")]
        reordered = [ReloadStep(), PromptStep(content="Create a Pi extension.")]

        assert derive_eval_group_key(first, 1) != derive_eval_group_key(second, 1)
        assert derive_eval_group_key(first, 1) != derive_eval_group_key(reordered, 1)
        assert derive_eval_group_key(first, 1) != derive_eval_group_key(first, 2)

    def test_is_stable_across_equal_instances(self) -> None:
        assert derive_eval_group_key([PromptStep(content="hello")], 1) == derive_eval_group_key(
            [PromptStep(content="hello")], 1
        )

    def test_takes_the_id_shortcut_like_a_plain_object(self) -> None:
        assert derive_eval_group_key(_IdentifiedInput(id=" input-1 "), 2) == derive_eval_group_key(
            {"id": "input-1", "prompt": "unused"}, 2
        )

    def test_rejects_a_dataclass_holding_a_non_json_value(self) -> None:
        with pytest.raises(TypeError, match="only plain objects and arrays"):
            derive_eval_group_key(_OpaqueInput(payload=object()), 1)
        with pytest.raises(TypeError, match="only plain objects and arrays"):
            derive_eval_group_key([PromptStep(content="ok"), _OpaqueInput(payload=object())], 1)
        with pytest.raises(TypeError, match="only finite numbers"):
            derive_eval_group_key(_OpaqueInput(payload=float("inf")), 1)

    def test_rejects_a_circular_dataclass(self) -> None:
        circular = _OpaqueInput(payload=None)
        circular.payload = [circular]

        with pytest.raises(TypeError, match="must not contain circular references"):
            derive_eval_group_key(circular, 1)


@dataclass
class _FakeOutput:
    harness: str
    input_id: str


def create_fake_harness(name: str) -> Harness:
    def run(*, input: object, signal: object, set_artifact: object) -> SimpleHarnessResult:
        assert isinstance(input, dict)
        input_id = input["id"]
        assert isinstance(input_id, str)
        return SimpleHarnessResult(
            output={"harness": name, "inputId": input_id},
            events=[
                TranscriptMessageEvent(role="user", content=input_id),
                TranscriptMessageEvent(role="assistant", content=name),
            ],
        )

    return create_harness(name=name, run=run)


harness_table = eval_harness_table(
    "local multi-harness eval",
    baseline=create_fake_harness("withoutSkill"),
    candidates=[create_fake_harness("withSkill")],
    repetitions=2,
)


class TestEvalHarnessTable:
    def test_plans_repetitions_in_declaration_order(self) -> None:
        assert [(row.name, row.repetition) for row in harness_table] == [
            ("withoutSkill", 1),
            ("withSkill", 1),
            ("withoutSkill", 2),
            ("withSkill", 2),
        ]

    def test_accepts_a_singular_candidate(self) -> None:
        rows = eval_harness_table(
            "singular candidate",
            baseline=create_fake_harness("baseline"),
            candidate=create_fake_harness("candidate"),
        )

        assert [row.name for row in rows] == ["baseline", "candidate"]

    def test_rejects_invalid_options(self) -> None:
        baseline = create_fake_harness("baseline")
        with pytest.raises(TypeError, match="evalSet must not be empty"):
            eval_harness_table("  ", baseline=baseline, candidate=create_fake_harness("candidate"))
        with pytest.raises(TypeError, match="At least one candidate harness is required"):
            eval_harness_table("empty candidates", baseline=baseline, candidates=[])
        with pytest.raises(TypeError, match="Harness names must be unique"):
            eval_harness_table("duplicate", baseline=baseline, candidate=create_fake_harness("baseline"))
        with pytest.raises(TypeError, match="repetitions must be a positive integer"):
            eval_harness_table(
                "bad repetitions",
                baseline=baseline,
                candidate=create_fake_harness("candidate"),
                repetitions=0,
            )
        with pytest.raises(TypeError, match="exactly one of candidate or candidates"):
            eval_harness_table("no candidate", baseline=baseline)

    async def test_attaches_iteration_metadata_to_every_wrapped_harness_run(self) -> None:
        for row in harness_table:
            context = HarnessContext()
            result = await row.harness.run({"id": "first"}, context)

            assert result.output == {"harness": row.name, "inputId": "first"}
            assert parse_eval_harness_iteration_artifact(
                result.artifacts.get(EVAL_HARNESS_ITERATION_ARTIFACT)
            ) == EvalHarnessIterationArtifact(
                eval_set="local multi-harness eval",
                group_key=derive_eval_group_key({"id": "first"}, row.repetition),
                harness=row.name,
                baseline="withoutSkill",
                candidates=["withSkill"],
                repetition=row.repetition,
            )

    def test_rejects_a_malformed_iteration_artifact(self) -> None:
        assert parse_eval_harness_iteration_artifact(None) is None
        assert parse_eval_harness_iteration_artifact(["not", "an", "object"]) is None
        assert parse_eval_harness_iteration_artifact({"schemaVersion": 2}) is None
        assert (
            parse_eval_harness_iteration_artifact(
                {
                    "schemaVersion": 1,
                    "evalSet": "set",
                    "groupKey": "key",
                    "harness": "harness",
                    "baseline": "baseline",
                    "candidates": ["candidate", 3],
                    "repetition": 1,
                }
            )
            is None
        )
