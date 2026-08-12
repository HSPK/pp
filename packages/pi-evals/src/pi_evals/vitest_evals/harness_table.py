"""Baseline/candidate/repetition table for comparative eval sets.

Python port of `packages/evals/src/vitest-evals/harness-table.ts`.

`eval_harness_table(...)` plans one row per (repetition, harness) pair in
declaration order and wraps each harness so every run records an iteration
artifact naming its eval set, group key, harness, baseline, candidate list and
repetition. The reporter reads that artifact back to pair baseline and
candidate observations.

The grouping key combines the repetition with a non-empty string `id` on the
input when available, otherwise with a SHA-256 hash of strict canonical JSON
input, so the same input run under different harnesses lands in the same
group.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Literal

from pi_evals.harness import (
    Harness,
    HarnessContext,
    HarnessRun,
    JsonValue,
    attach_harness_run_to_error,
    canonical_json,
    get_harness_run_from_error,
)

EVAL_HARNESS_ITERATION_ARTIFACT = "vitestEvalsHarnessIteration"
"""Artifact name. Kept in its TypeScript spelling: it is a JSON key on disk."""


@dataclass
class EvalHarnessIterationArtifact:
    """Port of `EvalHarnessIterationArtifact`.

    `to_json` emits the exact camelCase keys the TypeScript writes, since this
    artifact ends up in `runs.jsonl`'s `metadata`.
    """

    eval_set: str
    group_key: str
    harness: str
    baseline: str
    candidates: list[str] = field(default_factory=list)
    repetition: int = 1
    schema_version: Literal[1] = 1

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "schemaVersion": self.schema_version,
            "evalSet": self.eval_set,
            "groupKey": self.group_key,
            "harness": self.harness,
            "baseline": self.baseline,
            "candidates": list(self.candidates),
            "repetition": self.repetition,
        }


@dataclass
class EvalHarnessTableRow:
    """Port of `EvalHarnessTableRow<TInput, TOutput>`."""

    harness: Harness
    name: str
    repetition: int


def parse_eval_harness_iteration_artifact(value: JsonValue) -> EvalHarnessIterationArtifact | None:
    """Port of `parseEvalHarnessIterationArtifact`: validate an untrusted artifact record."""
    if not isinstance(value, dict):
        return None
    schema_version = value.get("schemaVersion")
    eval_set = value.get("evalSet")
    group_key = value.get("groupKey")
    harness = value.get("harness")
    baseline = value.get("baseline")
    candidates = value.get("candidates")
    repetition = value.get("repetition")
    if (
        schema_version != 1
        or not isinstance(eval_set, str)
        or not isinstance(group_key, str)
        or not isinstance(harness, str)
        or not isinstance(baseline, str)
        or not isinstance(candidates, list)
        or not all(isinstance(name, str) for name in candidates)
        or not isinstance(repetition, (int, float))
        or isinstance(repetition, bool)
    ):
        return None
    return EvalHarnessIterationArtifact(
        eval_set=eval_set,
        group_key=group_key,
        harness=harness,
        baseline=baseline,
        candidates=list(candidates),
        repetition=int(repetition),
    )


def _canonicalize_json(value: object, ancestors: list[int]) -> JsonValue:
    """Port of `canonicalizeJson`.

    Rejects anything that is not plain JSON data, so a group key can never
    silently depend on an object's identity or iteration order. Integral
    floats become ints so the canonical form matches JavaScript's, where
    `JSON.stringify(1.0)` is `"1"`.

    Dataclass instances are canonicalized exactly like dicts -- their fields
    sorted by name, values recursed into -- because that is how this port
    models TypeScript's plain step objects: `PiCodingAgentInput` is
    `Array<{type:"prompt";content:string}|{type:"reload"}>` there and
    `list[PromptStep | ReloadStep]` here, so both must hash to the same key.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise TypeError("Eval input must contain only finite numbers.")
        return int(value) if value.is_integer() else value
    is_dataclass_instance = is_dataclass(value) and not isinstance(value, type)
    if not is_dataclass_instance and not isinstance(value, (list, tuple, dict)):
        raise TypeError("Eval input must contain only plain objects and arrays.")
    if id(value) in ancestors:
        raise TypeError("Eval input must not contain circular references.")

    ancestors.append(id(value))
    try:
        if isinstance(value, (list, tuple)):
            return [_canonicalize_json(item, ancestors) for item in value]
        if is_dataclass_instance:
            entries: list[tuple[str, object]] = [
                (dataclass_field.name, getattr(value, dataclass_field.name)) for dataclass_field in fields(value)
            ]
        else:
            entries = []
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("Eval input must contain only plain objects and arrays.")
                entries.append((key, item))
        return {key: _canonicalize_json(item, ancestors) for key, item in sorted(entries, key=lambda pair: pair[0])}
    finally:
        ancestors.pop()


def _derive_input_key(input: object) -> str:
    """Port of `deriveInputKey`.

    TypeScript's `"id" in input` test covers any non-array object, so a
    dataclass instance with an `id` field takes the shortcut too -- the same
    reason `_canonicalize_json` treats dataclasses as plain objects.
    """
    identifier: object = None
    if isinstance(input, dict):
        identifier = input.get("id")
    elif is_dataclass(input) and not isinstance(input, type):
        identifier = getattr(input, "id", None)
    if isinstance(identifier, str) and identifier.strip():
        return identifier.strip()
    canonical_input = canonical_json(_canonicalize_json(input, []))
    return hashlib.sha256(canonical_input.encode("utf-8")).hexdigest()


def derive_eval_group_key(input: object, repetition: int) -> str:
    """Port of `deriveEvalGroupKey`."""
    return canonical_json([_derive_input_key(input), repetition])


def _validate_options(
    eval_set: str,
    baseline: Harness,
    candidates: Sequence[Harness],
    repetitions: int,
) -> None:
    if not eval_set.strip():
        raise TypeError("evalSet must not be empty.")
    if len(candidates) == 0:
        raise TypeError("At least one candidate harness is required.")
    harnesses = [baseline, *candidates]
    names = {harness.name for harness in harnesses}
    if len(names) != len(harnesses):
        raise TypeError("Harness names must be unique within an eval set.")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise TypeError("repetitions must be a positive integer.")


@dataclass
class _IterationHarness:
    """Wraps a harness so every run carries its iteration artifact."""

    name: str
    _harness: Harness
    _eval_set: str
    _baseline: str
    _candidates: list[str]
    _repetition: int

    def _artifact(self, input: object) -> EvalHarnessIterationArtifact:
        return EvalHarnessIterationArtifact(
            eval_set=self._eval_set,
            group_key=derive_eval_group_key(input, self._repetition),
            harness=self.name,
            baseline=self._baseline,
            candidates=list(self._candidates),
            repetition=self._repetition,
        )

    async def run(self, input: object, context: HarnessContext) -> HarnessRun:
        artifact = self._artifact(input).to_json()
        context.set_artifact(EVAL_HARNESS_ITERATION_ARTIFACT, artifact)

        def attach(run: HarnessRun) -> HarnessRun:
            run.artifacts = {
                **context.artifacts,
                **run.artifacts,
                EVAL_HARNESS_ITERATION_ARTIFACT: artifact,
            }
            return run

        try:
            return attach(await self._harness.run(input, context))
        except Exception as error:
            partial_run = get_harness_run_from_error(error)
            if partial_run is not None:
                raise attach_harness_run_to_error(error, attach(partial_run)) from None
            raise


def eval_harness_table(
    eval_set: str,
    *,
    baseline: Harness,
    candidate: Harness | None = None,
    candidates: Sequence[Harness] | None = None,
    repetitions: int = 1,
) -> list[EvalHarnessTableRow]:
    """Port of `evalHarnessTable`.

    TypeScript's two overloads (`{ baseline, candidate }` and
    `{ baseline, candidates }`) become two mutually exclusive keyword
    arguments.
    """
    if (candidate is None) == (candidates is None):
        raise TypeError("Provide exactly one of candidate or candidates.")
    resolved_candidates: list[Harness] = [candidate] if candidate is not None else list(candidates or [])
    _validate_options(eval_set, baseline, resolved_candidates, repetitions)

    rows: list[EvalHarnessTableRow] = []
    harnesses = [baseline, *resolved_candidates]
    candidate_names = [harness.name for harness in resolved_candidates]
    for repetition in range(1, repetitions + 1):
        for harness in harnesses:
            rows.append(
                EvalHarnessTableRow(
                    harness=_IterationHarness(
                        name=harness.name,
                        _harness=harness,
                        _eval_set=eval_set,
                        _baseline=baseline.name,
                        _candidates=candidate_names,
                        _repetition=repetition,
                    ),
                    name=harness.name,
                    repetition=repetition,
                )
            )
    return rows


__all__ = [
    "EVAL_HARNESS_ITERATION_ARTIFACT",
    "EvalHarnessIterationArtifact",
    "EvalHarnessTableRow",
    "derive_eval_group_key",
    "eval_harness_table",
    "parse_eval_harness_iteration_artifact",
]
