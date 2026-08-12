"""Comparative baseline/candidate eval summary.

Python port of `packages/evals/src/vitest-evals/summary.ts`.

Pure logic: it takes the per-run observations the reporter collected and
computes, per eval set, the paired baseline/candidate comparison -- pass-rate
lift from judge scores plus separate token/latency/cost paired deltas -- and
formats that as the terminal report. Observations that cannot be paired or
scored become diagnostics instead of being coerced into failures or zeros.

`styleText` (`node:util`) is ported as raw ANSI SGR sequences, matching the
colours Node emits, so `strip_ansi` in the tests sees the same plain text the
TypeScript test asserts on.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal

from pi_evals.harness import canonical_json

HarnessObservationOutcome = Literal["scored", "unscored", "skipped", "pending", "errored"]

HarnessComparisonDiagnosticReason = Literal[
    "missing-observation",
    "duplicate-observation",
    "harness-error",
    "missing-score",
    "unscorable-outcome",
]

_ANSI_CODES: dict[str, tuple[str, str]] = {
    "bold": ("\x1b[1m", "\x1b[22m"),
    "gray": ("\x1b[90m", "\x1b[39m"),
    "green": ("\x1b[32m", "\x1b[39m"),
    "red": ("\x1b[31m", "\x1b[39m"),
    "yellow": ("\x1b[33m", "\x1b[39m"),
}

_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def style_text(style: str, value: str) -> str:
    """Port of `node:util`'s `styleText` for the five styles this report uses."""
    open_code, close_code = _ANSI_CODES[style]
    return f"{open_code}{value}{close_code}"


def strip_ansi(value: str) -> str:
    """Port of `node:util`'s `stripVTControlCharacters`, as far as the report needs it."""
    return _ANSI_PATTERN.sub("", value)


@dataclass
class HarnessObservation:
    """One harness run's contribution to a comparison.

    TypeScript models the `outcome`/`score` pair as a discriminated union
    (`{ outcome: "scored"; score: number } | { outcome: ...; score?: never }`);
    here `score` is simply `None` for every non-`"scored"` outcome.
    """

    eval_set: str
    group_key: str
    test_name: str
    file: str
    harness: str
    baseline: str
    candidates: list[str] = field(default_factory=list)
    repetition: int = 1
    outcome: HarnessObservationOutcome = "scored"
    score: float | None = None
    total_tokens: float | None = None
    total_ms: float | None = None
    estimated_cost_usd: float | None = None


@dataclass
class PairedMetricSummary:
    total_pairs: int
    eligible_pairs: int
    baseline_mean: float | None
    candidate_mean: float | None
    mean_delta: float | None


@dataclass
class CorrectnessLiftSummary:
    total_pairs: int
    eligible_pairs: int
    baseline_pass_rate: float | None
    candidate_pass_rate: float | None
    lift: float | None
    baseline_wins: int
    candidate_wins: int
    ties: int


@dataclass
class HarnessPairComparison:
    baseline: str
    candidate: str
    correctness: CorrectnessLiftSummary
    total_tokens: PairedMetricSummary
    total_ms: PairedMetricSummary
    estimated_cost_usd: PairedMetricSummary


@dataclass
class HarnessComparisonDiagnostic:
    eval_set: str
    group_key: str
    test_name: str
    file: str
    repetition: int
    harness: str
    reason: HarnessComparisonDiagnosticReason


@dataclass
class HarnessEvalSetReport:
    eval_set: str
    comparisons: list[HarnessPairComparison]


@dataclass
class HarnessComparisonReport:
    eval_sets: list[HarnessEvalSetReport]
    diagnostics: list[HarnessComparisonDiagnostic]
    schema_version: Literal[1] = 1


@dataclass
class _HarnessDescriptor:
    name: str
    index: int


@dataclass
class _ObservationGroup:
    eval_set: str
    group_key: str
    test_name: str
    file: str
    repetition: int
    observations_by_harness: dict[str, list[HarnessObservation]] = field(default_factory=dict)


@dataclass
class _EvalSetData:
    baseline: _HarnessDescriptor
    candidates_by_name: dict[str, _HarnessDescriptor] = field(default_factory=dict)
    groups_by_key: dict[str, _ObservationGroup] = field(default_factory=dict)


@dataclass
class _ObservationPair:
    baseline: HarnessObservation
    candidate: HarnessObservation


def _mean(values: Sequence[float]) -> float | None:
    return None if len(values) == 0 else sum(values) / len(values)


def _precise_difference(left: float, right: float) -> float:
    """Port of `Number((left - right).toPrecision(15))`.

    Rounding to 15 significant digits removes the binary-floating-point noise
    that would otherwise make an exactly-equal delta print as `1e-17`.
    """
    difference = left - right
    if difference == 0 or difference != difference or difference in (float("inf"), float("-inf")):
        return difference
    return float(f"{difference:.15g}")


def _group_observations(observations: Iterable[HarnessObservation]) -> dict[str, _EvalSetData]:
    eval_sets: dict[str, _EvalSetData] = {}
    for observation in observations:
        eval_set = eval_sets.get(observation.eval_set)
        if eval_set is None:
            eval_set = _EvalSetData(baseline=_HarnessDescriptor(name=observation.baseline, index=0))
            eval_sets[observation.eval_set] = eval_set

        for index, name in enumerate(observation.candidates):
            existing = eval_set.candidates_by_name.get(name)
            if existing is None or index < existing.index:
                eval_set.candidates_by_name[name] = _HarnessDescriptor(name=name, index=index)

        key = canonical_json([observation.file, observation.test_name, observation.group_key])
        group = eval_set.groups_by_key.get(key)
        if group is None:
            group = _ObservationGroup(
                eval_set=observation.eval_set,
                group_key=observation.group_key,
                test_name=observation.test_name,
                file=observation.file,
                repetition=observation.repetition,
            )
            eval_set.groups_by_key[key] = group
        group.observations_by_harness.setdefault(observation.harness, []).append(observation)
    return eval_sets


def _ordered_candidates(eval_set: _EvalSetData) -> list[_HarnessDescriptor]:
    return sorted(eval_set.candidates_by_name.values(), key=lambda descriptor: (descriptor.index, descriptor.name))


def _ordered_harnesses(eval_set: _EvalSetData) -> list[_HarnessDescriptor]:
    return [eval_set.baseline, *_ordered_candidates(eval_set)]


def _ordered_groups(eval_set: _EvalSetData) -> list[_ObservationGroup]:
    return sorted(eval_set.groups_by_key.values(), key=lambda group: (group.group_key, group.repetition))


def _collect_diagnostics(
    harnesses: Sequence[_HarnessDescriptor],
    groups: Sequence[_ObservationGroup],
) -> list[HarnessComparisonDiagnostic]:
    diagnostics: list[HarnessComparisonDiagnostic] = []
    for group in groups:
        for descriptor in harnesses:
            observations = group.observations_by_harness.get(descriptor.name, [])
            reason: HarnessComparisonDiagnosticReason | None = None
            if len(observations) == 0:
                reason = "missing-observation"
            elif len(observations) > 1:
                reason = "duplicate-observation"
            elif observations[0].outcome == "errored":
                reason = "harness-error"
            elif observations[0].outcome == "unscored":
                reason = "missing-score"
            elif observations[0].outcome != "scored":
                reason = "unscorable-outcome"
            if reason is None:
                continue
            diagnostics.append(
                HarnessComparisonDiagnostic(
                    eval_set=group.eval_set,
                    group_key=group.group_key,
                    test_name=group.test_name,
                    file=group.file,
                    repetition=group.repetition,
                    harness=descriptor.name,
                    reason=reason,
                )
            )
    return diagnostics


def _pair_observations(
    groups: Sequence[_ObservationGroup],
    baseline_harness: str,
    candidate_harness: str,
) -> list[_ObservationPair]:
    pairs: list[_ObservationPair] = []
    for group in groups:
        baseline = group.observations_by_harness.get(baseline_harness, [])
        candidate = group.observations_by_harness.get(candidate_harness, [])
        if len(baseline) == 1 and len(candidate) == 1:
            pairs.append(_ObservationPair(baseline=baseline[0], candidate=candidate[0]))
    return pairs


def _is_finite(value: float | None) -> bool:
    return value is not None and value == value and value not in (float("inf"), float("-inf"))


def _summarize_metric(
    pairs: Sequence[_ObservationPair],
    select: Callable[[HarnessObservation], float | None],
    total_pairs: int,
) -> PairedMetricSummary:
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    for pair in pairs:
        if pair.baseline.outcome != "scored" or pair.candidate.outcome != "scored":
            continue
        baseline_value = select(pair.baseline)
        candidate_value = select(pair.candidate)
        if not _is_finite(baseline_value) or not _is_finite(candidate_value):
            continue
        assert baseline_value is not None and candidate_value is not None
        baseline_values.append(baseline_value)
        candidate_values.append(candidate_value)

    baseline_mean = _mean(baseline_values)
    candidate_mean = _mean(candidate_values)
    return PairedMetricSummary(
        total_pairs=total_pairs,
        eligible_pairs=len(baseline_values),
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        mean_delta=(
            None
            if baseline_mean is None or candidate_mean is None
            else _precise_difference(candidate_mean, baseline_mean)
        ),
    )


def _summarize_correctness(pairs: Sequence[_ObservationPair], total_pairs: int) -> CorrectnessLiftSummary:
    eligible_pairs = 0
    baseline_passes = 0
    candidate_passes = 0
    baseline_wins = 0
    candidate_wins = 0
    ties = 0

    for pair in pairs:
        if pair.baseline.outcome != "scored" or pair.candidate.outcome != "scored":
            continue
        eligible_pairs += 1
        baseline_passed = (pair.baseline.score or 0) >= 1
        candidate_passed = (pair.candidate.score or 0) >= 1
        if baseline_passed:
            baseline_passes += 1
        if candidate_passed:
            candidate_passes += 1
        if baseline_passed == candidate_passed:
            ties += 1
        elif baseline_passed:
            baseline_wins += 1
        else:
            candidate_wins += 1

    baseline_pass_rate = None if eligible_pairs == 0 else baseline_passes / eligible_pairs
    candidate_pass_rate = None if eligible_pairs == 0 else candidate_passes / eligible_pairs
    return CorrectnessLiftSummary(
        total_pairs=total_pairs,
        eligible_pairs=eligible_pairs,
        baseline_pass_rate=baseline_pass_rate,
        candidate_pass_rate=candidate_pass_rate,
        lift=(
            None
            if baseline_pass_rate is None or candidate_pass_rate is None
            else _precise_difference(candidate_pass_rate, baseline_pass_rate)
        ),
        baseline_wins=baseline_wins,
        candidate_wins=candidate_wins,
        ties=ties,
    )


def _compare_harnesses(
    baseline: _HarnessDescriptor,
    candidate: _HarnessDescriptor,
    groups: Sequence[_ObservationGroup],
) -> HarnessPairComparison:
    pairs = _pair_observations(groups, baseline.name, candidate.name)
    return HarnessPairComparison(
        baseline=baseline.name,
        candidate=candidate.name,
        correctness=_summarize_correctness(pairs, len(groups)),
        total_tokens=_summarize_metric(pairs, lambda observation: observation.total_tokens, len(groups)),
        total_ms=_summarize_metric(pairs, lambda observation: observation.total_ms, len(groups)),
        estimated_cost_usd=_summarize_metric(pairs, lambda observation: observation.estimated_cost_usd, len(groups)),
    )


def summarize_harness_comparisons(observations: Sequence[HarnessObservation]) -> HarnessComparisonReport:
    """Port of `summarizeHarnessComparisons`."""
    eval_sets: list[HarnessEvalSetReport] = []
    diagnostics: list[HarnessComparisonDiagnostic] = []
    for eval_set_name, data in sorted(_group_observations(observations).items(), key=lambda item: item[0]):
        harnesses = _ordered_harnesses(data)
        candidates = _ordered_candidates(data)
        groups = _ordered_groups(data)
        eval_sets.append(
            HarnessEvalSetReport(
                eval_set=eval_set_name,
                comparisons=[_compare_harnesses(data.baseline, candidate, groups) for candidate in candidates],
            )
        )
        diagnostics.extend(_collect_diagnostics(harnesses, groups))

    diagnostics.sort(
        key=lambda diagnostic: (
            diagnostic.eval_set,
            diagnostic.file,
            diagnostic.group_key,
            diagnostic.repetition,
            diagnostic.harness,
        )
    )
    return HarnessComparisonReport(eval_sets=eval_sets, diagnostics=diagnostics)


def _to_fixed(value: float, fraction_digits: int) -> str:
    """Port of `Number.prototype.toFixed`.

    Python's `format` rounds halves to even; JavaScript rounds them away from
    zero (the spec negates a negative value first, then picks the larger
    candidate). `Decimal(value)` is the exact binary value of the double, so
    quantizing it with `ROUND_HALF_UP` reproduces V8 digit for digit.
    """
    quantum = Decimal(1).scaleb(-fraction_digits)
    return str(Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP))


def _format_percentage(value: float | None) -> str:
    return "unavailable" if value is None else f"{_to_fixed(value * 100, 1)}%"


def _format_signed(value: float, fraction_digits: int) -> str:
    return f"{'+' if value >= 0 else ''}{_to_fixed(value, fraction_digits)}"


def _format_coverage(eligible_pairs: int, total_pairs: int) -> str:
    return style_text("gray", f"({eligible_pairs}/{total_pairs} pairs)")


def _format_report_line(label: str, value: str) -> str:
    return f"    {style_text('gray', label.rjust(9))}  {value}"


def _color_delta(value: float, formatted: str, positive_is_better: bool) -> str:
    if value == 0:
        return style_text("gray", formatted)
    improved = value > 0 if positive_is_better else value < 0
    return style_text("green" if improved else "red", formatted)


def _format_metric(
    label: str,
    metric: PairedMetricSummary,
    format_value: Callable[[float], str],
    format_delta: Callable[[float], str],
    comparison_pairs: int,
) -> str:
    coverage = (
        ""
        if metric.eligible_pairs == 0 or metric.eligible_pairs == comparison_pairs
        else f" {_format_coverage(metric.eligible_pairs, metric.total_pairs)}"
    )
    if metric.baseline_mean is None or metric.candidate_mean is None or metric.mean_delta is None:
        return _format_report_line(label, f"{style_text('yellow', 'unavailable')}{coverage}")
    delta = _color_delta(metric.mean_delta, format_delta(metric.mean_delta), False)
    values = style_text(
        "gray",
        f"(candidate {format_value(metric.candidate_mean)}, baseline {format_value(metric.baseline_mean)})",
    )
    return _format_report_line(label, f"{delta} {values}{coverage}")


def format_harness_comparison_report(report: HarnessComparisonReport) -> str:
    """Port of `formatHarnessComparisonReport`."""
    if all(len(eval_set.comparisons) == 0 for eval_set in report.eval_sets):
        return ""
    lines = [style_text("bold", "Eval Comparisons")]
    for eval_set in report.eval_sets:
        lines.append(f"  {eval_set.eval_set}")
        for index, comparison in enumerate(eval_set.comparisons):
            if index > 0:
                lines.append("")
            correctness = comparison.correctness
            lines.append(_format_report_line("Baseline", comparison.baseline))
            lines.append(
                _format_report_line(
                    "Candidate",
                    f"{comparison.candidate} {_format_coverage(correctness.eligible_pairs, correctness.total_pairs)}",
                )
            )
            if correctness.lift is None:
                lines.append(_format_report_line("Pass rate", style_text("yellow", "unavailable")))
            else:
                lift = correctness.lift * 100
                delta = _color_delta(lift, f"{_format_signed(lift, 1)} pp", True)
                values = style_text(
                    "gray",
                    f"(candidate {_format_percentage(correctness.candidate_pass_rate)}, "
                    f"baseline {_format_percentage(correctness.baseline_pass_rate)})",
                )
                lines.append(_format_report_line("Pass rate", f"{delta} {values}"))
            lines.append(
                _format_metric(
                    "Tokens",
                    comparison.total_tokens,
                    lambda value: _to_fixed(value, 1),
                    lambda value: _format_signed(value, 1),
                    correctness.eligible_pairs,
                )
            )
            lines.append(
                _format_metric(
                    "Latency",
                    comparison.total_ms,
                    lambda value: f"{_to_fixed(value, 1)}ms",
                    lambda value: f"{_format_signed(value, 1)}ms",
                    correctness.eligible_pairs,
                )
            )
            lines.append(
                _format_metric(
                    "Est. cost",
                    comparison.estimated_cost_usd,
                    lambda value: f"${_to_fixed(value, 4)}",
                    lambda value: f"{'+' if value >= 0 else '-'}${_to_fixed(abs(value), 4)}",
                    correctness.eligible_pairs,
                )
            )
    if report.diagnostics:
        lines.append(f"  {style_text('yellow', 'Incomplete observations')}")
        for diagnostic in report.diagnostics:
            lines.append(
                f"    {diagnostic.reason}: {diagnostic.file}/{diagnostic.test_name} "
                f"repetition {diagnostic.repetition}, harness {diagnostic.harness}"
            )
    return "\n".join(lines)


__all__ = [
    "CorrectnessLiftSummary",
    "HarnessComparisonDiagnostic",
    "HarnessComparisonReport",
    "HarnessEvalSetReport",
    "HarnessObservation",
    "HarnessPairComparison",
    "PairedMetricSummary",
    "format_harness_comparison_report",
    "strip_ansi",
    "style_text",
    "summarize_harness_comparisons",
]
