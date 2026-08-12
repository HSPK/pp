"""Tests for the comparative eval summary.

Python port of `packages/evals/test/vitest-evals/summary.test.ts`.
"""

from __future__ import annotations

import json
from typing import Literal

from pi_evals.vitest_evals.summary import (
    CorrectnessLiftSummary,
    HarnessObservation,
    PairedMetricSummary,
    format_harness_comparison_report,
    strip_ansi,
    summarize_harness_comparisons,
)

ObservationResult = Literal["passed", "failed", "unscored", "skipped", "pending", "errored"]


def observation(
    harness: str,
    test_name: str,
    result: ObservationResult,
    *,
    total_tokens: float | None = None,
    total_ms: float | None = None,
    estimated_cost_usd: float | None = None,
    baseline: str = "without-tools",
    candidates: list[str] | None = None,
    file: str = "src/tool-access.eval.ts",
) -> HarnessObservation:
    return HarnessObservation(
        eval_set="tool access",
        group_key=json.dumps([test_name, 1], separators=(",", ":")),
        test_name=test_name,
        file=file,
        harness=harness,
        baseline=baseline,
        candidates=list(candidates or ["with-tools"]),
        repetition=1,
        outcome="scored" if result in ("passed", "failed") else result,
        score=(1 if result == "passed" else 0) if result in ("passed", "failed") else None,
        total_tokens=total_tokens,
        total_ms=total_ms,
        estimated_cost_usd=estimated_cost_usd,
    )


class TestSummarizeHarnessComparisons:
    def test_computes_paired_correctness_lift_separately_from_efficiency_deltas(self) -> None:
        report = summarize_harness_comparisons(
            [
                observation(
                    "without-tools", "create", "failed", total_tokens=100, total_ms=1000, estimated_cost_usd=0.01
                ),
                observation("with-tools", "create", "passed", total_tokens=120, total_ms=800, estimated_cost_usd=0.02),
                observation("without-tools", "inspect", "passed", total_tokens=200),
                observation("with-tools", "inspect", "passed", total_tokens=180),
            ]
        )

        assert len(report.eval_sets) == 1
        comparison = report.eval_sets[0].comparisons[0]
        assert comparison.baseline == "without-tools"
        assert comparison.candidate == "with-tools"
        assert comparison.correctness == CorrectnessLiftSummary(
            total_pairs=2,
            eligible_pairs=2,
            baseline_pass_rate=0.5,
            candidate_pass_rate=1,
            lift=0.5,
            baseline_wins=0,
            candidate_wins=1,
            ties=1,
        )
        assert comparison.total_tokens == PairedMetricSummary(
            total_pairs=2, eligible_pairs=2, baseline_mean=150, candidate_mean=150, mean_delta=0
        )
        assert comparison.total_ms == PairedMetricSummary(
            total_pairs=2, eligible_pairs=1, baseline_mean=1000, candidate_mean=800, mean_delta=-200
        )
        assert comparison.estimated_cost_usd == PairedMetricSummary(
            total_pairs=2, eligible_pairs=1, baseline_mean=0.01, candidate_mean=0.02, mean_delta=0.01
        )
        assert report.diagnostics == []

    def test_reports_missing_observations_without_coercing_them(self) -> None:
        report = summarize_harness_comparisons(
            [
                observation("without-tools", "create", "failed"),
                observation("with-tools", "create", "passed"),
                observation("without-tools", "inspect", "passed"),
            ]
        )
        comparison = report.eval_sets[0].comparisons[0]

        assert comparison.correctness == CorrectnessLiftSummary(
            total_pairs=2,
            eligible_pairs=1,
            baseline_pass_rate=0,
            candidate_pass_rate=1,
            lift=1,
            baseline_wins=0,
            candidate_wins=1,
            ties=0,
        )
        assert comparison.total_tokens == PairedMetricSummary(
            total_pairs=2, eligible_pairs=0, baseline_mean=None, candidate_mean=None, mean_delta=None
        )
        assert any(
            diagnostic.test_name == "inspect"
            and diagnostic.harness == "with-tools"
            and diagnostic.reason == "missing-observation"
            for diagnostic in report.diagnostics
        )

    def test_keeps_identical_inputs_in_different_test_files_separate(self) -> None:
        report = summarize_harness_comparisons(
            [
                observation("without-tools", "shared", "failed"),
                observation("with-tools", "shared", "passed"),
                observation("without-tools", "shared", "passed", file="src/other.eval.ts"),
                observation("with-tools", "shared", "passed", file="src/other.eval.ts"),
            ]
        )
        correctness = report.eval_sets[0].comparisons[0].correctness

        assert (correctness.total_pairs, correctness.eligible_pairs) == (2, 2)
        assert report.diagnostics == []

    def test_does_not_score_harness_errors_as_correctness_failures(self) -> None:
        report = summarize_harness_comparisons(
            [
                observation("without-tools", "create", "errored", total_tokens=100),
                observation("with-tools", "create", "passed", total_tokens=100),
            ]
        )
        comparison = report.eval_sets[0].comparisons[0]

        assert (comparison.correctness.total_pairs, comparison.correctness.eligible_pairs) == (1, 0)
        assert comparison.total_tokens.eligible_pairs == 0
        assert any(
            diagnostic.harness == "without-tools" and diagnostic.reason == "harness-error"
            for diagnostic in report.diagnostics
        )

    def test_does_not_derive_correctness_from_completed_tests_without_judge_scores(self) -> None:
        report = summarize_harness_comparisons(
            [
                observation("without-tools", "create", "unscored"),
                observation("with-tools", "create", "unscored"),
            ]
        )

        assert report.eval_sets[0].comparisons[0].correctness.eligible_pairs == 0
        assert [(diagnostic.harness, diagnostic.reason) for diagnostic in report.diagnostics] == [
            ("with-tools", "missing-score"),
            ("without-tools", "missing-score"),
        ]

    def test_compares_each_candidate_with_the_declared_baseline(self) -> None:
        candidates = ["second", "third"]
        report = summarize_harness_comparisons(
            [
                observation("first", "input", "passed", baseline="first", candidates=candidates),
                observation("second", "input", "passed", baseline="first", candidates=candidates),
                observation("third", "input", "passed", baseline="first", candidates=candidates),
            ]
        )

        assert [(comparison.baseline, comparison.candidate) for comparison in report.eval_sets[0].comparisons] == [
            ("first", "second"),
            ("first", "third"),
        ]

    def test_retains_a_declared_harness_with_no_completed_observations(self) -> None:
        report = summarize_harness_comparisons([observation("without-tools", "create", "failed")])

        assert len(report.eval_sets[0].comparisons) == 1
        assert report.eval_sets[0].comparisons[0].correctness.eligible_pairs == 0
        assert any(
            diagnostic.test_name == "create"
            and diagnostic.harness == "with-tools"
            and diagnostic.reason == "missing-observation"
            for diagnostic in report.diagnostics
        )

    def test_reports_duplicate_and_unscorable_observations_once(self) -> None:
        candidates = ["second", "third"]
        report = summarize_harness_comparisons(
            [
                observation("first", "duplicate", "passed", baseline="first", candidates=candidates),
                observation("first", "duplicate", "failed", baseline="first", candidates=candidates),
                observation("second", "duplicate", "passed", baseline="first", candidates=candidates),
                observation("third", "duplicate", "passed", baseline="first", candidates=candidates),
                observation("first", "skipped", "skipped", baseline="first", candidates=candidates),
                observation("second", "skipped", "passed", baseline="first", candidates=candidates),
                observation("third", "skipped", "passed", baseline="first", candidates=candidates),
            ]
        )

        assert [
            diagnostic.test_name for diagnostic in report.diagnostics if diagnostic.reason == "duplicate-observation"
        ] == ["duplicate"]
        assert [
            (diagnostic.test_name, diagnostic.harness)
            for diagnostic in report.diagnostics
            if diagnostic.reason == "unscorable-outcome"
        ] == [("skipped", "first")]

    def test_formats_lift_and_telemetry_availability_for_the_terminal_report(self) -> None:
        report = summarize_harness_comparisons(
            [
                observation("without-tools", "create", "failed", total_ms=34853.7),
                observation("with-tools", "create", "passed", total_ms=30694.2),
            ]
        )

        formatted = strip_ansi(format_harness_comparison_report(report))
        assert "Eval Comparisons" in formatted
        assert " Baseline  without-tools" in formatted
        assert "Candidate  with-tools (1/1 pairs)" in formatted
        assert "Pass rate  +100.0 pp (candidate 100.0%, baseline 0.0%)" in formatted
        assert "   Tokens  unavailable" in formatted
        assert "  Latency  -4159.5ms (candidate 30694.2ms, baseline 34853.7ms)" in formatted

    def test_formats_cost_deltas_and_diagnostics(self) -> None:
        report = summarize_harness_comparisons(
            [
                observation("without-tools", "create", "failed", estimated_cost_usd=0.05),
                observation("with-tools", "create", "passed", estimated_cost_usd=0.02),
                observation("without-tools", "inspect", "passed"),
            ]
        )

        formatted = strip_ansi(format_harness_comparison_report(report))
        # No coverage suffix: the metric covers every pair the comparison itself covers.
        assert "Est. cost  -$0.0300 (candidate $0.0200, baseline $0.0500)\n" in f"{formatted}\n"
        assert "Incomplete observations" in formatted
        assert "missing-observation: src/tool-access.eval.ts/inspect repetition 1, harness with-tools" in formatted

    def test_formats_nothing_without_comparisons(self) -> None:
        assert format_harness_comparison_report(summarize_harness_comparisons([])) == ""
