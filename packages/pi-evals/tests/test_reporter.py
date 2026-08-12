"""End-to-end test of the eval declaration helper and the pytest reporter plugin.

Covers the pytest substitutions this port makes for
`packages/evals/src/vitest-evals/reporter.ts`,
`packages/evals/src/vitest-evals/setup.ts` and `vitest-evals`' own
`describeEval`: a real pytest run must collect the generated tests, apply the
judges, write the `.eval/runs.jsonl` index plus the `sessions/` attachment
layout, and print the comparison report.
"""

from __future__ import annotations

import json
from pathlib import Path

pytest_plugins = ["pytester"]

_EVAL_MODULE = """
from pi_evals.harness import (
    EvalCase,
    EvalOptions,
    HarnessRun,
    HarnessTimings,
    HarnessUsage,
    JudgeContext,
    JudgeResult,
    SimpleHarnessResult,
    create_harness,
    create_judge,
    describe_eval,
)
from pi_evals.vitest_evals.harness_table import eval_harness_table


def make_harness(name, score_output, total_tokens):
    def run(*, input, signal, set_artifact):
        set_artifact("runId", f"run-{name}")
        set_artifact("piSessionJsonl", '{"type":"session"}\\n')
        return SimpleHarnessResult(
            output=score_output,
            usage=HarnessUsage(
                provider="faux",
                model="faux-model",
                input_tokens=1,
                output_tokens=2,
                total_tokens=total_tokens,
                tool_calls=0,
                metadata={"estimatedCostUsd": 0.5},
            ),
            timings=HarnessTimings(total_ms=10.0),
        )

    return create_harness(name=name, run=run)


ExactJudge = create_judge(
    "ExactJudge",
    lambda context: JudgeResult(score=1 if context.output == "ok" else 0),
)

table = eval_harness_table(
    "example eval set",
    baseline=make_harness("baseline", "bad", 100),
    candidate=make_harness("candidate", "ok", 80),
)


def declare(row):
    def define(it):
        async def body(case: EvalCase) -> None:
            result = await case.run({"id": "case-1"})
            assert result.usage.provider == "faux"

        it("answers the question", body)

    describe_eval(
        "example eval set",
        EvalOptions(harness=row.harness, judges=[ExactJudge], judge_threshold=None),
        define,
        suffix=f"{row.name} repetition {row.repetition}",
        namespace=globals(),
    )


for _row in table:
    declare(_row)
"""


def test_reporter_writes_runs_jsonl_and_prints_the_comparison(pytester, monkeypatch, tmp_path: Path) -> None:
    artifact_directory = tmp_path / "eval-artifacts"
    monkeypatch.setenv("PI_EVAL_ARTIFACT_DIR", str(artifact_directory))
    pytester.makeini("[pytest]\nasyncio_mode = auto\n")
    pytester.makepyfile(test_example_eval=_EVAL_MODULE)

    result = pytester.runpytest_subprocess("-q")
    result.assert_outcomes(passed=2)

    # Keyed by harness, not indexed: records are appended in completion order,
    # which pytest (like Vitest upstream) does not promise.
    records = {
        json.loads(line)["harness"]: json.loads(line)
        for line in (artifact_directory / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert sorted(records) == ["baseline", "candidate"]
    first = records["baseline"]
    assert first["schemaVersion"] == 1
    assert first["runId"] == "run-baseline"
    assert first["test"]["name"] == "answers the question"
    assert first["test"]["fullName"] == "example eval set > answers the question"
    assert first["test"]["status"] == "passed"
    assert first["test"]["file"] == "test_example_eval.py"
    assert first["usage"] == {
        "provider": "faux",
        "model": "faux-model",
        "inputTokens": 1,
        "outputTokens": 2,
        "totalTokens": 100,
        "toolCalls": 0,
        "metadata": {"estimatedCostUsd": 0.5},
    }
    assert first["timings"] == {"totalMs": 10.0}
    assert first["metadata"]["vitestEvalsHarnessIteration"]["evalSet"] == "example eval set"
    assert first["artifacts"][0]["name"] == "session.jsonl"

    session_path = artifact_directory / first["artifacts"][0]["path"]
    assert session_path.read_text(encoding="utf-8") == '{"type":"session"}\n'

    result.stdout.fnmatch_lines(["*Eval Comparisons*", "*Pass rate*+100.0 pp*"])


def test_reporter_loads_without_xdist_installed(pytester, monkeypatch, tmp_path: Path) -> None:
    """`pytest_testnodedown` is an xdist hookspec.

    Declaring it at module level makes pytest reject the whole plugin with
    `PluginValidationError: unknown hook` whenever xdist is absent, so the
    controller-side collector is registered only when xdist is loaded.
    """
    artifact_directory = tmp_path / "eval-artifacts"
    monkeypatch.setenv("PI_EVAL_ARTIFACT_DIR", str(artifact_directory))
    pytester.makeini("[pytest]\nasyncio_mode = auto\n")
    pytester.makepyfile(test_example_eval=_EVAL_MODULE)

    result = pytester.runpytest_subprocess("-q", "-p", "no:xdist")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*Eval Comparisons*"])


def test_reporter_survives_xdist_workers(pytester, monkeypatch, tmp_path: Path) -> None:
    """Under `-n auto` the tests, and their observations, live in the workers.

    `pytest_terminal_summary` only runs in the controller, so without the
    `workeroutput` handover the comparison report would silently disappear --
    and `runs.jsonl` would be appended by several processes at once.
    """
    artifact_directory = tmp_path / "eval-artifacts"
    monkeypatch.setenv("PI_EVAL_ARTIFACT_DIR", str(artifact_directory))
    pytester.makeini("[pytest]\nasyncio_mode = auto\n")
    pytester.makepyfile(test_example_eval=_EVAL_MODULE)

    result = pytester.runpytest_subprocess("-q", "-n", "2")
    result.assert_outcomes(passed=2)

    records = {
        json.loads(line)["harness"]: json.loads(line)
        for line in (artifact_directory / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert sorted(records) == ["baseline", "candidate"]
    assert {record["test"]["status"] for record in records.values()} == {"passed"}

    result.stdout.fnmatch_lines(["*Eval Comparisons*", "*Pass rate*+100.0 pp*"])
