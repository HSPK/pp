"""Eval reporter, as a pytest plugin.

Python port of `packages/evals/src/vitest-evals/reporter.ts` and
`packages/evals/src/vitest-evals/setup.ts`.

There is no Vitest here, so the TypeScript `Reporter` class becomes a pytest
plugin registered through this package's `pytest11` entry point:

| Vitest | pytest |
| --- | --- |
| `setup.ts`'s `afterEach` recording the Pi session snapshot | `pytest_runtest_teardown` |
| `Reporter.onTestCaseResult` appending `runs.jsonl` | `pytest_runtest_teardown`, after the snapshot is recorded |
| `Reporter.onTestRunEnd` printing the comparison report | `pytest_terminal_summary` |

The on-disk layout is unchanged: with `PI_EVAL_ARTIFACT_DIR` set, every
completed harness run appends one JSON object to `<dir>/runs.jsonl` and its
attachments are written under `<dir>/sessions/` and `<dir>/sources/`. Records
keep the TypeScript key spelling (`schemaVersion`, `runId`, `fullName`,
`totalTokens`, ...) so both implementations' artifact directories can be read
by the same tooling.

**Record order is completion order, not declaration order**, exactly as
upstream: `onTestCaseResult` appends as each test finishes and Vitest runs
test files concurrently. Nothing downstream depends on it -- `summary.py`
pairs observations by group key and sorts eval sets, candidates and groups
itself.

**One difference forced by pytest-xdist.** Vitest reporters run in the main
process, so upstream both appends and summarizes from one place. Under
`-n auto` the plugin is loaded in every worker: `pytest_runtest_teardown`
(and therefore the append) runs in the worker that executed the test, while
`pytest_terminal_summary` only runs in the controller, which never saw those
tests. Workers therefore ship their observations to the controller through
xdist's `workeroutput` channel, and the append takes an exclusive `flock` so
concurrent workers cannot interleave partial lines. The controller-side
collector is a separate plugin object registered only when xdist is loaded,
because `pytest_testnodedown` is an xdist hookspec and pytest rejects a plugin
declaring hooks it does not know.
"""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pytest

from pi_evals.harness import EVAL_META_KEY, EvalMeta, JsonValue, is_harness_run
from pi_evals.vitest_evals.artifacts import (
    PI_SESSION_SNAPSHOT_ARTIFACT,
    persist_eval_artifact_references,
    record_eval_session_artifact,
)
from pi_evals.vitest_evals.harness_table import (
    EVAL_HARNESS_ITERATION_ARTIFACT,
    parse_eval_harness_iteration_artifact,
)
from pi_evals.vitest_evals.summary import (
    HarnessObservation,
    format_harness_comparison_report,
    summarize_harness_comparisons,
)

_COMPLETED_METAS: pytest.StashKey[list[EvalMeta]] = pytest.StashKey()
"""Session-level stash holding every eval test's meta, for the final summary."""

_WORKER_OBSERVATIONS: pytest.StashKey[list[HarnessObservation]] = pytest.StashKey()
"""Controller-side stash holding observations shipped back by xdist workers."""

WORKER_OUTPUT_KEY = "pi_evals_observations"
"""Key under which a worker hands its observations to the controller."""

_XDIST_COLLECTOR_NAME = "pi_evals_xdist_observations"
"""Plugin name for the controller-side collector, registered only under xdist."""

_interrupted = False
"""Set by `pytest_keyboard_interrupt`; the comparison report is unavailable then."""


def _read_finite_number(value: object) -> float | None:
    """Port of `readFiniteNumber`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _artifact_directory() -> str | None:
    directory = os.environ.get("PI_EVAL_ARTIFACT_DIR", "").strip()
    return directory or None


def _relative_module_id(item: pytest.Item) -> str:
    """Vitest's `module.relativeModuleId`: the test file relative to the run root."""
    path = Path(str(getattr(item, "path", item.fspath)))
    try:
        return str(path.relative_to(Path(str(item.config.rootpath))))
    except ValueError:
        return str(path)


def append_harness_run_report(item: pytest.Item, meta: EvalMeta) -> dict[str, JsonValue] | None:
    """Port of `appendHarnessRunReport`.

    Returns the appended record (or `None` when there was nothing to append),
    which keeps the record shape directly testable.
    """
    artifact_directory = _artifact_directory()
    if artifact_directory is None:
        return None
    harness = meta.harness
    if harness is None or not is_harness_run(harness.run):
        return None

    run = harness.run
    assert run is not None
    artifact_run_id = run.artifacts.get("runId")
    run_id = artifact_run_id if isinstance(artifact_run_id, str) else str(uuid.uuid4())
    metadata = {
        name: value for name, value in run.artifacts.items() if name not in ("runId", PI_SESSION_SNAPSHOT_ARTIFACT)
    }
    record: dict[str, JsonValue] = {
        "schemaVersion": 1,
        "runId": run_id,
        "test": {
            "id": item.nodeid,
            "file": _relative_module_id(item),
            "name": meta.case_name,
            "fullName": f"{meta.eval_name} > {meta.case_name}",
            "status": meta.status,
        },
        "harness": harness.name,
        "usage": run.usage.to_json(),
    }
    if run.timings is not None:
        record["timings"] = run.timings.to_json()
    if run.errors:
        record["errors"] = list(run.errors)
    record["artifacts"] = [
        dict(reference)
        for reference in persist_eval_artifact_references(meta.task.artifacts, run_id, artifact_directory)
    ]
    if metadata:
        record["metadata"] = metadata

    directory = Path(artifact_directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    runs_path = directory / "runs.jsonl"
    existed = runs_path.exists()
    line = f"{json.dumps(record, ensure_ascii=False, separators=(',', ':'))}\n"
    # Under `-n auto` several workers append concurrently, so take an
    # exclusive lock: an append larger than a pipe buffer is not atomic, and a
    # torn line would make `runs.jsonl` unparseable.
    with runs_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    if not existed:
        runs_path.chmod(0o600)
    return record


def collect_harness_observations(metas: Iterable[EvalMeta]) -> list[HarnessObservation]:
    """Port of `collectHarnessObservations`."""
    observations: list[HarnessObservation] = []
    for meta in metas:
        harness = meta.harness
        if harness is None or not is_harness_run(harness.run):
            continue
        run = harness.run
        assert run is not None
        iteration = parse_eval_harness_iteration_artifact(run.artifacts.get(EVAL_HARNESS_ITERATION_ARTIFACT))
        if iteration is None:
            continue
        score = _read_finite_number(meta.eval.avg_score) if meta.eval is not None else None
        estimated_cost_usd = _read_finite_number(run.usage.metadata.get("estimatedCostUsd"))
        observation = HarnessObservation(
            eval_set=iteration.eval_set,
            group_key=iteration.group_key,
            test_name=meta.case_name,
            file=meta.file,
            harness=iteration.harness,
            baseline=iteration.baseline,
            candidates=list(iteration.candidates),
            repetition=iteration.repetition,
            total_tokens=run.usage.total_tokens,
            total_ms=run.timings.total_ms if run.timings is not None else None,
            estimated_cost_usd=estimated_cost_usd,
            outcome="scored",
        )
        if run.errors:
            observation.outcome = "errored"
        elif score is not None:
            observation.score = score
        elif meta.status == "passed":
            observation.outcome = "unscored"
        elif meta.status == "failed":
            observation.outcome = "errored"
        else:
            observation.outcome = meta.status
        observations.append(observation)
    return observations


# ==========================================================================
# pytest hooks
# ==========================================================================


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_COMPLETED_METAS] = []
    config.stash[_WORKER_OBSERVATIONS] = []
    # `pytest_testnodedown` is an xdist hookspec, and pytest rejects a plugin
    # declaring hooks it does not know, so it lives on a separate object that
    # is only registered when xdist is actually loaded.
    if config.pluginmanager.hasplugin("xdist") and not config.pluginmanager.hasplugin(_XDIST_COLLECTOR_NAME):
        config.pluginmanager.register(_XdistObservationCollector(), _XDIST_COLLECTOR_NAME)


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> None:
    meta = item.stash.get(EVAL_META_KEY, None)
    if meta is None or call.when != "call":
        return
    if call.excinfo is None:
        meta.status = "passed"
    elif call.excinfo.errisinstance(pytest.skip.Exception):
        meta.status = "skipped"
    else:
        meta.status = "failed"


def pytest_runtest_teardown(item: pytest.Item) -> None:
    meta = item.stash.get(EVAL_META_KEY, None)
    if meta is None:
        return
    meta.file = _relative_module_id(item)
    # Port of `setup.ts`'s eval-only `afterEach`: register the Pi session
    # snapshot against this test before the reporter persists artifacts.
    if meta.harness is not None and meta.harness.run is not None:
        record_eval_session_artifact(meta.task, meta.harness.run)
    append_harness_run_report(item, meta)
    item.config.stash[_COMPLETED_METAS].append(meta)


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Hand this xdist worker's observations to the controller.

    Only workers have `config.workeroutput`; in a serial run this does
    nothing and `pytest_terminal_summary` summarizes the local metas.
    """
    worker_output = getattr(session.config, "workeroutput", None)
    if worker_output is None:
        return
    metas: Sequence[EvalMeta] = session.config.stash.get(_COMPLETED_METAS, [])
    worker_output[WORKER_OUTPUT_KEY] = [asdict(observation) for observation in collect_harness_observations(metas)]


class _XdistObservationCollector:
    """Collects each finished xdist worker's observations on the controller."""

    def pytest_testnodedown(self, node: Any, error: object) -> None:
        del error
        shipped = getattr(node, "workeroutput", {}).get(WORKER_OUTPUT_KEY)
        if not shipped:
            return
        node.config.stash[_WORKER_OBSERVATIONS].extend(observations_from_json(shipped))


def observations_from_json(records: Iterable[dict[str, object]]) -> list[HarnessObservation]:
    """Rebuild observations shipped across the xdist worker/controller boundary."""
    known = {observation_field.name for observation_field in fields(HarnessObservation)}
    return [HarnessObservation(**{key: value for key, value in record.items() if key in known}) for record in records]


def pytest_keyboard_interrupt(excinfo: pytest.ExceptionInfo[BaseException]) -> None:
    global _interrupted
    _interrupted = True


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """Port of `Reporter.onTestRunEnd`."""
    if _interrupted:
        terminalreporter.write_line("\nEval comparisons unavailable: test run interrupted.")
        return
    metas: Sequence[EvalMeta] = config.stash.get(_COMPLETED_METAS, [])
    observations = [*config.stash.get(_WORKER_OBSERVATIONS, []), *collect_harness_observations(metas)]
    if not observations:
        return
    report = summarize_harness_comparisons(observations)
    formatted = format_harness_comparison_report(report)
    if formatted:
        terminalreporter.write_line(f"\n{formatted}")


__all__ = [
    "WORKER_OUTPUT_KEY",
    "append_harness_run_report",
    "collect_harness_observations",
    "observations_from_json",
]
