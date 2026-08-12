"""End-to-end test of the `pp-evals` runner against the ported evals.

Drives `pi_evals.run_evals.main` -- the port of
`packages/evals/scripts/run-evals.mjs` -- over the real eval modules in
`pi_evals.evals` (ports of `packages/evals/src/smoke.eval.ts` and
`packages/evals/src/extensions.eval.ts`), with `pi_ai`'s scripted `faux`
provider substituted for the model runtime, so the whole path runs offline:
argument parsing, artifact-directory creation, pytest collection of
`*_eval.py`, group-key derivation from the evals' dataclass step input, the
`AgentSession` harness, and the reporter's `.eval/runs.jsonl` index.

`uv run pp-evals` previously failed here with a canonicalizer `TypeError`,
which no unit test caught, so this exercises the runner itself rather than its
parts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pi_evals.run_evals import EVALS_PATH, main
from pi_evals.vitest_evals.summary import strip_ansi

_FAUX_PLUGIN = '''
"""Redirects the Pi harness at an authenticated `faux` provider."""

from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_coding_agent.core.model_runtime import ModelRuntime

from pi_evals import pi_harness

AGENT_DIR = {agent_dir!r}

_faux = faux_provider()
_faux.set_responses([faux_assistant_message("Paris") for _ in range(64)])


class _FauxModelRuntime:
    @staticmethod
    async def create():
        runtime = await ModelRuntime.create(agent_dir=AGENT_DIR, providers=[_faux.provider])
        await runtime.login(_faux.provider.id, "faux-key")
        return runtime


def pytest_configure(config):
    pi_harness.ModelRuntime = _FauxModelRuntime
'''


@pytest.fixture
def offline_runner(tmp_path: Path, monkeypatch):
    """Run the runner with the faux provider injected into its pytest subprocess."""
    plugin_directory = tmp_path / "plugins"
    plugin_directory.mkdir()
    agent_directory = tmp_path / "runtime-agent"
    agent_directory.mkdir()
    (plugin_directory / "faux_eval_plugin.py").write_text(
        _FAUX_PLUGIN.format(agent_dir=str(agent_directory)), encoding="utf-8"
    )
    artifact_directory = tmp_path / "artifacts"

    monkeypatch.setenv("PYTHONPATH", str(plugin_directory))
    monkeypatch.setenv("PYTEST_ADDOPTS", "-p faux_eval_plugin")
    monkeypatch.setenv("PI_EVAL_ARTIFACT_DIR", str(artifact_directory))
    monkeypatch.delenv("PI_PROVIDER", raising=False)
    monkeypatch.delenv("PI_MODEL", raising=False)
    return artifact_directory


def _records(artifact_directory: Path) -> list[dict]:
    return [json.loads(line) for line in (artifact_directory / "runs.jsonl").read_text(encoding="utf-8").splitlines()]


def test_runs_the_smoke_eval_and_indexes_its_session(offline_runner: Path) -> None:
    exit_code = main(["--provider", "faux", "--model", "faux-1", str(EVALS_PATH / "smoke_eval.py")])

    assert exit_code == 0
    records = _records(offline_runner)
    assert len(records) == 1
    record = records[0]
    assert record["schemaVersion"] == 1
    assert record["harness"] == "pi-coding-agent"
    assert record["test"] == {
        "id": (
            "packages/pi-evals/src/pi_evals/evals/smoke_eval.py"
            "::test_pi_coding_agent_smoke__runs_a_basic_prompt_end_to_end"
        ),
        "file": "packages/pi-evals/src/pi_evals/evals/smoke_eval.py",
        "name": "runs a basic prompt end to end",
        "fullName": "Pi Coding Agent smoke > runs a basic prompt end to end",
        "status": "passed",
    }
    assert record["usage"]["provider"] == "faux"
    assert record["usage"]["model"] == "faux-1"
    assert record["usage"]["totalTokens"] > 0
    assert record["timings"]["totalMs"] >= 0
    assert "errors" not in record
    assert record["artifacts"][0]["name"] == "session.jsonl"

    session_path = offline_runner / record["artifacts"][0]["path"]
    assert json.loads(session_path.read_text(encoding="utf-8").splitlines()[0])


def test_derives_group_keys_for_the_comparative_extensions_eval(offline_runner: Path) -> None:
    """The comparative eval's input is a list of step dataclasses.

    Deriving its group key is what `uv run pp-evals` used to fail on, and the
    reporter cannot pair baseline with candidate without it.

    `runs.jsonl` is keyed by harness rather than indexed: records are appended
    in completion order (`Reporter.onTestCaseResult` upstream,
    `pytest_runtest_teardown` here), and the runner inherits `-n auto`, so the
    two rows finish in whichever order their workers happen to finish.
    """
    exit_code = main(["--provider", "faux", "--model", "faux-1", str(EVALS_PATH / "extensions_eval.py")])

    assert exit_code == 0
    records = {record["harness"]: record for record in _records(offline_runner)}
    assert sorted(records) == ["default-system-prompt", "system-prompt-without-docs"]
    assert all(record["test"]["status"] == "passed" for record in records.values())

    iterations = {harness: record["metadata"]["vitestEvalsHarnessIteration"] for harness, record in records.items()}
    for harness, iteration in iterations.items():
        assert iteration["harness"] == harness
        assert iteration["evalSet"] == "Pi extension authoring system prompt"
        assert iteration["repetition"] == 1
        assert iteration["baseline"] == "system-prompt-without-docs"
        assert iteration["candidates"] == ["default-system-prompt"]
    # One group per input: both harnesses ran the same step sequence, so the
    # reporter can pair them.
    assert len({iteration["groupKey"] for iteration in iterations.values()}) == 1


def test_prints_the_comparison_report_from_parallel_workers(offline_runner: Path, capfd) -> None:
    """The report must survive pytest-xdist.

    Vitest reporters run in the main process; here `pytest_terminal_summary`
    runs in the xdist controller while the tests -- and their observations --
    live in the workers, so the workers ship them back over `workeroutput`.
    Without that the runner silently prints no comparison at all.
    """
    exit_code = main(["--provider", "faux", "--model", "faux-1", "-n", "2", str(EVALS_PATH / "extensions_eval.py")])

    assert exit_code == 0
    # `capfd`, not `capsys`: the report is written by the runner's pytest
    # subprocess, on the inherited file descriptor.
    output = strip_ansi(capfd.readouterr().out)
    assert "Eval Comparisons" in output
    assert "Pi extension authoring system prompt" in output
    assert "Baseline  system-prompt-without-docs" in output
    assert "Candidate  default-system-prompt (1/1 pairs)" in output


def test_reports_a_missing_default_model_selection(offline_runner: Path, capsys) -> None:
    assert main(["--provider", "faux"]) == 1
    assert "CLI model selection requires both --provider and --model." in capsys.readouterr().err
    assert not (offline_runner / "runs.jsonl").exists()
