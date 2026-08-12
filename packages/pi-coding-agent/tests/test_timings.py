"""Coverage for `packages/coding-agent/src/core/timings.ts` and its call sites.

There is no `test/timings.test.ts` upstream; the module is pinned here because
the Python port shipped a faithful translation of it that *nothing called*, so
`PI_TIMING=1` printed a startup profile in TypeScript and printed nothing at
all in Python. The last case runs the real CLI in a subprocess so it exercises
the wiring in `cli/entry.py`, not just the recorder.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from _cli_process import run_cli


@pytest.fixture(autouse=True)
def _restore_timings_module():
    """Undo this module's `importlib.reload` before another test file runs.

    `reload` mutates the single shared `core.timings` module object: its
    env-derived `ENABLED` and its module-level timing store. Production
    importers bind `time`/`reset_timings` at import (`sdk.py`,
    `extensions/loader.py`, `cli/entry.py`), so a left-over reloaded state is
    visible to every later test in the same xdist worker -- which is how
    `test_time_starts_a_namespace_that_was_never_reset` failed roughly one run
    in ten while passing in isolation.

    Reloading here, after the test's `monkeypatch` has already restored
    `PI_TIMING`, puts the module back in the state the rest of the suite
    expects.
    """
    yield
    importlib.reload(importlib.import_module("pi_coding_agent.core.timings"))


def _load_timings(enabled: bool, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Re-import `core.timings` with `PI_TIMING` set.

    `ENABLED` is captured at module load in both languages
    (`const ENABLED = process.env.PI_TIMING === "1"`), so toggling it needs a
    fresh module object rather than a `monkeypatch.setenv` alone.
    """
    if enabled:
        monkeypatch.setenv("PI_TIMING", "1")
    else:
        monkeypatch.delenv("PI_TIMING", raising=False)
    module = importlib.import_module("pi_coding_agent.core.timings")
    return importlib.reload(module)


def test_records_nothing_when_disabled(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    timings = _load_timings(False, monkeypatch)
    timings.reset_timings()
    timings.time("parseArgs")
    timings.print_timings()
    assert capsys.readouterr().err == ""


def test_prints_a_group_per_namespace_when_enabled(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    timings = _load_timings(True, monkeypatch)
    timings.reset_timings()
    timings.time("parseArgs")
    timings.time("runMigrations")
    timings.reset_timings("extensions")
    timings.time("a.py factory", "extensions")
    timings.print_timings()

    err = capsys.readouterr().err
    assert "--- Startup Timings: main ---" in err
    assert "--- Startup Timings: extensions ---" in err
    assert "  parseArgs: " in err
    assert "  runMigrations: " in err
    assert "  a.py factory: " in err
    assert err.count("  TOTAL: ") == 2


def test_time_starts_a_namespace_that_was_never_reset(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`time()` on an unknown namespace calls `resetTimings` first.

    The first label therefore measures from that moment, i.e. ~0ms, and is
    still printed (the `ms >= 0` filter keeps it).
    """
    timings = _load_timings(True, monkeypatch)
    timings.reset_timings()
    timings.time("first", "extensions")
    timings.print_timings()
    assert "  first: " in capsys.readouterr().err


def test_empty_namespaces_print_no_header(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    timings = _load_timings(True, monkeypatch)
    timings.reset_timings()
    timings.reset_timings("extensions")
    timings.print_timings()
    assert capsys.readouterr().err == ""


def test_cli_prints_startup_timings_with_pi_timing(tmp_path: Path) -> None:
    """`PI_TIMING=1 pi -p ...` prints the profile `main.ts` prints.

    Without a model configured the run fails, but the timing calls up to that
    point have already been recorded and the labels must appear.
    """
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    project = tmp_path / "project"
    project.mkdir()

    result = run_cli(
        ["-p", "hello"],
        cwd=str(project),
        agent_dir=str(agent_dir),
        env={"PI_TIMING": "1"},
    )

    assert "--- Startup Timings: main ---" in result.stderr
    assert "parseArgs: " in result.stderr
    assert "runMigrations: " in result.stderr
    assert "readPipedStdin: " in result.stderr
    assert "TOTAL: " in result.stderr


def test_cli_prints_nothing_without_pi_timing(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    project = tmp_path / "project"
    project.mkdir()

    env = {k: v for k, v in os.environ.items() if k != "PI_TIMING"}
    result = run_cli(["-p", "hello"], cwd=str(project), agent_dir=str(agent_dir), env=env)

    assert "Startup Timings" not in result.stderr
