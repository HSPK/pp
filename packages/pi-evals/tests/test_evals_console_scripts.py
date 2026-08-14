"""Smoke test that runs this package's console script as a real subprocess.

Every other test here imports the code under test, which misses wiring that
only executes when the entry point actually runs. `pp-ai login github-copilot`
once crashed with "'coroutine' object has no attribute 'login'" while the whole
suite was green, because the CLI's own tests replaced the provider table with a
stub. Nothing ever ran the real command.

This used to live in `pp-coding-agent`'s suite, which covered every script in
the monorepo workspace. `pp-evals` depends on `pp-coding-agent`, so its script
is not installed in that package's environment once the two are separate
distributions -- the check has to run from the side that owns the script.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(sys.executable).parent
"""The `bin/` of the interpreter running the suite.

Resolving the script here rather than through `uv run` tests the environment
the suite is actually running in, and needs no workspace on disk.
"""


def test_console_script_runs() -> None:
    executable = SCRIPT_DIR / "pp-evals"
    assert executable.exists(), f"pp-evals is not installed in {SCRIPT_DIR}"

    result = subprocess.run(
        [str(executable), "--help"],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        text=True,
        timeout=300,
        # Closed stdin, so a command that waits for input fails fast instead of
        # hanging the suite.
        stdin=subprocess.DEVNULL,
        env={**os.environ, "PI_OFFLINE": "1", "NO_COLOR": "1"},
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "usage" in combined.lower(), combined


def test_entry_point_target_is_importable_and_callable() -> None:
    """`project.scripts` must resolve, so a rename cannot break the script."""
    result = subprocess.run(
        [sys.executable, "-c", "import pi_evals.run_evals as m; assert callable(m.main)"],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, f"pi_evals.run_evals:main is not importable/callable:\n{result.stderr}"
