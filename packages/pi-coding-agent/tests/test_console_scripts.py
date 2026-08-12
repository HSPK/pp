"""Smoke tests that run each installed console script as a real subprocess.

Every other test in this repo imports the code under test. That misses a whole
class of failure: wiring that only executes when the entry point actually runs.
`pp-ai login github-copilot` crashed with "'coroutine' object has no
attribute 'login'" while the entire suite was green, because the CLI's own
tests replaced the provider table with a stub whose `login` was reachable
synchronously. Nothing ever ran the real command.

So these run the scripts through `subprocess`, exactly as a user does. They
must stay fast and offline: only commands that neither hit the network nor
block on input belong here. Stdin is closed, so a command that waits for input
fails fast instead of hanging.

This file covers every `project.scripts` entry in the workspace, not just
`pi-ai`'s, which is why it lives here rather than in one package's tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# (script, args, substring that must appear in the output)
SCRIPT_INVOCATIONS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("pp-ai", ("--help",), "login"),
    ("pp-ai", ("list",), "anthropic"),
    ("pp", ("--help",), "Usage"),
    ("pp", ("--version",), "."),
    ("pp-evals", ("--help",), "usage"),
)

_IDS = [f"{script} {' '.join(args)}".strip() for script, args, _ in SCRIPT_INVOCATIONS]


def run_script(script: str, args: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    """Run an installed console script with stdin closed, so a hang fails fast."""
    environment = {**os.environ, "PI_OFFLINE": "1", "NO_COLOR": "1"}
    return subprocess.run(
        ["uv", "run", script, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        stdin=subprocess.DEVNULL,
        env=environment,
        check=False,
    )


@pytest.mark.parametrize(("script", "args", "expected"), SCRIPT_INVOCATIONS, ids=_IDS)
def test_console_script_runs(script: str, args: tuple[str, ...], expected: str) -> None:
    result = run_script(script, args)
    combined = result.stdout + result.stderr

    assert result.returncode == 0, f"{script} {' '.join(args)} exited {result.returncode}:\n{combined}"
    assert expected.lower() in combined.lower(), combined


@pytest.mark.parametrize(("script", "args", "_expected"), SCRIPT_INVOCATIONS, ids=_IDS)
def test_console_script_never_leaves_a_coroutine_unawaited(script: str, args: tuple[str, ...], _expected: str) -> None:
    """A forgotten `await` surfaces as a RuntimeWarning, not a non-zero exit.

    The `pp-ai login` bug printed `coroutine ... was never awaited` on stderr
    while still exiting cleanly, so an exit-code check alone would have missed
    it.
    """
    result = run_script(script, args)
    combined = result.stdout + result.stderr

    assert "was never awaited" not in combined, combined
    assert "object has no attribute" not in combined, combined


def test_login_reaches_the_provider_flow() -> None:
    """`pp-ai login <provider>` must get as far as prompting the user.

    Running the real command with stdin closed means the flow starts, asks its
    first question, reads EOF and stops. That proves the CLI resolved a real
    flow object rather than an unawaited coroutine, without completing a login.
    """
    result = run_script("pp-ai", ("login", "github-copilot"))
    combined = result.stdout + result.stderr

    assert "object has no attribute" not in combined, combined
    assert "was never awaited" not in combined, combined
    # The GitHub Copilot flow's first step is the Enterprise-domain question.
    assert "github" in combined.lower(), combined


@pytest.mark.parametrize(
    "target",
    ["pi_ai.cli:main", "pi_coding_agent.cli:main", "pi_evals.run_evals:main"],
)
def test_entry_point_target_is_importable_and_callable(target: str) -> None:
    """Each `project.scripts` target must resolve, so a rename cannot break the script."""
    module_name, attribute = target.split(":")
    result = subprocess.run(
        [sys.executable, "-c", f"import {module_name} as m; assert callable(m.{attribute})"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, f"{target} is not importable/callable:\n{result.stderr}"


def test_list_models_builds_every_provider_for_real() -> None:
    """`pp --list-models` runs the whole provider-composition path in a real process.

    `ModelRuntime.create` composes every built-in provider with its
    `models.json` overlay, which is where auth methods are assembled. Unit
    tests reach that code with hand-built providers; this reaches it with the
    real catalog, so a provider whose composition raises cannot stay hidden
    behind a test fixture. Offline: listing reads the local catalog only.
    """
    result = run_script("pp", ("--list-models",))
    combined = result.stdout + result.stderr

    assert result.returncode == 0, combined
    assert combined.strip(), "expected either a model listing or the no-models hint"
    assert "traceback" not in combined.lower(), combined
    assert "no authentication method configured" not in combined, combined
