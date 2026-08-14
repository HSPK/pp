"""Helper for tests that need to run the CLI in a real subprocess.

Mirrors the `spawn(process.execPath, [cliPath, ...])` pattern used by
`packages/coding-agent/test/session-file-invalid.test.ts` and
`packages/coding-agent/test/session-id-readonly.test.ts`: those tests assert on
process exit codes and on stderr *not* containing stack traces, which only an
out-of-process run can show.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from pi_coding_agent.core.config import ENV_AGENT_DIR

_RUNNER = "import sys; from pi_coding_agent.cli import main; sys.exit(main())"


@dataclass
class CliResult:
    code: int
    stdout: str
    stderr: str


def run_cli(
    args: list[str],
    cwd: str,
    agent_dir: str,
    env: dict[str, str] | None = None,
    timeout: float = 60.0,
    stdin: str = "",
) -> CliResult:
    """Run the CLI with `agent_dir` as its agent directory and no network access.

    `timeout` mirrors the TypeScript watchdog that SIGKILLs a hung child. A
    signal-killed child would surface as a negative `code`, which is how the
    `expect(result.signal).toBeNull()` assertions are covered here.

    `stdin` is always a pipe, never inherited, and defaults to empty so the
    child sees EOF immediately. Inheriting is actively dangerous: under
    `pytest-xdist` the runner's stdin is the execnet control channel, so a CLI
    that reads stdin -- `--mode rpc` reads it as its command channel, and print
    mode reads it as a piped prompt -- consumes the worker's instructions and
    deadlocks the whole run.
    """
    process_env = {
        **os.environ,
        ENV_AGENT_DIR: agent_dir,
        "PI_OFFLINE": "1",
        **(env or {}),
    }
    completed = subprocess.run(
        [sys.executable, "-c", _RUNNER, *args],
        cwd=cwd,
        env=process_env,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return CliResult(code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
