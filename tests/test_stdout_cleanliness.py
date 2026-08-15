"""Python port of `packages/coding-agent/test/stdout-cleanliness.test.ts`."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from _cli_process import CliResult, run_cli


def _run(tmp_path: Path, args: list[str]) -> CliResult:
    agent_dir = tmp_path / "agent"
    project_dir = tmp_path / "project"
    project_config_dir = project_dir / ".pi"
    agent_dir.mkdir(parents=True)
    project_config_dir.mkdir(parents=True)

    fake_npm = tmp_path / "fake_npm.py"
    fake_npm.write_text('print("changed 1 package in 471ms")\nprint("found 0 vulnerabilities")\n')
    (project_config_dir / "settings.json").write_text(
        json.dumps({"packages": ["npm:fake-package"], "npmCommand": [sys.executable, str(fake_npm)]}, indent=2)
    )

    return run_cli(args, str(project_dir), str(agent_dir), env={"PI_OFFLINE": ""})


def test_prints_version_to_stdout_when_stdout_is_redirected(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--version"])

    assert result.code == 0
    assert re.match(r"^\d+\.\d+\.\d+", result.stdout.strip())
    assert result.stderr == ""


def test_prints_plain_help_to_stdout_when_stdout_is_redirected(tmp_path: Path) -> None:
    result = _run(tmp_path, ["--help"])

    assert result.code == 0
    assert "Usage:" in result.stdout
    assert "Usage:" not in result.stderr


def test_keeps_stdout_empty_for_mode_json_help(tmp_path: Path) -> None:
    # TS also asserts the npm chatter ("changed 1 package in 471ms" / "found 0
    # vulnerabilities") is routed to stderr, because TS builds the whole runtime -- and
    # therefore installs `settings.packages` via ResourceLoader -> PackageManager.resolve()
    # -- before printing `--help`. This port's ResourceLoader deliberately takes explicit
    # resource paths instead of calling PackageManager.resolve() (see the scope note in
    # core/resource_loader.py), and PackageManager is reachable only from the explicit
    # `pi install`/`pi update` subcommands, so no install runs at startup and there is no
    # chatter to route. The stdout-must-stay-empty half of the case -- what the file is
    # actually about -- is asserted below.
    result = _run(tmp_path, ["--mode", "json", "--help", "--approve"])

    assert result.code == 0
    assert result.stdout == ""
    assert "Usage:" in result.stderr


def test_keeps_stdout_empty_for_print_help(tmp_path: Path) -> None:
    # TS also asserts the npm chatter ("changed 1 package in 471ms" / "found 0
    # vulnerabilities") is routed to stderr, because TS builds the whole runtime -- and
    # therefore installs `settings.packages` via ResourceLoader -> PackageManager.resolve()
    # -- before printing `--help`. This port's ResourceLoader deliberately takes explicit
    # resource paths instead of calling PackageManager.resolve() (see the scope note in
    # core/resource_loader.py), and PackageManager is reachable only from the explicit
    # `pi install`/`pi update` subcommands, so no install runs at startup and there is no
    # chatter to route. The stdout-must-stay-empty half of the case -- what the file is
    # actually about -- is asserted below.
    result = _run(tmp_path, ["-p", "--help", "--approve"])

    assert result.code == 0
    assert result.stdout == ""
    assert "Usage:" in result.stderr


def test_ignores_untrusted_project_package_installs_for_help(tmp_path: Path) -> None:
    result = _run(tmp_path, ["-p", "--help"])

    assert result.code == 0
    assert result.stdout == ""
    assert "changed 1 package in 471ms" not in result.stderr
    assert "found 0 vulnerabilities" not in result.stderr
    assert "Usage:" in result.stderr
