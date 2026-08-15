"""`--mode rpc` is advertised in `--help`, so it has to work.

The mode existed in the parser (`VALID_MODES`), in `resolve_app_mode` and in the
help text; `run_app` answered it with "RPC mode is not ported" and exit 2. That
is the flag-shaped defect `test_cli_dead_flags.py` covers, one level up: a whole
mode the CLI offers and refuses.

These cases cover the CLI's side of the wiring. What the mode *does* once
running is covered by `tests/suite/test_rpc_mode.py` and
`tests/suite/test_rpc_mode_driver.py`.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from _cli_process import run_cli

from pi_coding_agent.cli import entry as cli
from pi_coding_agent.cli.args import HELP_TEXT, parse_args


def test_help_advertises_rpc() -> None:
    """If this ever stops being true, the tests below are no longer required."""
    assert "rpc" in HELP_TEXT


def test_rpc_resolves_to_its_own_app_mode() -> None:
    parsed = parse_args(["--mode", "rpc"])
    assert cli.resolve_app_mode(parsed, stdin_is_tty=False, stdout_is_tty=False) == "rpc"


def test_rpc_mode_serves_the_protocol_end_to_end(tmp_path: Path) -> None:
    """The whole point of the mode: spawn it, speak JSON lines, get answers.

    This is the case the old behaviour failed -- `--mode rpc` exited 2 with
    "RPC mode is not ported" before reaching any of it.
    """
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    commands = [
        {"id": "1", "type": "get_state"},
        {"id": "2", "type": "bash", "command": "echo hello-from-rpc"},
        {"id": "3", "type": "set_session_name", "name": "smoke"},
        {"id": "4", "type": "nope"},
    ]
    stdin = "".join(json.dumps(command) + "\n" for command in commands) + "{bad json\n"

    result = run_cli(["--mode", "rpc"], cwd=str(tmp_path), agent_dir=str(agent_dir), timeout=120, stdin=stdin)

    assert result.code == 0
    assert "not ported" not in result.stderr

    # Every stdout line must be a JSON record: a stray print would break the
    # host's parser, which is why the mode takes over stdout.
    lines = [json.loads(line) for line in result.stdout.splitlines()]
    responses = {line["id"]: line for line in lines if line.get("type") == "response" and line.get("id")}

    assert responses["1"]["success"] is True
    assert responses["1"]["data"]["isStreaming"] is False
    assert "hello-from-rpc" in responses["2"]["data"]["output"]
    assert responses["3"]["success"] is True
    assert responses["4"] == {
        "id": "4",
        "type": "response",
        "command": "nope",
        "success": False,
        "error": "Unknown command: nope",
    }

    parse_errors = [line for line in lines if line.get("command") == "parse"]
    assert parse_errors, "the malformed trailing line should have been answered"


def test_rpc_mode_rejects_file_arguments(tmp_path: Path) -> None:
    """`@file` prepends file contents to a prompt, and RPC mode has no prompt
    to prepend to -- the host sends messages as commands instead. TypeScript
    exits 1 here rather than silently dropping the files.
    """
    target = tmp_path / "notes.txt"
    target.write_text("hello")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()

    result = run_cli(["--mode", "rpc", f"@{target}"], cwd=str(tmp_path), agent_dir=str(agent_dir), timeout=60)
    assert result.code == 1
    assert "@file arguments are not supported in RPC mode" in result.stderr


def test_rpc_mode_does_not_drain_stdin(monkeypatch, tmp_path: Path) -> None:
    """Stdin is the command channel, so reading it as a piped prompt would eat
    the host's first commands and then report "No prompt given".
    """
    reads: list[bool] = []
    monkeypatch.setattr(cli, "read_piped_stdin", lambda tty: reads.append(tty) or "piped text")

    modes = _run_main_capturing_mode(monkeypatch, tmp_path, ["--mode", "rpc"])
    assert modes == ["rpc"]
    assert reads == [], "stdin was drained before the RPC loop could read it"


def test_other_non_interactive_modes_still_read_piped_stdin(monkeypatch, tmp_path: Path) -> None:
    """The guard above must be specific to RPC; print mode still needs stdin."""
    reads: list[bool] = []
    monkeypatch.setattr(cli, "read_piped_stdin", lambda tty: (reads.append(tty), "piped text")[1])

    modes = _run_main_capturing_mode(monkeypatch, tmp_path, ["--mode", "json", "hello"])
    assert modes == ["json"]
    assert reads == [False]


def test_rpc_mode_starts_without_a_prompt(monkeypatch, tmp_path: Path) -> None:
    """Print and JSON modes need a prompt up front; RPC mode gets its work over
    the wire after startup, so requiring one would make the mode unusable.
    """
    monkeypatch.setattr(cli, "read_piped_stdin", lambda _tty: None)
    assert _run_main_capturing_mode(monkeypatch, tmp_path, ["--mode", "rpc"]) == ["rpc"]


def test_a_non_rpc_mode_without_a_prompt_is_still_refused(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(cli, "read_piped_stdin", lambda _tty: None)
    assert _run_main_capturing_mode(monkeypatch, tmp_path, ["--mode", "json"]) == []
    assert "No prompt given" in capsys.readouterr().err


def _run_main_capturing_mode(monkeypatch, tmp_path: Path, argv: list[str]) -> list[str]:
    """Run `main` up to the point it would start a session, recording the mode.

    `run_app` is where startup would need a provider, a model and the network,
    none of which these cases are about.
    """
    seen: list[str] = []

    async def fake_run_app(_parsed, app_mode, *_args, **_kwargs) -> int:
        seen.append(app_mode)
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_app", fake_run_app)
    cli.main(argv)
    return seen


def test_rpc_mode_takes_over_stdout() -> None:
    """The protocol owns stdout: a stray `print` would land between JSON lines
    and break the host's parser.
    """
    source = inspect.getsource(cli.run_app)
    assert 'if app_mode in ("print", "json", "rpc"):' in source
