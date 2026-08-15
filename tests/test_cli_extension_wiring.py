"""The CLI must actually load extensions and hand them to the session.

TypeScript loads extensions inside `DefaultResourceLoader` and passes the
result down from `main.ts:769` (`additionalExtensionPaths` / `noExtensions`,
consumed at `resource-loader.ts:451` and `:555`). This port deliberately keeps
extensions *out* of `ResourceLoader` -- `CreateAgentSessionOptions.extensions`
takes an already-loaded list -- which makes the CLI the caller responsible for
loading them.

That responsibility was missed: `--extension`/`-e` and `--no-extensions` were
parsed into `ParsedArgs` and read nowhere, and no production code path called
`discover_and_load_extensions` at all. The extension system was fully
implemented and completely unreachable from the binary -- a project's
`.pi/extensions/*.py` never loaded, `-e path` was silently discarded, and
`--no-extensions` disabled something that was never enabled.

These tests pin the wiring end to end: a real extension file on disk, loaded
through the real `build_session_runtime`, arriving in the options that build
the session. They are deliberately not written against a stubbed loader --
a stub would have been satisfied by the broken code too.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from pi_coding_agent.cli import entry as cli
from pi_coding_agent.cli.args import Args

COMMAND_EXTENSION = """
def pi_extension(pi):
    async def _handler(args, ctx):
        return None

    pi.register_command("probe", handler=_handler)
"""


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def agent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An agent dir with one env-var-backed provider, so no network is needed."""
    directory = tmp_path / "agent"
    directory.mkdir()
    (directory / "models.json").write_text(
        json.dumps(
            {
                "providers": {
                    "probe-provider": {
                        "baseUrl": "https://example.invalid/v1",
                        "api": "openai-completions",
                        "apiKey": "$PROBE_API_KEY",
                        "models": [{"id": "probe-model", "name": "Probe"}],
                    }
                }
            }
        )
    )
    monkeypatch.setenv("PROBE_API_KEY", "probe-key")
    return directory


@pytest.fixture
def captured_options(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture the real `CreateAgentSessionOptions` the CLI builds."""
    captured: dict[str, Any] = {}

    async def fake_create_agent_session(options: Any) -> Any:
        captured["options"] = options
        raise _StopBuild

    monkeypatch.setattr(cli, "create_agent_session", fake_create_agent_session)
    return captured


class _StopBuild(Exception):
    """Ends the build once the options are captured; the session is not the subject."""


async def _build(parsed: Args, cwd: Path, agent_dir: Path) -> None:
    try:
        await cli.build_session_runtime(parsed, str(cwd), str(agent_dir))
    except _StopBuild:
        return
    raise AssertionError("create_agent_session was never called")


def _args(**kwargs: Any) -> Args:
    return Args(model="probe-provider/probe-model", **kwargs)


async def test_loads_project_local_extensions(tmp_path, agent_dir, captured_options):
    """`.pi/extensions/*.py` must reach the session through the real CLI path."""
    cwd = tmp_path / "project"
    _write(cwd / ".pi" / "extensions" / "probe.py", COMMAND_EXTENSION)

    await _build(_args(), cwd, agent_dir)

    extensions = captured_options["options"].extensions
    assert extensions is not None
    assert [os.path.basename(e.path) for e in extensions] == ["probe.py"]


async def test_loads_extension_paths_given_with_the_extension_flag(tmp_path, agent_dir, captured_options):
    """`-e/--extension` was parsed and discarded; it must now be honored."""
    cwd = tmp_path / "project"
    cwd.mkdir()
    explicit = _write(tmp_path / "elsewhere" / "explicit.py", COMMAND_EXTENSION)

    await _build(_args(extensions=[str(explicit)]), cwd, agent_dir)

    extensions = captured_options["options"].extensions
    assert [os.path.basename(e.path) for e in extensions] == ["explicit.py"]


async def test_no_extensions_suppresses_discovery(tmp_path, agent_dir, captured_options):
    cwd = tmp_path / "project"
    _write(cwd / ".pi" / "extensions" / "probe.py", COMMAND_EXTENSION)

    await _build(_args(no_extensions=True), cwd, agent_dir)

    assert captured_options["options"].extensions == []


async def test_no_extensions_keeps_explicitly_requested_paths(tmp_path, agent_dir, captured_options):
    """`--no-extensions` suppresses discovery only, not `-e`.

    `resource-loader.ts:451` is `noExtensions ? cliEnabledExtensions :
    mergePaths(cliEnabledExtensions, enabledExtensions)` -- the CLI-supplied
    paths survive either way. Suppressing them too would be a stricter flag
    than upstream's.
    """
    cwd = tmp_path / "project"
    _write(cwd / ".pi" / "extensions" / "discovered.py", COMMAND_EXTENSION)
    explicit = _write(tmp_path / "elsewhere" / "explicit.py", COMMAND_EXTENSION)

    await _build(_args(extensions=[str(explicit)], no_extensions=True), cwd, agent_dir)

    extensions = captured_options["options"].extensions
    assert [os.path.basename(e.path) for e in extensions] == ["explicit.py"]


async def test_extension_load_failure_is_fatal(tmp_path, agent_dir, captured_options):
    """A typo'd `-e` path must stop startup, not produce a degraded agent.

    `main.ts:894-899` treats extension load errors as fatal startup
    diagnostics: it prints the hint and exits 1. Printing and continuing --
    which this port did at first -- hands the user a running agent silently
    missing the extension they asked for, which is the one failure mode they
    cannot detect from the prompt.
    """
    cwd = tmp_path / "project"
    cwd.mkdir()
    missing = tmp_path / "nope.py"

    with pytest.raises(SystemExit) as excinfo:
        await _build(_args(extensions=[str(missing)]), cwd, agent_dir)

    assert excinfo.value.code == 1
    assert "options" not in captured_options


async def test_extension_load_failure_prints_the_recovery_hint(tmp_path, agent_dir, captured_options, capsys):
    """The flag that recovers from the failure is the whole point of the message."""
    cwd = tmp_path / "project"
    cwd.mkdir()

    with pytest.raises(SystemExit):
        await _build(_args(extensions=[str(tmp_path / "nope.py")]), cwd, agent_dir)

    stderr = capsys.readouterr().err
    assert "Failed to load extension" in stderr
    assert "-ne" in stderr


async def test_missing_explicit_path_is_reported_as_a_missing_path(tmp_path, agent_dir, captured_options, capsys):
    """`resource-loader.ts:455-461` pre-checks explicitly requested paths.

    Without the pre-check the loader's raw errno surfaces -- "[Errno 2] No
    such file or directory" -- which reads as an internal fault rather than
    "the file you asked for isn't there".
    """
    cwd = tmp_path / "project"
    cwd.mkdir()
    missing = tmp_path / "nope.py"

    with pytest.raises(SystemExit):
        await _build(_args(extensions=[str(missing)]), cwd, agent_dir)

    stderr = capsys.readouterr().err
    assert f"Extension path does not exist: {missing}" in stderr
    assert "Errno 2" not in stderr
