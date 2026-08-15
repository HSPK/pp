"""End-to-end tests for the interactive TUI mode.

These drive the real `InteractiveMode` against a `FakeTerminal` and the `faux`
scripted provider: no TTY, no network, no real editor or clipboard.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pi_tui.testing import FakeTerminal

from pi_coding_agent.core.agent_session_runtime import AgentSessionRuntime
from pi_coding_agent.core.bash_executor import BashResult
from pi_coding_agent.core.extensions.types import Extension, UserBashEventResult
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.modes.interactive.interactive_mode import (
    CompactionQueuedMessage,
    InteractiveMode,
    InteractiveModeOptions,
    create_interactive_tui,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[=>]")


def _visible(terminal: FakeTerminal) -> str:
    return _ANSI_RE.sub("", "".join(terminal.writes))


async def _make_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    script: list[Any] | None = None,
    extensions: list[Any] | None = None,
) -> tuple[InteractiveMode, FakeTerminal]:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    cwd = tmp_path / "project"
    cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("PI_OFFLINE", "1")

    from pi_ai.providers.faux import faux_assistant_message, faux_provider

    from pi_coding_agent.core.model_runtime import ModelRuntime

    faux = faux_provider()
    if script:
        faux.set_responses([faux_assistant_message(text) for text in script])
    model_runtime = await ModelRuntime.create(agent_dir=str(agent_dir), providers=[faux.provider])
    await model_runtime.login(faux.provider.id, "faux-key")

    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=str(cwd),
            agent_dir=str(agent_dir),
            model=faux.models[0],
            model_runtime=model_runtime,
            extensions=extensions,
        )
    )

    async def create_runtime(**_kwargs: Any) -> Any:
        return await create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(cwd),
                agent_dir=str(agent_dir),
                model=faux.models[0],
                model_runtime=model_runtime,
                extensions=extensions,
            )
        )

    runtime = AgentSessionRuntime(result.session, str(agent_dir), create_runtime, result.model_fallback_message)

    terminal = FakeTerminal(columns=80, rows=24)
    mode = InteractiveMode(runtime, InteractiveModeOptions(verbose=False), terminal=terminal)
    return mode, terminal


def _run(coro: Any, timeout: float = 20.0) -> Any:
    return asyncio.run(asyncio.wait_for(coro, timeout=timeout))


# --------------------------------------------------------------------------
# composition root
# --------------------------------------------------------------------------


def test_create_interactive_tui_uses_the_given_terminal():
    terminal = FakeTerminal()
    tui = create_interactive_tui(terminal=terminal)
    assert tui.terminal is terminal


def test_interactive_mode_options_defaults():
    options = InteractiveModeOptions()
    assert options.migrated_providers == []
    assert options.initial_messages == []
    assert options.verbose is False


# --------------------------------------------------------------------------
# init / layout
# --------------------------------------------------------------------------


def test_init_mounts_the_layout_and_focuses_the_editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            assert mode.is_initialized is True
            assert mode.ui.get_focused_component() is mode.editor
            roots = mode.ui.children
            assert mode.document_container in roots
            assert mode.editor_container in roots
            assert mode.footer_container in roots
            assert len(terminal.writes) > 0
        finally:
            await mode.shutdown()

    _run(scenario())


def test_init_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            children_before = len(mode.ui.children)
            await mode.init()
            assert len(mode.ui.children) == children_before
        finally:
            await mode.shutdown()

    _run(scenario())


def test_header_is_minimal_when_quiet_startup_is_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        mode.settings_manager.set_quiet_startup(True)
        try:
            await mode.init()
            rendered = "\n".join(mode.header_container.render(80))
            assert "commands" not in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_verbose_header_lists_hints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        mode.options.verbose = True
        try:
            await mode.init()
            rendered = _ANSI_RE.sub("", "\n".join(mode.header_container.render(80)))
            assert "commands" in rendered
            assert "bash" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_terminal_title_includes_the_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            assert any("project" in title for title in terminal.titles)
        finally:
            await mode.shutdown()

    _run(scenario())


# --------------------------------------------------------------------------
# status lines
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["show_status", "show_warning", "show_error"])
def test_status_helpers_append_to_the_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            before = len(mode.chat_container.children)
            getattr(mode, method)("hello there")
            assert len(mode.chat_container.children) == before + 2
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(60)))
            assert "hello there" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


# --------------------------------------------------------------------------
# submit handling
# --------------------------------------------------------------------------


def test_plain_text_is_delivered_to_get_user_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            pending = asyncio.ensure_future(mode.get_user_input())
            await asyncio.sleep(0)
            await mode._handle_submit("hello world")
            assert await asyncio.wait_for(pending, timeout=5) == "hello world"
        finally:
            await mode.shutdown()

    _run(scenario())


def test_blank_submit_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await mode._handle_submit("   ")
            assert mode.pending_user_inputs == []
        finally:
            await mode.shutdown()

    _run(scenario())


def test_submitted_text_is_added_to_history_and_clears_the_editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.editor.set_text("remembered")
            await mode._handle_submit("remembered")
            assert mode.editor.get_text() == ""
            assert mode.pending_user_inputs == ["remembered"]
        finally:
            await mode.shutdown()

    _run(scenario())


# --------------------------------------------------------------------------
# slash commands
# --------------------------------------------------------------------------


def test_unsupported_slash_commands_warn_instead_of_prompting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await mode._handle_submit("/arminsayshi")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "/arminsayshi is not available" in rendered
            assert mode.pending_user_inputs == []
        finally:
            await mode.shutdown()

    _run(scenario())


def test_name_command_sets_and_reports_the_session_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await mode._handle_submit("/name My Session")
            assert mode.session_manager.get_session_name() == "My Session"

            await mode._handle_submit("/name")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "My Session" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_hotkeys_command_lists_bindings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await mode._handle_submit("/hotkeys")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(100)))
            assert "Keybindings" in rendered
            assert "app.interrupt" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_copy_command_reports_when_there_is_nothing_to_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await mode._handle_submit("/copy")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "No agent messages to copy yet" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_copy_command_copies_the_last_assistant_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            copied: list[str] = []
            import pi_coding_agent.modes.interactive.interactive_mode as module

            async def fake_copy(text: str, *_args: object, **_kwargs: object) -> None:
                copied.append(text)

            monkeypatch.setattr(module, "copy_to_clipboard", fake_copy)
            monkeypatch.setattr(type(mode.session), "get_last_assistant_text", lambda _self: "the answer")
            await mode._handle_submit("/copy")
            assert copied == ["the answer"]
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "Copied last agent message" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_changelog_command_renders_a_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await mode._handle_submit("/changelog")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "What's New" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_debug_command_writes_a_log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await mode._handle_submit("/debug")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(120)))
            assert "Debug log written" in rendered

            log_path = next((tmp_path / "agent").glob("*debug.log"), None)
            assert log_path is not None
            body = log_path.read_text(encoding="utf-8")
            assert "Terminal: 80x24" in body
            assert "All rendered lines" in body
        finally:
            await mode.shutdown()

    _run(scenario())


def test_tree_command_opens_the_tree_selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, script=["hi"])
        try:
            await mode.init()
            await mode.session.prompt("question")
            await asyncio.sleep(0.05)
            await mode._handle_submit("/tree")
            assert mode._active_selector is not None
            rendered = _ANSI_RE.sub("", "\n".join(mode._active_selector.render(80)))
            assert "Session Tree" in rendered
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_tree_command_reports_an_empty_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            monkeypatch.setattr(type(mode.session_manager), "get_tree", lambda _self: [])
            await mode._handle_submit("/tree")
            assert mode._active_selector is None
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "Session tree is empty" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_scoped_models_command_opens_the_selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await mode._handle_submit("/scoped-models")
            assert mode._active_selector is not None
            rendered = _ANSI_RE.sub("", "\n".join(mode._active_selector.render(80)))
            assert "Model Configuration" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_reload_command_reloads_settings_and_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            reloaded: list[str] = []
            monkeypatch.setattr(
                type(mode.session.resource_loader),
                "reload",
                lambda _self: reloaded.append("resources"),
            )
            await mode._handle_submit("/reload")
            assert reloaded == ["resources"]
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(100)))
            assert "Reloaded keybindings" in rendered
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_reload_is_refused_while_streaming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            monkeypatch.setattr(type(mode.session), "is_streaming", property(lambda _self: True))
            await mode._handle_submit("/reload")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(100)))
            assert "Wait for the current response" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_export_command_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, script=["hi"])
        try:
            await mode.init()
            await mode.session.prompt("question")
            await asyncio.sleep(0.05)
            target = tmp_path / "out.jsonl"
            await mode._handle_submit(f"/export {target}")
            assert target.exists()
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(100)))
            assert "Session exported to" in rendered
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_export_to_html_reports_the_unported_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            # HTML export is deliberately not ported; the session raises and the
            # command surfaces that instead of failing silently.
            await mode._handle_submit("/export /tmp/session.html")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(90)))
            assert "Failed to export session" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_export_command_reports_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()

            def boom(_self: object, _path: object = None) -> str:
                raise RuntimeError("disk full")

            monkeypatch.setattr(type(mode.session), "export_to_jsonl", boom)
            await mode._handle_submit("/export /tmp/x.jsonl")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(100)))
            assert "Failed to export session: disk full" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_import_command_requires_a_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await mode._handle_submit("/import")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "Usage: /import <path.jsonl>" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_import_command_round_trips_an_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, script=["hi"])
        try:
            await mode.init()
            await mode.session.prompt("question")
            await asyncio.sleep(0.05)
            target = tmp_path / "out.jsonl"
            await mode._handle_submit(f"/export {target}")

            # `/import` now asks for confirmation first, as upstream does; answer Yes
            # by driving the real dialog rather than stubbing the prompt away.
            submit = asyncio.ensure_future(mode._handle_submit(f"/import {target}"))
            for _ in range(100):
                await asyncio.sleep(0)
                if mode._active_selector is not None:
                    break
            assert mode._active_selector is not None
            mode._active_selector.handle_input("\r")
            await submit
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(100)))
            assert "Session imported from" in rendered
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=40)


def test_clone_command_reports_an_empty_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await mode._handle_submit("/clone")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            # An unsaved session has no fork point yet; the runtime says so.
            assert "has not been saved yet" in rendered or "Nothing to clone yet" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_clone_command_forks_the_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, script=["hi"])
        try:
            await mode.init()
            await mode.session.prompt("question")
            await asyncio.sleep(0.05)

            forks: list[tuple[str, str]] = []

            async def fake_fork(entry_id: str, position: str = "before") -> dict[str, bool]:
                forks.append((entry_id, position))
                return {"cancelled": False}

            monkeypatch.setattr(mode.runtime_host, "fork", fake_fork)
            await mode._handle_submit("/clone")
            assert len(forks) == 1
            assert forks[0][1] == "at"
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "Cloned to new session" in rendered
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_share_command_requires_the_gh_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            import pi_coding_agent.modes.interactive.interactive_mode as module

            monkeypatch.setattr(module.shutil, "which", lambda _name: None)
            await mode._handle_submit("/share")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(100)))
            assert "GitHub CLI (gh) is not installed" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_share_command_reports_a_logged_out_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            import pi_coding_agent.modes.interactive.interactive_mode as module

            monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/gh")
            monkeypatch.setattr(
                module.subprocess,
                "run",
                lambda *_a, **_kw: SimpleNamespace(returncode=1, stdout=b"", stderr=b""),
            )
            await mode._handle_submit("/share")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(100)))
            assert "not logged in" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_share_command_publishes_a_gist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            import pi_coding_agent.modes.interactive.interactive_mode as module

            monkeypatch.setattr(module.shutil, "which", lambda _name: "/usr/bin/gh")

            def fake_run(command: list[str], *_a: object, **_kw: object) -> SimpleNamespace:
                if command[:2] == ["gh", "auth"]:
                    return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
                return SimpleNamespace(returncode=0, stdout=b"https://gist.github.com/u/abc123\n", stderr=b"")

            monkeypatch.setattr(module.subprocess, "run", fake_run)

            async def fake_export(_self: object, path: str | None = None) -> str:
                if path:
                    Path(path).write_text("<html></html>", encoding="utf-8")
                return path or ""

            monkeypatch.setattr(type(mode.session), "export_to_html", fake_export)
            await mode._handle_submit("/share")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(120)))
            assert "Share URL:" in rendered
            assert "abc123" in rendered
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_quit_command_shuts_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        await mode.init()
        await mode._handle_submit("/quit")
        assert mode.shutdown_requested is True

    _run(scenario())


@pytest.mark.parametrize(
    ("command", "attribute"),
    [
        ("/settings", "show_settings_selector"),
        ("/model", "show_model_selector"),
        ("/resume", "show_session_selector"),
        ("/fork", "show_user_message_selector"),
        ("/trust", "show_trust_selector"),
    ],
)
def test_slash_commands_open_their_selectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, attribute: str
):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            calls: list[Any] = []
            monkeypatch.setattr(mode, attribute, lambda *args: calls.append(args))
            handled = await mode._handle_slash_command(command)
            assert handled is True
            assert len(calls) == 1
        finally:
            await mode.shutdown()

    _run(scenario())


def test_unknown_slash_command_is_passed_through_as_a_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            assert await mode._handle_slash_command("/unknown-thing") is False
        finally:
            await mode.shutdown()

    _run(scenario())


# --------------------------------------------------------------------------
# bash mode
# --------------------------------------------------------------------------


def test_bang_prefix_switches_the_editor_into_bash_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            assert mode.is_bash_mode is False
            mode._on_editor_change("!ls")
            assert mode.is_bash_mode is True
            mode._on_editor_change("ls")
            assert mode.is_bash_mode is False
        finally:
            await mode.shutdown()

    _run(scenario())


def test_bash_command_streams_output_into_a_component(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await mode._handle_submit("!printf 'hello-bash'")
            # Idle `!command` goes straight into the transcript, matching
            # `handleBashCommand`'s `isDeferred` branch.
            assert mode.pending_bash_components == []
            component = mode.chat_container.children[-1]
            assert "hello-bash" in component.get_output()
            assert component.status == "complete"
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_bash_components_move_to_the_transcript_on_the_next_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            # A bash command issued while the assistant is answering is shown
            # above the editor and only migrates into the transcript when the
            # user submits their next message.
            monkeypatch.setattr(type(mode.session), "is_streaming", property(lambda _self: True))
            await mode._handle_submit("!printf hi")
            component = mode.pending_bash_components[0]
            assert component in mode.pending_messages_container.children
            assert component not in mode.chat_container.children

            monkeypatch.undo()
            await mode._handle_submit("next message")
            assert mode.pending_bash_components == []
            assert component in mode.chat_container.children
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_user_bash_extension_result_replaces_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`handleBashCommand` emits `user_bash` first; a handler returning a `result`
    short-circuits execution entirely and the result is still recorded in the session."""
    seen: list[tuple[str, bool, str]] = []

    async def handler(event, _ctx):
        seen.append((event.command, event.exclude_from_context, event.cwd))
        return UserBashEventResult(
            result=BashResult(output="from-extension", exit_code=0, cancelled=False, truncated=False)
        )

    extension = Extension(path="inline.py", resolved_path="inline.py", handlers={"user_bash": [handler]})

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, extensions=[extension])
        try:
            await mode.init()
            await mode._handle_submit("!printf 'never-runs'")
            component = mode.chat_container.children[-1]
            assert "from-extension" in component.get_output()
            assert "never-runs" not in component.get_output()
            assert component.status == "complete"
            assert [entry[0] for entry in seen] == ["printf 'never-runs'"]
            assert seen[0][1] is False
            assert seen[0][2] == mode.session_manager.get_cwd()
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_user_bash_extension_operations_replace_the_execution_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A handler returning only `operations` still runs the normal execution path,
    but through the extension's backend rather than a real shell."""

    class _Operations:
        async def exec(self, command, cwd, on_data, signal, timeout, env):
            on_data(f"ran:{command}".encode())
            return 0

    async def handler(_event, _ctx):
        return UserBashEventResult(operations=_Operations())

    extension = Extension(path="inline.py", resolved_path="inline.py", handlers={"user_bash": [handler]})

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, extensions=[extension])
        try:
            await mode.init()
            await mode._handle_submit("!printf 'never-runs'")
            component = mode.chat_container.children[-1]
            assert "ran:printf 'never-runs'" in component.get_output()
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_bash_is_refused_while_another_command_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            monkeypatch.setattr(type(mode.session), "is_bash_running", property(lambda _self: True))
            await mode._handle_submit("!sleep 1")
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(90)))
            assert "already running" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


# --------------------------------------------------------------------------
# key handlers
# --------------------------------------------------------------------------


def test_escape_clears_bash_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.editor.set_text("!ls")
            mode.is_bash_mode = True
            mode._handle_escape()
            assert mode.editor.get_text() == ""
            assert mode.is_bash_mode is False
        finally:
            await mode.shutdown()

    _run(scenario())


def test_double_escape_opens_the_fork_selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.settings_manager.set_double_escape_action("fork")
            calls: list[int] = []
            monkeypatch.setattr(mode, "show_user_message_selector", lambda: calls.append(1))

            mode._handle_escape()
            assert calls == []
            mode._handle_escape()
            assert calls == [1]
        finally:
            await mode.shutdown()

    _run(scenario())


def test_double_escape_respects_the_none_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.settings_manager.set_double_escape_action("none")
            calls: list[int] = []
            monkeypatch.setattr(mode, "show_user_message_selector", lambda: calls.append(1))
            mode._handle_escape()
            mode._handle_escape()
            assert calls == []
        finally:
            await mode.shutdown()

    _run(scenario())


def test_ctrl_c_clears_text_then_requires_a_second_press(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.editor.set_text("draft")
            mode._handle_ctrl_c()
            assert mode.editor.get_text() == ""
            assert mode.shutdown_requested is False

            mode._handle_ctrl_c()
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "again to exit" in rendered
            assert mode.shutdown_requested is False
        finally:
            await mode.shutdown()

    _run(scenario())


def test_toggle_tool_expansion_propagates_to_components(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            from pi_coding_agent.modes.interactive.components.tool_execution import (
                ToolExecutionComponent,
            )

            component = ToolExecutionComponent("read", "id", {}, cwd=str(tmp_path))
            mode.chat_container.add_child(component)

            mode._toggle_tool_output_expansion()
            assert mode.tool_output_expanded is True
            assert component.expanded is True

            mode._toggle_tool_output_expansion()
            assert component.expanded is False
        finally:
            await mode.shutdown()

    _run(scenario())


def test_toggle_thinking_visibility_persists_the_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            before = mode.hide_thinking_block
            mode._toggle_thinking_block_visibility()
            assert mode.hide_thinking_block is not before
            assert mode.settings_manager.get_hide_thinking_block() is not before
        finally:
            await mode.shutdown()

    _run(scenario())


def test_cycle_thinking_level_reports_unsupported_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            # The faux model does not advertise reasoning support.
            assert mode.session.supports_thinking() is False
            mode._cycle_thinking_level()
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "does not support thinking" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_cycle_thinking_level_advances_for_capable_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            monkeypatch.setattr(type(mode.session), "supports_thinking", lambda _self: True)
            monkeypatch.setattr(
                type(mode.session),
                "get_available_thinking_levels",
                lambda _self: ["off", "low", "high"],
            )
            mode._cycle_thinking_level()
            assert mode.session.thinking_level == "low"
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "Thinking level: low" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


# --------------------------------------------------------------------------
# selectors
# --------------------------------------------------------------------------


def test_selector_replaces_the_editor_and_restores_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.show_thinking_selector()
            assert mode._active_selector is not None
            assert mode.editor not in mode.editor_container.children
            assert mode.ui.get_focused_component() is mode._active_selector

            mode._hide_selector()
            assert mode._active_selector is None
            assert mode.editor in mode.editor_container.children
            assert mode.ui.get_focused_component() is mode.editor
        finally:
            await mode.shutdown()

    _run(scenario())


def test_opening_a_second_selector_disposes_the_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.show_thinking_selector()
            first = mode._active_selector
            mode.show_settings_selector()
            assert mode._active_selector is not first
            assert len(mode.editor_container.children) == 1
        finally:
            await mode.shutdown()

    _run(scenario())


def test_thinking_selector_applies_the_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.show_thinking_selector()
            selector = mode._active_selector
            assert selector is not None
            levels = mode.session.get_available_thinking_levels()
            selector.select_list.set_selected_index(len(levels) - 1)
            selector.handle_input("\r")
            assert mode._active_selector is None
            assert mode.session.thinking_level == levels[-1]
        finally:
            await mode.shutdown()

    _run(scenario())


def test_fork_selector_reports_when_there_is_nothing_to_fork(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.show_user_message_selector()
            assert mode._active_selector is None
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "No user messages" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_trust_selector_renders_the_current_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.show_trust_selector()
            assert mode._active_selector is not None
            rendered = _ANSI_RE.sub("", "\n".join(mode._active_selector.render(80)))
            assert "Project trust" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_settings_selector_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.show_settings_selector()
            rendered = _ANSI_RE.sub("", "\n".join(mode._active_selector.render(80)))
            assert "Auto-compact" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_model_selector_renders_available_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.show_model_selector()
            assert mode._active_selector is not None
        finally:
            await mode.shutdown()

    _run(scenario())


def test_session_selector_loads_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.show_session_selector()
            await asyncio.sleep(0.1)
            assert mode._active_selector is not None
            rendered = _ANSI_RE.sub("", "\n".join(mode._active_selector.render(80)))
            assert "Resume Session" in rendered
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


# --------------------------------------------------------------------------
# agent events end to end
# --------------------------------------------------------------------------


def test_prompt_renders_the_assistant_reply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, script=["Hello from the model"])
        try:
            await mode.init()
            await mode.session.prompt("say hi")
            await asyncio.sleep(0.05)
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "say hi" in rendered
            assert "Hello from the model" in rendered
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_working_indicator_appears_then_clears(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, script=["done"])
        try:
            await mode.init()
            seen: list[str | None] = []
            original = mode._show_status_indicator

            def record(indicator: Any) -> None:
                seen.append(indicator.kind)
                original(indicator)

            monkeypatch.setattr(mode, "_show_status_indicator", record)
            await mode.session.prompt("go")
            await asyncio.sleep(0.05)
            assert "working" in seen
            assert mode.active_status_indicator is None
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_run_processes_initial_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, script=["first reply"])
        mode.options.initial_message = "initial question"
        try:
            task = asyncio.ensure_future(mode.run())
            await asyncio.sleep(0.2)
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(80)))
            assert "initial question" in rendered
            await mode.shutdown()
            await asyncio.wait_for(task, timeout=5)
        finally:
            if not mode.shutdown_requested:
                await mode.shutdown()

    _run(scenario(), timeout=30)


def test_run_surfaces_warnings_from_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        mode.options.migrated_providers = ["openai"]
        mode.options.model_fallback_message = "fell back"
        try:
            task = asyncio.ensure_future(mode.run())
            # `init()` awaits the theme controller's terminal-background probe,
            # which waits out its 100ms timeout against a fake terminal, so poll
            # rather than assume a fixed startup delay.
            rendered = ""
            for _ in range(200):
                await asyncio.sleep(0.01)
                rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(90)))
                if "Migrated credentials to auth.json: openai" in rendered:
                    break
            assert "Migrated credentials to auth.json: openai" in rendered
            assert "fell back" in rendered
            await mode.shutdown()
            await asyncio.wait_for(task, timeout=5)
        finally:
            if not mode.shutdown_requested:
                await mode.shutdown()

    _run(scenario(), timeout=30)


# --------------------------------------------------------------------------
# shutdown
# --------------------------------------------------------------------------


def test_shutdown_is_idempotent_and_stops_the_ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, terminal = await _make_mode(tmp_path, monkeypatch)
        await mode.init()
        await mode.shutdown()
        assert mode.shutdown_requested is True
        writes_after_first = len(terminal.writes)
        await mode.shutdown()
        assert len(terminal.writes) == writes_after_first

    _run(scenario())


def test_shutdown_releases_a_pending_get_user_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        await mode.init()
        pending = asyncio.ensure_future(mode.get_user_input())
        await asyncio.sleep(0)
        await mode.shutdown()
        assert await asyncio.wait_for(pending, timeout=5) == ""

    _run(scenario())


def test_shutdown_stops_the_footer_watcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        await mode.init()
        await mode.shutdown()
        assert mode.footer_data_provider._watch_task is None

    _run(scenario())


def test_settings_selector_reflects_every_persisted_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`/settings` rows must show the saved value, not a hardcoded default.

    Ported from the `showSettingsSelector` config block in
    `packages/coding-agent/src/modes/interactive/interactive-mode.ts`.
    """

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            settings = mode.settings_manager
            settings.set_transport("websocket")
            settings.set_mermaid_rendering_mode("streaming")
            settings.set_show_cache_miss_notices(False)
            settings.set_collapse_changelog(True)
            settings.set_enable_install_telemetry(False)
            settings.set_double_escape_action("fork")
            settings.set_tree_filter_mode("no-tools")
            settings.set_quiet_startup(True)
            settings.set_fullscreen_exit_output("resume-hint")
            settings.set_fullscreen_scrollbar("always")
            mode.session.set_steering_mode("all")
            mode.session.set_follow_up_mode("all")

            mode.show_settings_selector()
            selector = mode._active_selector
            assert selector is not None
            values = {item.id: item.current_value for item in selector.settings_list.items}

            assert values["transport"] == "websocket"
            assert values["mermaid-rendering"] == "streaming"
            assert values["cache-miss-notices"] == "false"
            assert values["collapse-changelog"] == "true"
            assert values["install-telemetry"] == "false"
            assert values["double-escape-action"] == "fork"
            assert values["tree-filter-mode"] == "no-tools"
            assert values["quiet-startup"] == "true"
            assert values["fullscreen-exit-output"] == "resume-hint"
            assert values["fullscreen-scrollbar"] == "always"
            assert values["steering-mode"] == "all"
            assert values["follow-up-mode"] == "all"
        finally:
            await mode.shutdown()

    _run(scenario())


def test_settings_selector_persists_every_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Changing a `/settings` row must reach the settings manager.

    Ported from the `showSettingsSelector` callbacks block in
    `packages/coding-agent/src/modes/interactive/interactive-mode.ts`.
    """

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            settings = mode.settings_manager
            mode.show_settings_selector()
            selector = mode._active_selector
            assert selector is not None
            change = selector.settings_list.on_change

            change("transport", "websocket")
            assert settings.get_transport() == "websocket"
            change("mermaid-rendering", "final")
            assert settings.get_mermaid_rendering_mode() == "final"
            change("cache-miss-notices", "false")
            assert settings.get_show_cache_miss_notices() is False
            change("collapse-changelog", "true")
            assert settings.get_collapse_changelog() is True
            change("install-telemetry", "false")
            assert settings.get_enable_install_telemetry() is False
            change("double-escape-action", "fork")
            assert settings.get_double_escape_action() == "fork"
            change("tree-filter-mode", "user-only")
            assert settings.get_tree_filter_mode() == "user-only"
            change("quiet-startup", "true")
            assert settings.get_quiet_startup() is True
            change("fullscreen-exit-output", "resume-hint")
            assert settings.get_fullscreen_exit_output() == "resume-hint"
            change("fullscreen-scrollbar", "always")
            assert settings.get_fullscreen_scrollbar() == "always"
            change("steering-mode", "all")
            assert settings.get_steering_mode() == "all"
            assert mode.session.steering_mode == "all"
            change("follow-up-mode", "all")
            assert settings.get_follow_up_mode() == "all"
            assert mode.session.follow_up_mode == "all"
            change("hide-thinking", "true")
            assert settings.get_hide_thinking_block() is True
            assert mode.hide_thinking_block is True
            change("http-idle-timeout", "disabled")
            assert settings.get_http_idle_timeout_ms() == 0
            change("tui-mode", "fullscreen")
            assert settings.get_tui_mode() == "fullscreen"
        finally:
            await mode.shutdown()

    _run(scenario())


def test_escape_while_streaming_restores_queued_messages_to_the_editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Port of `restoreQueuedMessagesToEditor({abort: true})` on the default escape handler.

    Aborting drops both queues, so the queued text has to land back in the
    editor instead of being lost.
    """

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            monkeypatch.setattr(type(mode.session), "is_streaming", property(lambda _self: True))
            mode.session._steering_messages = ["steered"]
            mode.session._follow_up_messages = ["followed up"]
            mode.compaction_queued_messages.append(CompactionQueuedMessage(text="held", mode="steer"))
            mode.editor.set_text("typing")

            mode._handle_escape()

            assert mode.editor.get_text() == "steered\n\nheld\n\nfollowed up\n\ntyping"
            assert mode.session._steering_messages == []
            assert mode.session._follow_up_messages == []
            assert mode.compaction_queued_messages == []
        finally:
            await mode.shutdown()

    _run(scenario())


def test_escape_while_streaming_with_empty_queues_only_aborts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            monkeypatch.setattr(type(mode.session), "is_streaming", property(lambda _self: True))
            aborts: list[bool] = []
            monkeypatch.setattr(mode.session.agent, "abort", lambda: aborts.append(True))
            mode.editor.set_text("typing")

            mode._handle_escape()

            assert mode.editor.get_text() == "typing"
            assert aborts == [True]
        finally:
            await mode.shutdown()

    _run(scenario())


def test_tree_navigation_aborts_an_in_flight_response_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TS: 'The user committed to navigating: stop the active response first.'

    Without the abort, `navigate_tree` rejects with "Wait for the current
    response to finish before navigating the session tree." and the selection
    is dropped.
    """

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, script=["hi"])
        try:
            await mode.init()
            # The summarize prompt is a separate concern; this case is about
            # aborting the in-flight response.
            mode.settings_manager.apply_overrides({"branchSummary": {"skipPrompt": True}})
            await mode.session.prompt("question")
            await asyncio.sleep(0.05)
            entry_id = mode.session_manager.get_branch()[0].id

            streaming = [True]
            monkeypatch.setattr(type(mode.session), "is_streaming", property(lambda _self: streaming[0]))
            aborted: list[bool] = []

            original_abort = mode.session.abort

            async def abort() -> None:
                aborted.append(True)
                streaming[0] = False
                await original_abort()

            monkeypatch.setattr(mode.session, "abort", abort)

            await mode._navigate_tree(entry_id)

            assert aborted == [True]
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(100)))
            assert "Wait for the current response" not in rendered
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_markdown_theme_uses_the_configured_code_block_indent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Port of `getMarkdownThemeWithSettings()`; `markdown.codeBlockIndent` was ignored."""

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            assert mode._markdown_theme().code_block_indent == "  "
            mode.settings_manager._settings["markdown"] = {"codeBlockIndent": ">>>>"}
            assert mode._markdown_theme().code_block_indent == ">>>>"
        finally:
            await mode.shutdown()

    _run(scenario())


def test_registers_every_app_action_keybinding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Every `app.*` action with a default keybinding must have a handler.

    Mirrors the `onAction(...)` registrations in `setupEditorHandlers`;
    `app.editor.external`, `app.message.followUp` and `app.message.dequeue`
    were defined in `DEFAULT_APP_KEYBINDINGS` and listed by `/hotkeys` but did
    nothing when pressed.
    """

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            registered = set(mode.default_editor.action_handlers)
            assert {
                "app.clear",
                "app.suspend",
                "app.thinking.cycle",
                "app.thinking.toggle",
                "app.tools.expand",
                "app.model.select",
                "app.model.cycleForward",
                "app.model.cycleBackward",
                "app.session.new",
                "app.session.tree",
                "app.session.fork",
                "app.session.resume",
                "app.message.copy",
                "app.message.followUp",
                "app.message.dequeue",
                "app.editor.external",
            } <= registered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_dequeue_restores_queued_messages_and_reports_the_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Port of `handleDequeue`."""

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode._handle_dequeue()
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(100)))
            assert "No queued messages to restore" in rendered

            mode.session._steering_messages = ["one", "two"]
            mode._handle_dequeue()
            assert mode.editor.get_text() == "one\n\ntwo"
            rendered = _ANSI_RE.sub("", "\n".join(mode.chat_container.render(100)))
            assert "Restored 2 queued messages to editor" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_follow_up_action_queues_as_follow_up_while_streaming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Port of `handleFollowUp`: alt+enter queues a follow-up, not a steering message."""

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            monkeypatch.setattr(type(mode.session), "is_streaming", property(lambda _self: True))
            prompts: list[tuple[str, object]] = []

            async def prompt(text: str, **kwargs: Any) -> None:
                prompts.append((text, kwargs.get("streaming_behavior")))

            monkeypatch.setattr(mode.session, "prompt", prompt)
            mode.editor.set_text("later please")

            await mode._handle_follow_up()

            assert prompts == [("later please", "followUp")]
            assert mode.editor.get_text() == ""
        finally:
            await mode.shutdown()

    _run(scenario())


def test_follow_up_action_acts_like_enter_when_idle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            submitted: list[str] = []
            mode.editor.on_submit = submitted.append
            mode.editor.set_text("now please")

            await mode._handle_follow_up()

            assert submitted == ["now please"]
            assert mode.editor.get_text() == ""
        finally:
            await mode.shutdown()

    _run(scenario())


def test_follow_up_action_holds_input_during_compaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            monkeypatch.setattr(type(mode.session), "is_compacting", property(lambda _self: True))
            mode.editor.set_text("hold me")

            await mode._handle_follow_up()

            assert [(m.text, m.mode) for m in mode.compaction_queued_messages] == [("hold me", "followUp")]
        finally:
            await mode.shutdown()

    _run(scenario())


def test_external_editor_action_replaces_the_editor_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Port of `handleOpenExternalEditor`; `getExternalEditorCommand` had no reader."""

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            seen: list[str] = []

            async def fake_edit(options: Any) -> Any:
                seen.append(options.command)
                seen.append(options.content)
                return SimpleNamespace(status="complete", content="edited elsewhere")

            monkeypatch.setattr("pi_coding_agent.modes.interactive.interactive_mode.edit_in_external_editor", fake_edit)
            mode.settings_manager._settings["externalEditor"] = "my-editor"
            mode.editor.set_text("draft")

            await mode._handle_open_external_editor()

            assert seen == ["my-editor", "draft"]
            assert mode.editor.get_text() == "edited elsewhere"
        finally:
            await mode.shutdown()

    _run(scenario())


# --------------------------------------------------------------------------
# startup theme resolution (InteractiveThemeController constructor)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("colorfgbg", "expected"),
    # The setting is `"<light-theme>/<dark-theme>"` = `"dark/light"`, so a dark
    # terminal (COLORFGBG background index 0) selects the theme named "light"
    # and vice versa. Inverting the pair this way means a fallback to the
    # env-detected default would produce the opposite name.
    [("15;0", "light"), ("0;15", "dark")],
)
def test_startup_resolves_an_auto_theme_pair_against_the_terminal_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, colorfgbg: str, expected: str
):
    """`InteractiveThemeController`'s constructor resolves the raw theme setting.

    TypeScript calls `resolveThemeSetting(settingsManager.getThemeSetting(),
    detectTerminalBackgroundFromEnv().theme)` before `initTheme`, so an
    `"<light>/<dark>"` auto pair picks a side. `getTheme()` returns `undefined`
    for such a pair, so passing it straight to `initTheme` drops the setting.
    """
    from pi_coding_agent.modes.interactive.theme.theme import theme as current_theme

    async def scenario() -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "settings.json").write_text('{"theme": "dark/light"}')
        monkeypatch.setenv("COLORFGBG", colorfgbg)

        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            assert mode.settings_manager.get_theme_setting() == "dark/light"
            assert mode.settings_manager.get_theme() is None
            assert current_theme.name == expected
        finally:
            await mode.shutdown()

    _run(scenario())


def test_startup_uses_a_plain_theme_setting_verbatim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A non-auto setting still reaches `init_theme` unchanged."""
    from pi_coding_agent.modes.interactive.theme.theme import theme as current_theme

    async def scenario() -> None:
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "settings.json").write_text('{"theme": "light"}')
        monkeypatch.setenv("COLORFGBG", "15;0")

        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            assert current_theme.name == "light"
        finally:
            await mode.shutdown()

    _run(scenario())


def test_switch_tui_mode_replaces_the_live_renderer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Port of `switchTuiMode` (`interactive-mode.ts:788`).

    Previously this port only persisted the setting and reported "applies on
    next start", so the two TUI modes could not be swapped in a running
    session at all.
    """

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            assert mode.tui_mode == "regular"
            before = mode.renderer

            assert mode.switch_tui_mode("fullscreen") is True

            assert mode.tui_mode == "fullscreen"
            assert mode.renderer is not before
            # The editor must survive the swap: components are re-parented onto
            # the new renderer, not rebuilt, or the user loses their typed text.
            assert mode.editor_container in mode.renderer.children or any(
                mode.editor_container is c or mode.editor_container in getattr(c, "children", [])
                for c in mode.renderer.children
            )
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_switch_tui_mode_is_a_no_op_for_the_current_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            before = mode.renderer

            assert mode.switch_tui_mode("regular") is True

            assert mode.renderer is before
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


# --------------------------------------------------------------------------
# vertical spacing above the editor
# --------------------------------------------------------------------------


def _rendered_rows(container: Any, width: int = 80) -> list[str]:
    rows: list[str] = []
    for child in container.children:
        rows.extend(child.render(width))
    return rows


def _plain(rows: list[str]) -> list[str]:
    return [_ANSI_RE.sub("", row).rstrip() for row in rows]


def test_idle_status_does_not_reserve_rows_before_any_status_indicator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`interactive-mode.ts` mounts `idleStatus` only from `clearStatusIndicator`.

    Reserving the two placeholder rows at startup put a permanent gap between
    the last response and the prompt that the TypeScript UI does not have.
    """

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()

            assert _rendered_rows(mode.status_container) == []
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_idle_status_reserves_rows_once_an_indicator_has_been_cleared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The other half of `clearStatusIndicator`: keep the height it was using."""

    async def scenario() -> None:
        from pi_coding_agent.modes.interactive.components.status_indicator import WorkingStatusIndicator

        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            mode.ui.set_clear_on_shrink(True)

            mode._show_status_indicator(WorkingStatusIndicator(mode.ui, lambda: None))
            mode._clear_status_indicator()

            assert _rendered_rows(mode.status_container) == [" " * 80] * 2
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_a_user_message_is_separated_from_the_message_above_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Port of the `chatContainer.children.length > 0` guard in `addMessageToChat`."""

    async def scenario() -> None:
        from pi_ai.types import AssistantMessage, TextContent, UserMessage

        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            # Startup may legitimately append status lines (a managed-tool
            # download, for one), and this case is about message spacing, so
            # start from an empty transcript rather than assuming one.
            mode.chat_container.clear()

            mode._add_message_to_chat(UserMessage(content="first"))
            # The transcript opens flush: no leading blank row before the first
            # message's own box padding.
            assert _plain(_rendered_rows(mode.chat_container)) == ["", " first", ""]

            mode._add_message_to_chat(AssistantMessage(content=[TextContent(text="reply")]))
            mode._add_message_to_chat(UserMessage(content="second"))

            assert _plain(_rendered_rows(mode.chat_container)) == [
                "",
                " first",
                "",
                # assistant component's own leading spacer
                "",
                " reply",
                # separator before the next user message
                "",
                # user message box padding
                "",
                " second",
                "",
            ]
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


# --------------------------------------------------------------------------
# extension widgets
# --------------------------------------------------------------------------


def test_widget_container_above_holds_one_spacer_when_no_widgets_are_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """TS `init()` calls `renderWidgets()` to seed the default spacer.

    That single blank row is the gap between the transcript and the prompt, so
    losing it makes the prompt sit flush against the last response.
    """

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()

            assert _plain(_rendered_rows(mode.widget_container_above)) == [""]
            assert _rendered_rows(mode.widget_container_below) == []
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_widgets_render_in_the_container_their_placement_selects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()

            mode.set_extension_widget("build", ["building..."])
            # Above the editor the widget is preceded by a separator row.
            assert _plain(_rendered_rows(mode.widget_container_above)) == ["", " building..."]

            # Re-registering the same key moves the widget instead of cloning it.
            mode.set_extension_widget("build", ["building..."], placement="belowEditor")
            assert _plain(_rendered_rows(mode.widget_container_above)) == [""]
            assert _plain(_rendered_rows(mode.widget_container_below)) == [" building..."]

            # ...and back again, so removal is checked in both directions.
            mode.set_extension_widget("build", ["done"])
            assert _plain(_rendered_rows(mode.widget_container_above)) == ["", " done"]
            assert _rendered_rows(mode.widget_container_below) == []

            mode.clear_extension_widgets()
            assert _plain(_rendered_rows(mode.widget_container_above)) == [""]
            assert _rendered_rows(mode.widget_container_below) == []
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_a_long_widget_is_truncated_so_it_cannot_push_the_editor_off_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()

            mode.set_extension_widget("noisy", [f"line {index}" for index in range(25)])

            rows = _plain(_rendered_rows(mode.widget_container_above))
            # separator + MAX_WIDGET_LINES + the truncation notice
            assert len(rows) == 1 + mode.MAX_WIDGET_LINES + 1
            assert rows[1] == " line 0"
            assert rows[mode.MAX_WIDGET_LINES] == " line 9"
            assert rows[-1] == " ... (widget truncated)"
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_a_widget_factory_receives_the_tui_and_theme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        from pi_tui.components.text import Text as TuiText

        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        seen: list[tuple[Any, Any]] = []
        try:
            await mode.init()

            def factory(tui: Any, thm: Any) -> Any:
                seen.append((tui, thm))
                return TuiText("from factory", 1, 0)

            mode.set_extension_widget("custom", factory)

            assert seen[0][0] is mode.ui
            assert seen[0][1] is not None
            assert _plain(_rendered_rows(mode.widget_container_above)) == ["", " from factory"]
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_an_extension_can_draw_a_widget_through_the_ui_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The wiring test: `set_ui_context` must reach a real extension handler."""

    def on_session_start(_event: Any, ctx: Any) -> None:
        ctx.ui.set_widget("hello", ["from an extension"])

    extension = Extension(path="inline.py", resolved_path="inline.py", handlers={"session_start": [on_session_start]})

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, extensions=[extension])
        try:
            await mode.init()

            assert _plain(_rendered_rows(mode.widget_container_above)) == ["", " from an extension"]
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


# --------------------------------------------------------------------------
# branch summarization on tree navigation
# --------------------------------------------------------------------------


async def _answer_summary_prompt(mode: Any, answer: str | None) -> list[str]:
    """Replace the selector with a scripted answer, recording what it offered."""
    seen: list[str] = []

    async def show_extension_selector(title: str, options: list[str], timeout: int | None = None) -> str | None:
        seen.append(title)
        seen.extend(options)
        return answer

    mode.show_extension_selector = show_extension_selector
    return seen


def test_tree_navigation_offers_to_summarize_the_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`navigate_tree` accepts `summarize`, but nothing ever asked the user for it.

    The setting that turns the question off (`branchSummary.skipPrompt`) had no
    reader either, which is what made the gap visible.
    """

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, script=["hi"])
        try:
            await mode.init()
            await mode.session.prompt("question")
            entry_id = mode.session_manager.get_branch()[0].id
            seen = await _answer_summary_prompt(mode, "Summarize")

            captured: dict[str, Any] = {}
            original = mode.session.navigate_tree

            async def navigate_tree(target_id: str, **kwargs: Any) -> Any:
                captured.update(kwargs)
                return await original(target_id)

            monkeypatch.setattr(mode.session, "navigate_tree", navigate_tree)

            await mode._navigate_tree(entry_id)

            assert seen[0] == "Summarize branch?"
            assert seen[1:] == ["No summary", "Summarize", "Summarize with custom prompt"]
            assert captured["summarize"] is True
            assert captured["custom_instructions"] is None
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_declining_the_summary_navigates_without_summarizing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, script=["hi"])
        try:
            await mode.init()
            await mode.session.prompt("question")
            entry_id = mode.session_manager.get_branch()[0].id
            await _answer_summary_prompt(mode, "No summary")

            captured: dict[str, Any] = {}
            original = mode.session.navigate_tree

            async def navigate_tree(target_id: str, **kwargs: Any) -> Any:
                captured.update(kwargs)
                return await original(target_id)

            monkeypatch.setattr(mode.session, "navigate_tree", navigate_tree)

            await mode._navigate_tree(entry_id)

            assert captured["summarize"] is False
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_escaping_the_summary_prompt_returns_to_the_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Escape must not navigate: the choice was never made."""

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, script=["hi"])
        try:
            await mode.init()
            await mode.session.prompt("question")
            entry_id = mode.session_manager.get_branch()[0].id
            await _answer_summary_prompt(mode, None)

            navigated: list[str] = []

            async def navigate_tree(target_id: str, **kwargs: Any) -> Any:
                navigated.append(target_id)
                raise AssertionError("must not navigate after escape")

            monkeypatch.setattr(mode.session, "navigate_tree", navigate_tree)
            reopened: list[str | None] = []
            monkeypatch.setattr(mode, "show_tree_selector", lambda selected=None: reopened.append(selected))

            await mode._navigate_tree(entry_id)

            assert navigated == []
            # The tree reopens on the entry the user was looking at.
            assert reopened == [entry_id]
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_skip_prompt_setting_navigates_without_asking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch, script=["hi"])
        try:
            await mode.init()
            mode.settings_manager.apply_overrides({"branchSummary": {"skipPrompt": True}})
            await mode.session.prompt("question")
            entry_id = mode.session_manager.get_branch()[0].id

            asked: list[str] = []

            async def show_extension_selector(title: str, options: list[str], timeout: int | None = None):
                asked.append(title)
                return "Summarize"

            mode.show_extension_selector = show_extension_selector

            captured: dict[str, Any] = {}
            original = mode.session.navigate_tree

            async def navigate_tree(target_id: str, **kwargs: Any) -> Any:
                captured.update(kwargs)
                return await original(target_id)

            monkeypatch.setattr(mode.session, "navigate_tree", navigate_tree)

            await mode._navigate_tree(entry_id)

            assert asked == []
            assert captured["summarize"] is False
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)
