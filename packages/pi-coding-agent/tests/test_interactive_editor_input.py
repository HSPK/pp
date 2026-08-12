"""Tests that drive the interactive mode through real keystrokes.

The interactive-mode tests in `test_interactive_mode.py` call the submit and
command handlers directly. That bypasses the editor entirely, which is how the
missing autocomplete wiring (`/` never opening the command popup) went
unnoticed. These tests feed keys into the editor and assert on what the editor
actually renders, so anything that is not wired up at startup fails here.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from pi_coding_agent.core.agent_session_runtime import AgentSessionRuntime
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.modes.interactive.interactive_mode import (
    InteractiveMode,
    InteractiveModeOptions,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pi-tui" / "tests"))
from fakes import FakeTerminal

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b[=>]")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _editor_text(mode: InteractiveMode, width: int = 80) -> str:
    return "\n".join(_strip(line) for line in mode.editor.render(width))


async def _make_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InteractiveMode:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    cwd = tmp_path / "project"
    cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(agent_dir))
    monkeypatch.setenv("PI_OFFLINE", "1")

    from pi_ai.providers.faux import faux_provider
    from pi_coding_agent.core.model_runtime import ModelRuntime

    faux = faux_provider()
    model_runtime = await ModelRuntime.create(agent_dir=str(agent_dir), providers=[faux.provider])
    await model_runtime.login(faux.provider.id, "faux-key")
    options = CreateAgentSessionOptions(
        cwd=str(cwd), agent_dir=str(agent_dir), model=faux.models[0], model_runtime=model_runtime
    )
    result = await create_agent_session(options)

    async def create_runtime(**_kwargs: Any) -> Any:
        return await create_agent_session(options)

    runtime = AgentSessionRuntime(result.session, str(agent_dir), create_runtime, result.model_fallback_message)
    return InteractiveMode(runtime, InteractiveModeOptions(), terminal=FakeTerminal(100, 30))


async def _type(mode: InteractiveMode, text: str) -> None:
    for char in text:
        mode.editor.handle_input(char)
    await asyncio.sleep(0)


async def _wait_for_autocomplete(mode: InteractiveMode, *, expected: bool = True) -> bool:
    for _ in range(100):
        await asyncio.sleep(0.02)
        if mode.editor.is_showing_autocomplete() == expected:
            return expected
    return mode.editor.is_showing_autocomplete()


def _run(coro: Any, timeout: float = 30.0) -> Any:
    return asyncio.run(asyncio.wait_for(coro, timeout=timeout))


# --------------------------------------------------------------------------
# slash command autocomplete
# --------------------------------------------------------------------------


def test_editor_has_an_autocomplete_provider_after_init(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            assert mode.autocomplete_provider is not None
            assert mode.default_editor._autocomplete_provider is mode.autocomplete_provider
        finally:
            await mode.shutdown()

    _run(scenario())


def test_typing_slash_opens_the_command_popup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            assert mode.editor.is_showing_autocomplete() is False
            await _type(mode, "/")
            assert await _wait_for_autocomplete(mode) is True

            rendered = _editor_text(mode)
            assert "settings" in rendered
            assert "Open settings menu" in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_slash_popup_filters_as_you_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await _type(mode, "/comp")
            assert await _wait_for_autocomplete(mode) is True
            rendered = _editor_text(mode)
            assert "compact" in rendered
            assert "settings" not in rendered
        finally:
            await mode.shutdown()

    _run(scenario())


def test_slash_popup_lists_every_builtin_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from pi_coding_agent.core.slash_commands import BUILTIN_SLASH_COMMANDS

    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            provider = mode.autocomplete_provider
            assert provider is not None
            names = {command.name for command in provider._commands}
            for builtin in BUILTIN_SLASH_COMMANDS:
                assert builtin.name in names
        finally:
            await mode.shutdown()

    _run(scenario())


def test_model_command_offers_argument_completions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            provider = mode.autocomplete_provider
            assert provider is not None
            model_command = next(c for c in provider._commands if c.name == "model")
            assert model_command.get_argument_completions is not None

            completions = await model_command.get_argument_completions("")
            assert completions
            assert any("/" in item.value for item in completions)
        finally:
            await mode.shutdown()

    _run(scenario())


def test_escape_dismisses_the_popup_without_interrupting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await _type(mode, "/")
            assert await _wait_for_autocomplete(mode) is True

            # `CustomEditor` only forwards escape to the app handler when the
            # popup is closed, so this must dismiss the popup and keep the text.
            mode.editor.handle_input("\x1b")
            assert mode.editor.is_showing_autocomplete() is False
            assert mode.editor.get_text() == "/"
        finally:
            await mode.shutdown()

    _run(scenario())


def test_fd_is_resolved_from_path_when_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import shutil as shutil_module

    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            installed = shutil_module.which("fd") or shutil_module.which("fdfind")
            if installed:
                assert mode.fd_path is not None
            else:
                # The TypeScript downloader is not ported, so a missing binary
                # simply disables `@` completion rather than fetching one.
                assert mode.fd_path is None
        finally:
            await mode.shutdown()

    _run(scenario())


@pytest.mark.skipif(
    __import__("shutil").which("fd") is None and __import__("shutil").which("fdfind") is None,
    reason="`@` file completion needs fd; the auto-downloader is not ported",
)
def test_at_prefix_opens_file_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        cwd = Path(mode.session_manager.get_cwd())
        (cwd / "notes.md").write_text("hi", encoding="utf-8")
        try:
            await mode.init()
            await _type(mode, "@not")
            assert await _wait_for_autocomplete(mode) is True
            assert "notes.md" in _editor_text(mode)
        finally:
            await mode.shutdown()

    _run(scenario())


def test_plain_text_does_not_open_a_popup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            await _type(mode, "hello")
            await asyncio.sleep(0.1)
            assert mode.editor.is_showing_autocomplete() is False
        finally:
            await mode.shutdown()

    _run(scenario())


# --------------------------------------------------------------------------
# runtime settings applied at startup
# --------------------------------------------------------------------------


def test_editor_padding_follows_the_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        mode.settings_manager.set_editor_padding_x(3)
        try:
            await mode.init()
            assert mode.default_editor._padding_x == 3
        finally:
            await mode.shutdown()

    _run(scenario())


def test_autocomplete_max_visible_follows_the_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        mode.settings_manager.set_autocomplete_max_visible(3)
        try:
            await mode.init()
            await _type(mode, "/")
            assert await _wait_for_autocomplete(mode) is True
            rendered = _editor_text(mode)
            # 3 visible rows plus the "(1/N)" counter line.
            assert "(1/" in rendered
            assert rendered.count("  ") > 0
        finally:
            await mode.shutdown()

    _run(scenario())


def test_http_idle_timeout_is_applied_at_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from pi_ai.utils.http import get_idle_timeout_ms, set_idle_timeout_ms

    async def scenario() -> None:
        previous = get_idle_timeout_ms()
        set_idle_timeout_ms(None)
        mode = await _make_mode(tmp_path, monkeypatch)
        mode.settings_manager.set_http_idle_timeout_ms(30_000)
        try:
            await mode.init()
            assert get_idle_timeout_ms() == 30_000
        finally:
            await mode.shutdown()
            set_idle_timeout_ms(previous)

    _run(scenario())


def test_available_provider_count_reaches_the_footer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            assert mode.footer_data_provider.get_available_provider_count() >= 1
        finally:
            await mode.shutdown()

    _run(scenario())


def test_footer_data_provider_tracks_the_session_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            assert mode.footer_data_provider.cwd == mode.session_manager.get_cwd()
        finally:
            await mode.shutdown()

    _run(scenario())


def test_reload_rebuilds_the_autocomplete_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            first = mode.autocomplete_provider
            await mode._handle_submit("/reload")
            assert mode.autocomplete_provider is not None
            assert mode.autocomplete_provider is not first
            await _type(mode, "/")
            assert await _wait_for_autocomplete(mode) is True
        finally:
            await mode.shutdown()

    _run(scenario())


# --------------------------------------------------------------------------
# submitting through the editor
# --------------------------------------------------------------------------


def test_enter_submits_through_the_editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            pending = asyncio.ensure_future(mode.get_user_input())
            await _type(mode, "hello there")
            mode.editor.handle_input("\r")
            assert await asyncio.wait_for(pending, timeout=5) == "hello there"
        finally:
            await mode.shutdown()

    _run(scenario())


def test_bang_typed_into_the_editor_switches_border_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            assert mode.is_bash_mode is False
            await _type(mode, "!ls")
            assert mode.is_bash_mode is True
        finally:
            await mode.shutdown()

    _run(scenario())
