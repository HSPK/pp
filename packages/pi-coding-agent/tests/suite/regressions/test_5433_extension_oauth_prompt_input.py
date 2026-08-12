"""Python port of `packages/coding-agent/test/suite/regressions/5433-extension-oauth-prompt-input.test.ts`."""

from __future__ import annotations

from typing import Any

import pytest
from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.modes.interactive.components.login_dialog import LoginDialogComponent
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi
from pi_tui.keybindings import get_keybindings, set_keybindings


class _FakeTui:
    def request_render(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _theme_and_keybindings(monkeypatch: pytest.MonkeyPatch):
    init_theme("dark")
    previous = get_keybindings()
    set_keybindings(KeybindingsManager())
    monkeypatch.setattr(
        "pi_coding_agent.modes.interactive.components.login_dialog.open_browser",
        lambda _url: None,
    )
    yield
    set_keybindings(previous)


def _create_dialog() -> LoginDialogComponent:
    return LoginDialogComponent(_FakeTui(), "prompt-repro", lambda *_args: None, "Prompt Repro")


def _render_dialog(dialog: LoginDialogComponent) -> list[str]:
    return [line.rstrip() for line in strip_ansi("\n".join(dialog.render(120))).split("\n")]


def _count_rendered_value(lines: list[str], value: str) -> int:
    return len([line for line in lines if line.strip() == f"> {value}"])


def _type(dialog: LoginDialogComponent, text: str) -> None:
    dialog.handle_input(text)


async def test_keeps_previous_prompt_input_stable_when_a_later_prompt_is_active() -> None:
    dialog = _create_dialog()

    first_prompt = dialog.show_prompt("First prompt:", "first-value")
    _type(dialog, "first-value")
    _type(dialog, "\r")
    assert await first_prompt == "first-value"

    second_prompt = dialog.show_prompt("Second prompt:")
    _type(dialog, "second-secret-demo")

    lines = _render_dialog(dialog)
    assert "First prompt:" in "\n".join(lines)
    assert "Second prompt:" in "\n".join(lines)
    assert _count_rendered_value(lines, "first-value") == 1
    assert _count_rendered_value(lines, "second-secret-demo") == 1

    _type(dialog, "\r")
    assert await second_prompt == "second-secret-demo"


async def test_preserves_auth_instructions_when_showing_a_prompt() -> None:
    dialog = _create_dialog()

    dialog.show_auth("https://example.invalid/login", "Authorize the extension")
    dialog.show_prompt("First prompt:")

    output = "\n".join(_render_dialog(dialog))
    assert "https://example.invalid/login" in output
    assert "Authorize the extension" in output
    assert "First prompt:" in output


async def test_preserves_neutral_information_and_links_when_showing_a_prompt() -> None:
    dialog = _create_dialog()

    links: list[dict[str, Any]] = [{"label": "Provider documentation", "url": "https://example.invalid/docs"}]
    dialog.show_info("Configure credentials outside pi.", links)
    dialog.show_prompt("Press Enter to continue:")

    output = "\n".join(_render_dialog(dialog))
    assert "Configure credentials outside pi." in output
    assert "Provider documentation: https://example.invalid/docs" in output
    assert "Press Enter to continue:" in output


async def test_preserves_setup_details_when_showing_a_prompt() -> None:
    dialog = _create_dialog()

    dialog.show_details(["AWS credential setup:", "providers.md"])
    dialog.show_prompt("Enter API key:")

    output = "\n".join(_render_dialog(dialog))
    assert "AWS credential setup:" in output
    assert "providers.md" in output
    assert "Enter API key:" in output


async def test_keeps_previous_manual_input_stable_when_a_later_prompt_is_active() -> None:
    dialog = _create_dialog()

    manual_input = dialog.show_manual_input("Paste callback URL:")
    _type(dialog, "callback-value")
    _type(dialog, "\r")
    assert await manual_input == "callback-value"

    prompt = dialog.show_prompt("Second prompt:")
    _type(dialog, "second-secret-demo")

    lines = _render_dialog(dialog)
    assert "Paste callback URL:" in "\n".join(lines)
    assert "Second prompt:" in "\n".join(lines)
    assert _count_rendered_value(lines, "callback-value") == 1
    assert _count_rendered_value(lines, "second-secret-demo") == 1

    _type(dialog, "\r")
    assert await prompt == "second-secret-demo"
