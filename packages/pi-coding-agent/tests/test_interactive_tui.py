"""Python port of `packages/coding-agent/test/interactive-tui.test.ts`.

Covers the renderer composition root (`create_interactive_tui`), right-click
paste, the copy-shortcut confirmation, and the clear-on-shrink status spacing.
"""

from __future__ import annotations

import asyncio

import pytest
from pi_coding_agent.modes.interactive.interactive_mode import InteractiveMode, create_interactive_tui
from pi_tui.component import Component, Container
from pi_tui.components.text import Text
from pi_tui.testing import FakeTerminal, MiniAltScreenModel
from pi_tui.tui import is_viewport_tui


async def wait_render(ui: object, timeout_s: float = 5.0) -> None:
    """Wait until the TUI's throttled render timer has actually drawn a frame.

    `Tui.request_render` sets `_render_requested` and schedules the draw
    through `loop.call_later(MIN_RENDER_INTERVAL_S - elapsed)`, clearing the
    flag once the frame is written. Sleeping a fixed 30 ms instead only works
    while the machine is idle: `call_later` guarantees a lower bound, not an
    upper one, so under the parallel suite's load the assertion can run before
    the frame exists. Watching the flag pins the same "a frame has been drawn"
    condition without a clock.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while getattr(ui, "_render_requested", False):
        if loop.time() > deadline:
            raise AssertionError("Timed out waiting for a render")
        await asyncio.sleep(0)


def _all_writes(terminal: FakeTerminal) -> str:
    return "".join(terminal.writes)


def _viewport(terminal: FakeTerminal, width: int, height: int) -> list[str]:
    model = MiniAltScreenModel(width, height)
    for write in terminal.writes:
        model.feed(write)
    return [line.rstrip() for line in model.screen()]


class TestCreateInteractiveTui:
    @pytest.mark.asyncio
    async def test_selects_the_alternate_screen_renderer_only_when_requested(self) -> None:
        main_terminal = FakeTerminal()
        main_tui = create_interactive_tui(
            tui_mode="regular",
            show_hardware_cursor=False,
            log_directory=None,
            terminal=main_terminal,
        )
        assert main_tui.mode == "regular"
        assert is_viewport_tui(main_tui) is False
        main_tui.start()
        await wait_render(main_tui)
        assert "\x1b[?1049h" not in _all_writes(main_terminal)
        main_tui.stop()

        alt_terminal = FakeTerminal()
        alt_tui = create_interactive_tui(
            tui_mode="fullscreen",
            show_hardware_cursor=False,
            log_directory=None,
            terminal=alt_terminal,
        )
        assert alt_tui.mode == "fullscreen"
        assert is_viewport_tui(alt_tui) is True
        alt_tui.start()
        await wait_render(alt_tui)
        assert "\x1b[?1049h" in _all_writes(alt_terminal)
        alt_tui.stop()

    @pytest.mark.skip(
        reason=(
            "TS: 'replaces the renderer and restores the previous screen for "
            "resume-hint exits'. Needs `InteractiveMode.switchTuiMode()` and "
            "`stopInteractiveTui('resume-hint')`. Verified against the source (not "
            "just the module docstring, which is stale on several other items in the "
            "same list -- `/export`/`/import`/`/clone`/`/share`/`/tree` it also lists "
            "as unported are in fact implemented below in the slash-command dispatch): "
            "`grep`ing `interactive_mode.py` for `switch_tui_mode`/`stop_interactive_tui` "
            "finds neither method, and `tui_mode` is set once in `__init__` and never "
            "reassigned, so runtime switching between regular and fullscreen genuinely "
            "does not exist yet. This gap is NOT listed in the top-level README's 'Not "
            "ported, by decision' section, so it is not a documented deliberate omission "
            "-- it is a real, unimplemented feature this review did not add (a full port "
            "needs render-state capture/restore, focus/layout-root migration and overlay "
            "handling well beyond this test-parity pass). `create_interactive_tui_reference` "
            "is ported and is exercised by `InteractiveMode.__init__`."
        )
    )
    def test_replaces_the_renderer_and_restores_the_previous_screen(self) -> None:
        raise AssertionError("unreachable")


class _PasteTarget(Component):
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def render(self, _width: int) -> list[str]:
        return []

    def invalidate(self) -> None:
        return None

    def handle_input(self, data: str) -> None:
        self.inputs.append(data)


class _PasteRenderer:
    def __init__(self, target: Component | None) -> None:
        self.target = target

    def get_focused_component(self) -> Component | None:
        return self.target


class _PasteUi:
    def __init__(self) -> None:
        self.render_requests = 0

    def request_render(self) -> None:
        self.render_requests += 1


class _PasteContext:
    _handle_right_click_paste = InteractiveMode._handle_right_click_paste

    def __init__(self, target: Component | None) -> None:
        self.renderer = _PasteRenderer(target)
        self.ui = _PasteUi()


class TestRightClickPaste:
    @pytest.mark.asyncio
    async def test_feeds_clipboard_text_as_a_bracketed_paste(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_read(_env: object = None) -> str:
            return "clipboard text"

        monkeypatch.setattr(
            "pi_coding_agent.modes.interactive.interactive_mode.read_clipboard_text",
            fake_read,
        )
        target = _PasteTarget()
        context = _PasteContext(target)

        await context._handle_right_click_paste()

        assert target.inputs == ["\x1b[200~clipboard text\x1b[201~"]
        assert context.ui.render_requests == 1


class _CopySession:
    def __init__(self, text: str | None) -> None:
        self.text = text

    def get_last_assistant_text(self) -> str | None:
        return self.text


class _CopyContext:
    _handle_copy_command = InteractiveMode._handle_copy_command

    def __init__(self, text: str | None, ui: object) -> None:
        self.session = _CopySession(text)
        self.ui = ui
        self.statuses: list[str] = []
        self.errors: list[str] = []

    def show_status(self, message: str) -> None:
        self.statuses.append(message)

    def show_error(self, message: str) -> None:
        self.errors.append(message)


@pytest.fixture
def copied_texts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    texts: list[str] = []

    async def fake_copy(text: str) -> None:
        texts.append(text)

    monkeypatch.setattr(
        "pi_coding_agent.modes.interactive.interactive_mode.copy_to_clipboard",
        fake_copy,
    )
    return texts


class TestCopyConfirmation:
    @pytest.mark.asyncio
    async def test_flashes_copied_for_the_copy_shortcut_in_fullscreen_mode(self, copied_texts: list[str]) -> None:
        terminal = FakeTerminal(40, 4)
        ui = create_interactive_tui(
            tui_mode="fullscreen",
            show_hardware_cursor=False,
            log_directory=None,
            terminal=terminal,
        )
        context = _CopyContext("assistant response", ui)

        ui.start()
        try:
            await wait_render(ui)
            await context._handle_copy_command(flash_confirmation=True)
            await wait_render(ui)

            assert copied_texts == ["assistant response"]
            assert context.statuses == []
            assert context.errors == []
            assert any("Copied!" in line for line in _viewport(terminal, 40, 4))
        finally:
            ui.stop()

    @pytest.mark.asyncio
    async def test_keeps_the_status_line_confirmation_in_regular_mode(self, copied_texts: list[str]) -> None:
        ui = create_interactive_tui(
            tui_mode="regular",
            show_hardware_cursor=False,
            log_directory=None,
            terminal=FakeTerminal(),
        )
        context = _CopyContext("assistant response", ui)

        await context._handle_copy_command(flash_confirmation=True)

        assert context.statuses == ["Copied last agent message to clipboard"]
        assert context.errors == []


class _StatusIndicatorStub:
    def __init__(self) -> None:
        self.kind = "working"
        self.dispose_count = 0

    def dispose(self) -> None:
        self.dispose_count += 1


class _ClearOnShrinkUi:
    # Deliberately has no `request_render`: the TS test's `ui` mock is typed as
    # `{ getClearOnShrink: () => boolean }` only, which proves `clearStatusIndicator`
    # itself never calls `ui.requestRender()` (callers do that separately). If the
    # Python port ever regresses to calling `self.ui.request_render()` inside
    # `_clear_status_indicator`, this stub makes that raise `AttributeError`.
    def get_clear_on_shrink(self) -> bool:
        return True


class _ClearStatusContext:
    _clear_status_indicator = InteractiveMode._clear_status_indicator

    def __init__(self, tui_mode: str) -> None:
        self.active_status_indicator = _StatusIndicatorStub()
        self.status_container = Container()
        self.tui_mode = tui_mode
        self.ui = _ClearOnShrinkUi()
        self.idle_status = Text("", 0, 0)


class TestClearOnShrinkStatusSpacing:
    @pytest.mark.parametrize(("tui_mode", "expected_children"), [("regular", 1), ("fullscreen", 0)])
    def test_reserves_status_height_only_on_the_main_screen_renderer(
        self, tui_mode: str, expected_children: int
    ) -> None:
        context = _ClearStatusContext(tui_mode)
        indicator = context.active_status_indicator
        assert indicator is not None

        context._clear_status_indicator()

        assert indicator.dispose_count == 1
        assert len(context.status_container.children) == expected_children
