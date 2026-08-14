"""Tests for the components ported for interactive mode.

Covers `components/text.py`, `spacer.py`, `truncated_text.py`, `v_stack.py`,
`h_stack.py`, `loader.py` and `timers.py`.
"""

from __future__ import annotations

import asyncio

import pytest
from pi_tui.components.h_stack import HStack
from pi_tui.components.loader import (
    DEFAULT_FRAMES,
    DEFAULT_INTERVAL_MS,
    CancellableLoader,
    Loader,
    LoaderIndicatorOptions,
)
from pi_tui.components.spacer import Spacer
from pi_tui.components.stack import StackEntry, StackOptions
from pi_tui.components.text import Text
from pi_tui.components.truncated_text import TruncatedText
from pi_tui.components.v_stack import VStack
from pi_tui.keybindings import (
    TUI_KEYBINDINGS,
    KeybindingsManager,
    get_keybindings,
    set_keybindings,
)
from pi_tui.testing import FakeTerminal, wait_until
from pi_tui.timers import schedule_interval
from pi_tui.tui_main_screen import TuiMainScreen
from pi_tui.utils import strip_terminal_sequences, visible_width

# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def test_text_wraps_and_pads_to_width():
    lines = Text("hello world this is long", 1, 0).render(12)
    assert all(visible_width(line) == 12 for line in lines)
    assert lines[0] == " hello      "


def test_text_renders_nothing_for_blank_content():
    assert Text("", 1, 1).render(10) == []
    assert Text("   \n\t ", 1, 1).render(10) == []


def test_text_adds_vertical_padding():
    lines = Text("hi", 0, 2).render(6)
    assert lines[:2] == ["      ", "      "]
    assert lines[-2:] == ["      ", "      "]
    assert lines[2] == "hi    "


def test_text_expands_tabs_to_three_spaces():
    assert Text("a\tb", 0, 0).render(10)[0].startswith("a   b")


def test_text_caches_until_invalidated():
    text = Text("hello", 0, 0)
    first = text.render(10)
    assert text.render(10) is first

    text.set_text("changed")
    assert text.render(10)[0].startswith("changed")

    second = text.render(10)
    text.invalidate()
    assert text.render(10) is not second


def test_text_cache_misses_on_width_change():
    text = Text("hello world", 0, 0)
    narrow = text.render(5)
    wide = text.render(20)
    assert narrow != wide


def test_text_applies_custom_background():
    text = Text("hi", 0, 1, lambda s: f"<{s}>")
    lines = text.render(6)
    assert lines[0].startswith("<") and lines[0].endswith(">")
    assert "hi" in lines[1]

    text.set_custom_bg_fn(None)
    assert "<" not in text.render(6)[1]


def test_text_content_width_never_drops_below_one():
    # padding_x larger than the viewport must not produce a zero/negative width.
    assert Text("abc", 10, 0).render(4)


# --------------------------------------------------------------------------
# Spacer
# --------------------------------------------------------------------------


def test_spacer_renders_requested_number_of_blank_lines():
    spacer = Spacer(3)
    assert spacer.render(20) == ["", "", ""]
    spacer.set_lines(0)
    assert spacer.render(20) == []
    spacer.invalidate()
    assert spacer.render(20) == []


# --------------------------------------------------------------------------
# TruncatedText
# --------------------------------------------------------------------------


def test_truncated_text_keeps_only_the_first_line():
    line = TruncatedText("first\nsecond", 0, 0).render(20)[0]
    assert "second" not in line
    assert line.startswith("first")


def test_truncated_text_pads_to_width():
    lines = TruncatedText("abc", 1, 1).render(10)
    assert len(lines) == 3
    assert all(visible_width(line) == 10 for line in lines)


def test_truncated_text_truncates_long_content():
    line = TruncatedText("x" * 40, 0, 0).render(10)[0]
    assert visible_width(line) == 10
    assert strip_terminal_sequences(line).endswith("...")


def test_truncated_text_invalidate_is_a_noop():
    component = TruncatedText("abc")
    component.invalidate()
    assert component.render(10)


# --------------------------------------------------------------------------
# VStack / HStack
# --------------------------------------------------------------------------


def test_vstack_concatenates_children_in_order():
    stack = VStack([Text("a", 0, 0), Text("b", 0, 0)])
    assert [line.rstrip() for line in stack.render(10)] == ["a", "b"]


def test_vstack_inserts_gap_lines_between_children():
    stack = VStack([Text("a", 0, 0), Text("b", 0, 0)], StackOptions(gap=2))
    assert [line.rstrip() for line in stack.render(10)] == ["a", "", "", "b"]


def test_vstack_skips_hidden_children():
    stack = VStack(
        [
            StackEntry(component=Text("shown", 0, 0)),
            StackEntry(component=Text("hidden", 0, 0), visible=lambda _viewport: False),
        ]
    )
    assert [line.rstrip() for line in stack.render(20)] == ["shown"]


def test_vstack_pads_children_up_to_their_allocated_size():
    stack = VStack([StackEntry(component=Text("a", 0, 0), min_size=3)])
    assert [line.rstrip() for line in stack.render(10)] == ["a", "", ""]


def test_vstack_truncates_children_to_max_size():
    stack = VStack([StackEntry(component=Text("a\nb\nc", 0, 0), max_size=2)])
    assert [line.rstrip() for line in stack.render(10)] == ["a", "b"]


def test_hstack_places_children_side_by_side():
    stack = HStack([Text("ab", 0, 0), Text("cd", 0, 0)])
    rendered = strip_terminal_sequences(stack.render(10)[0])
    # Text pads itself to its allocated width, so the children are separated by
    # that padding rather than packed tight; order and total width are what matter.
    assert rendered.replace(" ", "") == "abcd"
    assert rendered.index("ab") < rendered.index("cd")
    assert visible_width(rendered) == 10


def test_hstack_honours_gap():
    without_gap = strip_terminal_sequences(HStack([Text("ab", 0, 0), Text("cd", 0, 0)]).render(12)[0])
    with_gap = strip_terminal_sequences(HStack([Text("ab", 0, 0), Text("cd", 0, 0)], StackOptions(gap=2)).render(12)[0])
    # The gap eats into the width available to the children.
    assert with_gap.replace(" ", "") == "abcd"
    assert with_gap.index("cd") >= without_gap.index("cd")


def test_hstack_returns_nothing_when_all_children_hidden():
    stack = HStack([StackEntry(component=Text("x", 0, 0), visible=lambda _viewport: False)])
    assert stack.render(10) == []


def test_hstack_aligns_shorter_children():
    tall = Text("1\n2\n3", 0, 0)
    short = Text("s", 0, 0)

    top = strip_terminal_sequences(HStack([tall, short], StackOptions(align="start")).render(10)[0])
    assert "s" in top

    bottom_lines = HStack([tall, short], StackOptions(align="end")).render(10)
    assert "s" in strip_terminal_sequences(bottom_lines[-1])

    center_lines = HStack([tall, short], StackOptions(align="center")).render(10)
    assert "s" in strip_terminal_sequences(center_lines[1])


def test_hstack_height_is_the_tallest_child():
    stack = HStack([Text("1\n2\n3", 0, 0), Text("x", 0, 0)])
    assert len(stack.render(10)) == 3


# --------------------------------------------------------------------------
# timers
# --------------------------------------------------------------------------


def test_schedule_interval_fires_repeatedly():
    async def scenario() -> int:
        ticks = 0

        def tick() -> None:
            nonlocal ticks
            ticks += 1

        handle = schedule_interval(tick, 0.005)
        assert handle.scheduled
        # Wait for a number of ticks instead of a duration: the assertion below
        # is about repetition, not about how long the repetition took.
        await wait_until(lambda: ticks >= 5, message="interval never fired 5 times")
        handle.cancel()
        assert not handle.scheduled
        captured = ticks
        await asyncio.sleep(0.03)
        assert ticks == captured
        return ticks

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=5)) >= 2


def test_schedule_interval_is_inert_without_a_running_loop():
    ticks: list[None] = []
    handle = schedule_interval(lambda: ticks.append(None), 0.001)
    assert handle.scheduled is False
    handle.cancel()
    assert ticks == []


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------


def _plain(text: str) -> str:
    return text


def test_loader_renders_leading_blank_line_and_message():
    loader = Loader(None, _plain, _plain, "Working...")
    lines = loader.render(30)
    assert lines[0] == ""
    assert "Working..." in lines[1]
    assert lines[1].strip().startswith(DEFAULT_FRAMES[0])


def test_loader_requests_render_on_updates():
    # Spy on the real `TuiMainScreen` rather than a one-method double: the
    # loader must call the render request the production TUI actually exposes,
    # with the arguments it actually accepts.
    ui = TuiMainScreen(FakeTerminal())
    real_request_render = ui.request_render
    calls: list[bool] = []

    def recording_request_render(force: bool = False) -> None:
        calls.append(force)
        real_request_render(force)

    ui.request_render = recording_request_render  # type: ignore[method-assign]

    loader = Loader(ui, _plain, _plain, "a")
    before = len(calls)
    loader.set_message("b")
    assert len(calls) == before + 1
    assert "b" in loader.render(30)[1]


def test_loader_colors_spinner_and_message_separately():
    loader = Loader(None, lambda s: f"<{s}>", lambda s: f"[{s}]", "msg")
    line = loader.render(30)[1]
    assert f"<{DEFAULT_FRAMES[0]}>" in line
    assert "[msg]" in line


def test_loader_custom_indicator_is_rendered_verbatim():
    loader = Loader(None, lambda s: f"<{s}>", _plain, "msg", LoaderIndicatorOptions(frames=["*"]))
    line = loader.render(30)[1]
    assert "*" in line
    assert "<*>" not in line


def test_loader_empty_frames_hide_the_indicator():
    loader = Loader(None, _plain, _plain, "msg", LoaderIndicatorOptions(frames=[]))
    assert loader.render(30)[1].strip() == "msg"


def test_loader_indicator_interval_falls_back_to_default():
    loader = Loader(None, _plain, _plain, "msg", LoaderIndicatorOptions(frames=["a", "b"], interval_ms=0))
    assert loader._interval_ms == DEFAULT_INTERVAL_MS

    loader.set_indicator(LoaderIndicatorOptions(frames=["a", "b"], interval_ms=10))
    assert loader._interval_ms == 10


def test_loader_animation_advances_frames():
    async def scenario() -> str:
        loader = Loader(None, _plain, _plain, "msg", LoaderIndicatorOptions(frames=["A", "B"], interval_ms=5))
        try:
            for _ in range(200):
                await asyncio.sleep(0.005)
                if "B" in loader.render(30)[1]:
                    return "advanced"
            return "stuck"
        finally:
            loader.stop()

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=10)) == "advanced"


def test_loader_single_frame_schedules_no_timer():
    loader = Loader(None, _plain, _plain, "msg", LoaderIndicatorOptions(frames=["A"], interval_ms=5))
    assert loader._interval is None


def test_loader_stop_is_idempotent():
    loader = Loader(None, _plain, _plain, "msg")
    loader.stop()
    loader.stop()
    assert loader._interval is None


def test_cancellable_loader_aborts_on_cancel_key():
    previous = get_keybindings()
    set_keybindings(KeybindingsManager(TUI_KEYBINDINGS))
    try:
        loader = CancellableLoader(None, _plain, _plain, "msg")
        calls: list[None] = []
        loader.on_abort = lambda: calls.append(None)

        assert loader.aborted is False
        loader.handle_input("x")
        assert loader.aborted is False
        assert calls == []

        loader.handle_input("\x1b")
        assert loader.aborted is True
        assert loader.signal.aborted is True
        assert calls == [None]

        # TS calls `onAbort` on every cancel key, even after the signal aborted.
        loader.handle_input("\x1b")
        assert calls == [None, None]

        loader.dispose()
    finally:
        set_keybindings(previous)


def test_abort_controller_notifies_late_listeners():
    loader = CancellableLoader(None, _plain, _plain, "msg")
    seen: list[str] = []
    loader.signal.add_listener(lambda: seen.append("early"))
    loader.signal.abort()
    loader.signal.add_listener(lambda: seen.append("late"))
    assert seen == ["early", "late"]


def test_cancellable_loader_without_callback_does_not_raise():
    previous = get_keybindings()
    set_keybindings(KeybindingsManager(TUI_KEYBINDINGS))
    try:
        loader = CancellableLoader(None, _plain, _plain, "msg")
        loader.handle_input("\x1b")
        assert loader.aborted is True
    finally:
        set_keybindings(previous)


@pytest.mark.parametrize("width", [1, 2, 5, 40])
def test_loader_renders_at_any_width(width: int):
    assert Loader(None, _plain, _plain, "message").render(width)
