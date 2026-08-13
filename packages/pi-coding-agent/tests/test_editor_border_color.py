"""The editor border must track the thinking level and bash mode.

shift+tab cycles the thinking level, and upstream signals that by recolouring
the editor border (`interactive-mode.ts:3994`). This port assigned the new
colour onto a freshly built `EditorTheme` and replaced `editor._theme`, but
`Editor` copies `theme.border_color` into its own public `border_color` at
construction and renders from that -- so the assignment changed a field nothing
reads and the border never moved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from test_interactive_mode import _make_mode, _run


def _border_line(mode) -> str:
    for line in mode.default_editor.render(40):
        if "─" in line:
            return line
    raise AssertionError("editor rendered no border")


def test_cycling_the_thinking_level_recolours_the_border(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            # The faux model clamps unsupported levels back to "off", so drive
            # the property the renderer actually reads rather than the session
            # state -- the defect under test is that the assignment never
            # reached the editor, not how the level is chosen.
            monkeypatch.setattr(type(mode.session), "thinking_level", property(lambda _self: "off"))
            mode._update_editor_border_color()
            off = _border_line(mode)

            monkeypatch.setattr(type(mode.session), "thinking_level", property(lambda _self: "high"))
            mode._update_editor_border_color()
            high = _border_line(mode)

            assert off != high, "border colour did not change with the thinking level"
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)


def test_bash_mode_overrides_the_thinking_colour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`!` bash mode owns the border regardless of thinking level."""

    async def scenario() -> None:
        mode, _terminal = await _make_mode(tmp_path, monkeypatch)
        try:
            await mode.init()
            monkeypatch.setattr(type(mode.session), "thinking_level", property(lambda _self: "high"))
            mode.is_bash_mode = False
            mode._update_editor_border_color()
            thinking = _border_line(mode)

            mode.is_bash_mode = True
            mode._update_editor_border_color()
            bash = _border_line(mode)

            assert bash != thinking
        finally:
            await mode.shutdown()

    _run(scenario(), timeout=30)
