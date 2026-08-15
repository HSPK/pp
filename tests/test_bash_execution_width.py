"""Python port of `packages/coding-agent/test/bash-execution-width.test.ts`.

Regression test for #2569: `BashExecutionComponent`'s collapsed output must
respect the render-time width, not a stale captured width.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pi_tui import visible_width

from pi_coding_agent.modes.interactive.components.bash_execution import BashExecutionComponent
from pi_coding_agent.modes.interactive.theme.theme import init_theme


@dataclass
class _Terminal:
    columns: int
    rows: int = 24


class _Disposable:
    def dispose(self) -> None:
        return None


class _TuiStub:
    """Minimal TUI stub that only exposes `terminal.columns` and interval hooks."""

    def __init__(self, columns: int) -> None:
        self.terminal = _Terminal(columns)

    def add_interval(self, callback: object, ms: int) -> _Disposable:
        return _Disposable()

    def remove_interval(self, handle: object) -> None:
        return None

    def request_render(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _theme() -> None:
    init_theme("dark")


def test_collapsed_preview_lines_respect_render_time_width() -> None:
    wide_width = 200
    narrow_width = 80

    component = BashExecutionComponent("pwd", _TuiStub(wide_width))  # type: ignore[arg-type]

    long_line = "x" * 150
    component.append_output(f"{long_line}\n{long_line}\n")

    component.set_complete(0, False)

    lines = component.render(narrow_width)

    for index, line in enumerate(lines):
        width = visible_width(line)
        assert width <= narrow_width, f"Line {index} visible_width={width} > {narrow_width}"


def test_recomputes_lines_when_width_changes_between_renders() -> None:
    component = BashExecutionComponent("echo hello", _TuiStub(200))  # type: ignore[arg-type]

    long_line = "abcdefghij" * 20
    component.append_output(f"{long_line}\n")
    component.set_complete(0, False)

    for line in component.render(200):
        assert visible_width(line) <= 200

    lines60 = component.render(60)
    for index, line in enumerate(lines60):
        width = visible_width(line)
        assert width <= 60, f"Line {index} visible_width={width} > 60"
