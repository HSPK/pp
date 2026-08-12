"""Python port of `packages/coding-agent/test/export-html-whitespace.test.ts`.

The first TypeScript case asserts on `template.css`; the HTML exporter's
document assembly (template.css/template.js) is not ported -- see the port
README -- so only its two renderer-level cases are portable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pi_coding_agent.core.export_html.ansi_to_html import ansi_lines_to_html
from pi_coding_agent.core.export_html.tool_renderer import create_tool_html_renderer
from pi_tui.component import Component


class _FixedComponent(Component):
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def render(self, width: int) -> list[str]:
        return list(self._lines)

    def invalidate(self) -> None:
        return None


@dataclass
class _Tool:
    name: str
    label: str
    description: str
    render_result: Any


@pytest.mark.skip(
    reason=(
        "`it('preserves whitespace for plain-text tool output lines without preserving template whitespace')` "
        "asserts three regexes against `src/core/export-html/template.css`. This port has no "
        "`export_html/template.css` (nor template.js/template.html): the HTML exporter's document "
        "assembly is deliberately omitted -- only `ansi_to_html.py`, `colors.py` and `tool_renderer.py` "
        "are ported. There is no file to assert against."
    )
)
def test_preserves_whitespace_for_plain_text_tool_output_lines() -> None:
    """The TypeScript assertions this stands in for, verbatim:

    1. css matches ``/\\.output-preview > div:not\\(\\.expand-hint\\),\\s*\\.output-full >
       div:not\\(\\.expand-hint\\) \\{[\\s\\S]*?white-space:\\s*pre-wrap;/``
    2. css matches ``/\\.ansi-line\\s*\\{[\\s\\S]*?white-space:\\s*pre;/``
    3. css does NOT match ``/\\.output-preview,\\s*\\.output-full\\s*\\{[\\s\\S]*?white-space:\\s*pre-wrap;/``
    """


def test_does_not_insert_source_whitespace_between_ansi_lines() -> None:
    assert ansi_lines_to_html(["one", "two"]) == '<div class="ansi-line">one</div><div class="ansi-line">two</div>'


def test_trims_tui_spacing_lines_from_custom_tool_result_html() -> None:
    component = _FixedComponent(["", "\x1b[31mone\x1b[0m", "two", ""])
    tool = _Tool(
        name="custom",
        label="custom",
        description="custom",
        render_result=lambda *_args: component,
    )
    renderer = create_tool_html_renderer(lambda _name: tool, object(), "/tmp")

    rendered = renderer.render_result("id", "custom", [], None, False)

    assert rendered is not None
    assert rendered.expanded == (
        '<div class="ansi-line"><span style="color:#800000">one</span></div><div class="ansi-line">two</div>'
    )
