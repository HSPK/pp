"""Built-in per-tool renderers, ported from `core/tools/*.ts`.

These were listed as unported, but the gap was not the formatting code alone:
`ToolExecutionComponent` looked for a `built_in_tool_definition` that no code
path ever supplied, so every built-in tool fell back to the generic renderer
and printed raw argument JSON.

`read` and `bash` collapse differently on purpose, and that difference is the
thing most worth pinning: a collapsed `read` renders nothing (the call line
already says which file it was), while a collapsed `bash` previews the *last*
few lines, because what a command just printed is usually the point.
"""

from __future__ import annotations

import re

import pytest
from pi_agent.types import AgentToolResult
from pi_ai.types import TextContent
from pi_coding_agent.modes.interactive.theme.theme import init_theme, theme
from pi_coding_agent.tools import create_all_tool_definitions
from pi_coding_agent.tools.bash import format_bash_call, format_bash_result_lines
from pi_coding_agent.tools.read import format_read_call, format_read_result

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


class _Options:
    def __init__(self, expanded: bool, is_partial: bool = False) -> None:
        self.expanded = expanded
        self.is_partial = is_partial


@pytest.fixture(autouse=True)
def _theme():
    init_theme("dark")


def _result(text: str, details=None) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=text)], details=details)


# -- registration ----------------------------------------------------------


def test_read_and_bash_have_registered_renderers():
    """Without registration the formatters exist but nothing calls them."""
    definitions = create_all_tool_definitions("/tmp")

    for name in ("read", "bash"):
        assert definitions[name].render_call is not None, name
        assert definitions[name].render_result is not None, name


# -- read ------------------------------------------------------------------


def test_read_call_shows_the_path_not_the_argument_json():
    assert _plain(format_read_call({"path": "/tmp/x.py"}, theme, "/tmp")) == "read /tmp/x.py"


def test_read_call_appends_an_explicit_line_range():
    assert _plain(format_read_call({"path": "/a.py", "offset": 5, "limit": 3}, theme, "/x")) == "read /a.py:5-7"
    assert _plain(format_read_call({"path": "/a.py", "offset": 5}, theme, "/x")) == "read /a.py:5"


def test_a_collapsed_read_renders_nothing():
    """Upstream shows only the call line until expanded (`read.ts:179`).

    Emitting the body here would make every read twice as tall as in
    TypeScript, which is exactly the reported divergence.
    """
    assert format_read_result({"path": "a.py"}, _result("x\ny"), _Options(False), theme, False, "/tmp", False) == ""


def test_an_expanded_read_renders_the_body():
    out = format_read_result({"path": "a.py"}, _result("one\ntwo"), _Options(True), theme, False, "/tmp", False)

    assert _plain(out) == "\none\ntwo"


def test_a_collapsed_read_still_renders_when_it_is_an_error():
    """Errors are the exception: they must be visible without expanding."""
    out = format_read_result({"path": "a.py"}, _result("boom"), _Options(False), theme, False, "/tmp", True)

    assert "boom" in _plain(out)


# -- bash ------------------------------------------------------------------


def test_bash_call_renders_a_shell_prompt():
    assert _plain(format_bash_call({"command": "ls -la"}, theme)) == "$ ls -la"


def test_bash_call_notes_an_explicit_timeout():
    assert _plain(format_bash_call({"command": "sleep 5", "timeout": 30}, theme)) == "$ sleep 5 (timeout 30s)"


def test_a_collapsed_bash_previews_the_last_lines():
    """The opposite of `read`: the tail is what the user wants to see."""
    body = "\n".join(f"line{i}" for i in range(1, 11))

    lines = [
        _plain(x).rstrip() for x in format_bash_result_lines(_result(body), _Options(False), theme, False, None, None)
    ]

    assert "line10" in lines
    assert "line1" not in lines
    assert any("5 earlier lines" in line for line in lines)


def test_an_expanded_bash_shows_every_line():
    body = "\n".join(f"line{i}" for i in range(1, 11))

    lines = [
        _plain(x).rstrip() for x in format_bash_result_lines(_result(body), _Options(True), theme, False, None, None)
    ]

    assert "line1" in lines and "line10" in lines
    assert not any("earlier lines" in line for line in lines)


def test_bash_reports_elapsed_time_while_running_and_total_when_done():
    running = format_bash_result_lines(_result(""), _Options(False, True), theme, False, 1000.0, 3500.0)
    finished = format_bash_result_lines(_result(""), _Options(False), theme, False, 1000.0, 3500.0)

    assert "Elapsed 2.5s" in _plain("\n".join(running))
    assert "Took 2.5s" in _plain("\n".join(finished))
