"""Tests for the transcript components of interactive mode.

Covers `footer.py`, `tool_execution.py`, `assistant_message.py`,
`bash_execution.py` and `tools/render_utils.py`.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
from pi_ai.types import AssistantMessage, TextContent, ThinkingContent, ToolCall, Usage
from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.modes.interactive.components.assistant_message import AssistantMessageComponent
from pi_coding_agent.modes.interactive.components.bash_execution import (
    PREVIEW_LINES,
    BashExecutionComponent,
)
from pi_coding_agent.modes.interactive.components.footer import (
    FooterComponent,
    format_cwd_for_footer,
    format_tokens,
    sanitize_status_text,
)
from pi_coding_agent.modes.interactive.components.tool_execution import (
    ToolExecutionComponent,
    ToolExecutionOptions,
    ToolResult,
)
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.tools.render_utils import (
    get_text_output,
    normalize_display_text,
    replace_tabs,
    shorten_path,
    str_arg,
)
from pi_coding_agent.utils.js_number import to_fixed as _to_fixed
from pi_tui.components.text import Text
from pi_tui.keybindings import get_keybindings, set_keybindings

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\x1b\]133;.\x07|\x1b\]8;;\x07")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _stripped(lines: list[str]) -> list[str]:
    return [_strip(line).rstrip() for line in lines]


@pytest.fixture(autouse=True)
def _theme_and_keybindings():
    init_theme("dark")
    previous = get_keybindings()
    set_keybindings(KeybindingsManager())
    yield
    set_keybindings(previous)


# --------------------------------------------------------------------------
# footer helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1000, "1.0k"),
        (1500, "1.5k"),
        (1950, "1.9k"),
        (9999, "10.0k"),
        (10000, "10k"),
        (10500, "11k"),
        (999999, "1000k"),
        (1000000, "1.0M"),
        (1500000, "1.5M"),
        (9999999, "10.0M"),
        (10000000, "10M"),
        (123456789, "123M"),
    ],
)
def test_format_tokens_matches_typescript(count: int, expected: str):
    assert format_tokens(count) == expected


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [(1.95, 1, "1.9"), (0.5, 0, "1"), (2.5, 0, "3"), (1.25, 1, "1.3"), (2.675, 2, "2.67"), (0.125, 2, "0.13")],
)
def test_to_fixed_matches_javascript(value: float, digits: int, expected: str):
    assert _to_fixed(value, digits) == expected


def test_format_cwd_replaces_home_with_tilde():
    assert format_cwd_for_footer("/home/u/proj", "/home/u") == "~/proj"
    assert format_cwd_for_footer("/home/u", "/home/u") == "~"


def test_format_cwd_leaves_paths_outside_home():
    assert format_cwd_for_footer("/tmp/x", "/home/u") == "/tmp/x"
    assert format_cwd_for_footer("/home/user2", "/home/u") == "/home/user2"


def test_format_cwd_without_home_returns_input():
    assert format_cwd_for_footer("/tmp/x", None) == "/tmp/x"
    assert format_cwd_for_footer("/tmp/x", "") == "/tmp/x"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("a\nb", "a b"), ("  x  ", "x"), ("tab\there", "tab here"), ("multi   spaces", "multi spaces"), ("", "")],
)
def test_sanitize_status_text(text: str, expected: str):
    assert sanitize_status_text(text) == expected


# --------------------------------------------------------------------------
# footer component
# --------------------------------------------------------------------------


class _FakeSessionManager:
    def __init__(self, entries=None, cwd="/tmp/project", session_name=None) -> None:
        self._entries = entries or []
        self._cwd = cwd
        self._session_name = session_name

    def get_entries(self):
        return self._entries

    def get_cwd(self) -> str:
        return self._cwd

    def get_session_name(self):
        return self._session_name


class _FakeModelRuntime:
    def __init__(self, subscription: bool = False) -> None:
        self._subscription = subscription

    def is_using_subscription(self, _provider: str) -> bool:
        return self._subscription


class _FakeFooterData:
    def __init__(self, branch=None, statuses=None, provider_count=1) -> None:
        self._branch = branch
        self._statuses = statuses or {}
        self._provider_count = provider_count

    def get_git_branch(self):
        return self._branch

    def get_extension_statuses(self):
        return self._statuses

    def get_available_provider_count(self) -> int:
        return self._provider_count


def _fake_session(*, entries=None, model=None, context_usage=None, **manager_kwargs):
    return SimpleNamespace(
        state=SimpleNamespace(model=model, thinking_level=None),
        session_manager=_FakeSessionManager(entries, **manager_kwargs),
        model_runtime=_FakeModelRuntime(),
        get_context_usage=lambda: context_usage,
    )


def _model(**kwargs):
    return SimpleNamespace(
        id=kwargs.get("id", "gpt-test"),
        provider=kwargs.get("provider", "openai"),
        context_window=kwargs.get("context_window", 100_000),
        reasoning=kwargs.get("reasoning", False),
    )


def test_footer_shows_cwd_branch_and_session_name():
    session = _fake_session(model=_model(), cwd="/tmp/project", session_name="my session")
    footer = FooterComponent(session, _FakeFooterData(branch="main"))
    line = _strip(footer.render(80)[0])
    assert "/tmp/project (main) • my session" in line


def test_footer_shows_model_name_right_aligned():
    session = _fake_session(model=_model(id="my-model"))
    stats_line = _strip(FooterComponent(session, _FakeFooterData()).render(80)[1])
    assert stats_line.rstrip().endswith("my-model")


def test_footer_falls_back_to_no_model():
    stats_line = _strip(FooterComponent(_fake_session(), _FakeFooterData()).render(80)[1])
    assert "no-model" in stats_line


def test_footer_missing_context_usage_reports_zero_not_question_mark():
    # Faithful to the TS `contextUsage?.percent !== null` check.
    session = _fake_session(model=_model(context_window=1000))
    assert "0.0%/1.0k" in _strip(FooterComponent(session, _FakeFooterData()).render(80)[1])


def test_footer_null_percent_reports_question_mark():
    usage = SimpleNamespace(tokens=None, context_window=1000, percent=None)
    session = _fake_session(model=_model(), context_usage=usage)
    assert "?/1.0k" in _strip(FooterComponent(session, _FakeFooterData()).render(80)[1])


def test_footer_auto_compact_indicator_can_be_disabled():
    session = _fake_session(model=_model())
    footer = FooterComponent(session, _FakeFooterData())
    assert "(auto)" in _strip(footer.render(80)[1])
    footer.set_auto_compact_enabled(False)
    assert "(auto)" not in _strip(footer.render(80)[1])


def test_footer_aggregates_usage_from_entries():
    usage = Usage(input=1200, output=340)
    entry = SimpleNamespace(
        type="message",
        message=SimpleNamespace(role="assistant", usage=usage),
    )
    session = _fake_session(entries=[entry], model=_model())
    stats = _strip(FooterComponent(session, _FakeFooterData()).render(100)[1])
    assert "↑1.2k" in stats
    assert "↓340" in stats


def test_footer_shows_cache_hit_rate_when_cache_used():
    usage = Usage(input=100, output=10, cache_read=300, cache_write=100)
    entry = SimpleNamespace(type="message", message=SimpleNamespace(role="assistant", usage=usage))
    session = _fake_session(entries=[entry], model=_model())
    stats = _strip(FooterComponent(session, _FakeFooterData()).render(120)[1])
    assert "R300" in stats
    assert "W100" in stats
    assert "CH60.0%" in stats


def test_footer_shows_provider_when_several_are_available():
    session = _fake_session(model=_model(provider="anthropic"))
    stats = _strip(FooterComponent(session, _FakeFooterData(provider_count=3)).render(80)[1])
    assert "(anthropic)" in stats


def test_footer_drops_provider_when_it_does_not_fit():
    session = _fake_session(model=_model(provider="a-very-long-provider-name", id="a-very-long-model-id"))
    stats = _strip(FooterComponent(session, _FakeFooterData(provider_count=3)).render(30)[1])
    assert "a-very-long-provider-name" not in stats


def test_footer_shows_thinking_level_for_reasoning_models():
    session = _fake_session(model=_model(reasoning=True))
    assert "thinking off" in _strip(FooterComponent(session, _FakeFooterData()).render(90)[1])

    session.state.thinking_level = "high"
    assert "• high" in _strip(FooterComponent(session, _FakeFooterData()).render(90)[1])


def test_footer_appends_sorted_extension_statuses():
    session = _fake_session(model=_model())
    footer_data = _FakeFooterData(statuses={"zeta": "z status", "alpha": "a\nstatus"})
    lines = FooterComponent(session, footer_data).render(80)
    assert len(lines) == 3
    assert _strip(lines[2]).startswith("a status z status")


def test_footer_omits_status_line_when_empty():
    session = _fake_session(model=_model())
    assert len(FooterComponent(session, _FakeFooterData()).render(80)) == 2


def test_footer_survives_a_very_narrow_terminal():
    session = _fake_session(model=_model())
    lines = FooterComponent(session, _FakeFooterData()).render(5)
    assert len(lines) == 2


def test_footer_invalidate_and_dispose_are_noops():
    footer = FooterComponent(_fake_session(), _FakeFooterData())
    footer.invalidate()
    footer.dispose()
    footer.set_session(_fake_session(model=_model(id="other")))
    assert "other" in _strip(footer.render(80)[1])


# --------------------------------------------------------------------------
# render_utils
# --------------------------------------------------------------------------


def test_get_text_output_joins_text_blocks_and_strips_ansi():
    result = {"content": [{"type": "text", "text": "\x1b[31mred\x1b[0m"}, {"type": "text", "text": "b"}]}
    assert get_text_output(result, True) == "red\nb"


def test_get_text_output_of_missing_result_is_empty():
    assert get_text_output(None, True) == ""


def test_get_text_output_drops_carriage_returns():
    assert get_text_output({"content": [{"type": "text", "text": "a\rb"}]}, True) == "ab"


def test_str_arg_semantics():
    assert str_arg("x") == "x"
    assert str_arg(None) == ""
    assert str_arg(3) is None
    assert str_arg([]) is None


def test_replace_tabs_and_normalize_display_text():
    assert replace_tabs("a\tb") == "a   b"
    assert normalize_display_text("a\r\nb") == "a\nb"


def test_shorten_path_handles_non_strings():
    assert shorten_path(3) == ""
    assert shorten_path(None) == ""


# --------------------------------------------------------------------------
# tool execution
# --------------------------------------------------------------------------


def test_tool_execution_renders_name_and_arguments():
    component = ToolExecutionComponent("read", "call-1", {"path": "/x.py"}, cwd="/tmp")
    rendered = "\n".join(_stripped(component.render(40)))
    assert "read" in rendered
    assert '"path": "/x.py"' in rendered


def test_tool_execution_appends_text_output():
    component = ToolExecutionComponent("read", "call-1", {}, cwd="/tmp")
    component.update_result(ToolResult(content=[{"type": "text", "text": "body"}]))
    assert "body" in "\n".join(_stripped(component.render(40)))


def test_tool_execution_uses_registered_renderers():
    definition = SimpleNamespace(
        render_call=lambda args, theme, ctx: Text(f"CALL {args['n']}", 0, 0),
        render_result=lambda result, options, theme, ctx: Text("RESULT", 0, 0),
        render_shell="default",
    )
    component = ToolExecutionComponent("x", "id", {"n": 1}, tool_definition=definition, cwd="/tmp")
    assert "CALL 1" in "\n".join(_stripped(component.render(40)))

    component.update_result(ToolResult(content=[]))
    rendered = "\n".join(_stripped(component.render(40)))
    assert "RESULT" in rendered


def test_tool_execution_falls_back_when_call_renderer_raises():
    def boom(*_args: object):
        raise RuntimeError("boom")

    definition = SimpleNamespace(render_call=boom, render_result=None, render_shell=None)
    component = ToolExecutionComponent("mytool", "id", {}, tool_definition=definition, cwd="/tmp")
    assert "mytool" in "\n".join(_stripped(component.render(40)))


def test_tool_execution_falls_back_when_result_renderer_raises():
    def boom(*_args: object):
        raise RuntimeError("boom")

    definition = SimpleNamespace(render_call=lambda *_a: Text("CALL", 0, 0), render_result=boom, render_shell=None)
    component = ToolExecutionComponent("mytool", "id", {}, tool_definition=definition, cwd="/tmp")
    component.update_result(ToolResult(content=[{"type": "text", "text": "fallback text"}]))
    assert "fallback text" in "\n".join(_stripped(component.render(40)))


def test_tool_execution_self_render_shell_skips_the_box():
    definition = SimpleNamespace(render_call=lambda *_a: Text("SELF", 0, 0), render_result=None, render_shell="self")
    component = ToolExecutionComponent("x", "id", {}, tool_definition=definition, cwd="/tmp")
    lines = _stripped(component.render(40))
    assert lines[0] == ""
    assert "SELF" in "\n".join(lines)


def test_tool_execution_hides_itself_when_renderers_produce_nothing():
    definition = SimpleNamespace(render_call=lambda *_a: _EmptyComponent(), render_result=None, render_shell="self")
    component = ToolExecutionComponent("x", "id", {}, tool_definition=definition, cwd="/tmp")
    assert component.render(40) == []


class _EmptyComponent(Text):
    def __init__(self) -> None:
        super().__init__("", 0, 0)


def test_tool_execution_renderer_context_reflects_state():
    seen = []

    definition = SimpleNamespace(
        render_call=lambda args, theme, ctx: (seen.append(ctx), Text("C", 0, 0))[1],
        render_result=None,
        render_shell=None,
    )
    component = ToolExecutionComponent("x", "call-9", {"a": 1}, tool_definition=definition, cwd="/work")
    component.mark_execution_started()
    component.set_args_complete()
    component.set_expanded(True)

    last = seen[-1]
    assert last.tool_call_id == "call-9"
    assert last.cwd == "/work"
    assert last.execution_started is True
    assert last.args_complete is True
    assert last.expanded is True
    assert last.is_partial is True


def test_tool_execution_renderer_state_is_shared_across_renders():
    def render_call(_args, _theme, ctx):
        ctx.state["count"] = ctx.state.get("count", 0) + 1
        return Text(str(ctx.state["count"]), 0, 0)

    definition = SimpleNamespace(render_call=render_call, render_result=None, render_shell=None)
    component = ToolExecutionComponent("x", "id", {}, tool_definition=definition, cwd="/tmp")
    component.set_expanded(True)
    assert component.renderer_state["count"] >= 2


def test_tool_execution_update_args_refreshes_display():
    component = ToolExecutionComponent("x", "id", {"a": 1}, cwd="/tmp")
    component.update_args({"a": 2})
    assert '"a": 2' in "\n".join(_stripped(component.render(40)))


def test_tool_execution_image_width_is_clamped():
    component = ToolExecutionComponent("x", "id", {}, ToolExecutionOptions(image_width_cells=10), cwd="/tmp")
    assert component.image_width_cells == 10
    component.set_image_width_cells(-5)
    assert component.image_width_cells == 1


def test_tool_execution_show_images_toggle():
    component = ToolExecutionComponent("x", "id", {}, ToolExecutionOptions(show_images=False), cwd="/tmp")
    assert component.show_images is False
    component.set_show_images(True)
    assert component.show_images is True


def test_tool_execution_invalidate_rebuilds():
    component = ToolExecutionComponent("x", "id", {}, cwd="/tmp")
    component.invalidate()
    assert component.render(40) is not None


# --------------------------------------------------------------------------
# assistant message
# --------------------------------------------------------------------------


def _assistant(content, **kwargs) -> AssistantMessage:
    return AssistantMessage(
        api="openai-completions",
        provider="p",
        model="m",
        content=content,
        usage=Usage(),
        **kwargs,
    )


def test_assistant_message_renders_markdown_text():
    component = AssistantMessageComponent(_assistant([TextContent(text="**bold**")]))
    assert "bold" in "\n".join(_stripped(component.render(40)))


def test_assistant_message_adds_osc133_markers_without_tool_calls():
    lines = AssistantMessageComponent(_assistant([TextContent(text="hi")])).render(40)
    assert lines[0].startswith("\x1b]133;A\x07")


def test_assistant_message_omits_markers_when_tool_calls_present():
    message = _assistant([TextContent(text="hi"), ToolCall(id="1", name="t", arguments={})])
    lines = AssistantMessageComponent(message).render(40)
    assert not lines[0].startswith("\x1b]133;A\x07")


def test_assistant_message_renders_thinking_blocks():
    component = AssistantMessageComponent(_assistant([ThinkingContent(thinking="deep thought")]))
    assert "deep thought" in "\n".join(_stripped(component.render(40)))


def test_assistant_message_hides_thinking_behind_a_label():
    component = AssistantMessageComponent(
        _assistant([ThinkingContent(thinking="deep thought")]), hide_thinking_block=True
    )
    rendered = "\n".join(_stripped(component.render(40)))
    assert "deep thought" not in rendered
    assert "Thinking..." in rendered

    component.set_hidden_thinking_label("Pondering")
    assert "Pondering" in "\n".join(_stripped(component.render(40)))


def test_assistant_message_merges_consecutive_thinking_blocks():
    message = _assistant([ThinkingContent(thinking="one"), ThinkingContent(thinking="two")])
    rendered = "\n".join(_stripped(AssistantMessageComponent(message).render(40)))
    assert "one" in rendered
    assert "two" in rendered


def test_assistant_message_skips_blank_thinking_blocks():
    message = _assistant([ThinkingContent(thinking="   "), TextContent(text="body")])
    rendered = "\n".join(_stripped(AssistantMessageComponent(message).render(40)))
    assert "body" in rendered


def test_assistant_message_shows_truncation_notice():
    component = AssistantMessageComponent(_assistant([TextContent(text="x")], stop_reason="length"))
    assert "Response was truncated" in "\n".join(_stripped(component.render(60)))


def test_assistant_message_shows_abort_notice():
    component = AssistantMessageComponent(_assistant([TextContent(text="x")], stop_reason="aborted"))
    assert "Operation aborted" in "\n".join(_stripped(component.render(60)))


def test_assistant_message_prefers_a_specific_abort_message():
    message = _assistant([TextContent(text="x")], stop_reason="aborted", error_message="user cancelled")
    assert "user cancelled" in "\n".join(_stripped(AssistantMessageComponent(message).render(60)))


def test_assistant_message_generic_abort_message_is_replaced():
    message = _assistant([TextContent(text="x")], stop_reason="aborted", error_message="Request was aborted")
    assert "Operation aborted" in "\n".join(_stripped(AssistantMessageComponent(message).render(60)))


def test_assistant_message_shows_error_notice():
    message = _assistant([TextContent(text="x")], stop_reason="error", error_message="boom")
    assert "Error: boom" in "\n".join(_stripped(AssistantMessageComponent(message).render(60)))


def test_assistant_message_error_without_text_says_unknown():
    message = _assistant([TextContent(text="x")], stop_reason="error")
    assert "Error: Unknown error" in "\n".join(_stripped(AssistantMessageComponent(message).render(60)))


def test_assistant_message_errors_are_suppressed_when_tool_calls_present():
    message = _assistant([ToolCall(id="1", name="t", arguments={})], stop_reason="error", error_message="boom")
    assert "Error: boom" not in "\n".join(_stripped(AssistantMessageComponent(message).render(60)))


def test_assistant_message_truncation_notice_survives_tool_calls():
    message = _assistant([ToolCall(id="1", name="t", arguments={})], stop_reason="length")
    assert "Response was truncated" in "\n".join(_stripped(AssistantMessageComponent(message).render(60)))


def test_assistant_message_output_pad_and_invalidate():
    component = AssistantMessageComponent(_assistant([TextContent(text="hi")]), output_pad=0)
    narrow = _strip(component.render(20)[1])
    component.set_output_pad(4)
    assert _strip(component.render(20)[1]).index("hi") > narrow.index("hi")
    component.invalidate()
    assert "hi" in "\n".join(_stripped(component.render(20)))


def test_assistant_message_applies_transformers():
    component = AssistantMessageComponent(
        _assistant([TextContent(text="x")]), markdown_transformers=[lambda m, _c: m + " done"]
    )
    assert "done" in "\n".join(_stripped(component.render(40)))


def test_assistant_message_with_no_message_renders_nothing():
    assert AssistantMessageComponent().render(40) == []


# --------------------------------------------------------------------------
# bash execution
# --------------------------------------------------------------------------


def test_bash_execution_shows_the_command_and_a_running_loader():
    component = BashExecutionComponent("ls -la", None)
    rendered = "\n".join(_stripped(component.render(50)))
    assert "$ ls -la" in rendered
    assert "Running..." in rendered


def test_bash_execution_streams_output():
    component = BashExecutionComponent("cmd", None)
    component.append_output("one\ntwo\n")
    rendered = "\n".join(_stripped(component.render(50)))
    assert "one" in rendered
    assert "two" in rendered


def test_bash_execution_joins_partial_lines():
    component = BashExecutionComponent("cmd", None)
    component.append_output("par")
    component.append_output("tial\n")
    assert component.get_output() == "partial\n"


def test_bash_execution_strips_ansi_and_normalizes_newlines():
    component = BashExecutionComponent("cmd", None)
    component.append_output("\x1b[31mred\x1b[0m\r\nnext\rlast")
    assert component.get_output() == "red\nnext\nlast"


def test_bash_execution_reports_exit_code():
    component = BashExecutionComponent("cmd", None)
    component.set_complete(3, False)
    assert component.status == "error"
    assert "(exit 3)" in "\n".join(_stripped(component.render(50)))


def test_bash_execution_reports_cancellation():
    component = BashExecutionComponent("cmd", None)
    component.set_complete(None, True)
    assert component.status == "cancelled"
    assert "(cancelled)" in "\n".join(_stripped(component.render(50)))


def test_bash_execution_success_has_no_status_suffix():
    component = BashExecutionComponent("cmd", None)
    component.set_complete(0, False)
    assert component.status == "complete"
    rendered = "\n".join(_stripped(component.render(50)))
    assert "(exit" not in rendered
    assert "(cancelled)" not in rendered


def test_bash_execution_collapsed_output_reports_hidden_lines():
    component = BashExecutionComponent("cmd", None)
    component.append_output("\n".join(f"line{i}" for i in range(PREVIEW_LINES + 10)))
    component.set_complete(0, False)
    assert "more lines" in "\n".join(_stripped(component.render(60)))


def test_bash_execution_expanded_shows_everything():
    component = BashExecutionComponent("cmd", None)
    component.append_output("\n".join(f"line{i}" for i in range(PREVIEW_LINES + 5)))
    component.set_complete(0, False)
    component.set_expanded(True)
    rendered = "\n".join(_stripped(component.render(60)))
    assert "line0" in rendered
    assert "to collapse" in rendered


def test_bash_execution_shows_full_output_path_when_truncated():
    from pi_coding_agent.tools.truncate import TruncationResult

    component = BashExecutionComponent("cmd", None)
    component.append_output("x")
    component.set_complete(
        0,
        False,
        TruncationResult(
            content="x",
            truncated=True,
            truncated_by="lines",
            total_lines=1,
            total_bytes=1,
            output_lines=1,
            output_bytes=1,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=1,
            max_bytes=1,
        ),
        "/tmp/full.log",
    )
    assert "/tmp/full.log" in "\n".join(_stripped(component.render(80)))


def test_bash_execution_get_command():
    assert BashExecutionComponent("echo hi", None).get_command() == "echo hi"


def test_bash_execution_invalidate_rebuilds():
    component = BashExecutionComponent("cmd", None)
    component.append_output("out")
    component.invalidate()
    assert "out" in "\n".join(_stripped(component.render(50)))
