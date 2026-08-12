"""Tests for interactive-mode components.

Covers `dynamic_border`, `visual_truncate`, `keybinding_hints`,
`countdown_timer`, `markdown_transform`, `summary_messages`, `user_message`,
`custom_message` and `diff` under
`pi_coding_agent/modes/interactive/components/`, plus the `utils/word_diff.py`
port of jsdiff's `diffWords`.
"""

from __future__ import annotations

import asyncio
import re

import pytest
from pi_agent.harness.messages import CustomMessage
from pi_agent.harness.session.context import BranchSummaryMessage, CompactionSummaryMessage
from pi_coding_agent.core.agent_session import ParsedSkillBlock
from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.core.session_manager import CustomEntry
from pi_coding_agent.modes.interactive.components import diff as diff_module
from pi_coding_agent.modes.interactive.components.countdown_timer import CountdownTimer
from pi_coding_agent.modes.interactive.components.custom_message import (
    CustomEntryComponent,
    CustomMessageComponent,
)
from pi_coding_agent.modes.interactive.components.diff import (
    parse_diff_line,
    render_diff,
    render_intra_line_diff,
)
from pi_coding_agent.modes.interactive.components.dynamic_border import DynamicBorder
from pi_coding_agent.modes.interactive.components.keybinding_hints import (
    KeyTextFormatOptions,
    format_key_text,
    key_display_text,
    key_hint,
    key_text,
    raw_key_hint,
)
from pi_coding_agent.modes.interactive.components.markdown_transform import (
    MarkdownTransformContext,
    apply_markdown_transformers,
    create_markdown_transform,
)
from pi_coding_agent.modes.interactive.components.summary_messages import (
    BranchSummaryMessageComponent,
    CompactionSummaryMessageComponent,
    SkillInvocationMessageComponent,
)
from pi_coding_agent.modes.interactive.components.user_message import (
    OSC133_ZONE_END,
    OSC133_ZONE_FINAL,
    OSC133_ZONE_START,
    UserMessageComponent,
)
from pi_coding_agent.modes.interactive.components.visual_truncate import truncate_to_visual_lines
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.word_diff import diff_words, tokenize_words
from pi_tui.component import Component
from pi_tui.components.text import Text
from pi_tui.keybindings import get_keybindings, set_keybindings

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\x1b\]133;.\x07|\x1b\]8;;\x07")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


@pytest.fixture(autouse=True)
def _theme_and_keybindings():
    init_theme("dark")
    previous = get_keybindings()
    set_keybindings(KeybindingsManager())
    yield
    set_keybindings(previous)


# --------------------------------------------------------------------------
# dynamic_border / visual_truncate
# --------------------------------------------------------------------------


def test_dynamic_border_fills_the_width():
    assert DynamicBorder(lambda s: s).render(7) == ["─" * 7]


def test_dynamic_border_never_renders_empty():
    assert DynamicBorder(lambda s: s).render(0) == ["─"]
    assert DynamicBorder(lambda s: s).render(-5) == ["─"]


def test_dynamic_border_defaults_to_the_theme_color():
    border = DynamicBorder()
    border.invalidate()
    assert "─" in _strip(border.render(4)[0])


def test_visual_truncate_returns_empty_for_empty_text():
    result = truncate_to_visual_lines("", 5, 20)
    assert result.visual_lines == []
    assert result.skipped_count == 0


def test_visual_truncate_keeps_everything_when_short_enough():
    result = truncate_to_visual_lines("a\nb", 5, 20)
    assert [line.strip() for line in result.visual_lines] == ["a", "b"]
    assert result.skipped_count == 0


def test_visual_truncate_keeps_the_last_lines():
    result = truncate_to_visual_lines("a\nb\nc\nd", 2, 20)
    assert [line.strip() for line in result.visual_lines] == ["c", "d"]
    assert result.skipped_count == 2


def test_visual_truncate_counts_wrapped_lines_not_source_lines():
    # One source line that wraps into several visual lines.
    result = truncate_to_visual_lines("word " * 20, 3, 12)
    assert len(result.visual_lines) == 3
    assert result.skipped_count > 0


def test_visual_truncate_honours_padding():
    padded = truncate_to_visual_lines("hello", 5, 20, 1).visual_lines[0]
    unpadded = truncate_to_visual_lines("hello", 5, 20, 0).visual_lines[0]
    assert padded.startswith(" ")
    assert unpadded.startswith("hello")


# --------------------------------------------------------------------------
# keybinding_hints
# --------------------------------------------------------------------------


def test_format_key_text_preserves_separators():
    assert format_key_text("ctrl+a/alt+b") == "ctrl+a/alt+b"


def test_format_key_text_capitalizes_each_part():
    assert format_key_text("ctrl+a/alt+b", KeyTextFormatOptions(capitalize=True)) == "Ctrl+A/Alt+B"


def test_format_key_text_capitalize_keeps_rest_of_word():
    # JS only uppercases the first character; Python's `str.capitalize` would
    # also lowercase the rest.
    assert format_key_text("pageUP", KeyTextFormatOptions(capitalize=True)) == "PageUP"


def test_format_key_text_renames_alt_to_option_on_macos(monkeypatch: pytest.MonkeyPatch):
    import pi_coding_agent.modes.interactive.components.keybinding_hints as module

    monkeypatch.setattr(module.sys, "platform", "darwin")
    assert format_key_text("alt+x") == "option+x"
    assert format_key_text("ALT+x") == "option+x"
    monkeypatch.setattr(module.sys, "platform", "linux")
    assert format_key_text("alt+x") == "alt+x"


def test_key_text_reads_the_active_keybindings():
    assert key_text("app.interrupt") == "escape"
    assert key_display_text("app.interrupt") == "Escape"


def test_key_text_is_empty_for_unbound_actions():
    set_keybindings(KeybindingsManager({"app.interrupt": []}))
    assert key_text("app.interrupt") == ""


def test_key_text_joins_multiple_bindings():
    set_keybindings(KeybindingsManager({"app.interrupt": ["escape", "ctrl+c"]}))
    assert key_text("app.interrupt") == "escape/ctrl+c"


def test_key_hint_includes_key_and_description():
    assert _strip(key_hint("app.interrupt", "stop")) == "escape stop"


def test_raw_key_hint_formats_a_literal_key():
    assert _strip(raw_key_hint("ctrl+z", "undo")) == "ctrl+z undo"


# --------------------------------------------------------------------------
# countdown_timer
# --------------------------------------------------------------------------


def test_countdown_timer_reports_initial_seconds_immediately():
    ticks: list[int] = []
    timer = CountdownTimer(2500, None, ticks.append, lambda: None)
    try:
        # `Math.ceil(2500 / 1000)` is 3.
        assert ticks == [3]
        assert timer.remaining_seconds == 3
    finally:
        timer.dispose()


def test_countdown_timer_ticks_down_and_expires():
    async def scenario() -> tuple[list[int], list[None]]:
        ticks: list[int] = []
        expired: list[None] = []
        timer = CountdownTimer(1000, None, ticks.append, lambda: expired.append(None))
        try:
            timer._interval.cancel()  # replace the 1s timer with manual ticks
            timer._tick()
            return ticks, expired
        finally:
            timer.dispose()

    ticks, expired = asyncio.run(asyncio.wait_for(scenario(), timeout=5))
    assert ticks == [1, 0]
    assert expired == [None]


def test_countdown_timer_requests_render_on_tick():
    class FakeTui:
        def __init__(self) -> None:
            self.renders = 0

        def request_render(self) -> None:
            self.renders += 1

    tui = FakeTui()
    timer = CountdownTimer(5000, tui, lambda _s: None, lambda: None)
    try:
        assert tui.renders == 0
        timer._tick()
        assert tui.renders == 1
    finally:
        timer.dispose()


def test_countdown_timer_dispose_is_idempotent():
    timer = CountdownTimer(5000, None, lambda _s: None, lambda: None)
    timer.dispose()
    timer.dispose()
    assert timer._interval is None


# --------------------------------------------------------------------------
# markdown_transform
# --------------------------------------------------------------------------


def test_markdown_transformers_run_in_order():
    transform = create_markdown_transform("user", False, [lambda m, _c: m + "1", lambda m, _c: m + "2"])
    assert transform("x", 40) == "x12"


def test_markdown_transformer_exception_is_skipped():
    def boom(_markdown: str, _context: MarkdownTransformContext) -> str:
        raise RuntimeError("boom")

    transform = create_markdown_transform("assistant", True, [boom, lambda m, _c: m.upper()])
    assert transform("ok", 40) == "OK"


def test_markdown_transformer_non_string_result_is_ignored():
    transform = create_markdown_transform("user", False, [lambda _m, _c: None, lambda m, _c: m + "!"])
    assert transform("x", 40) == "x!"


def test_markdown_transform_context_is_passed_through():
    seen: list[MarkdownTransformContext] = []

    def record(markdown: str, context: MarkdownTransformContext) -> str:
        seen.append(context)
        return markdown

    create_markdown_transform("custom", True, [record])("body", 33)
    assert seen[0].message_type == "custom"
    assert seen[0].is_streaming is True
    assert seen[0].available_width == 33


def test_apply_markdown_transformers_without_transformers():
    context = MarkdownTransformContext("user", False, 10)
    assert apply_markdown_transformers("x", context, []) == "x"


# --------------------------------------------------------------------------
# summary messages
# --------------------------------------------------------------------------


def _rendered(component: Component, width: int = 60) -> str:
    return "\n".join(_strip(line) for line in component.render(width))


def test_branch_summary_collapsed_then_expanded():
    component = BranchSummaryMessageComponent(BranchSummaryMessage(summary="Body", from_id="x", timestamp=0))
    collapsed = _rendered(component)
    assert "[branch]" in collapsed
    assert "Branch summary (escape" not in collapsed
    assert "to expand)" in collapsed
    assert "Body" not in collapsed

    component.set_expanded(True)
    expanded = _rendered(component)
    assert "Branch Summary" in expanded
    assert "Body" in expanded


def test_compaction_summary_formats_token_count_with_separators():
    component = CompactionSummaryMessageComponent(
        CompactionSummaryMessage(summary="Body", tokens_before=1234567, timestamp=0)
    )
    assert "Compacted from 1,234,567 tokens" in _rendered(component)

    component.set_expanded(True)
    expanded = _rendered(component)
    assert "Compacted from 1,234,567 tokens" in expanded
    assert "Body" in expanded


def test_skill_invocation_collapsed_is_a_single_content_line():
    component = SkillInvocationMessageComponent(ParsedSkillBlock(name="my-skill", location="l", content="Body"))
    collapsed = _rendered(component)
    assert "[skill] my-skill" in collapsed
    assert "Body" not in collapsed

    component.set_expanded(True)
    expanded = _rendered(component)
    assert "[skill]" in expanded
    assert "my-skill" in expanded
    assert "Body" in expanded


def test_summary_components_rebuild_on_invalidate():
    component = BranchSummaryMessageComponent(BranchSummaryMessage(summary="Body", from_id="x", timestamp=0))
    component.set_expanded(True)
    component.invalidate()
    assert "Body" in _rendered(component)


def test_summary_expand_hint_follows_the_configured_keybinding():
    set_keybindings(KeybindingsManager({"app.tools.expand": "ctrl+x"}))
    component = SkillInvocationMessageComponent(ParsedSkillBlock(name="s", location="l", content="c"))
    assert "ctrl+x to expand" in _rendered(component)


# --------------------------------------------------------------------------
# user_message
# --------------------------------------------------------------------------


def test_user_message_wraps_output_in_osc133_zone_markers():
    lines = UserMessageComponent("hello").render(30)
    assert lines[0].startswith(OSC133_ZONE_START)
    assert lines[-1].startswith(OSC133_ZONE_END + OSC133_ZONE_FINAL)
    assert "hello" in _strip("\n".join(lines))


def test_user_message_renders_markdown():
    assert "bold" in _strip("\n".join(UserMessageComponent("**bold**").render(30)))


def test_user_message_applies_transformers():
    component = UserMessageComponent("x", markdown_transformers=[lambda m, _c: m + " transformed"])
    assert "transformed" in _strip("\n".join(component.render(40)))


def test_user_message_output_pad_changes_indentation():
    narrow = _strip(UserMessageComponent("hi", output_pad=0).render(20)[1])
    wide = _strip(UserMessageComponent("hi", output_pad=4).render(20)[1])
    assert wide.index("hi") > narrow.index("hi")

    component = UserMessageComponent("hi", output_pad=0)
    component.set_output_pad(4)
    assert _strip(component.render(20)[1]).index("hi") == wide.index("hi")


def test_user_message_empty_content_renders_nothing_special():
    # No lines means no zone markers to attach.
    assert UserMessageComponent("").render(20) is not None


# --------------------------------------------------------------------------
# custom_message / custom_entry
# --------------------------------------------------------------------------


def _custom_message(content: object = "body") -> CustomMessage:
    return CustomMessage(custom_type="note", content=content, display=True, timestamp=0)


def test_custom_message_default_rendering_shows_label_and_body():
    rendered = _rendered(CustomMessageComponent(_custom_message()), 40)
    assert "[note]" in rendered
    assert "body" in rendered


def test_custom_message_joins_text_content_parts():
    from pi_ai.types import TextContent

    message = _custom_message([TextContent(text="one"), TextContent(text="two")])
    rendered = _rendered(CustomMessageComponent(message), 40)
    assert "one" in rendered
    assert "two" in rendered


def test_custom_message_uses_a_custom_renderer():
    component = CustomMessageComponent(_custom_message(), lambda *_args: Text("CUSTOM", 0, 0))
    rendered = _rendered(component, 40)
    assert "CUSTOM" in rendered
    assert "[note]" not in rendered


def test_custom_message_falls_back_when_renderer_raises():
    def boom(*_args: object) -> Component:
        raise RuntimeError("boom")

    rendered = _rendered(CustomMessageComponent(_custom_message(), boom), 40)
    assert "[note]" in rendered


def test_custom_message_falls_back_when_renderer_returns_none():
    rendered = _rendered(CustomMessageComponent(_custom_message(), lambda *_a: None), 40)
    assert "[note]" in rendered


def test_custom_message_renderer_receives_expanded_and_pad():
    seen: list[object] = []

    def renderer(_message: CustomMessage, context: object, _theme: object) -> Component:
        seen.append(context)
        return Text("X", 0, 0)

    component = CustomMessageComponent(_custom_message(), renderer, output_pad=2)
    component.set_expanded(True)
    component.set_expanded(True)  # no-op, must not rebuild
    component.set_output_pad(3)
    component.set_output_pad(3)  # no-op

    assert [(c.expanded, c.output_pad) for c in seen] == [  # type: ignore[attr-defined]
        (False, 2),
        (True, 2),
        (True, 3),
    ]


def test_custom_entry_renders_the_extension_component():
    entry = CustomEntry(id="1", parent_id=None, timestamp="t", custom_type="x")
    component = CustomEntryComponent(entry, lambda *_a: Text("ENTRY", 0, 0))
    assert component.has_content() is True
    assert "ENTRY" in _rendered(component, 40)


def test_custom_entry_without_content_renders_nothing():
    entry = CustomEntry(id="1", parent_id=None, timestamp="t", custom_type="x")
    component = CustomEntryComponent(entry, lambda *_a: None)
    assert component.has_content() is False
    assert component.render(40) == []


def test_custom_entry_shows_renderer_failures_inline():
    def boom(*_args: object) -> Component:
        raise RuntimeError("kaboom")

    entry = CustomEntry(id="1", parent_id=None, timestamp="t", custom_type="broken")
    rendered = _rendered(CustomEntryComponent(entry, boom), 60)
    assert "[broken] renderer failed: kaboom" in rendered


def test_custom_entry_rebuilds_on_expand_and_invalidate():
    seen: list[bool] = []

    def renderer(_entry: CustomEntry, context: object, _theme: object) -> Component:
        seen.append(context.expanded)  # type: ignore[attr-defined]
        return Text("E", 0, 0)

    entry = CustomEntry(id="1", parent_id=None, timestamp="t", custom_type="x")
    component = CustomEntryComponent(entry, renderer)
    component.set_expanded(True)
    component.set_expanded(True)
    component.invalidate()
    assert seen == [False, True, True]


# --------------------------------------------------------------------------
# word_diff (ported jsdiff `diffWords`)
# --------------------------------------------------------------------------


def test_tokenize_carries_whitespace_on_both_sides():
    # jsdiff keeps the whitespace both as a suffix of the previous token and a
    # prefix of the next; `_join_tokens` strips the duplicate when rejoining.
    assert tokenize_words("foo bar") == ["foo ", " bar"]


def test_tokenize_gives_whitespace_only_text_a_single_token():
    assert tokenize_words("   ") == ["   "]


def test_tokenize_splits_punctuation():
    assert tokenize_words("a(b)") == ["a", "(", "b", ")"]


def test_diff_words_marks_the_changed_word_only():
    changes = diff_words("foo bar baz", "foo qux baz")
    assert [(c.value, c.added, c.removed) for c in changes] == [
        ("foo ", False, False),
        ("bar", False, True),
        ("qux", True, False),
        (" baz", False, False),
    ]


def test_diff_words_of_identical_text_is_a_single_keep():
    changes = diff_words("same text", "same text")
    assert len(changes) == 1
    assert changes[0].added is False
    assert changes[0].removed is False


def test_diff_words_of_empty_inputs():
    assert diff_words("", "") == []
    assert [(c.value, c.removed) for c in diff_words("x", "")] == [("x", True)]
    assert [(c.value, c.added) for c in diff_words("", "x")] == [("x", True)]


def test_diff_words_dedupes_whitespace_around_a_deletion():
    changes = diff_words("foo bar baz", "foo baz")
    assert [(c.value, c.added, c.removed) for c in changes] == [
        ("foo ", False, False),
        ("bar ", False, True),
        ("baz", False, False),
    ]


def test_diff_words_reconstructs_the_new_text():
    old, new = "  const value = compute(a, b);", "  const result = compute(a, c);"
    assert "".join(c.value for c in diff_words(old, new) if not c.removed) == new
    assert "".join(c.value for c in diff_words(old, new) if not c.added) == old


# --------------------------------------------------------------------------
# diff rendering
# --------------------------------------------------------------------------


def test_parse_diff_line_extracts_prefix_number_and_content():
    parsed = parse_diff_line("+ 12 content here")
    assert parsed is not None
    assert (parsed.prefix, parsed.line_num, parsed.content) == ("+", " 12", "content here")


def test_parse_diff_line_rejects_non_diff_lines():
    assert parse_diff_line("@@ hunk @@") is None
    assert parse_diff_line("") is None


def test_render_diff_colors_each_line_kind():
    rendered = render_diff("  1 ctx\n-  2 old\n+  2 new")
    assert "toolDiffContext" in rendered or "\x1b[" in rendered
    assert len(rendered.split("\n")) == 3


def test_render_diff_passes_through_unparseable_lines():
    assert _strip(render_diff("@@ hunk @@")) == "@@ hunk @@"


def test_render_diff_expands_tabs():
    assert "\t" not in _strip(render_diff("  1 a\tb"))
    assert "a   b" in _strip(render_diff("  1 a\tb"))


def test_render_diff_shows_blocks_separately_when_not_one_to_one():
    rendered = _strip(render_diff("-  1 a\n-  2 b\n+  3 c"))
    # The captured line-number group keeps its original leading spaces.
    assert rendered.split("\n") == ["-  1 a", "-  2 b", "+  3 c"]


def test_render_intra_line_diff_inverts_only_changed_tokens(monkeypatch: pytest.MonkeyPatch):
    class FakeTheme:
        def fg(self, name: str, text: str) -> str:
            return text

        def inverse(self, text: str) -> str:
            return f"[{text}]"

    monkeypatch.setattr(diff_module, "theme", FakeTheme())
    removed, added = render_intra_line_diff("  foo bar baz", "  foo qux baz")
    assert removed == "  foo [bar] baz"
    assert added == "  foo [qux] baz"


def test_render_intra_line_diff_does_not_highlight_indentation(monkeypatch: pytest.MonkeyPatch):
    class FakeTheme:
        def fg(self, name: str, text: str) -> str:
            return text

        def inverse(self, text: str) -> str:
            return f"[{text}]"

    monkeypatch.setattr(diff_module, "theme", FakeTheme())
    removed, added = render_intra_line_diff("    old", "    new")
    assert removed.startswith("    [")
    assert added.startswith("    [")


def test_render_diff_of_empty_input_is_empty():
    assert _strip(render_diff("")) == ""
