"""Tests ported from packages/tui/test/select-list.test.ts."""

from pi_tui.components.select_list import SelectItem, SelectList, SelectListLayoutOptions, SelectListTheme
from pi_tui.utils import visible_width


def _test_theme() -> SelectListTheme:
    return SelectListTheme(
        selected_prefix=lambda text: text,
        selected_text=lambda text: text,
        description=lambda text: text,
        scroll_info=lambda text: text,
        no_match=lambda text: text,
    )


def _visible_index_of(line: str, text: str) -> int:
    index = line.find(text)
    assert index != -1
    return visible_width(line[:index])


class TestSelectList:
    def test_normalizes_multiline_descriptions_to_single_line(self):
        items = [
            SelectItem(
                value="test",
                label="test",
                description="Line one\nLine two\nLine three",
            ),
        ]

        select_list = SelectList(items, 5, _test_theme())
        rendered = select_list.render(100)

        assert len(rendered) > 0
        assert "\n" not in rendered[0]
        assert "Line one Line two Line three" in rendered[0]

    def test_keeps_descriptions_aligned_when_primary_text_truncated(self):
        items = [
            SelectItem(value="short", label="short", description="short description"),
            SelectItem(
                value="very-long-command-name-that-needs-truncation",
                label="very-long-command-name-that-needs-truncation",
                description="long description",
            ),
        ]

        select_list = SelectList(items, 5, _test_theme())
        rendered = select_list.render(80)

        assert _visible_index_of(rendered[0], "short description") == _visible_index_of(rendered[1], "long description")

    def test_uses_configured_minimum_primary_column_width(self):
        items = [
            SelectItem(value="a", label="a", description="first"),
            SelectItem(value="bb", label="bb", description="second"),
        ]

        select_list = SelectList(
            items,
            5,
            _test_theme(),
            SelectListLayoutOptions(min_primary_column_width=12, max_primary_column_width=20),
        )
        rendered = select_list.render(80)

        assert rendered[0].index("first") == 14
        assert rendered[1].index("second") == 14

    def test_uses_configured_maximum_primary_column_width(self):
        items = [
            SelectItem(
                value="very-long-command-name-that-needs-truncation",
                label="very-long-command-name-that-needs-truncation",
                description="first",
            ),
            SelectItem(value="short", label="short", description="second"),
        ]

        select_list = SelectList(
            items,
            5,
            _test_theme(),
            SelectListLayoutOptions(min_primary_column_width=12, max_primary_column_width=20),
        )
        rendered = select_list.render(80)

        assert _visible_index_of(rendered[0], "first") == 22
        assert _visible_index_of(rendered[1], "second") == 22

    def test_allows_overriding_primary_truncation_while_preserving_alignment(self):
        items = [
            SelectItem(
                value="very-long-command-name-that-needs-truncation",
                label="very-long-command-name-that-needs-truncation",
                description="first",
            ),
            SelectItem(value="short", label="short", description="second"),
        ]

        def truncate_primary(ctx):
            if len(ctx.text) <= ctx.max_width:
                return ctx.text
            return ctx.text[: max(0, ctx.max_width - 1)] + "…"

        select_list = SelectList(
            items,
            5,
            _test_theme(),
            SelectListLayoutOptions(
                min_primary_column_width=12,
                max_primary_column_width=12,
                truncate_primary=truncate_primary,
            ),
        )
        rendered = select_list.render(80)

        assert "…" in rendered[0]
        assert _visible_index_of(rendered[0], "first") == _visible_index_of(rendered[1], "second")


class TestSelectListExtra:
    def test_no_match_message_when_filter_matches_nothing(self):
        items = [SelectItem(value="apple", label="apple")]
        select_list = SelectList(items, 5, _test_theme())
        select_list.set_filter("zzz")

        rendered = select_list.render(80)
        assert rendered == ["  No matching commands"]

    def test_set_filter_matches_prefix_case_insensitively(self):
        items = [
            SelectItem(value="Apple", label="Apple"),
            SelectItem(value="banana", label="banana"),
            SelectItem(value="apricot", label="apricot"),
        ]
        select_list = SelectList(items, 5, _test_theme())
        select_list.set_filter("ap")

        rendered = select_list.render(80)
        assert len(rendered) == 2

    def test_handle_input_up_wraps_to_bottom(self):
        items = [SelectItem(value=str(i), label=str(i)) for i in range(3)]
        select_list = SelectList(items, 5, _test_theme())
        select_list.set_selected_index(0)

        select_list.handle_input("\x1b[A")  # up arrow
        assert select_list.get_selected_item().value == "2"

    def test_handle_input_down_wraps_to_top(self):
        items = [SelectItem(value=str(i), label=str(i)) for i in range(3)]
        select_list = SelectList(items, 5, _test_theme())
        select_list.set_selected_index(2)

        select_list.handle_input("\x1b[B")  # down arrow
        assert select_list.get_selected_item().value == "0"

    def test_handle_input_enter_triggers_on_select(self):
        items = [SelectItem(value="a", label="a")]
        select_list = SelectList(items, 5, _test_theme())

        selected = []
        select_list.on_select = selected.append
        select_list.handle_input("\r")

        assert selected == [items[0]]

    def test_handle_input_escape_triggers_on_cancel(self):
        items = [SelectItem(value="a", label="a")]
        select_list = SelectList(items, 5, _test_theme())

        cancelled = []
        select_list.on_cancel = lambda: cancelled.append(True)
        select_list.handle_input("\x1b")

        assert cancelled == [True]

    def test_scroll_indicator_shown_when_more_items_than_max_visible(self):
        items = [SelectItem(value=str(i), label=str(i)) for i in range(10)]
        select_list = SelectList(items, 3, _test_theme())

        rendered = select_list.render(80)
        assert len(rendered) == 4  # 3 items + scroll indicator
        assert "1/10" in rendered[-1]
