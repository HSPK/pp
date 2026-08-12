"""Tests ported from packages/tui/test/settings-list.test.ts."""

from pi_tui.component import Component
from pi_tui.components.settings_list import SettingItem, SettingsList, SettingsListOptions, SettingsListTheme

_test_theme = SettingsListTheme(
    label=lambda text, selected: text,
    value=lambda text, selected: text,
    description=lambda text: text,
    cursor="> ",
    hint=lambda text: text,
)


def _make_items() -> list[SettingItem]:
    return [
        SettingItem(
            id="tui-mode",
            label="TUI mode",
            current_value="regular",
            values=["regular", "fullscreen"],
        )
    ]


class TestSettingsList:
    def test_includes_spaces_in_active_search_instead_of_changing_setting(self):
        changes: list[tuple[str, str]] = []
        list_ = SettingsList(
            _make_items(),
            10,
            _test_theme,
            lambda id_, value: changes.append((id_, value)),
            lambda: None,
            SettingsListOptions(enable_search=True),
        )

        for character in "TUI mode":
            list_.handle_input(character)

        assert changes == []
        assert "TUI mode" in list_.render(80)[0]

        list_.handle_input("\r")
        assert changes == [("tui-mode", "fullscreen")]

    def test_keeps_space_as_change_shortcut_before_search_query_entered(self):
        changes: list[tuple[str, str]] = []
        list_ = SettingsList(
            _make_items(),
            10,
            _test_theme,
            lambda id_, value: changes.append((id_, value)),
            lambda: None,
            SettingsListOptions(enable_search=True),
        )

        list_.handle_input(" ")

        assert changes == [("tui-mode", "fullscreen")]


class TestSettingsListExtra:
    """Extra tests covering branches not exercised by the upstream TS suite."""

    def test_navigates_up_and_down_wrapping(self):
        items = [
            SettingItem(id="a", label="A", current_value="1"),
            SettingItem(id="b", label="B", current_value="2"),
            SettingItem(id="c", label="C", current_value="3"),
        ]
        list_ = SettingsList(items, 10, _test_theme, lambda i, v: None, lambda: None)

        assert list_.selected_index == 0
        list_.handle_input("\x1b[A")  # Up wraps to last
        assert list_.selected_index == 2
        list_.handle_input("\x1b[B")  # Down wraps to first
        assert list_.selected_index == 0
        list_.handle_input("\x1b[B")
        assert list_.selected_index == 1

    def test_cancel_invokes_on_cancel(self):
        cancelled = []
        list_ = SettingsList(_make_items(), 10, _test_theme, lambda i, v: None, lambda: cancelled.append(True))
        list_.handle_input("\x1b")
        assert cancelled == [True]

    def test_cycles_through_values_with_enter(self):
        changes: list[tuple[str, str]] = []
        list_ = SettingsList(_make_items(), 10, _test_theme, lambda i, v: changes.append((i, v)), lambda: None)
        list_.handle_input("\r")
        assert changes == [("tui-mode", "fullscreen")]
        list_.handle_input("\r")
        assert changes == [("tui-mode", "fullscreen"), ("tui-mode", "regular")]

    def test_no_settings_available_renders_hint(self):
        list_ = SettingsList([], 10, _test_theme, lambda i, v: None, lambda: None)
        lines = list_.render(80)
        assert "No settings available" in lines[0]

    def test_no_matching_settings_renders_hint_when_search_filters_all(self):
        list_ = SettingsList(
            _make_items(),
            10,
            _test_theme,
            lambda i, v: None,
            lambda: None,
            SettingsListOptions(enable_search=True),
        )
        for ch in "zzz_no_match":
            list_.handle_input(ch)
        lines = list_.render(80)
        assert any("No matching settings" in line for line in lines)

    def test_submenu_replaces_render_and_close_restores_selection(self):
        opened = {}

        class _FakeSubmenu(Component):
            # `SettingItem.submenu` is declared to return a `Component`, so the
            # double subclasses it rather than duck-typing the two methods the
            # settings list happens to call today.
            def __init__(self, done):
                self._done = done

            def render(self, width: int) -> list[str]:
                return ["submenu line"]

            def invalidate(self) -> None:
                pass

            def handle_input(self, data: str) -> None:
                if data == "\r":
                    self._done("fullscreen")

        def make_submenu(current_value, done):
            opened["current_value"] = current_value
            return _FakeSubmenu(done)

        items = [
            SettingItem(id="a", label="A", current_value="regular", submenu=make_submenu),
            SettingItem(id="b", label="B", current_value="x"),
        ]
        changes: list[tuple[str, str]] = []
        list_ = SettingsList(items, 10, _test_theme, lambda i, v: changes.append((i, v)), lambda: None)

        # Move to second item then back, then activate submenu on first item.
        list_.selected_index = 0
        list_.handle_input("\r")
        assert opened["current_value"] == "regular"
        assert list_.render(80) == ["submenu line"]

        # Submenu handles input directly, invoking done() which closes it.
        list_.handle_input("\r")
        assert changes == [("a", "fullscreen")]
        assert items[0].current_value == "fullscreen"
        # Selection restored to the item that opened the submenu.
        assert list_.selected_index == 0
        assert list_.render(80) != ["submenu line"]

    def test_update_value_changes_matching_item(self):
        items = _make_items()
        list_ = SettingsList(items, 10, _test_theme, lambda i, v: None, lambda: None)
        list_.update_value("tui-mode", "fullscreen")
        assert items[0].current_value == "fullscreen"
        list_.update_value("nonexistent", "x")
        assert items[0].current_value == "fullscreen"

    def test_render_shows_description_for_selected_item(self):
        items = [
            SettingItem(id="a", label="A", current_value="1", description="A helpful description"),
        ]
        list_ = SettingsList(items, 10, _test_theme, lambda i, v: None, lambda: None)
        lines = list_.render(80)
        assert any("A helpful description" in line for line in lines)

    def test_render_scroll_indicator_when_more_items_than_max_visible(self):
        items = [SettingItem(id=str(i), label=f"item{i}", current_value="v") for i in range(10)]
        list_ = SettingsList(items, 3, _test_theme, lambda i, v: None, lambda: None)
        lines = list_.render(80)
        assert any("/10)" in line for line in lines)
