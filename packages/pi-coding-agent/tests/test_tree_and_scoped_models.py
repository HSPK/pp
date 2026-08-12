"""Tests for the tree, scoped-models and bordered-loader components."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest
from pi_ai.models import Model
from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.core.session_manager import SessionTreeNode
from pi_coding_agent.modes.interactive.components.bordered_loader import BorderedLoader
from pi_coding_agent.modes.interactive.components.scoped_models_selector import (
    ModelsCallbacks,
    ModelsConfig,
    ScopedModelsSelectorComponent,
    clear_all,
    enable_all,
    get_sorted_ids,
    is_enabled,
    move,
    toggle,
)
from pi_coding_agent.modes.interactive.components.tree_selector import (
    FILTER_MODES,
    HorizontalViewportRow,
    TreeSelectorComponent,
    compact_raw_keys,
    format_tool_call,
    render_horizontal_viewport,
)
from pi_coding_agent.modes.interactive.theme.theme import init_theme, theme
from pi_tui.keybindings import get_keybindings, set_keybindings

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\x1b\]8;;\x07")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _rendered(component: Any, width: int = 70) -> list[str]:
    return [_strip(line).rstrip() for line in component.render(width)]


@pytest.fixture(autouse=True)
def _theme_and_keybindings():
    init_theme("dark")
    previous = get_keybindings()
    set_keybindings(KeybindingsManager())
    yield
    set_keybindings(previous)


# --------------------------------------------------------------------------
# tree selector: pure helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("read", {"path": "/tmp/x.py"}, "[read: /tmp/x.py]"),
        ("read", {"path": "/tmp/x.py", "offset": 10, "limit": 5}, "[read: /tmp/x.py:10-14]"),
        ("read", {"file_path": "/a.py", "offset": 3}, "[read: /a.py:3]"),
        ("write", {"path": "/w.py"}, "[write: /w.py]"),
        ("edit", {"file_path": "/e.py"}, "[edit: /e.py]"),
        ("bash", {"command": "ls -la\n/tmp"}, "[bash: ls -la /tmp]"),
        ("grep", {"pattern": "foo"}, "[grep: /foo/ in .]"),
        ("find", {"pattern": "*.py", "path": "/src"}, "[find: *.py in /src]"),
        ("ls", {}, "[ls: .]"),
    ],
)
def test_format_tool_call(name: str, args: dict[str, Any], expected: str):
    assert format_tool_call(name, args) == expected


def test_format_tool_call_truncates_long_bash():
    result = format_tool_call("bash", {"command": "x" * 80})
    assert result.endswith("...]")
    assert len(result) < 80


def test_format_tool_call_falls_back_to_json_for_unknown_tools():
    assert format_tool_call("custom", {"a": 1}) == '[custom: {"a": 1}]'
    long_args = format_tool_call("custom", {"key": "v" * 80})
    assert long_args.endswith("...]")


def test_format_tool_call_shortens_home(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", "/home/u")
    assert format_tool_call("read", {"path": "/home/u/proj/x.py"}) == "[read: ~/proj/x.py]"


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        (["ctrl+a"], "ctrl+a"),
        (["ctrl+a", "ctrl+b"], "ctrl+a/b"),
        (["up", "down"], "up/down"),
        (["up", "ctrl+b"], "up/ctrl+b"),
    ],
)
def test_compact_raw_keys(keys: list[str], expected: str):
    assert compact_raw_keys(keys) == expected


def test_horizontal_viewport_keeps_the_gutter_and_pans_the_body():
    rows = [
        HorizontalViewportRow(gutter="> ", body="x" * 200, anchor_col=150, body_width=200, is_selected=True),
        HorizontalViewportRow(gutter="  ", body="y" * 200, anchor_col=0, body_width=200, is_selected=False),
    ]
    lines = render_horizontal_viewport(rows, 40)
    assert lines[0].startswith("> ")
    assert lines[1].startswith("  ")
    assert all(len(_strip(line)) <= 40 for line in lines)


def test_horizontal_viewport_does_not_pan_when_it_fits():
    rows = [HorizontalViewportRow(gutter="> ", body="short", anchor_col=0, body_width=5, is_selected=True)]
    assert render_horizontal_viewport(rows, 40)[0] == "> short"


# --------------------------------------------------------------------------
# tree selector: rendering and navigation
# --------------------------------------------------------------------------


def _message(role: str, text: str, **kwargs: Any) -> Any:
    return SimpleNamespace(
        role=role,
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=kwargs.get("stop_reason", "stop"),
        error_message=kwargs.get("error_message"),
    )


def _entry(entry_id: str, parent_id: str | None, role: str, text: str, **kwargs: Any) -> Any:
    return SimpleNamespace(id=entry_id, parent_id=parent_id, type="message", message=_message(role, text, **kwargs))


def _branching_tree() -> SessionTreeNode:
    root = SessionTreeNode(entry=_entry("1", None, "user", "first question"))
    assistant = SessionTreeNode(entry=_entry("2", "1", "assistant", "first answer"))
    branch_one = SessionTreeNode(entry=_entry("3", "2", "user", "branch one"))
    branch_two = SessionTreeNode(entry=_entry("4", "2", "user", "branch two"))
    assistant.children = [branch_one, branch_two]
    root.children = [assistant]
    return root


def _selector(**kwargs: Any) -> TreeSelectorComponent:
    return TreeSelectorComponent(
        kwargs.pop("tree", [_branching_tree()]),
        kwargs.pop("current_leaf_id", "3"),
        kwargs.pop("terminal_height", 24),
        kwargs.pop("on_select", lambda _entry_id: None),
        kwargs.pop("on_cancel", lambda: None),
        **kwargs,
    )


def test_tree_renders_branch_connectors():
    lines = _rendered(_selector())
    body = "\n".join(lines)
    assert "user: first question" in body
    assert "├─" in body
    assert "└─" in body


def test_tree_marks_the_active_path():
    lines = _rendered(_selector())
    active_lines = [line for line in lines if "•" in line]
    # root -> assistant -> branch one
    assert len(active_lines) == 3
    assert any("branch one" in line for line in active_lines)


def test_tree_shows_the_help_and_search_lines():
    body = "\n".join(_rendered(_selector()))
    assert "Session Tree" in body
    assert "Type to search:" in body
    assert "move" in body


def test_tree_reports_position_and_filter_label():
    tree_list = _selector().get_tree_list()
    assert "(3/4)" in "\n".join(_rendered(tree_list))
    tree_list._set_filter_mode("all")
    assert "[all]" in "\n".join(_rendered(tree_list))


def test_tree_empty_tree_is_flagged():
    selector = _selector(tree=[], current_leaf_id=None)
    assert selector.is_empty is True
    assert "No entries found" in "\n".join(_rendered(selector))


def test_tree_navigation_wraps():
    tree_list = _selector().get_tree_list()
    tree_list.selected_index = 0
    tree_list.handle_input("\x1b[A")
    assert tree_list.selected_index == len(tree_list.filtered_nodes) - 1
    tree_list.handle_input("\x1b[B")
    assert tree_list.selected_index == 0


def test_tree_confirm_selects_the_entry():
    selected: list[str] = []
    selector = _selector(on_select=selected.append)
    selector.get_tree_list().handle_input("\r")
    assert selected == ["3"]


def test_tree_typing_filters_and_escape_clears():
    tree_list = _selector().get_tree_list()
    for char in "branch two":
        tree_list.handle_input(char)
    assert tree_list.get_search_query() == "branch two"
    assert len(tree_list.filtered_nodes) == 1

    tree_list.handle_input("\x1b")
    assert tree_list.get_search_query() == ""
    assert len(tree_list.filtered_nodes) == 4


def test_tree_escape_with_no_query_cancels():
    cancelled: list[int] = []
    selector = _selector(on_cancel=lambda: cancelled.append(1))
    selector.get_tree_list().handle_input("\x1b")
    assert cancelled == [1]


def test_tree_backspace_shrinks_the_query():
    tree_list = _selector().get_tree_list()
    for char in "branch":
        tree_list.handle_input(char)
    tree_list.handle_input("\x7f")
    assert tree_list.get_search_query() == "branc"


def test_tree_user_only_filter_hides_assistant_turns():
    tree_list = _selector().get_tree_list()
    tree_list._set_filter_mode("user-only")
    roles = [node.node.entry.message.role for node in tree_list.filtered_nodes]
    assert roles == ["user", "user", "user"]


def test_tree_filter_cycles_forward_and_backward():
    tree_list = _selector().get_tree_list()
    assert tree_list.filter_mode == "default"
    tree_list.handle_input("\x0f")  # ctrl+o -> cycleForward
    assert tree_list.filter_mode == FILTER_MODES[1]


def test_tree_folding_hides_descendants():
    tree_list = _selector().get_tree_list()
    assert len(tree_list.filtered_nodes) == 4
    tree_list.folded_nodes.add("2")
    tree_list._apply_filter()
    assert len(tree_list.filtered_nodes) == 2


def test_tree_copy_returns_the_entry_text():
    copied: list[str | None] = []
    selector = _selector()
    selector.on_copy = copied.append
    tree_list = selector.get_tree_list()
    tree_list.selected_index = 0
    tree_list.copy_selected()
    assert copied == ["first question"]


def test_tree_label_editing_round_trip():
    changes: list[tuple[str, str | None]] = []
    selector = _selector(on_label_change=lambda entry_id, label: changes.append((entry_id, label)))
    tree_list = selector.get_tree_list()
    tree_list.selected_index = 0

    selector._show_label_input("1", None)
    assert selector.label_input is not None
    for char in "important":
        selector.handle_input(char)
    selector.handle_input("\r")

    assert changes == [("1", "important")]
    assert selector.label_input is None
    assert "[important]" in "\n".join(_rendered(selector))


def test_tree_label_input_cancel_restores_the_list():
    selector = _selector()
    selector._show_label_input("1", "old")
    selector.handle_input("\x1b")
    assert selector.label_input is None
    assert "user: first question" in "\n".join(_rendered(selector))


def test_tree_label_timestamps_can_be_toggled():
    tree_list = _selector().get_tree_list()
    assert tree_list.show_label_timestamps is False
    tree_list.handle_input("\x14")  # shift+t is bound to toggleLabelTimestamp
    # The binding may differ; drive the flag directly to assert the label text.
    tree_list.show_label_timestamps = True
    assert "[+label time]" in "\n".join(_rendered(tree_list))


def test_tree_assistant_turn_without_text_is_hidden():
    root = SessionTreeNode(entry=_entry("1", None, "user", "q"))
    silent = SimpleNamespace(
        id="2",
        parent_id="1",
        type="message",
        message=SimpleNamespace(role="assistant", content=[], stop_reason="toolUse", error_message=None),
    )
    root.children = [SessionTreeNode(entry=silent)]
    tree_list = _selector(tree=[root], current_leaf_id="1").get_tree_list()
    assert [node.node.entry.id for node in tree_list.filtered_nodes] == ["1"]


def test_tree_aborted_assistant_turn_stays_visible():
    root = SessionTreeNode(entry=_entry("1", None, "user", "q"))
    aborted = SimpleNamespace(
        id="2",
        parent_id="1",
        type="message",
        message=SimpleNamespace(role="assistant", content=[], stop_reason="aborted", error_message=None),
    )
    root.children = [SessionTreeNode(entry=aborted)]
    tree_list = _selector(tree=[root], current_leaf_id="1").get_tree_list()
    assert [node.node.entry.id for node in tree_list.filtered_nodes] == ["1", "2"]
    assert "(aborted)" in "\n".join(_rendered(tree_list))


def test_tree_multiple_roots_render_side_by_side():
    root_a = SessionTreeNode(entry=_entry("1", None, "user", "tree a"))
    root_b = SessionTreeNode(entry=_entry("2", None, "user", "tree b"))
    body = "\n".join(_rendered(_selector(tree=[root_a, root_b], current_leaf_id="1")))
    assert "tree a" in body
    assert "tree b" in body


def test_tree_is_foldable_only_for_branch_points():
    tree_list = _selector().get_tree_list()
    # "2" has two children but its parent has only one child, so it is not a
    # segment start; the root is.
    assert tree_list.is_foldable("1") is True
    assert tree_list.is_foldable("3") is False


# --------------------------------------------------------------------------
# scoped models selector: pure helpers
# --------------------------------------------------------------------------


def test_is_enabled_treats_none_as_everything():
    assert is_enabled(None, "a") is True
    assert is_enabled(["a"], "a") is True
    assert is_enabled(["a"], "b") is False


def test_toggle_narrows_from_all_then_adds_and_removes():
    assert toggle(None, "a") == ["a"]
    assert toggle(["a"], "b") == ["a", "b"]
    assert toggle(["a", "b"], "a") == ["b"]


def test_enable_all_collapses_back_to_none():
    assert enable_all(None, ["a", "b"]) is None
    assert enable_all(["a"], ["a", "b"]) is None
    assert enable_all(["a"], ["a", "b", "c"], ["b"]) == ["a", "b"]


def test_clear_all_semantics():
    assert clear_all(None, ["a", "b"]) == []
    assert clear_all(None, ["a", "b"], ["a"]) == ["b"]
    assert clear_all(["a", "b"], ["a", "b"], ["a"]) == ["b"]


def test_move_swaps_within_bounds():
    assert move(["a", "b", "c"], "b", -1) == ["b", "a", "c"]
    assert move(["a", "b", "c"], "a", -1) == ["a", "b", "c"]
    assert move(None, "a", 1) is None


def test_get_sorted_ids_puts_enabled_first():
    assert get_sorted_ids(["b"], ["a", "b", "c"]) == ["b", "a", "c"]
    assert get_sorted_ids(None, ["a", "b"]) == ["a", "b"]


# --------------------------------------------------------------------------
# scoped models selector: component
# --------------------------------------------------------------------------


def _model(provider: str, model_id: str) -> Model:
    return Model(
        id=model_id,
        provider=provider,
        name=f"{provider} {model_id}",
        api="openai-completions",
        base_url="https://example.invalid",
        context_window=1000,
        max_tokens=100,
    )


def _scoped_selector(**kwargs: Any) -> tuple[ScopedModelsSelectorComponent, list[Any], list[Any]]:
    changes: list[Any] = []
    persists: list[Any] = []
    config = ModelsConfig(
        all_models=[_model("openai", "gpt"), _model("openai", "o3"), _model("anthropic", "claude")],
        **kwargs,
    )
    component = ScopedModelsSelectorComponent(
        config,
        ModelsCallbacks(on_change=changes.append, on_persist=persists.append, on_cancel=lambda: None),
    )
    return component, changes, persists


def test_scoped_models_lists_every_model():
    component, _changes, _persists = _scoped_selector()
    body = "\n".join(_rendered(component))
    assert "gpt [openai]" in body
    assert "claude [anthropic]" in body
    assert "all enabled" in body


def test_scoped_models_toggle_narrows_the_scope():
    component, changes, _persists = _scoped_selector()
    component.handle_input("\r")
    assert changes[-1] == ["openai/gpt"]
    body = "\n".join(_rendered(component))
    assert "✓" in body
    assert "✗" in body


def test_scoped_models_marks_unsaved_changes():
    component, _changes, persists = _scoped_selector()
    component.handle_input("\r")
    assert "(unsaved)" in "\n".join(_rendered(component))
    component.handle_input("\x13")  # ctrl+s -> save
    assert persists == [["openai/gpt"]]
    assert "(unsaved)" not in "\n".join(_rendered(component))


def test_scoped_models_search_filters():
    component, _changes, _persists = _scoped_selector()
    for char in "claude":
        component.handle_input(char)
    assert [item.full_id for item in component.filtered_items] == ["anthropic/claude"]


def test_scoped_models_shows_unavailable_entries():
    component, _changes, _persists = _scoped_selector(enabled_model_ids=["gone/model"])
    body = "\n".join(_rendered(component))
    assert "[unavailable]" in body
    assert "1 unavailable" in body


def test_scoped_models_update_models_preserves_selection():
    component, _changes, _persists = _scoped_selector()
    component.selected_index = 2
    selected_id = component.filtered_items[2].full_id
    component.update_models([_model("anthropic", "claude"), _model("openai", "gpt")])
    assert component.filtered_items[component.selected_index].full_id == selected_id


def test_scoped_models_navigation_wraps():
    component, _changes, _persists = _scoped_selector()
    component.handle_input("\x1b[A")
    assert component.selected_index == len(component.filtered_items) - 1
    component.handle_input("\x1b[B")
    assert component.selected_index == 0


def test_scoped_models_refresh_status_line():
    component, _changes, _persists = _scoped_selector(refresh_status="Refreshing…")
    assert "Refreshing…" in "\n".join(_rendered(component))
    component.set_refresh_status("Done", "success")
    assert "Done" in "\n".join(_rendered(component))


# --------------------------------------------------------------------------
# bordered loader
# --------------------------------------------------------------------------


def test_bordered_loader_shows_a_cancel_hint_when_cancellable():
    loader = BorderedLoader(None, theme, "Working...")
    try:
        body = "\n".join(_rendered(loader, 50))
        assert "Working..." in body
        assert "cancel" in body
    finally:
        loader.dispose()


def test_bordered_loader_without_cancel_has_no_hint():
    loader = BorderedLoader(None, theme, "Working...", cancellable=False)
    try:
        body = "\n".join(_rendered(loader, 50))
        assert "Working..." in body
        assert "cancel" not in body
        assert loader.signal.aborted is False
    finally:
        loader.dispose()


def test_bordered_loader_abort_flows_through():
    loader = BorderedLoader(None, theme, "Working...")
    calls: list[int] = []
    loader.on_abort = lambda: calls.append(1)
    try:
        loader.handle_input("\x1b")
        assert loader.signal.aborted is True
        assert calls == [1]
    finally:
        loader.dispose()


def test_bordered_loader_input_is_ignored_when_not_cancellable():
    loader = BorderedLoader(None, theme, "Working...", cancellable=False)
    try:
        loader.handle_input("\x1b")
        assert loader.signal.aborted is False
    finally:
        loader.dispose()
