"""Python port of `packages/coding-agent/test/tree-selector.test.ts`."""

from __future__ import annotations

import time
from datetime import datetime

import pytest
from pi_ai.types import AssistantMessage, Cost, TextContent, ToolCall, Usage, UserMessage
from pi_tui.keybindings import set_keybindings
from pi_tui.utils import visible_width

from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.core.session_manager import (
    ModelChangeEntry,
    SessionEntry,
    SessionMessageEntry,
    SessionTreeNode,
    ThinkingLevelChangeEntry,
)
from pi_coding_agent.modes.interactive.components.tree_selector import TreeSelectorComponent
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi

# Key escape sequences
UP = "\x1b[A"
DOWN = "\x1b[B"
CTRL_LEFT = "\x1b[1;5D"
CTRL_RIGHT = "\x1b[1;5C"
ALT_LEFT = "\x1b[1;3D"
ALT_RIGHT = "\x1b[1;3C"
CTRL_U = "\x15"
CTRL_D = "\x04"
CTRL_L = "\x0c"
CTRL_X = "\x18"


@pytest.fixture(autouse=True)
def _theme_and_keybindings() -> None:
    init_theme("dark")
    # Ensure test isolation: keybindings are a global singleton.
    set_keybindings(KeybindingsManager.create())


def _now_iso() -> str:
    return datetime.now().isoformat()


def _usage() -> Usage:
    return Usage(input=0, output=0, cache_read=0, cache_write=0, total_tokens=0, cost=Cost())


def user_message(entry_id: str, parent_id: str | None, content: str) -> SessionMessageEntry:
    return SessionMessageEntry(
        id=entry_id,
        parent_id=parent_id,
        timestamp=_now_iso(),
        message=UserMessage(content=content, timestamp=int(time.time() * 1000)),
    )


def assistant_message(entry_id: str, parent_id: str | None, text: str) -> SessionMessageEntry:
    return SessionMessageEntry(
        id=entry_id,
        parent_id=parent_id,
        timestamp=_now_iso(),
        message=AssistantMessage(
            role="assistant",
            content=[TextContent(text=text)],
            api="anthropic-messages",
            provider="anthropic",
            model="claude-sonnet-4",
            usage=_usage(),
            stop_reason="stop",
            timestamp=int(time.time() * 1000),
        ),
    )


def tool_call_only_assistant(entry_id: str, parent_id: str | None) -> SessionMessageEntry:
    return SessionMessageEntry(
        id=entry_id,
        parent_id=parent_id,
        timestamp=_now_iso(),
        message=AssistantMessage(
            role="assistant",
            content=[ToolCall(id=f"tc-{entry_id}", name="read", arguments={"path": "test.ts"})],
            api="anthropic-messages",
            provider="anthropic",
            model="claude-sonnet-4",
            usage=_usage(),
            stop_reason="toolUse",
            timestamp=int(time.time() * 1000),
        ),
    )


def model_change(entry_id: str, parent_id: str | None) -> ModelChangeEntry:
    return ModelChangeEntry(
        id=entry_id,
        parent_id=parent_id,
        timestamp=_now_iso(),
        provider="anthropic",
        model_id="claude-sonnet-4",
    )


def build_tree(entries: list[SessionEntry]) -> list[SessionTreeNode]:
    if not entries:
        return []

    nodes = [SessionTreeNode(entry=entry, children=[]) for entry in entries]
    by_id = {node.entry.id: node for node in nodes}

    roots: list[SessionTreeNode] = []
    for node in nodes:
        if node.entry.parent_id is None:
            roots.append(node)
        else:
            parent = by_id.get(node.entry.parent_id)
            if parent is not None:
                parent.children.append(node)
    return roots


def _selector(tree: list[SessionTreeNode], current_leaf_id: str | None) -> TreeSelectorComponent:
    return TreeSelectorComponent(
        tree,
        current_leaf_id,
        24,
        lambda _entry_id: None,
        lambda: None,
    )


def _selected_id(selector: TreeSelectorComponent) -> str | None:
    node = selector.get_tree_list().get_selected_node()
    return node.entry.id if node is not None else None


class TestInitialSelectionWithMetadataEntries:
    def test_focuses_nearest_visible_ancestor_for_a_model_change_leaf(self) -> None:
        entries: list[SessionEntry] = [
            user_message("user-1", None, "hello"),
            assistant_message("asst-1", "user-1", "hi"),
            user_message("user-2", "asst-1", "active branch"),
            model_change("model-1", "user-2"),
            user_message("user-3", "asst-1", "sibling branch"),
        ]
        selector = _selector(build_tree(entries), "model-1")

        # Should focus on user-2 (parent of model-1), not user-3 (last item).
        assert _selected_id(selector) == "user-2"

    def test_focuses_nearest_visible_ancestor_for_a_thinking_level_change_leaf(self) -> None:
        entries: list[SessionEntry] = [
            user_message("user-1", None, "hello"),
            assistant_message("asst-1", "user-1", "hi"),
            user_message("user-2", "asst-1", "active branch"),
            ThinkingLevelChangeEntry(id="thinking-1", parent_id="user-2", timestamp=_now_iso(), thinking_level="high"),
            user_message("user-3", "asst-1", "sibling branch"),
        ]
        selector = _selector(build_tree(entries), "thinking-1")

        assert _selected_id(selector) == "user-2"


class TestFilterSwitchingWithParentTraversal:
    @staticmethod
    def _entries() -> list[SessionEntry]:
        return [
            user_message("user-1", None, "hello"),
            assistant_message("asst-1", "user-1", "hi"),
            user_message("user-2", "asst-1", "active branch"),
            assistant_message("asst-2", "user-2", "response"),
            user_message("user-3", "asst-1", "sibling branch"),
        ]

    def test_switches_to_nearest_visible_user_message_for_the_user_only_filter(self) -> None:
        selector = _selector(build_tree(self._entries()), "asst-2")
        assert _selected_id(selector) == "asst-2"

        selector.handle_input(CTRL_U)

        # Should now be on user-2 (the parent user message), not user-3.
        assert _selected_id(selector) == "user-2"

    def test_returns_to_nearest_visible_ancestor_when_switching_back_to_default(self) -> None:
        selector = _selector(build_tree(self._entries()), "asst-2")
        assert _selected_id(selector) == "asst-2"

        selector.handle_input(CTRL_U)
        assert _selected_id(selector) == "user-2"

        selector.handle_input(CTRL_D)
        assert _selected_id(selector) == "user-2"


class TestHelp:
    def test_renders_semantic_help_rows_without_truncating_narrow_terminal_controls(self) -> None:
        tree = build_tree([user_message("user-1", None, "hello"), assistant_message("asst-1", "user-1", "hi")])
        selector = _selector(tree, "asst-1")

        plain_lines = [strip_ansi(line) for line in selector.render(30)]
        plain = "\n".join(plain_lines)
        assert "branch" in plain
        assert "copy" in plain
        assert "filters" in plain
        assert "cycle" in plain
        assert "label time" in plain
        assert "..." not in plain
        assert all(visible_width(line) <= 30 for line in plain_lines)


class TestCopy:
    def test_copies_the_full_selected_message_with_ctrl_x(self) -> None:
        message = "long message " * 30 + "\nsecond line"
        tree = build_tree([user_message("user-1", None, "hello"), assistant_message("asst-1", "user-1", message)])
        selector = _selector(tree, "asst-1")
        copied: list[str | None] = []
        selector.on_copy = copied.append

        selector.handle_input(CTRL_X)

        assert copied == [message]


class TestLabelTimestamps:
    def test_toggles_label_timestamps_for_labeled_nodes(self) -> None:
        tree = build_tree([user_message("user-1", None, "hello"), assistant_message("asst-1", "user-1", "hi")])
        tree[0].label = "checkpoint"
        tree[0].label_timestamp = datetime(2026, 3, 28, 14, 32, 0).isoformat()

        selector = _selector(tree, "asst-1")
        tree_list = selector.get_tree_list()

        render = "\n".join(tree_list.render(200))
        assert "[checkpoint]" in render
        assert "3/28 14:32" not in render
        assert "[+label time]" not in render

        selector.handle_input("T")

        render = "\n".join(tree_list.render(200))
        assert "3/28 14:32" in render
        assert "[+label time]" in render


class TestEmptyFilterPreservation:
    def test_preserves_selection_when_switching_to_an_empty_labeled_filter_and_back(self) -> None:
        entries: list[SessionEntry] = [
            user_message("user-1", None, "hello"),
            assistant_message("asst-1", "user-1", "hi"),
            user_message("user-2", "asst-1", "bye"),
            assistant_message("asst-2", "user-2", "goodbye"),
        ]
        selector = _selector(build_tree(entries), "asst-2")
        assert _selected_id(selector) == "asst-2"

        selector.handle_input(CTRL_L)
        assert selector.get_tree_list().get_selected_node() is None

        selector.handle_input(CTRL_D)
        assert _selected_id(selector) == "asst-2"

    def test_preserves_selection_through_multiple_empty_filter_switches(self) -> None:
        tree = build_tree([user_message("user-1", None, "hello"), assistant_message("asst-1", "user-1", "hi")])
        selector = _selector(tree, "asst-1")
        assert _selected_id(selector) == "asst-1"

        selector.handle_input(CTRL_L)  # labeled-only (empty)
        assert selector.get_tree_list().get_selected_node() is None

        selector.handle_input(CTRL_L)  # toggle back to default
        assert _selected_id(selector) == "asst-1"

        selector.handle_input(CTRL_L)  # labeled-only again
        assert selector.get_tree_list().get_selected_node() is None

        selector.handle_input(CTRL_D)
        assert _selected_id(selector) == "asst-1"


def _build_branching_tree() -> list[SessionTreeNode]:
    entries: list[SessionEntry] = [
        user_message("user-1", None, "first message"),
        assistant_message("asst-1", "user-1", "response 1"),
        user_message("user-2", "asst-1", "second message"),
        assistant_message("asst-2", "user-2", "response 2"),
        # Branch A (active)
        user_message("user-3a", "asst-2", "branch A start"),
        assistant_message("asst-3a", "user-3a", "branch A response"),
        user_message("user-4a", "asst-3a", "branch A deep"),
        assistant_message("asst-4a", "user-4a", "branch A leaf"),
        # Branch B
        user_message("user-3b", "asst-2", "branch B start"),
        assistant_message("asst-3b", "user-3b", "branch B response"),
        user_message("user-4b", "asst-3b", "branch B deep"),
    ]
    return build_tree(entries)


class TestBranchNavigationAndFolding:
    def test_ctrl_right_unfolds_a_folded_node_then_does_a_segment_jump(self) -> None:
        selector = _selector(_build_branching_tree(), "asst-4a")

        selector.handle_input(CTRL_LEFT)  # asst-4a -> user-3a
        assert _selected_id(selector) == "user-3a"

        selector.handle_input(CTRL_LEFT)  # fold user-3a
        assert _selected_id(selector) == "user-3a"

        selector.handle_input(DOWN)  # user-3a -> user-3b (children hidden)
        assert _selected_id(selector) == "user-3b"

        selector.handle_input(UP)  # user-3b -> user-3a
        assert _selected_id(selector) == "user-3a"

        selector.handle_input(CTRL_RIGHT)  # unfold user-3a
        assert _selected_id(selector) == "user-3a"

        selector.handle_input(DOWN)  # user-3a -> asst-3a (children restored)
        assert _selected_id(selector) == "asst-3a"

        selector.handle_input(CTRL_LEFT)  # asst-3a -> user-3a
        assert _selected_id(selector) == "user-3a"

        selector.handle_input(CTRL_RIGHT)  # user-3a -> asst-4a (segment jump to leaf)
        assert _selected_id(selector) == "asst-4a"

    def test_alt_left_right_are_aliases_for_fold_and_unfold_navigation(self) -> None:
        selector = _selector(_build_branching_tree(), "asst-4a")

        selector.handle_input(ALT_LEFT)  # asst-4a -> user-3a
        assert _selected_id(selector) == "user-3a"

        selector.handle_input(ALT_LEFT)  # fold user-3a
        assert _selected_id(selector) == "user-3a"

        selector.handle_input(ALT_RIGHT)  # unfold user-3a
        assert _selected_id(selector) == "user-3a"

        selector.handle_input(ALT_RIGHT)  # user-3a -> asst-4a
        assert _selected_id(selector) == "asst-4a"

    def test_folding_root_hides_the_entire_subtree_and_preserves_nested_folds(self) -> None:
        selector = _selector(_build_branching_tree(), "asst-4a")

        selector.handle_input(CTRL_LEFT)  # asst-4a -> user-3a
        assert _selected_id(selector) == "user-3a"

        selector.handle_input(CTRL_LEFT)  # fold user-3a
        assert _selected_id(selector) == "user-3a"

        selector.handle_input(CTRL_LEFT)  # user-3a (folded) -> user-1
        assert _selected_id(selector) == "user-1"

        selector.handle_input(CTRL_LEFT)  # fold user-1
        assert _selected_id(selector) == "user-1"

        selector.handle_input(DOWN)  # wrap (only visible node)
        assert _selected_id(selector) == "user-1"

        selector.handle_input(CTRL_RIGHT)  # unfold user-1
        assert _selected_id(selector) == "user-1"

        selector.handle_input(CTRL_RIGHT)  # user-1 -> user-3a (segment jump, user-3a still folded)
        assert _selected_id(selector) == "user-3a"

        selector.handle_input(DOWN)  # user-3a -> user-3b (user-3a still folded)
        assert _selected_id(selector) == "user-3b"

    def test_fold_and_navigate_on_a_non_active_branch(self) -> None:
        selector = _selector(_build_branching_tree(), "asst-4a")

        found = False
        for _ in range(20):
            selector.handle_input(DOWN)
            if _selected_id(selector) == "user-3b":
                found = True
                break
        assert found is True

        selector.handle_input(CTRL_RIGHT)  # user-3b -> user-4b (segment jump to leaf)
        assert _selected_id(selector) == "user-4b"

        selector.handle_input(CTRL_LEFT)  # user-4b -> user-3b
        assert _selected_id(selector) == "user-3b"

        selector.handle_input(CTRL_LEFT)  # fold user-3b
        assert _selected_id(selector) == "user-3b"

        selector.handle_input(CTRL_LEFT)  # user-3b (folded) -> user-1
        assert _selected_id(selector) == "user-1"

    def test_fold_and_navigate_with_multiple_roots(self) -> None:
        entries: list[SessionEntry] = [
            user_message("user-1", None, "first root"),
            assistant_message("asst-1", "user-1", "response 1"),
            user_message("user-2", None, "second root"),
            assistant_message("asst-2", "user-2", "response 2"),
        ]
        selector = _selector(build_tree(entries), "asst-1")

        assert _selected_id(selector) == "asst-1"

        selector.handle_input(CTRL_LEFT)  # asst-1 -> user-1
        assert _selected_id(selector) == "user-1"

        selector.handle_input(CTRL_LEFT)  # fold user-1
        assert _selected_id(selector) == "user-1"

        selector.handle_input(DOWN)  # user-1 -> user-2 (children hidden)
        assert _selected_id(selector) == "user-2"

        selector.handle_input(CTRL_RIGHT)  # user-2 -> asst-2 (segment jump to leaf)
        assert _selected_id(selector) == "asst-2"

        selector.handle_input(CTRL_LEFT)  # asst-2 -> user-2
        assert _selected_id(selector) == "user-2"

        selector.handle_input(CTRL_LEFT)  # fold user-2
        assert _selected_id(selector) == "user-2"

        selector.handle_input(CTRL_LEFT)  # user-2 (folded, root) -> stays
        assert _selected_id(selector) == "user-2"

    def test_folding_root_hides_descendants_even_when_intermediate_nodes_are_filtered_out(self) -> None:
        entries: list[SessionEntry] = [
            user_message("user-1", None, "hello"),
            tool_call_only_assistant("tool-asst-1", "user-1"),
            user_message("user-2", "tool-asst-1", "follow up"),
            assistant_message("asst-2", "user-2", "response"),
        ]
        selector = _selector(build_tree(entries), "asst-2")

        selector.handle_input(CTRL_LEFT)  # asst-2 -> user-1
        assert _selected_id(selector) == "user-1"

        selector.handle_input(CTRL_LEFT)  # fold user-1
        assert _selected_id(selector) == "user-1"

        selector.handle_input(DOWN)  # wrap (only visible node)
        assert _selected_id(selector) == "user-1"

    def test_search_resets_fold_state(self) -> None:
        selector = _selector(_build_branching_tree(), "asst-4a")

        selector.handle_input(CTRL_LEFT)  # asst-4a -> user-3a
        selector.handle_input(CTRL_LEFT)  # fold user-3a

        selector.handle_input(DOWN)  # user-3a -> user-3b (children hidden)
        assert _selected_id(selector) == "user-3b"

        selector.handle_input("b")  # search resets folds
        selector.handle_input("\x1b")  # clear search

        current_id = ""
        for _ in range(20):
            selector.handle_input(DOWN)
            current_id = _selected_id(selector) or ""
            if current_id == "user-3a":
                break
        assert current_id == "user-3a"

        selector.handle_input(DOWN)  # user-3a -> asst-3a (not user-3b)
        assert _selected_id(selector) == "asst-3a"

    def test_filter_mode_change_resets_fold_state(self) -> None:
        selector = _selector(_build_branching_tree(), "asst-4a")

        selector.handle_input(CTRL_LEFT)  # asst-4a -> user-3a
        selector.handle_input(CTRL_LEFT)  # fold user-3a

        selector.handle_input(CTRL_U)  # user-only filter resets folds
        selector.handle_input(CTRL_D)  # back to default

        current_id = ""
        for _ in range(20):
            selector.handle_input(DOWN)
            current_id = _selected_id(selector) or ""
            if current_id == "user-3a":
                break
        assert current_id == "user-3a"

        selector.handle_input(DOWN)  # user-3a -> asst-3a (not user-3b)
        assert _selected_id(selector) == "asst-3a"
