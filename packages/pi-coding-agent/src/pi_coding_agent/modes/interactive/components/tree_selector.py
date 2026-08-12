"""The `/tree` session-graph selector.

Ported from ``packages/coding-agent/src/modes/interactive/components/tree-selector.ts``.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pi_tui.component import Component, Container
from pi_tui.components.input import Input
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text
from pi_tui.keybindings import get_keybindings
from pi_tui.utils import slice_by_column, truncate_to_width, visible_width, wrap_text_with_ansi

from ..theme.theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import format_key_text, key_hint

if TYPE_CHECKING:
    from ....core.session_manager import SessionTreeNode

FilterMode = Literal["default", "no-tools", "user-only", "labeled-only", "all"]
FILTER_MODES: list[FilterMode] = ["default", "no-tools", "user-only", "labeled-only", "all"]

TREE_GUTTER_WIDTH = 2
MIN_VISIBLE_ANCHOR_CONTENT_WIDTH = 4
MAX_VISIBLE_ANCHOR_CONTENT_WIDTH = 20
MIN_ANCHOR_CONTEXT_WIDTH = 2
MAX_ANCHOR_CONTEXT_WIDTH = 12


@dataclass
class GutterInfo:
    position: int
    """Display-indent level where the connector was drawn."""
    show: bool
    """True draws ``│``, False draws spaces."""


@dataclass
class FlatNode:
    node: SessionTreeNode
    indent: int
    show_connector: bool
    is_last: bool
    gutters: list[GutterInfo] = field(default_factory=list)
    is_virtual_root_child: bool = False


@dataclass
class HorizontalViewportRow:
    gutter: str
    body: str
    anchor_col: int
    body_width: int
    is_selected: bool


@dataclass
class ToolCallInfo:
    name: str
    arguments: dict[str, Any]


def render_horizontal_viewport(rows: list[HorizontalViewportRow], width: int) -> list[str]:
    """Clip rows into a horizontally scrolled viewport.

    The tree gutter stays pinned; only the body pans, and only when the
    selected row's anchor would otherwise push its content off-screen.
    """
    viewport_width = max(0, width - TREE_GUTTER_WIDTH)
    max_body_width = max((row.body_width for row in rows), default=0)
    max_horizontal_scroll = max(0, max_body_width - viewport_width)
    selected_row = next((row for row in rows if row.is_selected), None)

    horizontal_scroll = 0
    if selected_row is not None and max_horizontal_scroll > 0:
        min_visible_anchor_content_width = min(
            MAX_VISIBLE_ANCHOR_CONTENT_WIDTH,
            max(MIN_VISIBLE_ANCHOR_CONTENT_WIDTH, math.floor(viewport_width / 3)),
        )
        if selected_row.anchor_col > viewport_width - min_visible_anchor_content_width:
            anchor_context_width = min(
                MAX_ANCHOR_CONTEXT_WIDTH,
                max(MIN_ANCHOR_CONTEXT_WIDTH, math.floor(viewport_width / 4)),
            )
            horizontal_scroll = min(max_horizontal_scroll, selected_row.anchor_col - anchor_context_width)

    lines: list[str] = []
    for row in rows:
        if horizontal_scroll > 0:
            body = slice_by_column(row.body, horizontal_scroll, viewport_width, True)
            line = f"{row.gutter}{body}\x1b[0m"
        else:
            line = row.gutter + row.body
        lines.append(truncate_to_width(line, width, ""))
    return lines


def _entry_attr(entry: Any, name: str, default: Any = None) -> Any:
    return getattr(entry, name, default)


def _normalize(text: str) -> str:
    return text.replace("\n", " ").replace("\t", " ").strip()


def _shorten_path(path: str) -> str:
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE") or ""
    if home and path.startswith(home):
        return f"~{path[len(home) :]}"
    return path


def format_tool_call(name: str, args: dict[str, Any]) -> str:
    if name == "read":
        path = _shorten_path(str(args.get("path") or args.get("file_path") or ""))
        offset = args.get("offset")
        limit = args.get("limit")
        display = path
        if offset is not None or limit is not None:
            start = offset if offset is not None else 1
            end = start + limit - 1 if limit is not None else ""
            display += f":{start}{f'-{end}' if end != '' else ''}"
        return f"[read: {display}]"
    if name == "write":
        return f"[write: {_shorten_path(str(args.get('path') or args.get('file_path') or ''))}]"
    if name == "edit":
        return f"[edit: {_shorten_path(str(args.get('path') or args.get('file_path') or ''))}]"
    if name == "bash":
        raw_command = str(args.get("command") or "")
        command = _normalize(raw_command)[:50]
        return f"[bash: {command}{'...' if len(raw_command) > 50 else ''}]"
    if name == "grep":
        pattern = str(args.get("pattern") or "")
        return f"[grep: /{pattern}/ in {_shorten_path(str(args.get('path') or '.'))}]"
    if name == "find":
        pattern = str(args.get("pattern") or "")
        return f"[find: {pattern} in {_shorten_path(str(args.get('path') or '.'))}]"
    if name == "ls":
        return f"[ls: {_shorten_path(str(args.get('path') or '.'))}]"
    encoded = json.dumps(args, ensure_ascii=False)
    return f"[{name}: {encoded[:40]}{'...' if len(encoded) > 40 else ''}]"


class TreeList(Component):
    """The scrollable tree body with filtering, folding and search."""

    def __init__(
        self,
        tree: list[SessionTreeNode],
        current_leaf_id: str | None,
        max_visible_lines: int,
        initial_selected_id: str | None = None,
        initial_filter_mode: FilterMode | None = None,
    ) -> None:
        self.current_leaf_id = current_leaf_id
        self.max_visible_lines = max_visible_lines
        self.filter_mode: FilterMode = initial_filter_mode or "default"
        self.search_query = ""
        self.tool_call_map: dict[str, ToolCallInfo] = {}
        self.multiple_roots = len(tree) > 1
        self.show_label_timestamps = False
        self.active_path_ids: set[str] = set()
        self.visible_parent_map: dict[str, str | None] = {}
        self.visible_children_map: dict[str | None, list[str]] = {}
        self.folded_nodes: set[str] = set()
        self.selected_index = 0
        self.filtered_nodes: list[FlatNode] = []

        self.on_select: Callable[[str], None] | None = None
        self.on_cancel: Callable[[], None] | None = None
        self.on_copy: Callable[[str | None], None] | None = None
        self.on_label_edit: Callable[[str, str | None], None] | None = None

        self.flat_nodes = self._flatten_tree(tree)
        self._build_active_path()
        self.last_selected_id: str | None = None
        self._apply_filter()

        target_id = initial_selected_id if initial_selected_id is not None else current_leaf_id
        self.selected_index = self._find_nearest_visible_index(target_id)
        self.last_selected_id = (
            self.filtered_nodes[self.selected_index].node.entry.id
            if 0 <= self.selected_index < len(self.filtered_nodes)
            else None
        )

    # -- structure ----------------------------------------------------------

    def _find_nearest_visible_index(self, entry_id: str | None) -> int:
        """Index of the nearest visible entry, walking up the parent chain."""
        if len(self.filtered_nodes) == 0:
            return 0

        entry_map = {flat.node.entry.id: flat for flat in self.flat_nodes}
        visible_id_to_index = {flat.node.entry.id: index for index, flat in enumerate(self.filtered_nodes)}

        current_id = entry_id
        while current_id is not None:
            index = visible_id_to_index.get(current_id)
            if index is not None:
                return index
            node = entry_map.get(current_id)
            if node is None:
                break
            current_id = _entry_attr(node.node.entry, "parent_id")

        return len(self.filtered_nodes) - 1

    def _build_active_path(self) -> None:
        self.active_path_ids.clear()
        if not self.current_leaf_id:
            return
        entry_map = {flat.node.entry.id: flat for flat in self.flat_nodes}
        current_id: str | None = self.current_leaf_id
        while current_id:
            self.active_path_ids.add(current_id)
            node = entry_map.get(current_id)
            if node is None:
                break
            current_id = _entry_attr(node.node.entry, "parent_id")

    def _collect_tool_calls(self, entry: Any) -> None:
        if _entry_attr(entry, "type") != "message":
            return
        message = _entry_attr(entry, "message")
        if message is None or getattr(message, "role", None) != "assistant":
            return
        for block in getattr(message, "content", None) or []:
            if getattr(block, "type", None) == "toolCall":
                self.tool_call_map[block.id] = ToolCallInfo(name=block.name, arguments=block.arguments)

    def _contains_active_map(self, roots: list[SessionTreeNode]) -> dict[int, bool]:
        """Which subtrees hold the active leaf, so the live branch sorts first."""
        all_nodes: list[SessionTreeNode] = []
        pre_order_stack = list(roots)
        while pre_order_stack:
            node = pre_order_stack.pop()
            all_nodes.append(node)
            for child in reversed(node.children):
                pre_order_stack.append(child)

        contains_active: dict[int, bool] = {}
        for node in reversed(all_nodes):
            has = self.current_leaf_id is not None and node.entry.id == self.current_leaf_id
            for child in node.children:
                if contains_active.get(id(child)):
                    has = True
            contains_active[id(node)] = has
        return contains_active

    def _flatten_tree(self, roots: list[SessionTreeNode]) -> list[FlatNode]:
        """Depth-first flatten with the TypeScript indentation rules:

        indent 0 stays flat unless the parent branches; the first generation
        after a branch shifts right for visual grouping; single-child chains
        never drift.
        """
        result: list[FlatNode] = []
        self.tool_call_map.clear()
        contains_active = self._contains_active_map(roots)

        multiple_roots = len(roots) > 1
        ordered_roots = sorted(roots, key=lambda node: not contains_active.get(id(node), False))

        # (node, indent, just_branched, show_connector, is_last, gutters, is_virtual_root_child)
        stack: list[tuple[SessionTreeNode, int, bool, bool, bool, list[GutterInfo], bool]] = []
        for index in range(len(ordered_roots) - 1, -1, -1):
            stack.append(
                (
                    ordered_roots[index],
                    1 if multiple_roots else 0,
                    multiple_roots,
                    multiple_roots,
                    index == len(ordered_roots) - 1,
                    [],
                    multiple_roots,
                )
            )

        while stack:
            node, indent, just_branched, show_connector, is_last, gutters, is_virtual_root_child = stack.pop()
            self._collect_tool_calls(node.entry)
            result.append(
                FlatNode(
                    node=node,
                    indent=indent,
                    show_connector=show_connector,
                    is_last=is_last,
                    gutters=gutters,
                    is_virtual_root_child=is_virtual_root_child,
                )
            )

            children = node.children
            multiple_children = len(children) > 1
            prioritized = [child for child in children if contains_active.get(id(child))]
            rest = [child for child in children if not contains_active.get(id(child))]
            ordered_children = [*prioritized, *rest]

            if multiple_children or (just_branched and indent > 0):
                child_indent = indent + 1
            else:
                child_indent = indent

            connector_displayed = show_connector and not is_virtual_root_child
            current_display_indent = max(0, indent - 1) if self.multiple_roots else indent
            connector_position = max(0, current_display_indent - 1)
            child_gutters = (
                [*gutters, GutterInfo(position=connector_position, show=not is_last)]
                if connector_displayed
                else gutters
            )

            for index in range(len(ordered_children) - 1, -1, -1):
                stack.append(
                    (
                        ordered_children[index],
                        child_indent,
                        multiple_children,
                        multiple_children,
                        index == len(ordered_children) - 1,
                        child_gutters,
                        False,
                    )
                )

        return result

    # -- filtering ----------------------------------------------------------

    def _is_settings_entry(self, entry: Any) -> bool:
        return _entry_attr(entry, "type") in (
            "label",
            "custom",
            "model_change",
            "thinking_level_change",
            "session_info",
        )

    def _passes_filter(self, flat_node: FlatNode) -> bool:
        entry = flat_node.node.entry
        entry_type = _entry_attr(entry, "type")
        is_settings_entry = self._is_settings_entry(entry)

        if self.filter_mode == "user-only":
            message = _entry_attr(entry, "message")
            return entry_type == "message" and getattr(message, "role", None) == "user"
        if self.filter_mode == "no-tools":
            message = _entry_attr(entry, "message")
            is_tool_result = entry_type == "message" and getattr(message, "role", None) == "toolResult"
            return not is_settings_entry and not is_tool_result
        if self.filter_mode == "labeled-only":
            return flat_node.node.label is not None
        if self.filter_mode == "all":
            return True
        return not is_settings_entry

    def _apply_filter(self) -> None:
        if len(self.filtered_nodes) > 0 and 0 <= self.selected_index < len(self.filtered_nodes):
            self.last_selected_id = self.filtered_nodes[self.selected_index].node.entry.id

        search_tokens = [token for token in self.search_query.lower().split() if token]

        filtered: list[FlatNode] = []
        for flat_node in self.flat_nodes:
            entry = flat_node.node.entry
            is_current_leaf = entry.id == self.current_leaf_id

            # Assistant turns that only carry tool calls add nothing here, but
            # the active leaf always stays visible so the cursor has a home.
            if _entry_attr(entry, "type") == "message" and not is_current_leaf:
                message = _entry_attr(entry, "message")
                if getattr(message, "role", None) == "assistant":
                    has_text = self._has_text_content(getattr(message, "content", None))
                    stop_reason = getattr(message, "stop_reason", None)
                    is_error_or_aborted = bool(stop_reason) and stop_reason not in ("stop", "toolUse")
                    if not has_text and not is_error_or_aborted:
                        continue

            if not self._passes_filter(flat_node):
                continue

            if search_tokens:
                node_text = self._get_searchable_text(flat_node.node).lower()
                if not all(token in node_text for token in search_tokens):
                    continue

            filtered.append(flat_node)

        self.filtered_nodes = filtered

        if self.folded_nodes:
            skip: set[str] = set()
            for flat_node in self.flat_nodes:
                entry_id = flat_node.node.entry.id
                parent_id = _entry_attr(flat_node.node.entry, "parent_id")
                if parent_id is not None and (parent_id in self.folded_nodes or parent_id in skip):
                    skip.add(entry_id)
            self.filtered_nodes = [
                flat_node for flat_node in self.filtered_nodes if flat_node.node.entry.id not in skip
            ]

        self._recalculate_visual_structure()

        if self.last_selected_id:
            self.selected_index = self._find_nearest_visible_index(self.last_selected_id)
        elif self.selected_index >= len(self.filtered_nodes):
            self.selected_index = max(0, len(self.filtered_nodes) - 1)

        if 0 <= self.selected_index < len(self.filtered_nodes):
            self.last_selected_id = self.filtered_nodes[self.selected_index].node.entry.id

    def _recalculate_visual_structure(self) -> None:
        """Recompute indent/connectors for the filtered view.

        Filtering can hide intermediate entries, so descendants re-attach to
        their nearest *visible* ancestor while keeping the same indentation
        semantics as the unfiltered flatten.
        """
        if len(self.filtered_nodes) == 0:
            return

        visible_ids = {flat.node.entry.id for flat in self.filtered_nodes}
        entry_map = {flat.node.entry.id: flat for flat in self.flat_nodes}

        def find_visible_ancestor(node_id: str) -> str | None:
            flat = entry_map.get(node_id)
            current_id = _entry_attr(flat.node.entry, "parent_id") if flat else None
            while current_id is not None:
                if current_id in visible_ids:
                    return current_id
                flat = entry_map.get(current_id)
                current_id = _entry_attr(flat.node.entry, "parent_id") if flat else None
            return None

        visible_parent: dict[str, str | None] = {}
        visible_children: dict[str | None, list[str]] = {None: []}
        for flat_node in self.filtered_nodes:
            node_id = flat_node.node.entry.id
            ancestor_id = find_visible_ancestor(node_id)
            visible_parent[node_id] = ancestor_id
            visible_children.setdefault(ancestor_id, []).append(node_id)

        visible_root_ids = visible_children[None]
        self.multiple_roots = len(visible_root_ids) > 1
        filtered_node_map = {flat.node.entry.id: flat for flat in self.filtered_nodes}

        stack: list[tuple[str, int, bool, bool, bool, list[GutterInfo], bool]] = []
        for index in range(len(visible_root_ids) - 1, -1, -1):
            stack.append(
                (
                    visible_root_ids[index],
                    1 if self.multiple_roots else 0,
                    self.multiple_roots,
                    self.multiple_roots,
                    index == len(visible_root_ids) - 1,
                    [],
                    self.multiple_roots,
                )
            )

        while stack:
            node_id, indent, just_branched, show_connector, is_last, gutters, is_virtual_root_child = stack.pop()
            flat_node = filtered_node_map.get(node_id)
            if flat_node is None:
                continue

            flat_node.indent = indent
            flat_node.show_connector = show_connector
            flat_node.is_last = is_last
            flat_node.gutters = gutters
            flat_node.is_virtual_root_child = is_virtual_root_child

            children = visible_children.get(node_id, [])
            multiple_children = len(children) > 1
            if multiple_children or (just_branched and indent > 0):
                child_indent = indent + 1
            else:
                child_indent = indent

            connector_displayed = show_connector and not is_virtual_root_child
            current_display_indent = max(0, indent - 1) if self.multiple_roots else indent
            connector_position = max(0, current_display_indent - 1)
            child_gutters = (
                [*gutters, GutterInfo(position=connector_position, show=not is_last)]
                if connector_displayed
                else gutters
            )

            for index in range(len(children) - 1, -1, -1):
                stack.append(
                    (
                        children[index],
                        child_indent,
                        multiple_children,
                        multiple_children,
                        index == len(children) - 1,
                        child_gutters,
                        False,
                    )
                )

        self.visible_parent_map = visible_parent
        self.visible_children_map = visible_children

    # -- text extraction ----------------------------------------------------

    def _extract_full_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        return "".join(getattr(block, "text", "") for block in content if getattr(block, "type", None) == "text")

    def _extract_content(self, content: Any) -> str:
        return self._extract_full_content(content)[:200]

    def _has_text_content(self, content: Any) -> bool:
        if isinstance(content, str):
            return len(content.strip()) > 0
        if isinstance(content, list):
            for block in content:
                if getattr(block, "type", None) == "text" and (getattr(block, "text", "") or "").strip():
                    return True
        return False

    def _get_searchable_text(self, node: SessionTreeNode) -> str:
        entry = node.entry
        entry_type = _entry_attr(entry, "type")
        parts: list[str] = []
        if node.label:
            parts.append(node.label)

        if entry_type == "message":
            message = _entry_attr(entry, "message")
            parts.append(getattr(message, "role", ""))
            content = getattr(message, "content", None)
            if content:
                parts.append(self._extract_content(content))
            if getattr(message, "role", None) == "bashExecution":
                command = getattr(message, "command", None)
                if command:
                    parts.append(command)
        elif entry_type == "custom_message":
            parts.append(_entry_attr(entry, "custom_type", ""))
            content = _entry_attr(entry, "content")
            parts.append(content if isinstance(content, str) else self._extract_content(content))
        elif entry_type == "compaction":
            parts.append("compaction")
        elif entry_type == "branch_summary":
            parts.extend(["branch summary", _entry_attr(entry, "summary", "")])
        elif entry_type == "session_info":
            parts.append("title")
            name = _entry_attr(entry, "name")
            if name:
                parts.append(name)
        elif entry_type == "model_change":
            parts.extend(["model", _entry_attr(entry, "model_id", "")])
        elif entry_type == "thinking_level_change":
            parts.extend(["thinking", _entry_attr(entry, "thinking_level", "")])
        elif entry_type == "custom":
            parts.extend(["custom", _entry_attr(entry, "custom_type", "")])
        elif entry_type == "label":
            parts.extend(["label", _entry_attr(entry, "label") or ""])

        return " ".join(parts)

    def _get_entry_display_text(self, node: SessionTreeNode, is_selected: bool) -> str:
        entry = node.entry
        entry_type = _entry_attr(entry, "type")
        result = ""

        if entry_type == "message":
            message = _entry_attr(entry, "message")
            role = getattr(message, "role", None)
            if role == "user":
                result = theme.fg("accent", "user: ") + _normalize(
                    self._extract_content(getattr(message, "content", None))
                )
            elif role == "assistant":
                text_content = _normalize(self._extract_content(getattr(message, "content", None)))
                if text_content:
                    result = theme.fg("success", "assistant: ") + text_content
                elif getattr(message, "stop_reason", None) == "aborted":
                    result = theme.fg("success", "assistant: ") + theme.fg("muted", "(aborted)")
                elif getattr(message, "error_message", None):
                    error = _normalize(message.error_message)[:80]
                    result = theme.fg("success", "assistant: ") + theme.fg("error", error)
                else:
                    result = theme.fg("success", "assistant: ") + theme.fg("muted", "(no content)")
            elif role == "toolResult":
                tool_call_id = getattr(message, "tool_call_id", None)
                tool_call = self.tool_call_map.get(tool_call_id) if tool_call_id else None
                if tool_call is not None:
                    result = theme.fg("muted", format_tool_call(tool_call.name, tool_call.arguments))
                else:
                    result = theme.fg("muted", f"[{getattr(message, 'tool_name', None) or 'tool'}]")
            elif role == "bashExecution":
                result = theme.fg("dim", f"[bash]: {_normalize(getattr(message, 'command', '') or '')}")
            else:
                result = theme.fg("dim", f"[{role}]")
        elif entry_type == "custom_message":
            content = _entry_attr(entry, "content")
            text = content if isinstance(content, str) else self._extract_full_content(content)
            result = theme.fg("customMessageLabel", f"[{_entry_attr(entry, 'custom_type', '')}]: ") + _normalize(text)
        elif entry_type == "compaction":
            tokens = math.floor(_entry_attr(entry, "tokens_before", 0) / 1000 + 0.5)
            result = theme.fg("borderAccent", f"[compaction: {tokens}k tokens]")
        elif entry_type == "branch_summary":
            result = theme.fg("warning", "[branch summary]: ") + _normalize(_entry_attr(entry, "summary", ""))
        elif entry_type == "model_change":
            result = theme.fg("dim", f"[model: {_entry_attr(entry, 'model_id', '')}]")
        elif entry_type == "thinking_level_change":
            result = theme.fg("dim", f"[thinking: {_entry_attr(entry, 'thinking_level', '')}]")
        elif entry_type == "custom":
            result = theme.fg("dim", f"[custom: {_entry_attr(entry, 'custom_type', '')}]")
        elif entry_type == "label":
            result = theme.fg("dim", f"[label: {_entry_attr(entry, 'label') or '(cleared)'}]")
        elif entry_type == "session_info":
            name = _entry_attr(entry, "name")
            if name:
                result = theme.fg("dim", "[title: ") + theme.fg("dim", name) + theme.fg("dim", "]")
            else:
                result = theme.fg("dim", "[title: ") + theme.italic(theme.fg("dim", "empty")) + theme.fg("dim", "]")

        return theme.bold(result) if is_selected else result

    def _get_entry_copy_text(self, node: SessionTreeNode) -> str | None:
        entry = node.entry
        entry_type = _entry_attr(entry, "type")
        text: str | None = None

        if entry_type == "message":
            message = _entry_attr(entry, "message")
            if getattr(message, "role", None) == "bashExecution":
                text = getattr(message, "command", None)
            elif hasattr(message, "content"):
                text = self._extract_full_content(message.content)
                if not text and getattr(message, "role", None) == "assistant":
                    text = getattr(message, "error_message", None)
        elif entry_type == "custom_message":
            text = self._extract_full_content(_entry_attr(entry, "content"))
        elif entry_type in ("compaction", "branch_summary"):
            text = _entry_attr(entry, "summary")

        return text if text and text.strip() else None

    # -- public helpers -----------------------------------------------------

    def invalidate(self) -> None:
        return None

    def get_search_query(self) -> str:
        return self.search_query

    def get_selected_node(self) -> SessionTreeNode | None:
        if 0 <= self.selected_index < len(self.filtered_nodes):
            return self.filtered_nodes[self.selected_index].node
        return None

    def copy_selected(self) -> None:
        node = self.get_selected_node()
        if self.on_copy is not None:
            self.on_copy(self._get_entry_copy_text(node) if node is not None else None)

    def update_node_label(self, entry_id: str, label: str | None, label_timestamp: str | None = None) -> None:
        for flat_node in self.flat_nodes:
            if flat_node.node.entry.id == entry_id:
                flat_node.node.label = label
                flat_node.node.label_timestamp = (label_timestamp or datetime.now().isoformat()) if label else None
                break

    def _get_status_labels(self) -> str:
        labels = {
            "no-tools": " [no-tools]",
            "user-only": " [user]",
            "labeled-only": " [labeled]",
            "all": " [all]",
        }.get(self.filter_mode, "")
        if self.show_label_timestamps:
            labels += " [+label time]"
        return labels

    def _format_label_timestamp(self, timestamp: str) -> str:
        try:
            date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return timestamp
        now = datetime.now(tz=date.tzinfo)
        time = f"{date.hour:02d}:{date.minute:02d}"
        if (date.year, date.month, date.day) == (now.year, now.month, now.day):
            return time
        if date.year == now.year:
            return f"{date.month}/{date.day} {time}"
        return f"{str(date.year)[-2:]}/{date.month}/{date.day} {time}"

    def is_foldable(self, entry_id: str) -> bool:
        """Foldable when it has visible children and is a root or a segment start."""
        children = self.visible_children_map.get(entry_id)
        if not children:
            return False
        parent_id = self.visible_parent_map.get(entry_id)
        if parent_id is None:
            return True
        siblings = self.visible_children_map.get(parent_id)
        return siblings is not None and len(siblings) > 1

    # -- rendering ----------------------------------------------------------

    def _build_prefix(self, flat_node: FlatNode, display_indent: int, connector: str) -> str:
        connector_position = display_indent - 1 if connector else -1
        entry_id = flat_node.node.entry.id
        is_folded = entry_id in self.folded_nodes
        prefix_chars: list[str] = []
        for position in range(display_indent * 3):
            level = position // 3
            position_in_level = position % 3
            gutter = next((g for g in flat_node.gutters if g.position == level), None)
            if gutter is not None:
                prefix_chars.append(("│" if gutter.show else " ") if position_in_level == 0 else " ")
            elif connector and level == connector_position:
                if position_in_level == 0:
                    prefix_chars.append("└" if flat_node.is_last else "├")
                elif position_in_level == 1:
                    prefix_chars.append("⊞" if is_folded else ("⊟" if self.is_foldable(entry_id) else "─"))
                else:
                    prefix_chars.append(" ")
            else:
                prefix_chars.append(" ")
        return "".join(prefix_chars)

    def render(self, width: int) -> list[str]:
        if len(self.filtered_nodes) == 0:
            return [
                truncate_to_width(theme.fg("muted", "  No entries found"), width),
                truncate_to_width(theme.fg("muted", f"  (0/0){self._get_status_labels()}"), width),
            ]

        start_index = max(
            0,
            min(
                self.selected_index - math.floor(self.max_visible_lines / 2),
                len(self.filtered_nodes) - self.max_visible_lines,
            ),
        )
        end_index = min(start_index + self.max_visible_lines, len(self.filtered_nodes))

        rendered_rows: list[HorizontalViewportRow] = []
        for index in range(start_index, end_index):
            flat_node = self.filtered_nodes[index]
            entry = flat_node.node.entry
            is_selected = index == self.selected_index
            cursor = theme.fg("accent", "› ") if is_selected else "  "

            # With multiple roots the whole tree shifts left by one so roots sit at 0.
            display_indent = max(0, flat_node.indent - 1) if self.multiple_roots else flat_node.indent
            connector = ""
            if flat_node.show_connector and not flat_node.is_virtual_root_child:
                connector = "└─ " if flat_node.is_last else "├─ "
            prefix = self._build_prefix(flat_node, display_indent, connector)

            shows_fold_in_connector = flat_node.show_connector and not flat_node.is_virtual_root_child
            is_folded = entry.id in self.folded_nodes
            fold_marker = theme.fg("accent", "⊞ ") if is_folded and not shows_fold_in_connector else ""
            path_marker = theme.fg("accent", "• ") if entry.id in self.active_path_ids else ""

            label = theme.fg("warning", f"[{flat_node.node.label}] ") if flat_node.node.label else ""
            label_timestamp = ""
            if self.show_label_timestamps and flat_node.node.label and flat_node.node.label_timestamp:
                label_timestamp = theme.fg("muted", f"{self._format_label_timestamp(flat_node.node.label_timestamp)} ")

            content = self._get_entry_display_text(flat_node.node, is_selected)
            prefix_part = theme.fg("dim", prefix) + fold_marker + path_marker
            gutter = cursor
            body = prefix_part + label + label_timestamp + content
            if is_selected:
                gutter = theme.bg("selectedBg", gutter)
                body = theme.bg("selectedBg", body)

            rendered_rows.append(
                HorizontalViewportRow(
                    gutter=gutter,
                    body=body,
                    anchor_col=visible_width(prefix_part),
                    body_width=visible_width(body),
                    is_selected=is_selected,
                )
            )

        lines = render_horizontal_viewport(rendered_rows, width)
        lines.append(
            truncate_to_width(
                theme.fg(
                    "muted",
                    f"  ({self.selected_index + 1}/{len(self.filtered_nodes)}){self._get_status_labels()}",
                ),
                width,
            )
        )
        return lines

    # -- navigation ---------------------------------------------------------

    def _find_branch_segment_start(self, direction: Literal["up", "down"]) -> int:
        if not (0 <= self.selected_index < len(self.filtered_nodes)):
            return self.selected_index
        selected_id = self.filtered_nodes[self.selected_index].node.entry.id
        index_by_entry_id = {flat.node.entry.id: index for index, flat in enumerate(self.filtered_nodes)}

        current_id = selected_id
        if direction == "down":
            while True:
                children = self.visible_children_map.get(current_id, [])
                if len(children) == 0:
                    return index_by_entry_id[current_id]
                if len(children) > 1:
                    return index_by_entry_id[children[0]]
                current_id = children[0]

        while True:
            parent_id = self.visible_parent_map.get(current_id)
            if parent_id is None:
                return index_by_entry_id[current_id]
            children = self.visible_children_map.get(parent_id, [])
            if len(children) > 1:
                segment_start = index_by_entry_id[current_id]
                if segment_start < self.selected_index:
                    return segment_start
            current_id = parent_id

    def _set_filter_mode(self, mode: FilterMode) -> None:
        self.filter_mode = mode
        self.folded_nodes.clear()
        self._apply_filter()

    def handle_input(self, key_data: str) -> None:
        keybindings = get_keybindings()

        if keybindings.matches(key_data, "tui.select.up"):
            self.selected_index = len(self.filtered_nodes) - 1 if self.selected_index == 0 else self.selected_index - 1
        elif keybindings.matches(key_data, "tui.select.down"):
            self.selected_index = 0 if self.selected_index == len(self.filtered_nodes) - 1 else self.selected_index + 1
        elif keybindings.matches(key_data, "app.tree.foldOrUp"):
            current = self.get_selected_node()
            current_id = current.entry.id if current is not None else None
            if current_id and self.is_foldable(current_id) and current_id not in self.folded_nodes:
                self.folded_nodes.add(current_id)
                self._apply_filter()
            else:
                self.selected_index = self._find_branch_segment_start("up")
        elif keybindings.matches(key_data, "app.tree.unfoldOrDown"):
            current = self.get_selected_node()
            current_id = current.entry.id if current is not None else None
            if current_id and current_id in self.folded_nodes:
                self.folded_nodes.discard(current_id)
                self._apply_filter()
            else:
                self.selected_index = self._find_branch_segment_start("down")
        elif keybindings.matches(key_data, "tui.editor.cursorLeft") or keybindings.matches(
            key_data, "tui.select.pageUp"
        ):
            self.selected_index = max(0, self.selected_index - self.max_visible_lines)
        elif keybindings.matches(key_data, "tui.editor.cursorRight") or keybindings.matches(
            key_data, "tui.select.pageDown"
        ):
            self.selected_index = min(len(self.filtered_nodes) - 1, self.selected_index + self.max_visible_lines)
        elif keybindings.matches(key_data, "tui.select.confirm"):
            node = self.get_selected_node()
            if node is not None and self.on_select is not None:
                self.on_select(node.entry.id)
        elif keybindings.matches(key_data, "app.message.copy"):
            self.copy_selected()
        elif keybindings.matches(key_data, "tui.select.cancel"):
            if self.search_query:
                self.search_query = ""
                self.folded_nodes.clear()
                self._apply_filter()
            elif self.on_cancel is not None:
                self.on_cancel()
        elif keybindings.matches(key_data, "app.tree.filter.default"):
            self._set_filter_mode("default")
        elif keybindings.matches(key_data, "app.tree.filter.noTools"):
            self._set_filter_mode("default" if self.filter_mode == "no-tools" else "no-tools")
        elif keybindings.matches(key_data, "app.tree.filter.userOnly"):
            self._set_filter_mode("default" if self.filter_mode == "user-only" else "user-only")
        elif keybindings.matches(key_data, "app.tree.filter.labeledOnly"):
            self._set_filter_mode("default" if self.filter_mode == "labeled-only" else "labeled-only")
        elif keybindings.matches(key_data, "app.tree.filter.all"):
            self._set_filter_mode("default" if self.filter_mode == "all" else "all")
        elif keybindings.matches(key_data, "app.tree.filter.cycleBackward"):
            index = FILTER_MODES.index(self.filter_mode)
            self._set_filter_mode(FILTER_MODES[(index - 1) % len(FILTER_MODES)])
        elif keybindings.matches(key_data, "app.tree.filter.cycleForward"):
            index = FILTER_MODES.index(self.filter_mode)
            self._set_filter_mode(FILTER_MODES[(index + 1) % len(FILTER_MODES)])
        elif keybindings.matches(key_data, "tui.editor.deleteCharBackward"):
            if len(self.search_query) > 0:
                self.search_query = self.search_query[:-1]
                self.folded_nodes.clear()
                self._apply_filter()
        elif keybindings.matches(key_data, "app.tree.editLabel"):
            node = self.get_selected_node()
            if node is not None and self.on_label_edit is not None:
                self.on_label_edit(node.entry.id, node.label)
        elif keybindings.matches(key_data, "app.tree.toggleLabelTimestamp"):
            self.show_label_timestamps = not self.show_label_timestamps
        else:
            has_control_chars = any(
                ord(char) < 32 or ord(char) == 0x7F or 0x80 <= ord(char) <= 0x9F for char in key_data
            )
            if not has_control_chars and len(key_data) > 0:
                self.search_query += key_data
                self.folded_nodes.clear()
                self._apply_filter()


class SearchLine(Component):
    def __init__(self, tree_list: TreeList) -> None:
        self.tree_list = tree_list

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        query = self.tree_list.get_search_query()
        prompt = theme.fg("muted", "Type to search:")
        if query:
            return [truncate_to_width(f"  {prompt} {theme.fg('accent', query)}", width)]
        return [truncate_to_width(f"  {prompt}", width)]

    def handle_input(self, _key_data: str) -> None:
        return None


@dataclass(frozen=True)
class TreeHelpItem:
    keys: tuple[str, ...]
    label: str
    label_first: bool = False


TREE_HELP_ITEMS: tuple[TreeHelpItem, ...] = (
    TreeHelpItem(("tui.select.up", "tui.select.down"), "move"),
    TreeHelpItem(("tui.editor.cursorLeft", "tui.editor.cursorRight"), "page"),
    TreeHelpItem(("app.tree.foldOrUp", "app.tree.unfoldOrDown"), "branch"),
    TreeHelpItem(("app.message.copy",), "copy"),
    TreeHelpItem(("app.tree.editLabel",), "label"),
    TreeHelpItem(("app.tree.toggleLabelTimestamp",), "label time"),
    TreeHelpItem(
        (
            "app.tree.filter.default",
            "app.tree.filter.noTools",
            "app.tree.filter.userOnly",
            "app.tree.filter.labeledOnly",
            "app.tree.filter.all",
        ),
        "filters",
        label_first=True,
    ),
    TreeHelpItem(("app.tree.filter.cycleForward", "app.tree.filter.cycleBackward"), "cycle", True),
)

_ARROW_REPLACEMENTS = (
    ("pageUp", "pgup"),
    ("pageDown", "pgdn"),
    ("up", "↑"),
    ("down", "↓"),
    ("left", "←"),
    ("right", "→"),
)


def compact_raw_keys(keys: Sequence[str]) -> str:
    if len(keys) == 1:
        return keys[0]
    parts = []
    for key in keys:
        separator_index = key.rfind("+")
        if separator_index == -1:
            parts.append(("", key))
        else:
            parts.append((key[: separator_index + 1], key[separator_index + 1 :]))
    prefix = parts[0][0]
    if prefix and all(part[0] == prefix for part in parts):
        return f"{prefix}{'/'.join(part[1] for part in parts)}"
    return "/".join(keys)


def format_help_keys(keybindings: Sequence[str]) -> str:
    keys: list[str] = []
    for keybinding in keybindings:
        bound = get_keybindings().get_keys(keybinding)
        if bound:
            keys.append(bound[0])
    if len(keys) == 0:
        return ""

    text = format_key_text(compact_raw_keys(keys))
    # Word-boundary replacements, matching the TypeScript `\b...\b` regexes.
    for source, replacement in _ARROW_REPLACEMENTS:
        text = re.sub(rf"\b{source}\b", replacement, text)
    return text


class TreeHelp(Component):
    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        items: list[str] = []
        for item in TREE_HELP_ITEMS:
            text = format_help_keys(item.keys)
            if not text:
                items.append(item.label)
            elif item.label_first:
                items.append(f"{item.label} {text}")
            else:
                items.append(f"{text} {item.label}")

        available_width = max(1, width)
        indent = "  "
        separator = " · "
        lines: list[str] = []
        current_line = ""

        for item_text in items:
            if current_line:
                candidate = f"{current_line}{separator}{item_text}"
            elif visible_width(f"{indent}{item_text}") <= available_width:
                candidate = f"{indent}{item_text}"
            else:
                candidate = item_text

            if not current_line or visible_width(candidate) <= available_width:
                current_line = candidate
                continue

            lines.extend(wrap_text_with_ansi(current_line.rstrip(), available_width))
            current_line = (
                f"{indent}{item_text}" if visible_width(f"{indent}{item_text}") <= available_width else item_text
            )

        if current_line:
            lines.extend(wrap_text_with_ansi(current_line.rstrip(), available_width))

        return [theme.fg("muted", line) for line in lines]


class LabelInput(Component):
    def __init__(self, entry_id: str, current_label: str | None) -> None:
        self.entry_id = entry_id
        self.input = Input()
        if current_label:
            self.input.set_value(current_label)
        self.on_submit: Callable[[str, str | None], None] | None = None
        self.on_cancel: Callable[[], None] | None = None
        self._focused = False

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self.input.focused = value

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        indent = "  "
        available_width = width - len(indent)
        lines = [truncate_to_width(f"{indent}{theme.fg('muted', 'Label (empty to remove):')}", width)]
        lines.extend(truncate_to_width(f"{indent}{line}", width) for line in self.input.render(available_width))
        lines.append(
            truncate_to_width(
                f"{indent}{key_hint('tui.select.confirm', 'save')}  {key_hint('tui.select.cancel', 'cancel')}",
                width,
            )
        )
        return lines

    def handle_input(self, key_data: str) -> None:
        keybindings = get_keybindings()
        if keybindings.matches(key_data, "tui.select.confirm"):
            if self.on_submit is not None:
                value = self.input.get_value().strip()
                self.on_submit(self.entry_id, value or None)
        elif keybindings.matches(key_data, "tui.select.cancel"):
            if self.on_cancel is not None:
                self.on_cancel()
        else:
            self.input.handle_input(key_data)


class TreeSelectorComponent(Container):
    def __init__(
        self,
        tree: list[SessionTreeNode],
        current_leaf_id: str | None,
        terminal_height: int,
        on_select: Callable[[str], None],
        on_cancel: Callable[[], None],
        on_label_change: Callable[[str, str | None], None] | None = None,
        initial_selected_id: str | None = None,
        initial_filter_mode: FilterMode | None = None,
    ) -> None:
        super().__init__()
        self._on_label_change = on_label_change
        self.on_copy: Callable[[str | None], None] | None = None
        self.label_input: LabelInput | None = None
        self._focused = False

        max_visible_lines = max(5, math.floor(terminal_height / 2))
        self.tree_list = TreeList(tree, current_leaf_id, max_visible_lines, initial_selected_id, initial_filter_mode)
        self.tree_list.on_select = on_select
        self.tree_list.on_cancel = on_cancel
        self.tree_list.on_copy = self._forward_copy
        self.tree_list.on_label_edit = self._show_label_input

        self.tree_container = Container()
        self.tree_container.add_child(self.tree_list)
        self.label_input_container = Container()

        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())
        self.add_child(Text(theme.bold("  Session Tree"), 1, 0))
        self.add_child(TreeHelp())
        self.add_child(SearchLine(self.tree_list))
        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.add_child(self.tree_container)
        self.add_child(self.label_input_container)
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

        # TypeScript auto-cancels on a 100ms timer for an empty tree; the
        # caller checks `is_empty` instead so no event loop is needed here.
        self.is_empty = len(tree) == 0

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        if self.label_input is not None:
            self.label_input.focused = value

    def _forward_copy(self, text: str | None) -> None:
        if self.on_copy is not None:
            self.on_copy(text)

    def _show_label_input(self, entry_id: str, current_label: str | None) -> None:
        label_input = LabelInput(entry_id, current_label)

        def submit(submitted_id: str, label: str | None) -> None:
            self.tree_list.update_node_label(submitted_id, label)
            if self._on_label_change is not None:
                self._on_label_change(submitted_id, label)
            self._hide_label_input()

        label_input.on_submit = submit
        label_input.on_cancel = self._hide_label_input
        label_input.focused = self._focused

        self.label_input = label_input
        self.tree_container.clear()
        self.label_input_container.clear()
        self.label_input_container.add_child(label_input)

    def _hide_label_input(self) -> None:
        self.label_input = None
        self.label_input_container.clear()
        self.tree_container.clear()
        self.tree_container.add_child(self.tree_list)

    def handle_input(self, key_data: str) -> None:
        if self.label_input is not None:
            self.label_input.handle_input(key_data)
        else:
            self.tree_list.handle_input(key_data)

    def get_tree_list(self) -> TreeList:
        return self.tree_list


__all__ = [
    "FILTER_MODES",
    "FilterMode",
    "FlatNode",
    "GutterInfo",
    "LabelInput",
    "SearchLine",
    "TreeHelp",
    "TreeList",
    "TreeSelectorComponent",
    "compact_raw_keys",
    "format_tool_call",
    "render_horizontal_viewport",
]
