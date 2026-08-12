"""Session picker: threaded tree, search, rename and delete.

Ported from ``packages/coding-agent/src/modes/interactive/components/session-selector.ts``.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pi_tui.component import Component, Container
from pi_tui.components.input import Input
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text
from pi_tui.keybindings import get_keybindings
from pi_tui.tasks import spawn
from pi_tui.utils import truncate_to_width, visible_width

from ....utils.paths import canonicalize_path as _canonicalize_path
from ..theme.theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, key_text
from .session_selector_search import (
    NameFilter,
    SortMode,
    filter_and_sort_sessions,
    has_session_name,
)

if TYPE_CHECKING:
    from ....core.session_manager import SessionInfo

SessionScope = Literal["current", "all"]

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def shorten_path(path: str) -> str:
    if not path:
        return path
    home = str(Path.home())
    if path.startswith(home):
        return f"~{path[len(home) :]}"
    return path


def format_session_date(date: datetime, now: datetime | None = None) -> str:
    now = now or datetime.now(tz=date.tzinfo)
    diff_ms = (now - date).total_seconds() * 1000
    diff_mins = math.floor(diff_ms / 60000)
    diff_hours = math.floor(diff_ms / 3600000)
    diff_days = math.floor(diff_ms / 86400000)

    if diff_mins < 1:
        return "now"
    if diff_mins < 60:
        return f"{diff_mins}m"
    if diff_hours < 24:
        return f"{diff_hours}h"
    if diff_days < 7:
        return f"{diff_days}d"
    if diff_days < 30:
        return f"{math.floor(diff_days / 7)}w"
    if diff_days < 365:
        return f"{math.floor(diff_days / 30)}mo"
    return f"{math.floor(diff_days / 365)}y"


def _canonicalize(path: str | None) -> str | None:
    if not path:
        return path
    return _canonicalize_path(path)


@dataclass
class StatusMessage:
    type: Literal["info", "error"]
    message: str


class SessionSelectorHeader(Component):
    def __init__(
        self,
        scope: SessionScope,
        sort_mode: SortMode,
        name_filter: NameFilter,
        request_render: Callable[[], None],
    ) -> None:
        self.scope = scope
        self.sort_mode = sort_mode
        self.name_filter = name_filter
        self.request_render = request_render
        self.loading = False
        self.load_progress: tuple[int, int] | None = None
        self.show_path = False
        self.confirming_delete_path: str | None = None
        self.status_message: StatusMessage | None = None
        self.show_rename_hint = False
        self._status_task: asyncio.Task[None] | None = None

    def set_scope(self, scope: SessionScope) -> None:
        self.scope = scope

    def set_sort_mode(self, sort_mode: SortMode) -> None:
        self.sort_mode = sort_mode

    def set_name_filter(self, name_filter: NameFilter) -> None:
        self.name_filter = name_filter

    def set_loading(self, loading: bool) -> None:
        self.loading = loading
        # Progress belongs to one load; clear it whenever loading state changes.
        self.load_progress = None

    def set_progress(self, loaded: int, total: int) -> None:
        self.load_progress = (loaded, total)

    def set_show_path(self, show_path: bool) -> None:
        self.show_path = show_path

    def set_show_rename_hint(self, show: bool) -> None:
        self.show_rename_hint = show

    def set_confirming_delete_path(self, path: str | None) -> None:
        self.confirming_delete_path = path

    def _clear_status_task(self) -> None:
        if self._status_task is not None and not self._status_task.done():
            self._status_task.cancel()
        self._status_task = None

    def set_status_message(self, message: StatusMessage | None, auto_hide_ms: int | None = None) -> None:
        self._clear_status_task()
        self.status_message = message
        if message is None or not auto_hide_ms:
            return

        async def hide() -> None:
            await asyncio.sleep(auto_hide_ms / 1000)
            self.status_message = None
            self._status_task = None
            self.request_render()

        try:
            self._status_task = spawn(hide())
        except RuntimeError:
            # No running loop (e.g. constructed outside the TUI); the message
            # simply stays until it is replaced.
            self._status_task = None

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        title = "Resume Session (Current Folder)" if self.scope == "current" else "Resume Session (All)"
        left_text = theme.bold(title)

        sort_label = {"threaded": "Threaded", "recent": "Recent"}.get(self.sort_mode, "Fuzzy")
        sort_text = theme.fg("muted", "Sort: ") + theme.fg("accent", sort_label)

        name_label = "All" if self.name_filter == "all" else "Named"
        name_text = theme.fg("muted", "Name: ") + theme.fg("accent", name_label)

        if self.loading:
            progress_text = f"{self.load_progress[0]}/{self.load_progress[1]}" if self.load_progress else "..."
            scope_text = theme.fg("muted", "○ Current Folder | ") + theme.fg("accent", f"Loading {progress_text}")
        elif self.scope == "current":
            scope_text = theme.fg("accent", "◉ Current Folder") + theme.fg("muted", " | ○ All")
        else:
            scope_text = theme.fg("muted", "○ Current Folder | ") + theme.fg("accent", "◉ All")

        right_text = truncate_to_width(f"{scope_text}  {name_text}  {sort_text}", width, "")
        available_left = max(0, width - visible_width(right_text) - 1)
        left = truncate_to_width(left_text, available_left, "")
        spacing = max(0, width - visible_width(left) - visible_width(right_text))

        if self.confirming_delete_path is not None:
            confirm_hint = (
                f"Delete session? {key_hint('tui.select.confirm', 'confirm')} · "
                f"{key_hint('tui.select.cancel', 'cancel')}"
            )
            hint_line_1 = theme.fg("error", truncate_to_width(confirm_hint, width, "…"))
            hint_line_2 = ""
        elif self.status_message is not None:
            color = "error" if self.status_message.type == "error" else "accent"
            hint_line_1 = theme.fg(color, truncate_to_width(self.status_message.message, width, "…"))
            hint_line_2 = ""
        else:
            path_state = "(on)" if self.show_path else "(off)"
            separator = theme.fg("muted", " · ")
            hint1 = (
                key_hint("tui.input.tab", "scope")
                + separator
                + theme.fg("muted", 're:<pattern> regex · "phrase" exact')
            )
            hint2_parts = [
                key_hint("app.session.toggleSort", "sort"),
                key_hint("app.session.toggleNamedFilter", "named"),
                key_hint("app.session.delete", "delete"),
                key_hint("app.session.togglePath", f"path {path_state}"),
            ]
            if self.show_rename_hint:
                hint2_parts.append(key_hint("app.session.rename", "rename"))
            hint_line_1 = truncate_to_width(hint1, width, "…")
            hint_line_2 = truncate_to_width(separator.join(hint2_parts), width, "…")

        return [f"{left}{' ' * spacing}{right_text}", hint_line_1, hint_line_2]


@dataclass
class SessionTreeNode:
    session: SessionInfo
    children: list[SessionTreeNode] = field(default_factory=list)
    latest_activity: float = 0.0


@dataclass
class FlatSessionNode:
    session: SessionInfo
    depth: int
    is_last: bool
    ancestor_continues: list[bool] = field(default_factory=list)


def build_session_tree(sessions: list[SessionInfo]) -> list[SessionTreeNode]:
    """Group sessions by ``parent_session_path``, newest subtree first."""
    by_path: dict[str, SessionTreeNode] = {}
    for session in sessions:
        session_path = _canonicalize(session.path) or session.path
        by_path[session_path] = SessionTreeNode(session=session, latest_activity=session.modified.timestamp())

    roots: list[SessionTreeNode] = []
    for session in sessions:
        session_path = _canonicalize(session.path) or session.path
        node = by_path[session_path]
        parent_path = _canonicalize(session.parent_session_path)
        if parent_path and parent_path in by_path:
            by_path[parent_path].children.append(node)
        else:
            roots.append(node)

    def update_latest_activity(node: SessionTreeNode) -> float:
        latest = node.session.modified.timestamp()
        for child in node.children:
            latest = max(latest, update_latest_activity(child))
        node.latest_activity = latest
        return latest

    for root in roots:
        update_latest_activity(root)

    def sort_nodes(nodes: list[SessionTreeNode]) -> None:
        nodes.sort(key=lambda node: -node.latest_activity)
        for node in nodes:
            sort_nodes(node.children)

    sort_nodes(roots)
    return roots


def flatten_session_tree(roots: list[SessionTreeNode]) -> list[FlatSessionNode]:
    result: list[FlatSessionNode] = []

    def walk(node: SessionTreeNode, depth: int, ancestor_continues: list[bool], is_last: bool) -> None:
        result.append(
            FlatSessionNode(session=node.session, depth=depth, is_last=is_last, ancestor_continues=ancestor_continues)
        )
        for index, child in enumerate(node.children):
            child_is_last = index == len(node.children) - 1
            # Continuation lines are only drawn for non-root ancestors.
            continues = (not is_last) if depth > 0 else False
            walk(child, depth + 1, [*ancestor_continues, continues], child_is_last)

    for index, root in enumerate(roots):
        walk(root, 0, [], index == len(roots) - 1)

    return result


@dataclass
class DeleteResult:
    ok: bool
    method: Literal["trash", "unlink"]
    error: str | None = None


async def delete_session_file(session_path: str) -> DeleteResult:
    """Try the ``trash`` CLI first, then fall back to unlinking."""
    trash_args = ["--", session_path] if session_path.startswith("-") else [session_path]
    trash_error: str | None = None
    trash_status: int | None = None
    if shutil.which("trash") is not None:
        try:
            completed = subprocess.run(
                ["trash", *trash_args],
                capture_output=True,
                check=False,
            )
            trash_status = completed.returncode
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            if stderr:
                trash_error = stderr.split("\n")[0]
        except OSError as error:
            trash_error = str(error)
    else:
        trash_error = "trash: command not found"

    if trash_status == 0 or not os.path.exists(session_path):
        return DeleteResult(ok=True, method="trash")

    try:
        os.unlink(session_path)
    except OSError as error:
        hint = f"trash: {trash_error[:200]}" if trash_error else None
        message = f"{error} ({hint})" if hint else str(error)
        return DeleteResult(ok=False, method="unlink", error=message)
    return DeleteResult(ok=True, method="unlink")


class SessionList(Component):
    """Multi-line session list with an embedded search input."""

    MAX_VISIBLE = 10

    def __init__(
        self,
        sessions: list[SessionInfo],
        show_cwd: bool,
        sort_mode: SortMode,
        name_filter: NameFilter,
        keybindings: object,
        current_session_file_path: str | None = None,
    ) -> None:
        self.all_sessions = sessions
        self.filtered_sessions: list[FlatSessionNode] = []
        self.selected_index = 0
        self.search_input = Input()
        self.show_cwd = show_cwd
        self.sort_mode = sort_mode
        self.name_filter = name_filter
        self.keybindings = keybindings
        self.show_path = False
        self.confirming_delete_path: str | None = None
        self.current_session_canonical_path = _canonicalize(current_session_file_path)
        self.max_visible = self.MAX_VISIBLE
        self._focused = False

        self.on_select: Callable[[str], None] | None = None
        self.on_cancel: Callable[[], None] | None = None
        self.on_exit: Callable[[], None] = lambda: None
        self.on_toggle_scope: Callable[[], None] | None = None
        self.on_toggle_sort: Callable[[], None] | None = None
        self.on_toggle_name_filter: Callable[[], None] | None = None
        self.on_toggle_path: Callable[[bool], None] | None = None
        self.on_delete_confirmation_change: Callable[[str | None], None] | None = None
        self.on_delete_session: Callable[[str], Awaitable[None]] | None = None
        self.on_rename_session: Callable[[str], None] | None = None
        self.on_error: Callable[[str], None] | None = None

        self._filter_sessions("")
        self.search_input.on_submit = self._submit_selected

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self.search_input.focused = value

    def get_selected_session_path(self) -> str | None:
        if 0 <= self.selected_index < len(self.filtered_sessions):
            return self.filtered_sessions[self.selected_index].session.path
        return None

    def _submit_selected(self, _value: str = "") -> None:
        if self.on_select is not None and 0 <= self.selected_index < len(self.filtered_sessions):
            self.on_select(self.filtered_sessions[self.selected_index].session.path)

    def set_sort_mode(self, sort_mode: SortMode) -> None:
        self.sort_mode = sort_mode
        self._filter_sessions(self.search_input.get_value())

    def set_name_filter(self, name_filter: NameFilter) -> None:
        self.name_filter = name_filter
        self._filter_sessions(self.search_input.get_value())

    def set_sessions(self, sessions: list[SessionInfo], show_cwd: bool) -> None:
        self.all_sessions = sessions
        self.show_cwd = show_cwd
        self._filter_sessions(self.search_input.get_value())

    def _filter_sessions(self, query: str) -> None:
        trimmed = query.strip()
        name_filtered = (
            self.all_sessions
            if self.name_filter == "all"
            else [session for session in self.all_sessions if has_session_name(session)]
        )

        if self.sort_mode == "threaded" and not trimmed:
            self.filtered_sessions = flatten_session_tree(build_session_tree(name_filtered))
        else:
            filtered = filter_and_sort_sessions(name_filtered, query, self.sort_mode, "all")
            self.filtered_sessions = [FlatSessionNode(session=session, depth=0, is_last=True) for session in filtered]
        self.selected_index = min(self.selected_index, max(0, len(self.filtered_sessions) - 1))

    def _set_confirming_delete_path(self, path: str | None) -> None:
        self.confirming_delete_path = path
        if self.on_delete_confirmation_change is not None:
            self.on_delete_confirmation_change(path)

    def _start_delete_confirmation(self) -> None:
        if not (0 <= self.selected_index < len(self.filtered_sessions)):
            return
        selected = self.filtered_sessions[self.selected_index]
        if self._is_current_session_path(selected.session.path):
            if self.on_error is not None:
                self.on_error("Cannot delete the currently active session")
            return
        self._set_confirming_delete_path(selected.session.path)

    def _is_current_session_path(self, path: str) -> bool:
        if not self.current_session_canonical_path:
            return False
        return (_canonicalize(path) or path) == self.current_session_canonical_path

    def invalidate(self) -> None:
        return None

    def _empty_message(self) -> str:
        if self.name_filter == "named":
            toggle_key = key_text("app.session.toggleNamedFilter")
            if self.show_cwd:
                return f"  No named sessions found. Press {toggle_key} to show all."
            return f"  No named sessions in current folder. Press {toggle_key} to show all, or Tab to view all."
        if self.show_cwd:
            return "  No sessions found"
        return "  No sessions in current folder. Press Tab to view all."

    def _build_tree_prefix(self, node: FlatSessionNode) -> str:
        if node.depth == 0:
            return ""
        parts = ["│  " if continues else "   " for continues in node.ancestor_continues]
        branch = "└─ " if node.is_last else "├─ "
        return "".join(parts) + branch

    def render(self, width: int) -> list[str]:
        lines: list[str] = []
        lines.extend(self.search_input.render(width))
        lines.append("")

        if len(self.filtered_sessions) == 0:
            lines.append(theme.fg("muted", truncate_to_width(self._empty_message(), width, "…")))
            return lines

        start_index = max(
            0,
            min(
                self.selected_index - math.floor(self.max_visible / 2),
                len(self.filtered_sessions) - self.max_visible,
            ),
        )
        end_index = min(start_index + self.max_visible, len(self.filtered_sessions))

        for index in range(start_index, end_index):
            node = self.filtered_sessions[index]
            session = node.session
            is_selected = index == self.selected_index
            is_confirming_delete = session.path == self.confirming_delete_path
            is_current = self._is_current_session_path(session.path)

            prefix = self._build_tree_prefix(node)
            has_name = bool(session.name)
            display_text = session.name if session.name else session.first_message
            normalized_message = _CONTROL_CHARS_RE.sub(" ", display_text).strip()

            right_part = f"{session.message_count} {format_session_date(session.modified)}"
            if self.show_cwd and session.cwd:
                right_part = f"{shorten_path(session.cwd)} {right_part}"
            if self.show_path:
                right_part = f"{shorten_path(session.path)} {right_part}"

            cursor = theme.fg("accent", "› ") if is_selected else "  "
            available_for_message = width - 2 - visible_width(prefix) - (visible_width(right_part) + 2)
            truncated_message = truncate_to_width(normalized_message, max(10, available_for_message), "…")

            message_color: str | None = None
            if is_confirming_delete:
                message_color = "error"
            elif is_current:
                message_color = "accent"
            elif has_name:
                message_color = "warning"
            styled_message = theme.fg(message_color, truncated_message) if message_color else truncated_message
            if is_selected:
                styled_message = theme.bold(styled_message)

            left_part = cursor + theme.fg("dim", prefix) + styled_message
            spacing = max(1, width - visible_width(left_part) - visible_width(right_part))
            styled_right = theme.fg("error" if is_confirming_delete else "dim", right_part)

            line = left_part + " " * spacing + styled_right
            if is_selected:
                line = theme.bg("selectedBg", line)
            lines.append(truncate_to_width(line, width))

        if start_index > 0 or end_index < len(self.filtered_sessions):
            scroll_text = f"  ({self.selected_index + 1}/{len(self.filtered_sessions)})"
            lines.append(theme.fg("muted", truncate_to_width(scroll_text, width, "")))

        return lines

    def handle_input(self, key_data: str) -> None:
        keybindings = get_keybindings()

        # Delete confirmation swallows every other key.
        if self.confirming_delete_path is not None:
            if keybindings.matches(key_data, "tui.select.confirm"):
                path_to_delete = self.confirming_delete_path
                self._set_confirming_delete_path(None)
                if self.on_delete_session is not None:
                    spawn(self.on_delete_session(path_to_delete))
            elif keybindings.matches(key_data, "tui.select.cancel"):
                self._set_confirming_delete_path(None)
            return

        if keybindings.matches(key_data, "tui.input.tab"):
            if self.on_toggle_scope is not None:
                self.on_toggle_scope()
            return

        if keybindings.matches(key_data, "app.session.toggleSort"):
            if self.on_toggle_sort is not None:
                self.on_toggle_sort()
            return

        if self.keybindings.matches(key_data, "app.session.toggleNamedFilter"):  # type: ignore[attr-defined]
            if self.on_toggle_name_filter is not None:
                self.on_toggle_name_filter()
            return

        if keybindings.matches(key_data, "app.session.togglePath"):
            self.show_path = not self.show_path
            if self.on_toggle_path is not None:
                self.on_toggle_path(self.show_path)
            return

        if keybindings.matches(key_data, "app.session.delete"):
            self._start_delete_confirmation()
            return

        if keybindings.matches(key_data, "app.session.rename"):
            if self.on_rename_session is not None and 0 <= self.selected_index < len(self.filtered_sessions):
                self.on_rename_session(self.filtered_sessions[self.selected_index].session.path)
            return

        # A convenience alias that only deletes when the query is empty;
        # otherwise it is a normal editing key for the search input.
        if keybindings.matches(key_data, "app.session.deleteNoninvasive"):
            if len(self.search_input.get_value()) > 0:
                self.search_input.handle_input(key_data)
                self._filter_sessions(self.search_input.get_value())
                return
            self._start_delete_confirmation()
            return

        if keybindings.matches(key_data, "tui.select.up"):
            self.selected_index = max(0, self.selected_index - 1)
        elif keybindings.matches(key_data, "tui.select.down"):
            self.selected_index = min(len(self.filtered_sessions) - 1, self.selected_index + 1)
        elif keybindings.matches(key_data, "tui.select.pageUp"):
            self.selected_index = max(0, self.selected_index - self.max_visible)
        elif keybindings.matches(key_data, "tui.select.pageDown"):
            self.selected_index = min(len(self.filtered_sessions) - 1, self.selected_index + self.max_visible)
        elif keybindings.matches(key_data, "tui.select.confirm"):
            self._submit_selected()
        elif keybindings.matches(key_data, "tui.select.cancel"):
            if self.on_cancel is not None:
                self.on_cancel()
        else:
            self.search_input.handle_input(key_data)
            self._filter_sessions(self.search_input.get_value())


SessionsLoader = Callable[..., Awaitable[list["SessionInfo"]]]


class SessionSelectorComponent(Container):
    def __init__(
        self,
        current_sessions_loader: SessionsLoader,
        all_sessions_loader: SessionsLoader,
        on_select: Callable[[str], None],
        on_cancel: Callable[[], None],
        on_exit: Callable[[], None],
        request_render: Callable[[], None],
        keybindings: object,
        rename_session: Callable[[str, str | None], Awaitable[None]] | None = None,
        show_rename_hint: bool | None = None,
        current_session_file_path: str | None = None,
    ) -> None:
        super().__init__()
        self.keybindings = keybindings
        self.current_sessions_loader = current_sessions_loader
        self.all_sessions_loader = all_sessions_loader
        self.request_render = request_render
        self.scope: SessionScope = "current"
        self.sort_mode: SortMode = "threaded"
        self.name_filter: NameFilter = "all"
        self.current_sessions: list[SessionInfo] | None = None
        self.all_sessions: list[SessionInfo] | None = None
        self.current_loading = False
        self.all_loading = False
        self.all_load_seq = 0
        self.mode: Literal["list", "rename"] = "list"
        self.rename_input = Input()
        self.rename_target_path: str | None = None
        self.rename_session = rename_session
        self.can_rename = rename_session is not None
        self._focused = False

        self.header = SessionSelectorHeader(self.scope, self.sort_mode, self.name_filter, request_render)
        self.header.set_show_rename_hint(self.can_rename if show_rename_hint is None else show_rename_hint)

        self.session_list = SessionList(
            [], False, self.sort_mode, self.name_filter, keybindings, current_session_file_path
        )
        self._build_base_layout(self.session_list)

        self.rename_input.on_submit = lambda value: spawn(self._confirm_rename(value))

        def clear_status() -> None:
            self.header.set_status_message(None)

        def select(session_path: str) -> None:
            clear_status()
            on_select(session_path)

        def cancel() -> None:
            clear_status()
            on_cancel()

        def exit_selector() -> None:
            clear_status()
            on_exit()

        self.session_list.on_select = select
        self.session_list.on_cancel = cancel
        self.session_list.on_exit = exit_selector
        self.session_list.on_toggle_scope = self._toggle_scope
        self.session_list.on_toggle_sort = self._toggle_sort_mode
        self.session_list.on_toggle_name_filter = self._toggle_name_filter
        self.session_list.on_rename_session = self._request_rename
        self.session_list.on_toggle_path = self._on_toggle_path
        self.session_list.on_delete_confirmation_change = self._on_delete_confirmation_change
        self.session_list.on_error = self._on_error
        self.session_list.on_delete_session = self._delete_session

        # Start loading current sessions immediately.
        self.load_current_sessions()

    # -- layout -------------------------------------------------------------

    def _build_base_layout(self, content: Component, show_header: bool = True) -> None:
        self.clear()
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder(lambda s: theme.fg("accent", s)))
        self.add_child(Spacer(1))
        if show_header:
            self.add_child(self.header)
            self.add_child(Spacer(1))
        self.add_child(content)
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder(lambda s: theme.fg("accent", s)))

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self.session_list.focused = value
        self.rename_input.focused = value or self.mode == "rename"

    def handle_input(self, data: str) -> None:
        if self.mode == "rename":
            if get_keybindings().matches(data, "tui.select.cancel"):
                self._exit_rename_mode()
                return
            self.rename_input.handle_input(data)
            return
        self.session_list.handle_input(data)

    def get_session_list(self) -> SessionList:
        return self.session_list

    # -- callbacks ----------------------------------------------------------

    def _on_toggle_path(self, show_path: bool) -> None:
        self.header.set_show_path(show_path)
        self.request_render()

    def _on_delete_confirmation_change(self, path: str | None) -> None:
        self.header.set_confirming_delete_path(path)
        self.request_render()

    def _on_error(self, message: str) -> None:
        self.header.set_status_message(StatusMessage(type="error", message=message), 3000)
        self.request_render()

    def _request_rename(self, session_path: str) -> None:
        if self.rename_session is None:
            return
        if self.scope == "current" and self.current_loading:
            return
        if self.scope == "all" and self.all_loading:
            return
        sessions = (self.all_sessions or []) if self.scope == "all" else (self.current_sessions or [])
        session = next((s for s in sessions if s.path == session_path), None)
        self._enter_rename_mode(session_path, session.name if session else None)

    async def _delete_session(self, session_path: str) -> None:
        result = await delete_session_file(session_path)

        if result.ok:
            if self.current_sessions is not None:
                self.current_sessions = [s for s in self.current_sessions if s.path != session_path]
            if self.all_sessions is not None:
                self.all_sessions = [s for s in self.all_sessions if s.path != session_path]

            sessions = (self.all_sessions or []) if self.scope == "all" else (self.current_sessions or [])
            self.session_list.set_sessions(sessions, self.scope == "all")

            message = "Session moved to trash" if result.method == "trash" else "Session deleted"
            self.header.set_status_message(StatusMessage(type="info", message=message), 2000)
            await self.load_scope(self.scope, "refresh")
        else:
            self.header.set_status_message(
                StatusMessage(type="error", message=f"Failed to delete: {result.error or 'Unknown error'}"),
                3000,
            )

        self.request_render()

    # -- rename -------------------------------------------------------------

    def _enter_rename_mode(self, session_path: str, current_name: str | None) -> None:
        self.mode = "rename"
        self.rename_target_path = session_path
        self.rename_input.set_value(current_name or "")
        self.rename_input.focused = True

        panel = Container()
        panel.add_child(Text(theme.bold("Rename Session"), 1, 0))
        panel.add_child(Spacer(1))
        panel.add_child(self.rename_input)
        panel.add_child(Spacer(1))
        panel.add_child(
            Text(
                theme.fg(
                    "muted",
                    f"{key_text('tui.select.confirm')} to save · {key_text('tui.select.cancel')} to cancel",
                ),
                1,
                0,
            )
        )
        self._build_base_layout(panel, show_header=False)
        self.request_render()

    def _exit_rename_mode(self) -> None:
        self.mode = "list"
        self.rename_target_path = None
        self._build_base_layout(self.session_list)
        self.request_render()

    async def _confirm_rename(self, value: str) -> None:
        next_name = value.strip()
        if not next_name:
            return
        target = self.rename_target_path
        if not target or self.rename_session is None:
            self._exit_rename_mode()
            return
        try:
            await self.rename_session(target, next_name)
            await self.load_scope(self.scope, "refresh")
        finally:
            self._exit_rename_mode()

    # -- loading ------------------------------------------------------------

    def load_current_sessions(self) -> None:
        self.start_scope_load("current", "initial")

    def start_scope_load(self, scope: SessionScope, reason: Literal["initial", "refresh", "toggle"]) -> None:
        """Fire-and-forget scope load whose bookkeeping applies immediately.

        TypeScript's `void this.loadScope(...)` runs the body up to its first
        `await` synchronously, so `allLoading`/the header scope are already
        updated when the next keystroke arrives. `spawn()` defers the whole
        coroutine, so the synchronous prologue is split out and run here;
        otherwise rapid scope toggling starts redundant loads and leaves the
        header showing the wrong scope.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No event loop yet (component built outside the TUI); the caller
            # can re-trigger the load once the loop is running.
            return
        seq = self._begin_scope_load(scope)
        spawn(self._run_scope_load(scope, reason, seq))

    async def load_scope(self, scope: SessionScope, reason: Literal["initial", "refresh", "toggle"]) -> None:
        await self._run_scope_load(scope, reason, self._begin_scope_load(scope))

    def _begin_scope_load(self, scope: SessionScope) -> int | None:
        if scope == "current":
            self.current_loading = True
            seq: int | None = None
        else:
            self.all_loading = True
            self.all_load_seq += 1
            seq = self.all_load_seq

        self.header.set_scope(scope)
        self.header.set_loading(True)
        self.request_render()
        return seq

    async def _run_scope_load(
        self,
        scope: SessionScope,
        reason: Literal["initial", "refresh", "toggle"],
        seq: int | None,
    ) -> None:
        show_cwd = scope == "all"

        def on_progress(loaded: int, total: int) -> None:
            if scope != self.scope:
                return
            if seq is not None and seq != self.all_load_seq:
                return
            self.header.set_progress(loaded, total)
            self.request_render()

        try:
            loader = self.current_sessions_loader if scope == "current" else self.all_sessions_loader
            sessions = await loader(on_progress)
        except Exception as error:
            if scope == "current":
                self.current_loading = False
            else:
                self.all_loading = False
            if scope != self.scope or (seq is not None and seq != self.all_load_seq):
                return
            self.header.set_loading(False)
            self.header.set_status_message(
                StatusMessage(type="error", message=f"Failed to load sessions: {error}"), 4000
            )
            if reason == "initial":
                self.session_list.set_sessions([], show_cwd)
            self.request_render()
            return

        if scope == "current":
            self.current_sessions = sessions
            self.current_loading = False
        else:
            self.all_sessions = sessions
            self.all_loading = False

        if scope != self.scope or (seq is not None and seq != self.all_load_seq):
            return

        self.header.set_loading(False)
        self.session_list.set_sessions(sessions, show_cwd)
        self.request_render()

    # -- toggles ------------------------------------------------------------

    def _toggle_sort_mode(self) -> None:
        # threaded -> recent -> relevance -> threaded
        self.sort_mode = (
            "recent" if self.sort_mode == "threaded" else "relevance" if self.sort_mode == "recent" else "threaded"
        )
        self.header.set_sort_mode(self.sort_mode)
        self.session_list.set_sort_mode(self.sort_mode)
        self.request_render()

    def _toggle_name_filter(self) -> None:
        self.name_filter = "named" if self.name_filter == "all" else "all"
        self.header.set_name_filter(self.name_filter)
        self.session_list.set_name_filter(self.name_filter)
        self.request_render()

    def _toggle_scope(self) -> None:
        if self.scope == "current":
            self.scope = "all"
            self.header.set_scope(self.scope)
            if self.all_sessions is not None:
                self.header.set_loading(False)
                self.session_list.set_sessions(self.all_sessions, True)
                self.request_render()
                return
            if not self.all_loading:
                self.start_scope_load("all", "toggle")
            return

        self.scope = "current"
        self.header.set_scope(self.scope)
        self.header.set_loading(self.current_loading)
        self.session_list.set_sessions(self.current_sessions or [], False)
        self.request_render()


__all__ = [
    "DeleteResult",
    "FlatSessionNode",
    "SessionList",
    "SessionSelectorComponent",
    "SessionSelectorHeader",
    "SessionTreeNode",
    "build_session_tree",
    "delete_session_file",
    "flatten_session_tree",
    "format_session_date",
    "shorten_path",
]
