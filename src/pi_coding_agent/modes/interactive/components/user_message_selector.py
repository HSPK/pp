"""Fork-from-message selector.

Ported from ``packages/coding-agent/src/modes/interactive/components/user-message-selector.ts``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from pi_tui.component import Component, Container
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text
from pi_tui.keybindings import get_keybindings
from pi_tui.utils import truncate_to_width

from ..theme.theme import theme
from .dynamic_border import DynamicBorder


@dataclass
class UserMessageItem:
    id: str
    """Entry ID in the session."""
    text: str
    timestamp: str | None = None


class UserMessageList(Component):
    """Scrolling list of user messages with a two-line entry per message."""

    MAX_VISIBLE = 10

    def __init__(self, messages: list[UserMessageItem], initial_selected_id: str | None = None) -> None:
        # Chronological order, oldest first.
        self.messages = messages
        self.on_select: Callable[[str], None] | None = None
        self.on_cancel: Callable[[], None] | None = None
        self.max_visible = self.MAX_VISIBLE

        initial_index = -1
        if initial_selected_id is not None:
            initial_index = next((i for i, message in enumerate(messages) if message.id == initial_selected_id), -1)
        self.selected_index = initial_index if initial_index >= 0 else max(0, len(messages) - 1)

    def invalidate(self) -> None:
        return None

    def render(self, width: int) -> list[str]:
        if len(self.messages) == 0:
            return [theme.fg("muted", "  No user messages found")]

        lines: list[str] = []
        start_index = max(
            0,
            min(
                self.selected_index - math.floor(self.max_visible / 2),
                len(self.messages) - self.max_visible,
            ),
        )
        end_index = min(start_index + self.max_visible, len(self.messages))

        for index in range(start_index, end_index):
            message = self.messages[index]
            is_selected = index == self.selected_index

            normalized_message = message.text.replace("\n", " ").strip()
            cursor = theme.fg("accent", "› ") if is_selected else "  "
            truncated = truncate_to_width(normalized_message, width - 2)
            lines.append(cursor + (theme.bold(truncated) if is_selected else truncated))
            lines.append(theme.fg("muted", f"  Message {index + 1} of {len(self.messages)}"))
            lines.append("")

        if start_index > 0 or end_index < len(self.messages):
            lines.append(theme.fg("muted", f"  ({self.selected_index + 1}/{len(self.messages)})"))

        return lines

    def handle_input(self, key_data: str) -> None:
        keybindings = get_keybindings()
        if keybindings.matches(key_data, "tui.select.up"):
            self.selected_index = len(self.messages) - 1 if self.selected_index == 0 else self.selected_index - 1
        elif keybindings.matches(key_data, "tui.select.down"):
            self.selected_index = 0 if self.selected_index == len(self.messages) - 1 else self.selected_index + 1
        elif keybindings.matches(key_data, "tui.select.confirm"):
            if self.on_select is not None and 0 <= self.selected_index < len(self.messages):
                self.on_select(self.messages[self.selected_index].id)
        elif keybindings.matches(key_data, "tui.select.cancel") and self.on_cancel is not None:
            self.on_cancel()


class UserMessageSelectorComponent(Container):
    def __init__(
        self,
        messages: list[UserMessageItem],
        on_select: Callable[[str], None],
        on_cancel: Callable[[], None],
        initial_selected_id: str | None = None,
    ) -> None:
        super().__init__()

        self.add_child(Spacer(1))
        self.add_child(Text(theme.bold("Fork from Message"), 1, 0))
        self.add_child(
            Text(
                theme.fg(
                    "muted",
                    "Select a user message to copy the active path up to that point into a new session",
                ),
                1,
                0,
            )
        )
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))

        self.message_list = UserMessageList(messages, initial_selected_id)
        self.message_list.on_select = on_select
        self.message_list.on_cancel = on_cancel
        self.add_child(self.message_list)

        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

        # TypeScript schedules the auto-cancel on a 100ms timer so the caller
        # can finish mounting first; here the caller checks `is_empty` instead,
        # which avoids a timer that would need an event loop at construction.
        self.is_empty = len(messages) == 0

    def get_message_list(self) -> UserMessageList:
        return self.message_list

    def handle_input(self, key_data: str) -> None:
        self.message_list.handle_input(key_data)


__all__ = ["UserMessageItem", "UserMessageList", "UserMessageSelectorComponent"]
