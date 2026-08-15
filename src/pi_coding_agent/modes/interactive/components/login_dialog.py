"""OAuth/API-key login dialog, shown in place of the editor.

Ported from ``packages/coding-agent/src/modes/interactive/components/login-dialog.ts``.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Sequence
from typing import Any

from pi_tui.component import Container
from pi_tui.components.input import Input
from pi_tui.components.spacer import Spacer
from pi_tui.components.text import Text
from pi_tui.keybindings import get_keybindings

from ....utils.open_browser import open_browser
from ..theme.theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint


class LoginCancelledError(Exception):
    """Raised into a pending prompt when the user cancels the dialog."""


class _AbortController:
    """Minimal stand-in for the DOM ``AbortController`` used by the TS source."""

    def __init__(self) -> None:
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


def _hyperlink(url: str, text: str) -> str:
    return f"\x1b]8;;{url}\x07{text}\x1b]8;;\x07"


def _click_hint() -> str:
    return "Cmd+click to open" if sys.platform == "darwin" else "Ctrl+click to open"


class LoginDialogComponent(Container):
    def __init__(
        self,
        tui: Any,
        provider_id: str,
        on_complete: Callable[..., None],
        provider_name_override: str | None = None,
        title_override: str | None = None,
    ) -> None:
        super().__init__()
        self.tui = tui
        self.on_complete = on_complete
        self._abort_controller = _AbortController()
        self._input_future: asyncio.Future[str] | None = None
        self._focused = False

        provider_name = provider_name_override or provider_id
        title = title_override if title_override is not None else f"Login to {provider_name}"

        self.add_child(DynamicBorder())
        self.add_child(Text(theme.fg("accent", theme.bold(title)), 1, 0))

        self.content_container = Container()
        self.add_child(self.content_container)

        # The input is always constructed and swapped into the content area
        # whenever a step needs it.
        self.input = Input()
        self.input.on_submit = self._submit_input
        self.input.on_escape = self.cancel

        self.add_child(DynamicBorder())

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self.input.focused = value

    @property
    def signal(self) -> _AbortController:
        return self._abort_controller

    def _request_render(self) -> None:
        if self.tui is not None:
            self.tui.request_render()

    def _submit_input(self, _value: str = "") -> None:
        if self._input_future is None or self._input_future.done():
            return
        value = self.input.get_value()
        self._replace_input_with_submitted_text(value)
        future = self._input_future
        self._input_future = None
        future.set_result(value)

    def _replace_input_with_submitted_text(self, value: str) -> None:
        self.content_container.children = [
            Text(f"> {value}", 0, 0) if child is self.input else child for child in self.content_container.children
        ]

    def cancel(self) -> None:
        self._abort_controller.abort()
        if self._input_future is not None and not self._input_future.done():
            future = self._input_future
            self._input_future = None
            future.set_exception(LoginCancelledError("Login cancelled"))
        self.on_complete(False, "Login cancelled")

    def _pending_input(self) -> asyncio.Future[str]:
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        self._input_future = future
        return future

    # -- flow steps ---------------------------------------------------------

    def show_auth(self, url: str, instructions: str | None = None) -> None:
        """``onAuth`` callback: show the URL and open the browser."""
        self.content_container.clear()
        self.content_container.add_child(Spacer(1))
        self.content_container.add_child(Text(theme.fg("accent", _hyperlink(url, url)), 1, 0))
        self.content_container.add_child(Text(theme.fg("dim", _hyperlink(url, _click_hint())), 1, 0))
        if instructions:
            self.content_container.add_child(Spacer(1))
            self.content_container.add_child(Text(theme.fg("warning", instructions), 1, 0))
        open_browser(url)
        self._request_render()

    def show_device_code(self, info: Any) -> None:
        """``onDeviceCode`` callback: show the verification URL and user code."""
        verification_uri = info["verificationUri"] if isinstance(info, dict) else getattr(info, "verification_uri", "")
        user_code = info["userCode"] if isinstance(info, dict) else getattr(info, "user_code", "")

        self.content_container.clear()
        self.content_container.add_child(Spacer(1))
        self.content_container.add_child(Text(theme.fg("accent", _hyperlink(verification_uri, verification_uri)), 1, 0))
        self.content_container.add_child(Text(theme.fg("dim", _hyperlink(verification_uri, _click_hint())), 1, 0))
        self.content_container.add_child(Spacer(1))
        self.content_container.add_child(Text(theme.fg("warning", f"Enter code: {user_code}"), 1, 0))
        self._request_render()

    def show_manual_input(self, prompt: str) -> asyncio.Future[str]:
        """Manual code/URL entry, for providers with a callback server."""
        self.input.set_value("")
        self.content_container.add_child(Spacer(1))
        self.content_container.add_child(Text(theme.fg("dim", prompt), 1, 0))
        self.content_container.add_child(self.input)
        self.content_container.add_child(Text(f"({key_hint('tui.select.cancel', 'to cancel')})", 1, 0))
        self._request_render()
        return self._pending_input()

    def show_prompt(self, message: str, placeholder: str | None = None) -> asyncio.Future[str]:
        """``onPrompt`` callback. Appends rather than clearing, so the URL from
        ``show_auth`` stays visible."""
        self.content_container.add_child(Spacer(1))
        self.content_container.add_child(Text(theme.fg("text", message), 1, 0))
        if placeholder:
            self.content_container.add_child(Text(theme.fg("dim", f"e.g., {placeholder}"), 1, 0))
        self.content_container.add_child(self.input)
        self.content_container.add_child(
            Text(
                f"({key_hint('tui.select.cancel', 'to cancel,')} {key_hint('tui.select.confirm', 'to submit')})",
                1,
                0,
            )
        )
        self.input.set_value("")
        self._request_render()
        return self._pending_input()

    def show_details(self, lines: Sequence[str]) -> None:
        self.content_container.clear()
        self.content_container.add_child(Spacer(1))
        for line in lines:
            self.content_container.add_child(Text(line, 1, 0))
        self._request_render()

    def show_info(self, message: str, links: Sequence[Any] = (), show_close_hint: bool = False) -> None:
        self.content_container.add_child(Spacer(1))
        self.content_container.add_child(Text(theme.fg("text", message), 1, 0))
        for link in links:
            url = link["url"] if isinstance(link, dict) else getattr(link, "url", "")
            label = link.get("label") if isinstance(link, dict) else getattr(link, "label", None)
            text = f"{label}: {url}" if label else url
            self.content_container.add_child(Text(theme.fg("accent", _hyperlink(url, text)), 1, 0))
        if show_close_hint:
            self.content_container.add_child(Spacer(1))
            self.content_container.add_child(Text(f"({key_hint('tui.select.cancel', 'to close')})", 1, 0))
        self._request_render()

    def show_waiting(self, message: str) -> None:
        self.content_container.add_child(Spacer(1))
        self.content_container.add_child(Text(theme.fg("dim", message), 1, 0))
        self.content_container.add_child(Text(f"({key_hint('tui.select.cancel', 'to cancel')})", 1, 0))
        self._request_render()

    def show_progress(self, message: str) -> None:
        self.content_container.add_child(Text(theme.fg("dim", message), 1, 0))
        self._request_render()

    def handle_input(self, data: str) -> None:
        if get_keybindings().matches(data, "tui.select.cancel"):
            self.cancel()
            return
        self.input.handle_input(data)


__all__ = ["LoginCancelledError", "LoginDialogComponent"]
