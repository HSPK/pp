"""`ExtensionUIContext` implemented over the RPC protocol.

Python port of `createExtensionUIContext()` in
`packages/coding-agent/src/modes/rpc/rpc-mode.ts`.

An extension asking the user something in RPC mode has no terminal to draw on,
so each request is emitted as an `extension_ui_request` line and the host
answers with an `extension_ui_response` carrying the same `id`. Fire-and-forget
methods (`notify`, `set_status`, `set_widget`, `set_title`) emit and return
immediately.

Scope note: this port's `ExtensionUIContext` is the narrowed protocol described
in `core/extensions/types.py`, so the TypeScript methods that exist only to be
refused in RPC mode (`setFooter`, `setWorkingMessage`, `addAutocompleteProvider`,
...) have nothing to refuse here -- they are not part of the protocol.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from typing import Any, Literal

from ...core.extensions.types import WidgetFactory, WidgetPlacement


class RpcExtensionUIContext:
    """Bridges extension UI calls onto the RPC wire."""

    def __init__(self, output: Callable[[dict[str, Any]], None]) -> None:
        self._output = output
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Host responses
    # ------------------------------------------------------------------

    def resolve(self, response: dict[str, Any]) -> bool:
        """Deliver an `extension_ui_response` to whoever is awaiting it.

        Returns whether the id matched a pending request; an unmatched id is
        the host answering twice or answering a request that already timed out
        on its side, and is dropped rather than raised.
        """
        request_id = response.get("id")
        if not isinstance(request_id, str):
            return False
        future = self._pending.pop(request_id, None)
        if future is None or future.done():
            return False
        future.set_result(response)
        return True

    def cancel_all(self) -> None:
        """Fail every in-flight request. Called when the host's input ends.

        Without this, an extension awaiting a dialog when stdin closes would
        block shutdown forever waiting for an answer that can no longer arrive.
        """
        for future in list(self._pending.values()):
            if not future.done():
                future.cancel()
        self._pending.clear()

    # ------------------------------------------------------------------
    # Request/response dialogs
    # ------------------------------------------------------------------

    async def _request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_id = str(uuid.uuid4())
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        self._output({"type": "extension_ui_request", "id": request_id, **request})
        try:
            return await future
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            return None

    async def select(self, title: str, options: list[str]) -> str | None:
        response = await self._request({"method": "select", "title": title, "options": list(options)})
        return _response_value(response)

    async def confirm(self, title: str, message: str) -> bool:
        response = await self._request({"method": "confirm", "title": title, "message": message})
        if response is None or response.get("cancelled"):
            return False
        return bool(response.get("confirmed", False))

    async def input(self, title: str, placeholder: str | None = None) -> str | None:
        response = await self._request({"method": "input", "title": title, "placeholder": placeholder})
        return _response_value(response)

    # ------------------------------------------------------------------
    # Fire and forget
    # ------------------------------------------------------------------

    def notify(self, message: str, type: Literal["info", "warning", "error"] = "info") -> None:
        self._emit({"method": "notify", "message": message, "notifyType": type})

    def set_status(self, key: str, text: str | None) -> None:
        self._emit({"method": "setStatus", "statusKey": key, "statusText": text})

    def set_title(self, title: str) -> None:
        self._emit({"method": "setTitle", "title": title})

    def set_widget(
        self,
        key: str,
        content: list[str] | WidgetFactory | None,
        placement: WidgetPlacement = "aboveEditor",
    ) -> None:
        # A factory builds a TUI component, and there is no TUI here to build
        # it against, so only plain lines can cross the wire. TypeScript drops
        # factories the same way.
        if content is not None and not isinstance(content, list):
            return
        self._emit(
            {
                "method": "setWidget",
                "widgetKey": key,
                "widgetLines": content,
                "widgetPlacement": placement,
            }
        )

    # ------------------------------------------------------------------
    # Unsupported without a TUI
    # ------------------------------------------------------------------

    def get_tools_expanded(self) -> bool:
        return False

    def set_tools_expanded(self, expanded: bool) -> None:
        pass

    def _emit(self, request: dict[str, Any]) -> None:
        self._output({"type": "extension_ui_request", "id": str(uuid.uuid4()), **request})


def _response_value(response: dict[str, Any] | None) -> str | None:
    if response is None or response.get("cancelled"):
        return None
    value = response.get("value")
    return value if isinstance(value, str) else None


__all__ = ["RpcExtensionUIContext"]
