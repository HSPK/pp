"""Print mode (single-shot): send prompts, output the result, exit.

Ported from ``packages/coding-agent/src/modes/print-mode.ts``.

Drives ``pi -p "prompt"`` (final text only) and ``pi --mode json "prompt"``
(the full event stream as newline-delimited JSON).

``rebind_session()`` ports TypeScript's ``rebindSession``: it calls
``session.bind_extensions()`` (which emits ``session_start`` to extensions)
before subscribing to the session's events, and is registered with
``runtime_host.set_rebind_session`` so an extension-driven session replacement
(``ctx.new_session()``, ``ctx.fork()``, ``ctx.switch_session()``) re-binds and
re-subscribes to the *new* session instead of leaving the mode attached to the
disposed one. The extension *UI host* is not
ported (see ``core/extensions``), so the ``uiContext`` /
``commandContextActions`` arguments TypeScript passes have no equivalent --
this port's ``bind_extensions()`` takes none. Everything else — the JSON
header, event subscription, stdout backpressure, signal handling and the exit
code derived from the last assistant message — is a direct port.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pi_tui.tasks import spawn

from ..core.output_guard import (
    flush_raw_stdout,
    wait_for_raw_stdout_backpressure,
    write_raw_stdout,
)
from ..utils.shell import kill_tracked_detached_children
from ..utils.wire import to_wire
from .json_event import to_json_event

if TYPE_CHECKING:
    from ..core.agent_session import AgentSession
    from ..core.agent_session_runtime import AgentSessionRuntime

PrintMode = Literal["text", "json"]


@dataclass
class PrintModeOptions:
    mode: PrintMode = "text"
    messages: list[str] = field(default_factory=list)
    initial_message: str | None = None
    initial_images: list[Any] = field(default_factory=list)


def _json_default(value: Any) -> Any:
    return repr(value)


async def run_print_mode(runtime_host: AgentSessionRuntime, options: PrintModeOptions) -> int:
    """Send the prompts and print the result. Returns a process exit code."""
    exit_code = 0
    session = runtime_host.session
    unsubscribe: Callable[[], None] | None = None
    disposed = False
    signal_cleanup: list[Callable[[], None]] = []

    async def dispose_runtime() -> None:
        nonlocal disposed
        if disposed:
            return
        disposed = True
        if unsubscribe is not None:
            with contextlib.suppress(Exception):
                unsubscribe()
        with contextlib.suppress(Exception):
            await runtime_host.dispose()

    def register_signal_handlers() -> None:
        signals = [signal.SIGTERM]
        if sys.platform != "win32":
            signals.append(signal.SIGHUP)

        loop = asyncio.get_event_loop()
        for sig in signals:
            exit_status = 129 if sig == getattr(signal, "SIGHUP", None) else 143

            def handler(exit_status: int = exit_status) -> None:
                # Bash children run in their own session and never see the
                # terminal's SIGHUP, so they must be killed explicitly.
                with contextlib.suppress(Exception):
                    kill_tracked_detached_children()

                async def finish() -> None:
                    await dispose_runtime()
                    await flush_raw_stdout()
                    sys.exit(exit_status)

                # `spawn` keeps a strong reference; a bare `create_task`
                # result can be garbage collected mid-flight.
                spawn(finish())

            try:
                loop.add_signal_handler(sig, handler)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            signal_cleanup.append(lambda sig=sig: loop.remove_signal_handler(sig))

    register_signal_handlers()

    async def rebind_session() -> None:
        nonlocal unsubscribe, session
        # TS reassigns `session = runtimeHost.session` first: after a
        # replacement the local must point at the new session before it is
        # bound and subscribed to, or the mode keeps driving the disposed one.
        session = runtime_host.session
        await session.bind_extensions()
        if unsubscribe is not None:
            with contextlib.suppress(Exception):
                unsubscribe()
        if options.mode != "json":
            unsubscribe = None
            return

        def on_event(event: Any) -> None:
            write_raw_stdout(json.dumps(to_json_event(event), default=_json_default, ensure_ascii=False) + "\n")

        unsubscribe = session.subscribe(on_event)

    async def on_session_replaced(_session: AgentSession) -> None:
        await rebind_session()

    runtime_host.set_rebind_session(on_session_replaced)

    try:
        if options.mode == "json":
            header = session.session_manager.get_header()
            if header:
                write_raw_stdout(json.dumps(to_wire(header), default=_json_default, ensure_ascii=False) + "\n")

        await rebind_session()

        if options.initial_message:
            await session.prompt(options.initial_message, images=options.initial_images or None)

        for message in options.messages:
            await session.prompt(message)

        if options.mode == "json":
            # Let every queued event drain before the process exits.
            await wait_for_raw_stdout_backpressure()

        if options.mode == "text":
            messages = session.state.messages
            last_message = messages[-1] if messages else None
            if last_message is not None and getattr(last_message, "role", None) == "assistant":
                stop_reason = getattr(last_message, "stop_reason", None)
                if stop_reason in ("error", "aborted"):
                    error_message = getattr(last_message, "error_message", None) or f"Request {stop_reason}"
                    print(error_message, file=sys.stderr)
                    exit_code = 1
                else:
                    for content in last_message.content:
                        if getattr(content, "type", None) == "text":
                            write_raw_stdout(f"{content.text}\n")

        return exit_code
    except Exception as error:
        print(str(error) or type(error).__name__, file=sys.stderr)
        return 1
    finally:
        for cleanup in signal_cleanup:
            with contextlib.suppress(Exception):
                cleanup()
        await dispose_runtime()
        await flush_raw_stdout()


__all__ = ["PrintMode", "PrintModeOptions", "run_print_mode"]
