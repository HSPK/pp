"""RPC mode: headless operation over a JSON-lines stdin/stdout protocol.

Python port of `runRpcMode()` in
`packages/coding-agent/src/modes/rpc/rpc-mode.ts`.

A host process spawns `pi --mode rpc`, writes one JSON command per line to
stdin, and reads two kinds of line back on stdout: `{"type": "response", ...}`
answers correlated by the command's `id`, and session events as they happen.
Extensions that need user input emit `extension_ui_request` lines the host
answers with `extension_ui_response`.

Command dispatch lives in `dispatcher.py`; this module owns only the parts that
need a real process -- stdout takeover, reading stdin without blocking the event
loop, session rebinding, signal handling and shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pi_tui.tasks import spawn

from ...core.output_guard import (
    flush_raw_stdout,
    take_over_stdout,
    wait_for_raw_stdout_backpressure,
    write_raw_stdout,
)
from ...utils.shell import kill_tracked_detached_children
from ..json_event import to_json_event
from .dispatcher import RpcDispatcher
from .jsonl import JsonlLineReader, serialize_json_line
from .ui_context import RpcExtensionUIContext

if TYPE_CHECKING:
    from ...core.agent_session import AgentSession
    from ...core.agent_session_runtime import AgentSessionRuntime


async def _read_stdin_lines(on_line: Callable[[str], None]) -> None:
    """Feed stdin to `on_line`, one JSONL record at a time, until EOF.

    `sys.stdin.buffer.read` is a blocking call, so it runs on a worker thread;
    `loop.connect_read_pipe` would avoid the thread but fails on a regular file
    or a Windows pipe, which are both normal ways for a host to feed us.

    `read1` returns as soon as one read syscall has data, so a command is acted
    on the moment it arrives instead of waiting for a full buffer. Not every
    binary stream provides it, so plain `read` is the fallback.
    """
    loop = asyncio.get_running_loop()
    reader = JsonlLineReader(on_line)
    stream = sys.stdin.buffer
    read = getattr(stream, "read1", None) or stream.read
    while True:
        chunk = await loop.run_in_executor(None, read, 65536)
        if not chunk:
            break
        reader.feed(chunk)
    reader.close()


async def run_rpc_mode(
    runtime_host: AgentSessionRuntime,
    read_input: Callable[[Callable[[str], None]], Any] | None = None,
) -> int:
    """Serve RPC commands until stdin ends or an extension requests shutdown.

    `read_input` exists so a test can drive the loop from a list of lines
    instead of the process's real stdin; production passes nothing.
    """
    take_over_stdout()

    unsubscribe: Callable[[], None] | None = None
    unsubscribe_backpressure: Callable[[], None] | None = None
    signal_cleanup: list[Callable[[], None]] = []
    disposed = False

    def output(payload: dict[str, Any]) -> None:
        write_raw_stdout(serialize_json_line(payload))

    ui_context = RpcExtensionUIContext(output)

    async def rebind_session() -> None:
        nonlocal unsubscribe, unsubscribe_backpressure
        session = runtime_host.session
        session.extension_runner.set_ui_context(ui_context, "rpc")
        session.set_extension_shutdown_handler(dispatcher.request_shutdown)
        session.extension_runner.on_error(
            lambda err: output(
                {
                    "type": "extension_error",
                    "extensionPath": err.extension_path,
                    "event": err.event,
                    "error": err.error,
                }
            )
        )
        await session.bind_extensions()

        if unsubscribe is not None:
            with contextlib.suppress(Exception):
                unsubscribe()
        if unsubscribe_backpressure is not None:
            with contextlib.suppress(Exception):
                unsubscribe_backpressure()

        def on_event(event: Any) -> None:
            output(to_json_event(event))
            if getattr(event, "type", None) == "agent_settled" and dispatcher.shutdown_requested:
                stop.set()

        unsubscribe = session.subscribe(on_event)
        unsubscribe_backpressure = _subscribe_backpressure(session)

    stop = asyncio.Event()
    dispatcher = RpcDispatcher(runtime_host, output, ui_context, rebind_session)

    async def on_session_replaced(_session: AgentSession) -> None:
        await rebind_session()

    runtime_host.set_rebind_session(on_session_replaced)

    async def dispose() -> None:
        nonlocal disposed
        if disposed:
            return
        disposed = True
        for cleanup in (unsubscribe, unsubscribe_backpressure):
            if cleanup is not None:
                with contextlib.suppress(Exception):
                    cleanup()
        ui_context.cancel_all()
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
                # Bash children run in their own process group and never see
                # the terminal's SIGHUP, so they must be killed explicitly.
                with contextlib.suppress(Exception):
                    kill_tracked_detached_children()
                nonlocal exit_code
                exit_code = exit_status
                stop.set()

            try:
                loop.add_signal_handler(sig, handler)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            signal_cleanup.append(lambda sig=sig: loop.remove_signal_handler(sig))

    exit_code = 0
    failure: list[BaseException] = []
    register_signal_handlers()

    # Commands are handled one at a time. Two `handle_input_line` coroutines
    # running concurrently could interleave a session replacement with a
    # command reading `self.session`, which is the disposed-session bug the
    # dispatcher's property lookup exists to avoid in the first place.
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def pump() -> None:
        while True:
            line = await queue.get()
            try:
                await dispatcher.handle_input_line(line)
                await wait_for_raw_stdout_backpressure()
            finally:
                queue.task_done()
            if dispatcher.shutdown_requested and runtime_host.session.is_idle:
                stop.set()

    async def feed() -> None:
        reader = read_input or _read_stdin_lines
        await reader(queue.put_nowait)
        await queue.join()

    async def until_stopped(work: Any) -> None:
        """Run a background half of the loop, and stop the mode when it ends.

        Both halves are load-bearing: if reading stdin dies, no further command
        can arrive, and if the pump dies, no command can be answered. Either
        way the run is over. Without this, an exception in a spawned task would
        leave `stop` unset and the mode would wait forever on a loop that is no
        longer running -- a silent hang instead of an error.
        """
        try:
            await work
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            failure.append(error)
        finally:
            stop.set()

    tasks: list[asyncio.Task[None]] = []
    try:
        await rebind_session()
        tasks.append(spawn(until_stopped(pump())))
        tasks.append(spawn(until_stopped(feed())))
        await stop.wait()
        if failure:
            raise failure[0]
        return exit_code
    finally:
        # In `finally` because the caller may cancel us while we wait on
        # `stop`; leaving these running would keep reading stdin and answering
        # commands for a session that is about to be disposed.
        for task in tasks:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        for cleanup in signal_cleanup:
            with contextlib.suppress(Exception):
                cleanup()
        await dispose()
        await flush_raw_stdout()


def _subscribe_backpressure(session: AgentSession) -> Callable[[], None] | None:
    """Pause the agent while stdout is congested.

    A fast model streaming into a slow host would otherwise grow the pending
    write buffer without bound.
    """
    agent = getattr(session, "agent", None)
    subscribe = getattr(agent, "subscribe", None)
    if subscribe is None:
        return None

    def on_agent_event(*_args: Any, **_kwargs: Any) -> None:
        spawn(wait_for_raw_stdout_backpressure())

    with contextlib.suppress(Exception):
        return subscribe(on_agent_event)
    return None


__all__ = ["run_rpc_mode"]
