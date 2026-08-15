"""Run a command directly (no shell) and collect stdout/stderr/exit code.

Ported from ``packages/coding-agent/src/core/exec.ts``, including the
termination handling in ``packages/coding-agent/src/utils/child-process.ts``
(``waitForChildProcess``).

This is the plumbing extensions and custom tools use when they need to shell
out to a helper binary: unlike the bash tool (which always runs
``/bin/bash -c "<command>"``), this executes ``command`` with its ``args``
directly, so no shell metacharacter interpretation, quoting, or globbing
happens. It supports both a timeout and cooperative cancellation via an
:class:`AbortSignal`.

Two behaviours are easy to get wrong and are load-bearing:

*Reading past exit.* A command can exit while a detached descendant still
holds the inherited stdout pipe open -- ``bash -c "echo hi; sleep 30 &"`` is
enough. Waiting for the pipes to reach EOF would block for the descendant's
whole lifetime, so after the process exits this waits only for the pipes to
fall *idle*: the grace timer restarts on every chunk, so a descendant that is
actively writing keeps us reading (its tail is not truncated), while a quiet
inherited handle releases us after 100ms.

*Exit codes.* Node reports ``null`` for a signal-terminated child and
``execCommand`` maps that to ``0``. Python reports ``-SIGTERM`` (``-15``), so
a caller testing ``code == 0`` would draw the opposite conclusion for every
timed-out or aborted command. Signal terminations are therefore normalized
back to ``0``.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal as signal_module
import time
from dataclasses import dataclass, field

from pi_ai.utils.abort import AbortSignal
from pi_ai.utils.tasks import spawn

_FORCE_KILL_DELAY_SECONDS = 5.0
_EXIT_STDIO_GRACE_SECONDS = 0.1
_IDLE_POLL_SECONDS = 0.01


@dataclass
class ExecOptions:
    """Options for executing shell commands."""

    signal: AbortSignal | None = None
    """AbortSignal to cancel the command."""
    timeout: float | None = None
    """Timeout in milliseconds."""
    cwd: str | None = None
    """Working directory. Not consumed by `exec_command` itself -- matching the
    TypeScript original, callers resolve the effective cwd themselves and pass
    it as the `cwd` positional argument."""


@dataclass
class ExecResult:
    """Result of executing a shell command."""

    stdout: str
    stderr: str
    code: int
    killed: bool


@dataclass
class _StreamCollector:
    """Accumulates one pipe's bytes and when it last produced output."""

    chunks: list[bytes] = field(default_factory=list)
    last_activity: float = field(default_factory=time.monotonic)

    def text(self) -> str:
        return b"".join(self.chunks).decode("utf-8", errors="replace")


async def _pump(stream: asyncio.StreamReader | None, collector: _StreamCollector) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        collector.chunks.append(chunk)
        collector.last_activity = time.monotonic()


async def _wait_for_exit(proc: asyncio.subprocess.Process) -> None:
    """Return as soon as the child has exited, ignoring its pipes.

    ``Process.wait()`` cannot be used here: asyncio only wakes its exit
    waiters once every pipe has disconnected, so a detached descendant holding
    the inherited stdout open keeps ``wait()`` pending for the descendant's
    whole lifetime -- the exact hang ``waitForChildProcess`` exists to avoid.
    ``returncode`` is populated the moment the child is reaped, so polling it
    observes the exit itself rather than the pipe lifetime.
    """
    while proc.returncode is None:
        await asyncio.sleep(_IDLE_POLL_SECONDS)


async def _drain_after_exit(pumps: list[asyncio.Task[None]], collectors: list[_StreamCollector]) -> None:
    """Wait for the pipes to reach EOF, or to fall idle for the grace period.

    Port of ``waitForChildProcess``'s post-exit idle timer: never block on a
    descendant that inherited the pipe and holds it open forever, but never
    truncate output that is still arriving either.
    """
    while True:
        if all(pump.done() for pump in pumps):
            return
        idle_for = time.monotonic() - max(collector.last_activity for collector in collectors)
        if idle_for >= _EXIT_STDIO_GRACE_SECONDS:
            for pump in pumps:
                pump.cancel()
            for pump in pumps:
                with contextlib.suppress(asyncio.CancelledError):
                    await pump
            return
        await asyncio.sleep(_IDLE_POLL_SECONDS)


def _close_transport(proc: asyncio.subprocess.Process) -> None:
    """Release the pipes, mirroring the TypeScript's ``stdout.destroy()``.

    When the pipes are abandoned while a descendant still holds them open,
    the transport survives until garbage collection and then tries to close
    itself against an already-closed event loop, printing a traceback from
    ``__del__``. Closing it here keeps that off the user's terminal. The
    transport handle is private in asyncio, so access is defensive.
    """
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        with contextlib.suppress(Exception):
            transport.close()


async def exec_command(
    command: str,
    args: list[str],
    cwd: str,
    options: ExecOptions | None = None,
) -> ExecResult:
    """Execute a shell command and return stdout/stderr/code.

    Supports timeout and abort signal. Never raises for a command that cannot
    be spawned (missing binary, bad ``cwd``): like the TypeScript, that is
    reported as ``code=1`` so extension code does not have to guard every call.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError:
        return ExecResult(stdout="", stderr="", code=1, killed=False)

    killed = False

    async def kill_process() -> None:
        nonlocal killed
        if killed:
            return
        killed = True
        with contextlib.suppress(ProcessLookupError):
            proc.send_signal(signal_module.SIGTERM)

        async def _force_kill_later() -> None:
            await asyncio.sleep(_FORCE_KILL_DELAY_SECONDS)
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()

        spawn(_force_kill_later())

    signal = options.signal if options is not None else None
    if signal is not None and signal.aborted:
        await kill_process()

    timeout = options.timeout if options is not None else None
    watchers: list[asyncio.Task[None]] = []
    if timeout is not None and timeout > 0:
        watchers.append(asyncio.ensure_future(asyncio.sleep(timeout / 1000)))
    if signal is not None:
        watchers.append(asyncio.ensure_future(signal.wait()))

    stdout_collector = _StreamCollector()
    stderr_collector = _StreamCollector()
    pumps = [
        asyncio.ensure_future(_pump(proc.stdout, stdout_collector)),
        asyncio.ensure_future(_pump(proc.stderr, stderr_collector)),
    ]
    wait_task = asyncio.ensure_future(_wait_for_exit(proc))

    try:
        if watchers:
            done, _pending = await asyncio.wait({wait_task, *watchers}, return_when=asyncio.FIRST_COMPLETED)
            if wait_task not in done:
                await kill_process()
                await wait_task
        else:
            await wait_task
    finally:
        for watcher in watchers:
            if not watcher.done():
                watcher.cancel()

    await _drain_after_exit(pumps, [stdout_collector, stderr_collector])
    _close_transport(proc)

    return ExecResult(
        stdout=stdout_collector.text(),
        stderr=stderr_collector.text(),
        # A signal termination is `null` in Node, which `execCommand` maps to 0.
        code=proc.returncode if (proc.returncode or 0) > 0 else 0,
        killed=killed,
    )


__all__ = ["ExecOptions", "ExecResult", "exec_command"]
