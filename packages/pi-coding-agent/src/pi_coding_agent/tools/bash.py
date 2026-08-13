"""Execute bash commands, with streaming output, truncation, timeout and abort support.

Python port of `packages/coding-agent/src/core/tools/bash.ts`. The TypeScript
version's `BashOperations` extensibility point exists so extensions can
redirect execution to remote systems (SSH); this port keeps only the
local-shell execution path, since an extension here registers a whole
replacement tool rather than swapping out the built-in's operations.
`BashSpawnHook` *is* ported (`create_bash_tool(..., spawn_hook=...)`), as is
the `PI_*` session environment it observes
(`create_bash_tool(..., session_environment=...)`).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import signal as signal_module
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent, now_ms
from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.core.experimental import get_experimental_tool_sampling
from pi_coding_agent.tools.output_accumulator import OutputAccumulator
from pi_coding_agent.tools.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size
from pi_coding_agent.utils.child_process import wait_for_child_streams
from pi_coding_agent.utils.shell import (
    get_shell_config,
    get_shell_env,
    track_detached_child_pid,
    untrack_detached_child_pid,
)

_MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000
_UPDATE_THROTTLE_SECONDS = 0.1

_SESSION_ENV_NAMES = (
    "PI_SESSION_ID",
    "PI_SESSION_FILE",
    "PI_PROVIDER",
    "PI_MODEL",
    "PI_REASONING_LEVEL",
)
"""Variables `resolveSpawnContext` clears before (re)populating them, so a
child never inherits a stale value from the host process."""


@dataclass
class BashSpawnContext:
    """Command, working directory and environment about to be spawned.

    Port of TypeScript's `BashSpawnContext`. A `BashSpawnHook` receives one and
    returns the (possibly adjusted) context that is actually executed.
    """

    command: str
    cwd: str
    env: dict[str, str]


BashSpawnHook = Callable[[BashSpawnContext], BashSpawnContext]


BASH_PROMPT_GUIDELINES = ["You can inspect PI_* environment variables for current model and session details."]
"""`bashToolSystemPromptContribution.guidelines`."""


@dataclass
class BashAgentTool(AgentTool):
    """`AgentTool` that also carries the bash tool's system-prompt guidelines.

    TypeScript's `createBashToolDefinition` returns a `ToolDefinition` whose
    `promptGuidelines` is `undefined` when `exposeSessionEnvironment` is false,
    so a bash tool that does not expose the `PI_*` variables never tells the
    model to inspect them. This port has no `ToolDefinition` layer -- prompt
    contributions live in `TOOL_PROMPT_CONTRIBUTIONS`, keyed by tool name --
    so the per-instance choice travels on the tool object instead, and
    `AgentSession._refresh_tool_registry` prefers it.
    """

    prompt_guidelines: list[str] = field(default_factory=list)


@dataclass
class BashToolDetails:
    truncation: TruncationResult | None = None
    full_output_path: str | None = None


def _resolve_timeout_seconds(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if timeout != timeout or timeout <= 0:
        raise ValueError("Invalid timeout: must be a finite number of seconds")
    if timeout > _MAX_TIMEOUT_SECONDS:
        raise ValueError(f"Invalid timeout: maximum is {_MAX_TIMEOUT_SECONDS} seconds")
    return timeout


async def _pump_stream(stream: asyncio.StreamReader, on_data: Callable[[bytes], None]) -> None:
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return
        on_data(chunk)


class _BashExecError(Exception):
    """Internal signal carrying the TS `Error("aborted")` / `Error("timeout:N")` convention."""


async def _wait_for_exit(proc: asyncio.subprocess.Process) -> int | None:
    """Resolve as soon as the child exits, even if a descendant holds its pipes open.

    `Process.wait()` only resolves once the process exited *and* every pipe has
    disconnected, so a detached descendant that inherited stdout would keep it
    pending long after the command finished. Node's `exit` event has no such
    coupling, which is what `waitForChildProcess` relies on.
    """
    waiter = asyncio.ensure_future(proc.wait())
    delay = 0.005
    try:
        while proc.returncode is None:
            done, _pending = await asyncio.wait({waiter}, timeout=delay)
            if done:
                break
            delay = min(delay * 2, 0.05)
        return proc.returncode
    finally:
        if not waiter.done():
            waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await waiter


def _close_child_pipes(proc: asyncio.subprocess.Process) -> None:
    """Close the child's stdio, matching TypeScript's `stream.destroy()` on finalize."""
    transport = getattr(proc, "_transport", None)
    if transport is None:
        return
    with contextlib.suppress(Exception):
        transport.close()


async def _exec_local(
    command: str,
    cwd: str,
    on_data: Callable[[bytes], None],
    signal: AbortSignal | None,
    timeout: float | None,
    env: dict[str, str],
    shell_path: str | None = None,
) -> int | None:
    timeout_seconds = _resolve_timeout_seconds(timeout)
    if signal is not None and signal.aborted:
        raise _BashExecError("aborted")
    # Resolved before the cwd check, as in TypeScript, so a bad `shellPath`
    # setting reports itself rather than being masked by a missing directory.
    shell_config = get_shell_config(shell_path)
    if not os.path.isdir(cwd):
        raise RuntimeError(f"Working directory does not exist: {cwd}\nCannot execute bash commands.")

    command_from_stdin = shell_config.command_transport == "stdin"
    args = list(shell_config.args) if command_from_stdin else [*shell_config.args, command]
    proc = await asyncio.create_subprocess_exec(
        shell_config.shell,
        *args,
        cwd=cwd,
        env=env,
        stdin=asyncio.subprocess.PIPE if command_from_stdin else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    if command_from_stdin and proc.stdin is not None:
        # TypeScript ignores stdin errors here (`child.stdin.on("error", ...)`):
        # a shell that exits before reading closes the pipe, which must not
        # take down the command that already ran.
        with contextlib.suppress(Exception):
            proc.stdin.write(command.encode("utf-8"))
            proc.stdin.close()
    track_detached_child_pid(proc.pid)
    timed_out = False
    aborted = False
    loop = asyncio.get_running_loop()
    last_data_at = loop.time()

    def track_data(data: bytes) -> None:
        nonlocal last_data_at
        last_data_at = loop.time()
        on_data(data)

    async def kill_process_group() -> None:
        if proc.returncode is not None:
            return
        try:
            os.killpg(proc.pid, signal_module.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()

    stdout_task = asyncio.ensure_future(_pump_stream(proc.stdout, track_data)) if proc.stdout else None
    stderr_task = asyncio.ensure_future(_pump_stream(proc.stderr, track_data)) if proc.stderr else None
    wait_task = asyncio.ensure_future(_wait_for_exit(proc))

    watchers: list[asyncio.Task] = []
    if timeout_seconds is not None:
        watchers.append(asyncio.ensure_future(asyncio.sleep(timeout_seconds)))
    if signal is not None:
        watchers.append(asyncio.ensure_future(signal.wait()))

    try:
        if watchers:
            done, _pending = await asyncio.wait({wait_task, *watchers}, return_when=asyncio.FIRST_COMPLETED)
            if wait_task not in done:
                if signal is not None and signal.aborted:
                    aborted = True
                else:
                    timed_out = True
                await kill_process_group()
                await wait_task
        else:
            await wait_task
    finally:
        for watcher in watchers:
            if not watcher.done():
                watcher.cancel()
        # The child exited, but a detached descendant may still hold the pipe
        # open: keep reading while output is arriving, and give up once the
        # pipes fall idle instead of blocking on the inherited handle.
        await wait_for_child_streams(
            [task for task in (stdout_task, stderr_task) if task is not None],
            lambda: last_data_at,
        )
        _close_child_pipes(proc)
        untrack_detached_child_pid(proc.pid)

    if signal is not None and signal.aborted:
        raise _BashExecError("aborted")
    if aborted:
        raise _BashExecError("aborted")
    if timed_out:
        raise _BashExecError(f"timeout:{timeout}")

    return proc.returncode


def _format_bash_output(
    output: OutputAccumulator,
    snapshot_content: str,
    truncation: TruncationResult,
    full_output_path: str | None,
    empty_text: str = "(no output)",
) -> tuple[str, BashToolDetails | None]:
    text = snapshot_content or empty_text
    details: BashToolDetails | None = None
    if truncation.truncated:
        details = BashToolDetails(truncation=truncation, full_output_path=full_output_path)
        start_line = truncation.total_lines - truncation.output_lines + 1
        end_line = truncation.total_lines
        if truncation.last_line_partial:
            last_line_size = format_size(output.get_last_line_bytes())
            text += (
                f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line} "
                f"(line is {last_line_size}). Full output: {full_output_path}]"
            )
        elif truncation.truncated_by == "lines":
            text += (
                f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                f"Full output: {full_output_path}]"
            )
        else:
            text += (
                f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {full_output_path}]"
            )
    return text, details


def create_bash_tool(
    cwd: str,
    command_prefix: str | None = None,
    *,
    session_environment: Callable[[], dict[str, str] | Awaitable[dict[str, str]]] | None = None,
    expose_session_environment: bool = True,
    spawn_hook: BashSpawnHook | None = None,
    shell_path: str | None = None,
) -> AgentTool:
    """Create the `bash` tool bound to a working directory.

    ``command_prefix`` is prepended (as its own line) to every command,
    matching the TypeScript `BashToolOptions.commandPrefix` hook.

    ``shell_path`` is the `shellPath` setting, resolved through
    `get_shell_config` exactly as TypeScript's `BashToolOptions.shellPath` is.

    ``session_environment`` supplies the `PI_*` variables TypeScript's
    `resolveSpawnContext` injects from the live `ExtensionContext`
    (`PI_SESSION_ID`, `PI_SESSION_FILE`, `PI_PROVIDER`, `PI_MODEL`,
    `PI_REASONING_LEVEL`). It is a callable, not a dict, because the model and
    thinking level change mid-session and the tool is built once. TypeScript
    reads them off `ctx` at spawn time for the same reason. The built-in
    system prompt tells the model "You can inspect PI_* environment variables
    for current model and session details", so leaving them unset makes the
    prompt lie. The callable may return a plain `dict` or an awaitable of one
    -- `server/create_harness.py` needs the latter because it reads the
    `AgentHarness`'s current model/thinking level, which are async getters.
    A value of `""` is still applied (only a missing key is treated as "leave
    unset"), matching TypeScript's `execution.env.PI_SESSION_FILE = sessionFile
    ?? ""` in `create-harness.ts`, which always sets the key even when there is
    no session file.

    ``expose_session_environment`` is TypeScript's
    `BashToolOptions.exposeSessionEnvironment`: when `False`, the `PI_*`
    variables are stripped from the child environment even if the host process
    already has them set.

    ``spawn_hook`` is TypeScript's `BashToolOptions.spawnHook`: it sees the
    resolved command, cwd and environment just before execution and returns the
    context that is actually spawned.
    """

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
    ) -> AgentToolResult:
        command = params["command"]
        timeout = params.get("timeout")
        resolved_command = f"{command_prefix}\n{command}" if command_prefix else command

        output = OutputAccumulator(temp_file_prefix="pi-bash")
        last_update_at = 0.0
        accepting_output = True

        def emit_output_update() -> None:
            if on_update is None:
                return
            snapshot = output.snapshot(persist_if_truncated=True)
            on_update(
                AgentToolResult(
                    content=[TextContent(text=snapshot.content or "")],
                    details=BashToolDetails(
                        truncation=snapshot.truncation if snapshot.truncation.truncated else None,
                        full_output_path=snapshot.full_output_path,
                    ),
                )
            )

        def handle_data(data: bytes) -> None:
            nonlocal last_update_at
            if not accepting_output:
                return
            output.append(data)
            if on_update is None:
                return
            now = time.monotonic()
            if now - last_update_at >= _UPDATE_THROTTLE_SECONDS:
                last_update_at = now
                emit_output_update()

        if on_update is not None:
            on_update(AgentToolResult(content=[]))

        def finish_output():
            nonlocal accepting_output
            accepting_output = False
            output.finish()
            emit_output_update()
            snapshot = output.snapshot(persist_if_truncated=True)
            output.close_temp_file()
            return snapshot

        def append_status(text: str, status: str) -> str:
            return f"{text}\n\n{status}" if text else status

        env = get_shell_env()
        for name in _SESSION_ENV_NAMES:
            env.pop(name, None)
        if expose_session_environment and session_environment is not None:
            resolved_env = session_environment()
            if inspect.isawaitable(resolved_env):
                resolved_env = await resolved_env
            # `value is not None` (not truthiness): a caller may deliberately set
            # e.g. PI_SESSION_FILE to "" to mean "no session file, but the key
            # is still present" -- see `create-harness.ts`'s `sessionFile ?? ""`.
            env.update({key: value for key, value in resolved_env.items() if value is not None})

        spawn_context = BashSpawnContext(command=resolved_command, cwd=cwd, env=env)
        if spawn_hook is not None:
            spawn_context = spawn_hook(spawn_context)
        resolved_command = spawn_context.command
        spawn_cwd = spawn_context.cwd
        env = spawn_context.env

        try:
            exit_code: int | None
            try:
                exit_code = await _exec_local(
                    resolved_command, spawn_cwd, handle_data, signal, timeout, env, shell_path
                )
            except _BashExecError as err:
                snapshot = finish_output()
                text, _ = _format_bash_output(
                    output, snapshot.content, snapshot.truncation, snapshot.full_output_path, empty_text=""
                )
                if str(err) == "aborted":
                    raise RuntimeError(append_status(text, "Command aborted")) from err
                if str(err).startswith("timeout:"):
                    timeout_secs = str(err).split(":", 1)[1]
                    raise RuntimeError(append_status(text, f"Command timed out after {timeout_secs} seconds")) from err
                raise

            snapshot = finish_output()
            output_text, details = _format_bash_output(
                output, snapshot.content, snapshot.truncation, snapshot.full_output_path
            )
            if exit_code is not None and exit_code != 0:
                raise RuntimeError(append_status(output_text, f"Command exited with code {exit_code}"))

            return AgentToolResult(content=[TextContent(text=output_text)], details=details)
        finally:
            output.close_temp_file()

    return BashAgentTool(
        name="bash",
        description=(
            "Execute a bash command in the current working directory. Returns stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
            "(whichever is hit first). If truncated, full output is saved to a temp file. "
            "Optionally provide a timeout in seconds."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
            },
            "required": ["command"],
        },
        execute=execute,
        constrained_sampling=get_experimental_tool_sampling(),
        prompt_guidelines=list(BASH_PROMPT_GUIDELINES) if expose_session_environment else [],
    )


# --------------------------------------------------------------------------
# Rendering
#
# Port of `bash.ts`'s `formatBashCall` and `rebuildBashResultRenderComponent`.
# Unlike `read`, a collapsed bash result is not empty: it previews the *last*
# few lines, because what a command just printed is usually what matters.
# --------------------------------------------------------------------------

BASH_PREVIEW_LINES = 5


def format_duration(ms: float) -> str:
    """Port of bash.ts's `formatDuration`: one decimal, seconds."""
    return f"{ms / 1000:.1f}s"


def format_bash_call(args: Any, theme: Any) -> str:
    """Port of `formatBashCall`."""
    from pi_coding_agent.tools.render_utils import invalid_arg_text, str_arg

    a = args if isinstance(args, dict) else {}
    command = str_arg(a.get("command"))
    timeout = a.get("timeout")
    timeout_suffix = theme.fg("muted", f" (timeout {timeout}s)") if timeout else ""
    if command is None:
        command_display = invalid_arg_text(theme)
    elif command:
        command_display = command
    else:
        command_display = theme.fg("toolOutput", "...")
    return theme.fg("toolTitle", theme.bold(f"$ {command_display}")) + timeout_suffix


def format_bash_result_lines(
    result: Any, options: Any, theme: Any, show_images: bool, started_at: float | None, ended_at: float | None
) -> list[str]:
    """Rendered lines for a bash result. Port of `rebuildBashResultRenderComponent`.

    Returns lines rather than components: this port's renderer hook hands back a
    single `Text`, and the upstream component tree exists only to cache the
    collapsed preview per width.
    """
    from pi_coding_agent.modes.interactive.components.keybinding_hints import key_hint
    from pi_coding_agent.modes.interactive.components.visual_truncate import truncate_to_visual_lines
    from pi_coding_agent.tools.render_utils import get_text_output

    expanded = bool(getattr(options, "expanded", False))
    is_partial = bool(getattr(options, "is_partial", False))
    details = getattr(result, "details", None)
    truncation = getattr(details, "truncation", None)
    full_output_path = getattr(details, "full_output_path", None)

    output = get_text_output(result, show_images).strip()
    if (
        not is_partial
        and truncation is not None
        and getattr(truncation, "truncated", False)
        and full_output_path
        and output.endswith("]")
    ):
        # Upstream drops the "[Full output: ...]" footer the tool already
        # appended, because the same information is re-rendered below as a
        # styled warning; leaving both shows it twice.
        footer_start = output.rfind("\n\n[")
        if footer_start != -1 and full_output_path in output[footer_start:]:
            output = output[:footer_start].rstrip()

    lines: list[str] = []
    if output:
        styled = "\n".join(theme.fg("toolOutput", line) for line in output.split("\n"))
        if expanded:
            lines.append("")
            lines.extend(styled.split("\n"))
        else:
            preview = truncate_to_visual_lines(styled, BASH_PREVIEW_LINES, 80)
            lines.append("")
            if preview.skipped_count:
                hint = (
                    theme.fg("muted", f"... ({preview.skipped_count} earlier lines,")
                    + " "
                    + key_hint("app.tools.expand", "to expand")
                    + theme.fg("muted", ")")
                )
                lines.append(hint)
            lines.extend(preview.visual_lines)

    if (truncation is not None and getattr(truncation, "truncated", False)) or full_output_path:
        warnings: list[str] = []
        if full_output_path:
            warnings.append(f"Full output: {full_output_path}")
        if truncation is not None and getattr(truncation, "truncated", False):
            if getattr(truncation, "truncated_by", None) == "lines":
                warnings.append(f"Truncated: showing {truncation.output_lines} of {truncation.total_lines} lines")
            else:
                max_bytes = getattr(truncation, "max_bytes", None) or DEFAULT_MAX_BYTES
                warnings.append(f"Truncated: {truncation.output_lines} lines shown ({format_size(max_bytes)} limit)")
        lines.append("")
        lines.append(theme.fg("warning", f"[{'. '.join(warnings)}]"))

    if started_at is not None:
        label = "Elapsed" if is_partial else "Took"
        end_time = ended_at if ended_at is not None else now_ms()
        lines.append("")
        lines.append(theme.fg("muted", f"{label} {format_duration(end_time - started_at)}"))

    return lines
