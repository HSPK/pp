"""Tests for the `bash` tool. Ported from `tools.test.ts` (describe("bash tool")).

The TypeScript suite's `BashOperations` fake-hook tests (for injecting
synthetic chatty output alongside a forced timeout/abort) rely on
`createBashTool(cwd, { operations })`, which this port intentionally does not
implement (see module docstring in `bash.py`). Those cases are replaced here
with equivalent tests that drive real chatty bash commands through actual
timeout and abort paths -- a stronger substitute, since they exercise the real
child process rather than a hand-rolled stand-in.
"""

from __future__ import annotations

import asyncio
import errno
import os
import re

import pytest
from pi_ai.utils.abort import AbortSignal
from pi_coding_agent.core.bash_executor import create_local_bash_operations, execute_bash_with_operations
from pi_coding_agent.tools import bash as bash_module
from pi_coding_agent.tools.bash import create_bash_tool
from pi_coding_agent.utils import shell as shell_module
from pi_coding_agent.utils.shell import _tracked_detached_child_pids


def get_text(result) -> str:
    return "\n".join(c.text for c in result.content if c.type == "text")


async def test_executes_simple_command(tmp_path):
    tool = create_bash_tool(str(tmp_path))

    result = await tool.execute("call-1", {"command": "echo 'test output'"})

    assert "test output" in get_text(result)
    assert result.details is None


async def test_command_error_exit_code(tmp_path):
    tool = create_bash_tool(str(tmp_path))

    with pytest.raises(RuntimeError, match="code 1"):
        await tool.execute("call-2", {"command": "exit 1"})


async def test_respects_timeout(tmp_path):
    tool = create_bash_tool(str(tmp_path))

    with pytest.raises(RuntimeError, match=r"(?i)timed out"):
        await tool.execute("call-3", {"command": "sleep 5", "timeout": 1})


async def test_working_directory_missing_raises(tmp_path):
    nonexistent_cwd = str(tmp_path / "does" / "not" / "exist")
    tool = create_bash_tool(nonexistent_cwd)

    with pytest.raises(RuntimeError, match="Working directory does not exist"):
        await tool.execute("call-4", {"command": "echo test"})


async def test_command_prefix_prepended(tmp_path):
    tool = create_bash_tool(str(tmp_path), command_prefix="export TEST_VAR=hello")

    result = await tool.execute("call-5", {"command": "echo $TEST_VAR"})

    assert get_text(result).strip() == "hello"


async def test_prefix_and_command_output_both_present(tmp_path):
    tool = create_bash_tool(str(tmp_path), command_prefix="echo prefix-output")

    result = await tool.execute("call-6", {"command": "echo command-output"})

    assert get_text(result).strip() == "prefix-output\ncommand-output"


async def test_works_without_command_prefix(tmp_path):
    tool = create_bash_tool(str(tmp_path))

    result = await tool.execute("call-7", {"command": "echo no-prefix"})

    assert get_text(result).strip() == "no-prefix"


async def test_line_truncation_and_full_output_path(tmp_path):
    tool = create_bash_tool(str(tmp_path))

    result = await tool.execute("call-8", {"command": "seq 3000"})
    output = get_text(result)

    assert result.details is not None
    assert result.details.truncation.truncated is True
    assert result.details.truncation.truncated_by == "lines"
    assert result.details.full_output_path is not None
    assert re.search(r"\[Showing lines \d+-\d+ of \d+\. Full output: ", output)
    assert "Full output: undefined" not in output

    for _ in range(20):
        if os.path.exists(result.details.full_output_path):
            break
        await asyncio.sleep(0.05)

    assert os.path.exists(result.details.full_output_path)
    with open(result.details.full_output_path) as f:
        full_output = f.read()
    assert "1\n2\n3" in full_output
    assert "2998\n2999\n3000" in full_output


async def test_trailing_newline_not_counted_as_extra_line(tmp_path):
    tool = create_bash_tool(str(tmp_path))
    command = "for i in $(seq -w 1 4000); do printf 'line-%s\\n' \"$i\"; done"

    result = await tool.execute("call-9", {"command": command})
    output = get_text(result)

    assert result.details.truncation.total_lines == 4000
    assert result.details.truncation.output_lines == 2000
    assert re.search(r"\[Showing lines 2001-4000 of 4000\. Full output: ", output)
    assert "4001" not in output


async def test_decodes_utf8_split_across_chunks(tmp_path):
    tool = create_bash_tool(str(tmp_path))

    result = await tool.execute("call-10", {"command": "printf '\\xe2\\x82\\xac\\n'"})

    assert get_text(result).strip() == "\u20ac"


async def test_timeout_error_includes_full_output_path_for_chatty_output(tmp_path):
    tool = create_bash_tool(str(tmp_path))
    command = "for i in $(seq 1 3000); do echo $i; done; sleep 5"

    with pytest.raises(RuntimeError) as exc_info:
        await tool.execute("call-11", {"command": command, "timeout": 1})

    message = str(exc_info.value)
    assert "Command timed out after 1 seconds" in message
    match = re.search(r"\[Showing lines \d+-\d+ of \d+\. Full output: ([^\]\n]+)\]", message)
    assert match is not None
    full_output_path = match.group(1)
    assert os.path.exists(full_output_path)
    with open(full_output_path) as f:
        full_output = f.read()
    assert "1\n2\n3" in full_output


async def test_abort_error_includes_full_output_path_for_chatty_output(tmp_path):
    tool = create_bash_tool(str(tmp_path))
    command = "for i in $(seq 1 3000); do echo $i; done; sleep 5"
    signal = AbortSignal()

    async def abort_soon():
        await asyncio.sleep(0.2)
        signal.abort()

    abort_task = asyncio.ensure_future(abort_soon())

    with pytest.raises(RuntimeError) as exc_info:
        await tool.execute("call-12", {"command": command}, signal)
    await abort_task

    message = str(exc_info.value)
    assert "Command aborted" in message
    match = re.search(r"\[Showing lines \d+-\d+ of \d+\. Full output: ([^\]\n]+)\]", message)
    assert match is not None
    full_output_path = match.group(1)
    assert os.path.exists(full_output_path)


async def test_coalesces_streaming_updates_for_chatty_output(tmp_path):
    tool = create_bash_tool(str(tmp_path))
    updates = []

    result = await tool.execute(
        "call-13",
        {"command": 'for i in $(seq 0 4999); do echo "line $i"; done'},
        None,
        lambda update: updates.append(update),
    )

    assert len(updates) < 25
    assert "line 4999" in get_text(result)


async def test_invalid_timeout_raises(tmp_path):
    tool = create_bash_tool(str(tmp_path))

    with pytest.raises(ValueError, match="Invalid timeout"):
        await tool.execute("call-14", {"command": "echo test", "timeout": -1})


async def test_already_aborted_signal_raises_immediately(tmp_path):
    tool = create_bash_tool(str(tmp_path))
    signal = AbortSignal()
    signal.abort()

    with pytest.raises(RuntimeError):
        await tool.execute("call-15", {"command": "echo test"}, signal)


async def test_exposes_local_bash_operations_for_extension_reuse(tmp_path):
    ops = create_local_bash_operations()
    chunks: list[bytes] = []

    exit_code = await ops.exec(
        "echo $TEST_LOCAL_BASH_OPS",
        str(tmp_path),
        chunks.append,
        None,
        None,
        {**os.environ, "TEST_LOCAL_BASH_OPS": "from-local-ops"},
    )

    assert exit_code == 0
    assert b"".join(chunks).decode("utf-8").strip() == "from-local-ops"


async def test_preserves_execute_bash_sanitization_with_local_bash_operations(tmp_path):
    result = await execute_bash_with_operations(
        "printf '\\033[31mred\\033[0m\\r\\n'",
        str(tmp_path),
        create_local_bash_operations(),
    )

    assert result.exit_code == 0
    assert result.output == "red\n"


async def test_execute_bash_persists_full_output_on_line_count_truncation(tmp_path):
    result = await execute_bash_with_operations("seq 3000", str(tmp_path), create_local_bash_operations())

    assert result.truncated is True
    assert result.full_output_path is not None

    for _ in range(20):
        if os.path.exists(result.full_output_path):
            break
        await asyncio.sleep(0.05)

    assert os.path.exists(result.full_output_path)
    with open(result.full_output_path) as f:
        full_output = f.read()
    assert "1\n2\n3" in full_output
    assert "2998\n2999\n3000" in full_output


async def test_handles_process_spawn_errors(tmp_path, monkeypatch):
    """TS 'should handle process spawn errors'.

    TypeScript stubs `getShellConfig` to a nonexistent binary and asserts the
    failure reaches the caller as ENOENT rather than being swallowed into an
    empty successful result.
    """
    monkeypatch.setattr(
        shell_module,
        "get_shell_config",
        lambda custom_shell_path=None: shell_module.ShellConfig(shell="/nonexistent-shell-path-xyz123", args=["-c"]),
    )
    monkeypatch.setattr(bash_module, "get_shell_config", shell_module.get_shell_config)

    tool = create_bash_tool(str(tmp_path))

    with pytest.raises(FileNotFoundError) as excinfo:
        await tool.execute("call-12", {"command": "echo test"})
    assert excinfo.value.errno == errno.ENOENT


async def test_passes_shell_path_through_to_shell_resolution(tmp_path):
    """TS 'should pass shellPath through to shell resolution'.

    A `shellPath` that does not exist must be reported, not silently ignored:
    falling back to the default shell would run the command under a shell the
    user explicitly did not ask for.
    """
    ops = create_local_bash_operations("/custom/bash")

    with pytest.raises(RuntimeError, match=r"^Custom shell path not found: /custom/bash$"):
        await ops.exec("echo test", str(tmp_path), lambda _data: None, None, None, None)

    # The same path through the tool: `shellPath` reaches `get_shell_config`.
    tool = create_bash_tool(str(tmp_path), shell_path="/custom/bash")
    with pytest.raises(RuntimeError, match="Custom shell path not found: /custom/bash"):
        await tool.execute("call-12b", {"command": "echo test"})


async def test_shell_path_is_used_for_execution_when_it_exists(tmp_path):
    """The other half of `getShellConfig(customShellPath)`: an existing path is used.

    TypeScript only reaches this through `getBashShellConfig(customShellPath)`;
    pinning it here is what stops `shell_path` regressing to "accepted and
    ignored", which is how this port previously behaved.
    """
    marker_shell = tmp_path / "marker-shell"
    marker_shell.write_text('#!/bin/bash\necho "custom-shell-ran: $2"\n')
    marker_shell.chmod(0o755)

    ops = create_local_bash_operations(str(marker_shell))
    chunks: list[bytes] = []
    exit_code = await ops.exec("echo original", str(tmp_path), chunks.append, None, None, None)

    assert exit_code == 0
    assert b"custom-shell-ran: echo original" in b"".join(chunks)


async def test_sends_commands_over_stdin_when_shell_resolution_requires_it(tmp_path, monkeypatch):
    """TS 'should send commands over stdin when shell resolution requires it'.

    The TypeScript stub is a node process that echoes back everything it reads
    on stdin; the assertion is that the command text arrives there verbatim
    instead of on argv. `cat` is the same stub with no interpreter needed.
    """
    monkeypatch.setattr(
        shell_module,
        "get_shell_config",
        lambda custom_shell_path=None: shell_module.ShellConfig(shell="/bin/cat", args=[], command_transport="stdin"),
    )
    monkeypatch.setattr(bash_module, "get_shell_config", shell_module.get_shell_config)

    chunks: list[bytes] = []
    ops = create_local_bash_operations("C:\\Windows\\System32\\bash.exe")
    command = 'name=\'World\'; echo "Hello, ${name}!"; count=3; for i in $(seq 1 ${count}); do echo "Iteration ${i} of ${count}"; done'

    exit_code = await ops.exec(command, str(tmp_path), chunks.append, None, None, None)

    assert exit_code == 0
    assert b"".join(chunks).decode("utf-8") == command


def test_resolves_legacy_wsl_bash_exe_to_stdin_command_transport(tmp_path, monkeypatch):
    """TS 'should resolve legacy WSL bash.exe to stdin command transport'.

    Legacy `C:\\Windows\\System32\\bash.exe` cannot take a `-c` command, so it
    has to be driven with `-s` and the command piped in. The custom-shell-path
    branch does not consult the platform, so (exactly as in TypeScript) this is
    checked by creating a file with that literal name relative to the cwd.
    """
    shell_path = "C:\\Windows\\System32\\bash.exe"
    (tmp_path / shell_path).write_text("")
    monkeypatch.chdir(tmp_path)

    assert shell_module.get_shell_config(shell_path) == shell_module.ShellConfig(
        shell=shell_path, args=["-s"], command_transport="stdin"
    )


@pytest.mark.parametrize(
    "path,expected_stdin",
    [
        ("C:\\Windows\\System32\\bash.exe", True),
        ("c:/windows/sysnative/BASH.EXE", True),
        ("C:\\Windows\\System32\\bash.exe.bak", False),
        ("C:\\Program Files\\Git\\bin\\bash.exe", False),
        ("/bin/bash", False),
    ],
)
def test_legacy_wsl_bash_path_detection(tmp_path, monkeypatch, path, expected_stdin):
    """Pins the `isLegacyWslBashPath` regex: slash/case normalisation and anchoring."""
    target = tmp_path / path
    if not os.path.isabs(path):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("")
    monkeypatch.chdir(tmp_path)

    config = shell_module.get_shell_config(path)
    assert config.shell == path
    assert (config.command_transport == "stdin") is expected_stdin
    assert config.args == (["-s"] if expected_stdin else ["-c"])


def test_unix_shell_resolution_prefers_bin_bash_then_path_then_sh(monkeypatch):
    """TS `getShellConfig()` Unix branch: /bin/bash, then bash on PATH, then sh.

    The final `sh` fallback is what keeps pi usable on images without bash; the
    port previously hard-coded /bin/bash, which would have failed outright.
    """
    assert shell_module.get_shell_config() == shell_module.ShellConfig(shell="/bin/bash", args=["-c"])

    monkeypatch.setattr(os.path, "exists", lambda path: False)
    monkeypatch.setattr(shell_module, "_find_bash_on_path", lambda: "/usr/local/bin/bash")
    assert shell_module.get_shell_config() == shell_module.ShellConfig(shell="/usr/local/bin/bash", args=["-c"])

    monkeypatch.setattr(shell_module, "_find_bash_on_path", lambda: None)
    assert shell_module.get_shell_config() == shell_module.ShellConfig(shell="sh", args=["-c"])


async def test_running_child_is_tracked_for_shutdown_and_untracked_when_it_exits(tmp_path):
    """Port of the `trackDetachedChildPid` calls in `src/tools/bash.ts`.

    The child is spawned in its own session, so a shutdown signal reaching the
    parent never reaches it: the pid has to be registered while it runs so
    `killTrackedDetachedChildren` can reap it.
    """
    tool = create_bash_tool(str(tmp_path))
    signal = AbortSignal()
    seen_while_running: set[int] = set()

    async def abort_once_tracked():
        for _ in range(200):
            if _tracked_detached_child_pids:
                seen_while_running.update(_tracked_detached_child_pids)
                break
            await asyncio.sleep(0.01)
        signal.abort()

    watcher = asyncio.ensure_future(abort_once_tracked())

    with pytest.raises(RuntimeError):
        await tool.execute("call-track", {"command": "sleep 5"}, signal)
    await watcher

    assert seen_while_running
    assert not _tracked_detached_child_pids


async def test_completed_child_is_untracked(tmp_path):
    tool = create_bash_tool(str(tmp_path))

    await tool.execute("call-untrack", {"command": "echo done"})

    assert not _tracked_detached_child_pids
