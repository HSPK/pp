"""Python port of `packages/coding-agent/test/bash-close-hang-windows.test.ts`.

The TypeScript suite is `describe.skipIf(process.platform !== "win32")`, so
these cases only run on Windows; the same guard applies here. They pin that a
shell whose grandchild inherits stdio and outlives it still resolves as soon as
the shell itself exits.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
from pathlib import Path

import pytest
from pi_ai.utils.abort import AbortController

from pi_coding_agent.core.bash_executor import create_local_bash_operations, execute_bash_with_operations
from pi_coding_agent.tools.bash import create_bash_tool

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only child-process close handling")


def to_bash_single_quoted_arg(value: str) -> str:
    escaped = value.replace("\\", "/").replace("'", "'\"'\"'")
    return f"'{escaped}'"


def create_inherited_stdio_command(pid_file: str) -> str:
    """A shell command that leaves a detached grandchild holding the stdio handles."""
    script = (
        "import subprocess,sys;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],"
        "creationflags=getattr(subprocess,'DETACHED_PROCESS',0));"
        "open(sys.argv[1],'w').write(str(child.pid));"
        "print('child-exiting')"
    )
    return f'{to_bash_single_quoted_arg(sys.executable)} -c "{script}" {to_bash_single_quoted_arg(pid_file)}'


def cleanup_detached_child(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    if pid <= 0:
        return
    # The process may have already exited.
    with contextlib.suppress(OSError):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def get_text_output(result: object) -> str:
    content = getattr(result, "content", None) or []
    return "\n".join(getattr(block, "text", "") or "" for block in content if getattr(block, "type", None) == "text")


async def test_execute_bash_resolves_after_the_shell_exits(tmp_path: Path) -> None:
    pid_file = tmp_path / "executor-grandchild.pid"
    command = create_inherited_stdio_command(str(pid_file))
    controller = AbortController()

    try:
        result = await asyncio.wait_for(
            execute_bash_with_operations(
                command,
                str(Path.cwd()),
                create_local_bash_operations(),
                signal=controller.signal,
            ),
            timeout=3,
        )
        assert "child-exiting" in result.output
        assert result.exit_code == 0
        assert result.cancelled is False
    finally:
        controller.abort()
        cleanup_detached_child(pid_file)


async def test_bash_tool_resolves_after_the_shell_exits(tmp_path: Path) -> None:
    pid_file = tmp_path / "tool-grandchild.pid"
    command = create_inherited_stdio_command(str(pid_file))
    controller = AbortController()
    bash_tool = create_bash_tool(str(tmp_path))

    try:
        result = await asyncio.wait_for(
            bash_tool.execute("test-call", {"command": command}, controller.signal),
            timeout=3,
        )
        assert "child-exiting" in get_text_output(result)
    finally:
        controller.abort()
        cleanup_detached_child(pid_file)
