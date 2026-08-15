"""Hand off prompt editing to the user's ``$EDITOR``.

Ported from ``packages/coding-agent/src/modes/interactive/external-editor.ts``.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ExternalEditorOptions:
    command: str
    content: str


@dataclass
class ExternalEditorResult:
    status: Literal["complete", "failed"]
    content: str = ""


def default_external_editor_command() -> str:
    return os.environ.get("VISUAL") or os.environ.get("EDITOR") or ("notepad" if sys.platform == "win32" else "nano")


async def edit_in_external_editor(options: ExternalEditorOptions) -> ExternalEditorResult:
    directory = tempfile.mkdtemp(prefix="pi-editor-")
    file_path = str(Path(directory) / "prompt.md")
    try:
        Path(file_path).write_text(options.content, encoding="utf-8")
        argv = options.command.split(" ")
        sys.stdout.write(f"Launching external editor: {options.command}\nPi will resume when the editor exits.\n")
        sys.stdout.flush()

        try:
            # `stdio: "inherit"` in TypeScript; the child owns the terminal
            # until it exits, which is exactly what the default here does.
            completed = subprocess.run(
                [*argv, file_path],
                shell=sys.platform == "win32",
                check=False,
            )
            exit_code: int | None = completed.returncode
        except OSError:
            exit_code = None

        if exit_code != 0:
            return ExternalEditorResult(status="failed")

        content = Path(file_path).read_text(encoding="utf-8")
        # TS strips a single trailing newline, not all of them.
        if content.endswith("\n"):
            content = content[:-1]
        return ExternalEditorResult(status="complete", content=content)
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(directory, ignore_errors=True)


__all__ = [
    "ExternalEditorOptions",
    "ExternalEditorResult",
    "default_external_editor_command",
    "edit_in_external_editor",
]
