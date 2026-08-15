"""Python port of `packages/coding-agent/test/external-editor.test.ts`."""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pi_coding_agent.modes.interactive.external_editor import (
    ExternalEditorOptions,
    ExternalEditorResult,
    edit_in_external_editor,
)

EDITOR_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "fake_external_editor.py"


@dataclass
class EditorCapture:
    file_path: str
    content: str
    entries: list[str]
    directory_mode: int


async def run_external_editor(
    tmp_path: Path, fixture_flag: str | None = None
) -> tuple[ExternalEditorResult, EditorCapture]:
    capture_path = tmp_path / "capture.json"
    command = f"{sys.executable} {EDITOR_FIXTURE_PATH} {capture_path}"
    if fixture_flag:
        command += f" {fixture_flag}"

    result = await edit_in_external_editor(ExternalEditorOptions(command=command, content="original"))
    raw = json.loads(capture_path.read_text(encoding="utf-8"))
    capture = EditorCapture(
        file_path=raw["filePath"],
        content=raw["content"],
        entries=raw["entries"],
        directory_mode=raw["directoryMode"],
    )
    return result, capture


async def test_edits_a_prompt_inside_a_private_temporary_directory(tmp_path: Path) -> None:
    result, capture = await run_external_editor(tmp_path)
    directory = Path(capture.file_path).parent

    assert result == ExternalEditorResult(status="complete", content="edited")
    assert str(directory.parent) == tempfile.gettempdir()
    assert directory.name.startswith("pi-editor-")
    assert len(directory.name) > len("pi-editor-")
    assert Path(capture.file_path).name == "prompt.md"
    assert capture.entries == ["prompt.md"]
    assert capture.content == "original"
    if sys.platform != "win32":
        assert capture.directory_mode & 0o077 == 0
    assert not directory.exists()


async def test_keeps_the_original_content_when_the_editor_fails(tmp_path: Path) -> None:
    result, capture = await run_external_editor(tmp_path, "--fail")

    assert result == ExternalEditorResult(status="failed")
    assert not Path(capture.file_path).parent.exists()


async def test_returns_empty_content_when_the_editor_clears_the_prompt(tmp_path: Path) -> None:
    result, _capture = await run_external_editor(tmp_path, "--empty")

    assert result == ExternalEditorResult(status="complete", content="")
