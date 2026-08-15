"""Python port of `packages/coding-agent/test/session-file-invalid.test.ts`."""

from __future__ import annotations

from pathlib import Path

from _cli_process import run_cli


def test_prints_a_friendly_error_and_preserves_non_session_file_content(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    project_dir = tmp_path / "project"
    session_file = tmp_path / "not-a-session.log"
    original_content = '{"type":"event","data":"not a session"}\n'
    agent_dir.mkdir()
    project_dir.mkdir()
    session_file.write_text(original_content)

    result = run_cli(["--session", str(session_file), "-p", "hi"], str(project_dir), str(agent_dir))

    assert result.code == 1
    assert f"Error: Session file is not a valid pi session: {session_file}" in result.stderr
    assert "SessionManager.open" not in result.stderr
    # TS asserts the stderr has no stack frames (`not.toContain("at ")`); the Python
    # equivalent of a stack dump is a traceback header plus `  File "..."` frame lines.
    assert "Traceback" not in result.stderr
    assert '  File "' not in result.stderr
    assert session_file.read_text() == original_content
