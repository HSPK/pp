"""Tests for the one-time startup migrations.

These assert on the real filesystem layout after each migration, not just the
return values: the point of a migration is where the bytes end up.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from pi_coding_agent.migrations import (
    MigrationResult,
    check_deprecated_extension_dirs,
    format_deprecation_warnings,
    migrate_auth_to_auth_json,
    migrate_commands_to_prompts,
    migrate_extension_system,
    migrate_keybindings_config_file,
    migrate_sessions_from_agent_root,
    migrate_tools_to_bin,
    run_migrations,
    show_deprecation_warnings,
)


@pytest.fixture
def agent_dir(tmp_path: Path) -> Path:
    path = tmp_path / "agent"
    path.mkdir()
    return path


# ---------------------------------------------------------------------------
# auth.json
# ---------------------------------------------------------------------------


def test_oauth_json_is_folded_into_auth_json(agent_dir: Path) -> None:
    (agent_dir / "oauth.json").write_text(json.dumps({"anthropic": {"access": "tok", "refresh": "r"}}))

    providers = migrate_auth_to_auth_json(str(agent_dir))

    assert providers == ["anthropic"]
    auth = json.loads((agent_dir / "auth.json").read_text())
    assert auth["anthropic"] == {"type": "oauth", "access": "tok", "refresh": "r"}
    assert not (agent_dir / "oauth.json").exists()
    assert (agent_dir / "oauth.json.migrated").exists()


def test_settings_api_keys_are_moved_and_removed(agent_dir: Path) -> None:
    settings_path = agent_dir / "settings.json"
    settings_path.write_text(json.dumps({"apiKeys": {"openai": "sk-1"}, "theme": "dark"}))

    providers = migrate_auth_to_auth_json(str(agent_dir))

    assert providers == ["openai"]
    assert json.loads((agent_dir / "auth.json").read_text())["openai"] == {
        "type": "api_key",
        "key": "sk-1",
    }
    remaining = json.loads(settings_path.read_text())
    assert "apiKeys" not in remaining
    assert remaining["theme"] == "dark", "unrelated settings must survive"


def test_oauth_wins_over_a_settings_api_key_for_the_same_provider(agent_dir: Path) -> None:
    (agent_dir / "oauth.json").write_text(json.dumps({"anthropic": {"access": "tok"}}))
    (agent_dir / "settings.json").write_text(json.dumps({"apiKeys": {"anthropic": "sk-1"}}))

    migrate_auth_to_auth_json(str(agent_dir))

    assert json.loads((agent_dir / "auth.json").read_text())["anthropic"]["type"] == "oauth"


def test_auth_json_is_written_owner_only(agent_dir: Path) -> None:
    """Credentials must not be world-readable."""
    (agent_dir / "oauth.json").write_text(json.dumps({"anthropic": {"access": "tok"}}))
    migrate_auth_to_auth_json(str(agent_dir))
    mode = stat.S_IMODE(os.stat(agent_dir / "auth.json").st_mode)
    assert mode == 0o600


def test_existing_auth_json_is_never_overwritten(agent_dir: Path) -> None:
    (agent_dir / "auth.json").write_text(json.dumps({"existing": {"type": "api_key"}}))
    (agent_dir / "oauth.json").write_text(json.dumps({"anthropic": {"access": "tok"}}))

    assert migrate_auth_to_auth_json(str(agent_dir)) == []
    assert json.loads((agent_dir / "auth.json").read_text()) == {"existing": {"type": "api_key"}}
    assert (agent_dir / "oauth.json").exists(), "nothing should be consumed"


def test_malformed_oauth_json_is_skipped(agent_dir: Path) -> None:
    (agent_dir / "oauth.json").write_text("{not json")
    assert migrate_auth_to_auth_json(str(agent_dir)) == []
    assert not (agent_dir / "auth.json").exists()


def test_nothing_to_migrate_writes_nothing(agent_dir: Path) -> None:
    assert migrate_auth_to_auth_json(str(agent_dir)) == []
    assert not (agent_dir / "auth.json").exists()


# ---------------------------------------------------------------------------
# stray sessions
# ---------------------------------------------------------------------------


def _session_file(agent_dir: Path, name: str, cwd: str) -> Path:
    path = agent_dir / name
    header = json.dumps({"type": "session", "id": "abc", "cwd": cwd, "timestamp": "t"})
    path.write_text(f"{header}\n")
    return path


def test_stray_session_is_filed_under_its_project(agent_dir: Path) -> None:
    _session_file(agent_dir, "s1.jsonl", "/home/u/proj")

    migrate_sessions_from_agent_root(str(agent_dir))

    moved = agent_dir / "sessions" / "--home-u-proj--" / "s1.jsonl"
    assert moved.exists()
    assert not (agent_dir / "s1.jsonl").exists()


def test_session_encoding_matches_the_session_manager(agent_dir: Path, tmp_path: Path) -> None:
    """The migration must land exactly where the session manager looks."""
    from pi_coding_agent.core.session_manager import get_default_session_dir

    cwd = str(tmp_path / "myproject")
    _session_file(agent_dir, "s1.jsonl", cwd)
    migrate_sessions_from_agent_root(str(agent_dir))

    expected = Path(get_default_session_dir(cwd, str(agent_dir)))
    assert (expected / "s1.jsonl").exists()


def test_existing_target_is_not_clobbered(agent_dir: Path) -> None:
    _session_file(agent_dir, "s1.jsonl", "/home/u/proj")
    target_dir = agent_dir / "sessions" / "--home-u-proj--"
    target_dir.mkdir(parents=True)
    (target_dir / "s1.jsonl").write_text("original")

    migrate_sessions_from_agent_root(str(agent_dir))

    assert (target_dir / "s1.jsonl").read_text() == "original"
    assert (agent_dir / "s1.jsonl").exists(), "the source is left in place"


def test_files_without_a_session_header_are_left_alone(agent_dir: Path) -> None:
    stray = agent_dir / "notes.jsonl"
    stray.write_text(json.dumps({"type": "other"}) + "\n")
    migrate_sessions_from_agent_root(str(agent_dir))
    assert stray.exists()


def test_empty_and_malformed_files_are_skipped(agent_dir: Path) -> None:
    (agent_dir / "empty.jsonl").write_text("")
    (agent_dir / "bad.jsonl").write_text("{not json\n")
    migrate_sessions_from_agent_root(str(agent_dir))
    assert (agent_dir / "empty.jsonl").exists()
    assert (agent_dir / "bad.jsonl").exists()


def test_missing_agent_dir_is_not_an_error(tmp_path: Path) -> None:
    migrate_sessions_from_agent_root(str(tmp_path / "nope"))


# ---------------------------------------------------------------------------
# commands/ -> prompts/
# ---------------------------------------------------------------------------


def test_commands_directory_is_renamed(tmp_path: Path) -> None:
    (tmp_path / "commands").mkdir()
    (tmp_path / "commands" / "x.md").write_text("hi")

    assert migrate_commands_to_prompts(str(tmp_path), "Global") is True
    assert (tmp_path / "prompts" / "x.md").read_text() == "hi"
    assert not (tmp_path / "commands").exists()


def test_existing_prompts_directory_blocks_the_rename(tmp_path: Path) -> None:
    (tmp_path / "commands").mkdir()
    (tmp_path / "prompts").mkdir()
    assert migrate_commands_to_prompts(str(tmp_path), "Global") is False
    assert (tmp_path / "commands").exists()


def test_missing_commands_directory_is_a_no_op(tmp_path: Path) -> None:
    assert migrate_commands_to_prompts(str(tmp_path), "Global") is False


def test_symlinked_commands_directory_is_migrated(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "x.md").write_text("hi")
    (tmp_path / "commands").symlink_to(real, target_is_directory=True)

    assert migrate_commands_to_prompts(str(tmp_path), "Global") is True
    assert (tmp_path / "prompts" / "x.md").read_text() == "hi"


# ---------------------------------------------------------------------------
# tools/ -> bin/
# ---------------------------------------------------------------------------


def test_managed_binaries_are_moved_to_bin(agent_dir: Path) -> None:
    tools = agent_dir / "tools"
    tools.mkdir()
    (tools / "fd").write_text("binary")
    (tools / "rg").write_text("binary")

    migrate_tools_to_bin(str(agent_dir))

    assert (agent_dir / "bin" / "fd").read_text() == "binary"
    assert (agent_dir / "bin" / "rg").read_text() == "binary"
    assert not (tools / "fd").exists()


def test_stale_binary_is_deleted_when_the_target_exists(agent_dir: Path) -> None:
    tools = agent_dir / "tools"
    tools.mkdir()
    (tools / "fd").write_text("old")
    bin_dir = agent_dir / "bin"
    bin_dir.mkdir()
    (bin_dir / "fd").write_text("current")

    migrate_tools_to_bin(str(agent_dir))

    assert (bin_dir / "fd").read_text() == "current"
    assert not (tools / "fd").exists()


def test_custom_tools_are_not_moved(agent_dir: Path) -> None:
    tools = agent_dir / "tools"
    tools.mkdir()
    (tools / "my-tool.js").write_text("code")

    migrate_tools_to_bin(str(agent_dir))

    assert (tools / "my-tool.js").exists()
    assert not (agent_dir / "bin" / "my-tool.js").exists()


def test_missing_tools_directory_is_a_no_op(agent_dir: Path) -> None:
    migrate_tools_to_bin(str(agent_dir))
    assert not (agent_dir / "bin").exists()


# ---------------------------------------------------------------------------
# keybindings
# ---------------------------------------------------------------------------


def test_missing_keybindings_file_is_a_no_op(agent_dir: Path) -> None:
    migrate_keybindings_config_file(str(agent_dir))
    assert not (agent_dir / "keybindings.json").exists()


def test_malformed_keybindings_file_is_left_alone(agent_dir: Path) -> None:
    path = agent_dir / "keybindings.json"
    path.write_text("[not an object]")
    migrate_keybindings_config_file(str(agent_dir))
    assert path.read_text() == "[not an object]"


def test_already_current_keybindings_are_not_rewritten(agent_dir: Path) -> None:
    path = agent_dir / "keybindings.json"
    original = json.dumps({"editor": {}})
    path.write_text(original)
    migrate_keybindings_config_file(str(agent_dir))
    assert path.read_text() == original


# ---------------------------------------------------------------------------
# deprecated directories
# ---------------------------------------------------------------------------


def test_hooks_directory_warns(tmp_path: Path) -> None:
    (tmp_path / "hooks").mkdir()
    warnings = check_deprecated_extension_dirs(str(tmp_path), "Global")
    assert any("hooks/" in warning for warning in warnings)


def test_tools_with_only_managed_binaries_does_not_warn(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "fd").write_text("x")
    (tools / "rg.exe").write_text("x")
    (tools / ".DS_Store").write_text("x")
    assert check_deprecated_extension_dirs(str(tmp_path), "Global") == []


def test_tools_with_custom_entries_warns(tmp_path: Path) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "fd").write_text("x")
    (tools / "custom.js").write_text("x")
    warnings = check_deprecated_extension_dirs(str(tmp_path), "Project")
    assert any("custom tools" in warning for warning in warnings)


def test_clean_directory_produces_no_warnings(tmp_path: Path) -> None:
    assert check_deprecated_extension_dirs(str(tmp_path), "Global") == []


def test_extension_system_checks_both_scopes(tmp_path: Path, agent_dir: Path) -> None:
    (agent_dir / "hooks").mkdir()
    project = tmp_path / "proj"
    (project / ".pi" / "hooks").mkdir(parents=True)

    warnings = migrate_extension_system(str(project), str(agent_dir))

    assert any(warning.startswith("Global") for warning in warnings)
    assert any(warning.startswith("Project") for warning in warnings)


# ---------------------------------------------------------------------------
# warning output
# ---------------------------------------------------------------------------


def test_no_warnings_produces_no_output(capsys: pytest.CaptureFixture[str]) -> None:
    show_deprecation_warnings([])
    assert capsys.readouterr().out == ""


def test_warnings_include_the_migration_links(capsys: pytest.CaptureFixture[str]) -> None:
    show_deprecation_warnings(["Global hooks/ directory found."])
    output = capsys.readouterr().out
    assert "Global hooks/ directory found." in output
    assert "Migration guide:" in output


def test_format_deprecation_warnings_is_empty_without_warnings() -> None:
    assert format_deprecation_warnings([]) == ""


# ---------------------------------------------------------------------------
# run_migrations
# ---------------------------------------------------------------------------


def test_run_migrations_reports_everything(tmp_path: Path, agent_dir: Path) -> None:
    (agent_dir / "oauth.json").write_text(json.dumps({"anthropic": {"access": "tok"}}))
    _session_file(agent_dir, "s1.jsonl", "/home/u/proj")
    (agent_dir / "hooks").mkdir()
    project = tmp_path / "proj"
    project.mkdir()

    result = run_migrations(str(project), str(agent_dir))

    assert isinstance(result, MigrationResult)
    assert result.migrated_auth_providers == ["anthropic"]
    assert any("hooks/" in warning for warning in result.deprecation_warnings)
    assert (agent_dir / "auth.json").exists()
    assert (agent_dir / "sessions" / "--home-u-proj--" / "s1.jsonl").exists()


def test_run_migrations_is_idempotent(tmp_path: Path, agent_dir: Path) -> None:
    (agent_dir / "oauth.json").write_text(json.dumps({"anthropic": {"access": "tok"}}))
    project = tmp_path / "proj"
    project.mkdir()

    first = run_migrations(str(project), str(agent_dir))
    second = run_migrations(str(project), str(agent_dir))

    assert first.migrated_auth_providers == ["anthropic"]
    assert second.migrated_auth_providers == []
    assert json.loads((agent_dir / "auth.json").read_text())["anthropic"]["type"] == "oauth"


def test_run_migrations_on_a_clean_install_does_nothing(tmp_path: Path, agent_dir: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    result = run_migrations(str(project), str(agent_dir))
    assert result.migrated_auth_providers == []
    assert result.deprecation_warnings == []
