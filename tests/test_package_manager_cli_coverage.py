"""Additional coverage tests for pi_coding_agent.cli.package_manager_cli.

Targets uncovered branches left after the main CLI test suite:
- print_config_command_help called directly
- print_package_command_help for "remove", "update", "list"
- parse_package_command: --no-approve / -na / -a flags
- resolve_project_trust: "never" and "always" defaults, explicit overrides
- create_command_settings_manager helper
- report_settings_errors with actual errors (malformed settings.json)
- handle_package_command: invalid_argument and conflicting_options rejections
- handle_package_command: external settings_manager supplied (skips
  internal SettingsManager creation and trust gate)
- handle_package_command remove: success path
- handle_package_command list: user-only, project-only, mixed, filtered
  entries, installed_path display
- handle_package_command update: source-specific update
- handle_config_command: unknown option, unexpected positional arg,
  approve / no-approve flags, external settings_manager supplied, and the
  write scope / project-mode flag it hands to `select_config`
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

from pi_coding_agent.cli.package_manager_cli import (
    PackageCommandOptions,
    create_command_settings_manager,
    get_package_command_usage,
    handle_config_command,
    handle_package_command,
    parse_package_command,
    print_config_command_help,
    print_package_command_help,
    report_settings_errors,
    resolve_project_trust,
)
from pi_coding_agent.core.config import CONFIG_DIR_NAME
from pi_coding_agent.core.settings_manager import SettingsManager, SettingsManagerCreateOptions
from pi_coding_agent.core.trust_manager import ProjectTrustStore


def _write(path: str, content: str = "") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def _ext() -> str:
    return "def pi_extension(pi):\n    pass\n"


class _SelectConfigRecorder:
    """Stands in for `cli.config_selector.select_config`.

    Shaped like the real function: an async callable taking exactly its
    keyword-only arguments, so a caller that forgets to await it, or passes a
    different argument set, fails here rather than passing silently.
    """

    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    async def __call__(
        self,
        *,
        resolved_paths,
        settings_manager,
        cwd: str,
        agent_dir: str,
        write_scope: str,
        project_mode_available: bool,
    ) -> None:
        self.kwargs.append(
            {
                "resolved_paths": resolved_paths,
                "settings_manager": settings_manager,
                "cwd": cwd,
                "agent_dir": agent_dir,
                "write_scope": write_scope,
                "project_mode_available": project_mode_available,
            }
        )


# ---------------------------------------------------------------------------
# print_config_command_help
# ---------------------------------------------------------------------------


def test_print_config_command_help_direct():
    out = io.StringIO()
    print_config_command_help(out)
    text = out.getvalue()
    assert "Usage:" in text
    assert "Open the resource configuration TUI" in text
    assert "settings.json" in text
    assert "-l, --local" in text


# ---------------------------------------------------------------------------
# print_package_command_help: remove / update / list branches
# ---------------------------------------------------------------------------


def test_print_package_command_help_remove():
    out = io.StringIO()
    print_package_command_help("remove", out)
    text = out.getvalue()
    assert "remove <source>" in text
    assert "uninstall" in text


def test_print_package_command_help_update():
    out = io.StringIO()
    print_package_command_help("update", out)
    text = out.getvalue()
    assert "update" in text
    # Either "not supported" or "self-update" — just confirm it printed something useful
    assert len(text) > 50


def test_print_package_command_help_list():
    out = io.StringIO()
    print_package_command_help("list", out)
    text = out.getvalue()
    assert "list" in text
    assert "--approve" in text


# ---------------------------------------------------------------------------
# parse_package_command: --no-approve / -na / -a
# ---------------------------------------------------------------------------


def test_parse_package_command_long_no_approve():
    opts = parse_package_command(["install", "foo", "--no-approve"])
    assert opts is not None
    assert opts.project_trust_override is False


def test_parse_package_command_short_no_approve():
    opts = parse_package_command(["install", "foo", "-na"])
    assert opts is not None
    assert opts.project_trust_override is False


def test_parse_package_command_short_approve():
    opts = parse_package_command(["install", "foo", "-a"])
    assert opts is not None
    assert opts.project_trust_override is True


def test_parse_package_command_update_no_approve():
    opts = parse_package_command(["update", "--no-approve"])
    assert opts is not None
    assert opts.project_trust_override is False


# ---------------------------------------------------------------------------
# resolve_project_trust: "never" and "always" defaults
# ---------------------------------------------------------------------------


def _trust_requiring_project(tmp_path) -> tuple[Path, Path]:
    """A cwd that actually has something to trust, plus a fresh agent dir.

    `resolveProjectTrusted` short-circuits to trusted when the project has no
    trust-requiring resources, so every case below that is *about* the
    remembered/default decision has to give the project a `.pi/settings.json`
    first -- otherwise it would pass for the wrong reason.
    """
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    _write(str(cwd / CONFIG_DIR_NAME / "settings.json"), "{}")
    return cwd, agent_dir


async def test_resolve_project_trust_never_default(tmp_path):
    cwd, agent_dir = _trust_requiring_project(tmp_path)
    _write(str(agent_dir / "settings.json"), json.dumps({"defaultProjectTrust": "never"}))
    sm = SettingsManager.create(str(cwd), str(agent_dir), SettingsManagerCreateOptions(project_trusted=False))
    assert await resolve_project_trust(sm, None, cwd=str(cwd), agent_dir=str(agent_dir)) is False


async def test_resolve_project_trust_always_default(tmp_path):
    cwd, agent_dir = _trust_requiring_project(tmp_path)
    _write(str(agent_dir / "settings.json"), json.dumps({"defaultProjectTrust": "always"}))
    sm = SettingsManager.create(str(cwd), str(agent_dir), SettingsManagerCreateOptions(project_trusted=False))
    assert await resolve_project_trust(sm, None, cwd=str(cwd), agent_dir=str(agent_dir)) is True


async def test_resolve_project_trust_explicit_true_override(tmp_path):
    cwd, agent_dir = _trust_requiring_project(tmp_path)
    sm = SettingsManager.create(str(cwd), str(agent_dir), SettingsManagerCreateOptions(project_trusted=False))
    assert await resolve_project_trust(sm, True, cwd=str(cwd), agent_dir=str(agent_dir)) is True


async def test_resolve_project_trust_explicit_false_override(tmp_path):
    cwd, agent_dir = _trust_requiring_project(tmp_path)
    _write(str(agent_dir / "settings.json"), json.dumps({"defaultProjectTrust": "always"}))
    sm = SettingsManager.create(str(cwd), str(agent_dir), SettingsManagerCreateOptions(project_trusted=False))
    # Explicit False wins over "always" default
    assert await resolve_project_trust(sm, False, cwd=str(cwd), agent_dir=str(agent_dir)) is False


async def test_resolve_project_trust_trusts_a_project_with_nothing_to_trust(tmp_path):
    """`resolveProjectTrusted` returns true before consulting anything else
    when `hasTrustRequiringProjectResources(cwd)` is false."""
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    _write(str(agent_dir / "settings.json"), json.dumps({"defaultProjectTrust": "never"}))
    sm = SettingsManager.create(str(cwd), str(agent_dir), SettingsManagerCreateOptions(project_trusted=False))
    assert await resolve_project_trust(sm, None, cwd=str(cwd), agent_dir=str(agent_dir)) is True


async def test_resolve_project_trust_uses_the_remembered_decision(tmp_path):
    cwd, agent_dir = _trust_requiring_project(tmp_path)
    ProjectTrustStore(str(agent_dir)).set(str(cwd), True)
    sm = SettingsManager.create(str(cwd), str(agent_dir), SettingsManagerCreateOptions(project_trusted=False))
    assert await resolve_project_trust(sm, None, cwd=str(cwd), agent_dir=str(agent_dir)) is True


async def test_remembered_distrust_beats_the_always_default(tmp_path):
    cwd, agent_dir = _trust_requiring_project(tmp_path)
    _write(str(agent_dir / "settings.json"), json.dumps({"defaultProjectTrust": "always"}))
    ProjectTrustStore(str(agent_dir)).set(str(cwd), False)
    sm = SettingsManager.create(str(cwd), str(agent_dir), SettingsManagerCreateOptions(project_trusted=False))
    assert await resolve_project_trust(sm, None, cwd=str(cwd), agent_dir=str(agent_dir)) is False


# ---------------------------------------------------------------------------
# create_command_settings_manager
# ---------------------------------------------------------------------------


async def test_create_command_settings_manager_approve(tmp_path):
    cwd, agent_dir = _trust_requiring_project(tmp_path)
    sm = await create_command_settings_manager(str(cwd), str(agent_dir), project_trust_override=True)
    assert sm.is_project_trusted() is True


async def test_create_command_settings_manager_no_approve(tmp_path):
    cwd, agent_dir = _trust_requiring_project(tmp_path)
    sm = await create_command_settings_manager(str(cwd), str(agent_dir), project_trust_override=False)
    assert sm.is_project_trusted() is False


async def test_create_command_settings_manager_no_override_defaults_untrusted(tmp_path):
    cwd, agent_dir = _trust_requiring_project(tmp_path)
    sm = await create_command_settings_manager(str(cwd), str(agent_dir), project_trust_override=None)
    assert sm.is_project_trusted() is False


async def test_create_command_settings_manager_use_saved_project_trust_only(tmp_path):
    """`useSavedProjectTrustOnly` (what `update` passes) must ignore
    `defaultProjectTrust` and consult only the flag or the saved decision."""
    cwd, agent_dir = _trust_requiring_project(tmp_path)
    _write(str(agent_dir / "settings.json"), json.dumps({"defaultProjectTrust": "always"}))

    sm = await create_command_settings_manager(
        str(cwd), str(agent_dir), project_trust_override=None, use_saved_project_trust_only=True
    )
    assert sm.is_project_trusted() is False

    ProjectTrustStore(str(agent_dir)).set(str(cwd), True)
    saved = await create_command_settings_manager(
        str(cwd), str(agent_dir), project_trust_override=None, use_saved_project_trust_only=True
    )
    assert saved.is_project_trusted() is True

    overridden = await create_command_settings_manager(
        str(cwd), str(agent_dir), project_trust_override=False, use_saved_project_trust_only=True
    )
    assert overridden.is_project_trusted() is False


# ---------------------------------------------------------------------------
# report_settings_errors: with actual parse errors
# ---------------------------------------------------------------------------


def test_report_settings_errors_with_malformed_global_settings(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    _write(str(agent_dir / "settings.json"), "not valid json {{{{")
    sm = SettingsManager.create(str(cwd), str(agent_dir), SettingsManagerCreateOptions(project_trusted=False))
    err = io.StringIO()
    report_settings_errors(sm, "test ctx", err)
    assert "Warning" in err.getvalue()


def test_report_settings_errors_empty_when_no_errors(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    sm = SettingsManager.create(str(cwd), str(agent_dir), SettingsManagerCreateOptions(project_trusted=False))
    err = io.StringIO()
    report_settings_errors(sm, "ctx", err)
    assert err.getvalue() == ""


# ---------------------------------------------------------------------------
# handle_package_command: invalid_argument
# ---------------------------------------------------------------------------


async def test_handle_package_command_extra_argument_rejected(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["install", "foo", "bar"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )
    assert code == 1
    assert "Unexpected argument bar" in err.getvalue()


async def test_handle_package_command_extra_arg_includes_usage(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    out, err = io.StringIO(), io.StringIO()
    await handle_package_command(
        ["remove", "source", "extra"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )
    assert get_package_command_usage("remove") in err.getvalue()


# ---------------------------------------------------------------------------
# handle_package_command: conflicting_options path (monkeypatched)
# ---------------------------------------------------------------------------


async def test_conflicting_options_shows_message_and_usage(tmp_path, monkeypatch):
    from pi_coding_agent.cli import package_manager_cli

    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()

    def _fake_parse(args):
        return PackageCommandOptions(
            command="install",
            conflicting_options="--approve and --no-approve are mutually exclusive.",
        )

    monkeypatch.setattr(package_manager_cli, "parse_package_command", _fake_parse)
    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["install", "foo"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 1
    assert "mutually exclusive" in err.getvalue()


# ---------------------------------------------------------------------------
# handle_package_command: external settings_manager supplied
# (owns_settings_manager = False → internal trust check is bypassed)
# ---------------------------------------------------------------------------


async def test_install_with_external_settings_manager_bypasses_trust_gate(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = tmp_path / "pkg"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())

    sm = await create_command_settings_manager(str(cwd), str(agent_dir), project_trust_override=True)
    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["install", str(pkg_dir)],
        cwd=str(cwd),
        agent_dir=str(agent_dir),
        settings_manager=sm,
        out=out,
        err=err,
    )
    assert code == 0, err.getvalue()
    assert f"Installed {pkg_dir}" in out.getvalue()


async def test_list_with_external_settings_manager(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    sm = await create_command_settings_manager(str(cwd), str(agent_dir), project_trust_override=None)
    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["list"],
        cwd=str(cwd),
        agent_dir=str(agent_dir),
        settings_manager=sm,
        out=out,
        err=err,
    )
    assert code == 0
    assert "No packages installed." in out.getvalue()


# ---------------------------------------------------------------------------
# handle_package_command remove: success path
# ---------------------------------------------------------------------------


async def test_remove_after_install_returns_success_and_message(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = tmp_path / "local-package"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["install", str(pkg_dir)], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )
    assert code == 0

    out2, err2 = io.StringIO(), io.StringIO()
    code2 = await handle_package_command(
        ["remove", str(pkg_dir)], cwd=str(cwd), agent_dir=str(agent_dir), out=out2, err=err2
    )
    assert code2 == 0
    assert f"Removed {pkg_dir}" in out2.getvalue()


# ---------------------------------------------------------------------------
# handle_package_command list: user-only, project-only, mixed, filtered
# ---------------------------------------------------------------------------


async def test_list_user_packages_printed_with_header(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg = tmp_path / "user-pkg"
    _write(str(pkg / "extensions" / "main.py"), _ext())
    _write(str(agent_dir / "settings.json"), json.dumps({"packages": [str(pkg)]}))

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["list"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    text = out.getvalue()
    assert "User packages:" in text
    assert str(pkg) in text


async def test_list_project_packages_with_approve_flag(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg = tmp_path / "proj-pkg"
    _write(str(pkg / "extensions" / "main.py"), _ext())
    _write(
        str(cwd / CONFIG_DIR_NAME / "settings.json"),
        json.dumps({"packages": [str(pkg)]}),
    )

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["list", "--approve"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    text = out.getvalue()
    assert "Project packages:" in text
    assert str(pkg) in text


async def test_list_mixed_user_and_project_packages_has_separator(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()

    user_pkg = tmp_path / "u-pkg"
    _write(str(user_pkg / "extensions" / "main.py"), _ext())
    proj_pkg = tmp_path / "p-pkg"
    _write(str(proj_pkg / "extensions" / "main.py"), _ext())

    _write(str(agent_dir / "settings.json"), json.dumps({"packages": [str(user_pkg)]}))
    _write(
        str(cwd / CONFIG_DIR_NAME / "settings.json"),
        json.dumps({"packages": [str(proj_pkg)]}),
    )

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["list", "--approve"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    text = out.getvalue()
    assert "User packages:" in text
    assert "Project packages:" in text
    # blank line separator between the two sections
    lines = text.splitlines()
    assert "" in lines


async def test_list_filtered_package_shows_filtered_label(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg = tmp_path / "filtered-pkg"
    _write(str(pkg / "extensions" / "main.py"), _ext())
    _write(
        str(agent_dir / "settings.json"),
        json.dumps({"packages": [{"source": str(pkg), "autoload": True}]}),
    )

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["list"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    assert "(filtered)" in out.getvalue()


async def test_list_package_with_installed_path_shows_it(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg = tmp_path / "installed-pkg"
    _write(str(pkg / "extensions" / "main.py"), _ext())
    _write(str(agent_dir / "settings.json"), json.dumps({"packages": [str(pkg)]}))

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["list"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    # installed_path is the real directory (exists) → should appear indented
    assert str(pkg) in out.getvalue()


# ---------------------------------------------------------------------------
# handle_package_command update: source-specific update with offline mode
# ---------------------------------------------------------------------------


async def test_update_specific_source_offline_prints_updated_message(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_OFFLINE", "1")
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    _write(str(agent_dir / "settings.json"), json.dumps({"packages": ["git:github.com/user/repo"]}))

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["update", "git:github.com/user/repo"],
        cwd=str(cwd),
        agent_dir=str(agent_dir),
        out=out,
        err=err,
    )
    assert code == 0
    assert "Updated git:github.com/user/repo" in out.getvalue()


async def test_update_all_offline_prints_generic_message(tmp_path, monkeypatch):
    monkeypatch.setenv("PI_OFFLINE", "1")
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    _write(str(agent_dir / "settings.json"), json.dumps({"packages": ["git:github.com/user/repo"]}))

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["update"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    assert "Updated packages" in out.getvalue()


# ---------------------------------------------------------------------------
# handle_config_command: unknown option, unexpected arg, approve/no-approve
# ---------------------------------------------------------------------------


async def test_config_command_unknown_option_prints_error(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    out, err = io.StringIO(), io.StringIO()
    code = await handle_config_command(
        ["config", "--unknown"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )
    assert code == 1
    assert 'Unknown option --unknown for "config"' in err.getvalue()


async def test_config_command_unexpected_positional_arg_prints_error(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    out, err = io.StringIO(), io.StringIO()
    code = await handle_config_command(["config", "extra"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 1
    assert "Unexpected argument extra" in err.getvalue()


async def test_config_command_approve_flag_opens_the_selector(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    calls = _SelectConfigRecorder()
    out, err = io.StringIO(), io.StringIO()
    code = await handle_config_command(
        ["config", "--approve"],
        cwd=str(cwd),
        agent_dir=str(agent_dir),
        out=out,
        err=err,
        select_config=calls,
    )
    assert code == 0
    assert calls.kwargs[0]["project_mode_available"] is True


async def test_config_command_no_approve_flag_opens_the_selector_without_project_mode(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    calls = _SelectConfigRecorder()
    out, err = io.StringIO(), io.StringIO()
    code = await handle_config_command(
        ["config", "--no-approve"],
        cwd=str(cwd),
        agent_dir=str(agent_dir),
        out=out,
        err=err,
        select_config=calls,
    )
    assert code == 0
    assert calls.kwargs[0]["project_mode_available"] is False


async def test_config_command_with_external_settings_manager(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    sm = await create_command_settings_manager(str(cwd), str(agent_dir), project_trust_override=True)
    calls = _SelectConfigRecorder()
    out, err = io.StringIO(), io.StringIO()
    code = await handle_config_command(
        ["config"],
        cwd=str(cwd),
        agent_dir=str(agent_dir),
        settings_manager=sm,
        out=out,
        err=err,
        select_config=calls,
    )
    assert code == 0
    assert calls.kwargs[0]["settings_manager"] is sm


async def test_config_command_local_approve_trusted_opens_the_selector_in_project_scope(tmp_path):
    cwd, agent_dir = tmp_path / "cwd", tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    calls = _SelectConfigRecorder()
    out, err = io.StringIO(), io.StringIO()
    code = await handle_config_command(
        ["config", "-l", "--approve"],
        cwd=str(cwd),
        agent_dir=str(agent_dir),
        out=out,
        err=err,
        select_config=calls,
    )
    assert code == 0
    assert calls.kwargs[0]["write_scope"] == "project"
