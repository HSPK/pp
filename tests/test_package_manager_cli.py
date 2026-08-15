"""Tests for pi_coding_agent.cli.package_manager_cli.

Ported from packages/coding-agent/test/package-command-paths.test.ts and
the "source parsing"/help-output cases in package-manager-cli.ts. Cases
relying on TypeScript-only behavior with no Python equivalent are skipped,
per package_manager_cli.py's module docstring:

- Self-update (`update --self`/`--force`), model-catalog refresh
  (`update --models`), and their combinations/conflicts -- no Python
  equivalent (see config.py's and package_manager.py's module docstrings).
- The extension-driven `project_trust` hook and the interactive trust
  prompt -- package commands in this port load no extensions and there is
  no TUI to prompt with. Everything else about the trust decision is the
  same: an explicit `--approve`/`--no-approve`, a project with no
  trust-requiring resources, the persisted `ProjectTrustStore`
  (`trust.json`) and `defaultProjectTrust` are all honoured, and `update`
  uses the saved decision only.
- The interactive resource-config TUI cycling test (`ConfigSelectorComponent`)
  is ported below and drives the real component.
- `npm:` package sources in these CLI-path tests are replaced with local
  path sources, since npm has no Python equivalent.

Uses real, file-backed `SettingsManager.create()` (no in-memory injection)
so these tests also exercise the on-disk settings.json read/write path that
`handle_package_command`/`handle_config_command` use by default.
"""

import asyncio
import io
import json
import os

import pytest

from pi_coding_agent.cli.entry import main
from pi_coding_agent.cli.package_manager_cli import (
    handle_config_command,
    handle_package_command,
    parse_package_command,
)
from pi_coding_agent.core.config import APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR
from pi_coding_agent.core.package_manager import PathMetadata, ResolvedPaths, ResolvedResource
from pi_coding_agent.core.settings_manager import (
    InMemorySettingsStorage,
    SettingsManager,
    SettingsManagerCreateOptions,
)
from pi_coding_agent.core.trust_manager import ProjectTrustStore
from pi_coding_agent.modes.interactive.components.config_selector import ConfigSelectorComponent
from pi_coding_agent.modes.interactive.theme.theme import init_theme


def _run(coro):
    return asyncio.run(coro)


def _write(path: str, content: str = "") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def _ext() -> str:
    return "def pi_extension(pi):\n    pass\n"


# ---------------------------------------------------------------------------
# parse_package_command
# ---------------------------------------------------------------------------


def test_parse_package_command_returns_none_for_non_package_args():
    assert parse_package_command([]) is None
    assert parse_package_command(["run", "something"]) is None


def test_parse_package_command_uninstall_is_alias_for_remove():
    options = parse_package_command(["uninstall", "foo"])
    assert options is not None
    assert options.command == "remove"
    assert options.source == "foo"


def test_parse_package_command_local_and_approve_flags():
    options = parse_package_command(["install", "foo", "-l", "--approve"])
    assert options.local is True
    assert options.project_trust_override is True


def test_parse_package_command_local_flag_invalid_for_list():
    options = parse_package_command(["list", "-l"])
    assert options.invalid_option == "-l"


def test_parse_package_command_unknown_option_recorded():
    options = parse_package_command(["install", "--unknown"])
    assert options.invalid_option == "--unknown"


def test_parse_package_command_extra_positional_recorded_as_invalid_argument():
    options = parse_package_command(["install", "foo", "bar"])
    assert options.source == "foo"
    assert options.invalid_argument == "bar"


# ---------------------------------------------------------------------------
# install / remove: persisted settings.json round trip (file-backed SettingsManager)
# ---------------------------------------------------------------------------


async def test_install_persists_global_relative_local_package_path(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = cwd / "packages" / "local-package"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["install", "./packages/local-package"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )
    assert code == 0, err.getvalue()

    settings_path = agent_dir / "settings.json"
    settings = json.loads(settings_path.read_text())
    assert len(settings.get("packages") or []) == 1
    stored = settings["packages"][0]
    resolved_from_settings = os.path.realpath(os.path.join(str(agent_dir), stored))
    assert resolved_from_settings == os.path.realpath(str(pkg_dir))


async def test_remove_local_package_using_trailing_slash(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = tmp_path / "local-package"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["install", f"{pkg_dir}/"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )
    assert code == 0, err.getvalue()
    settings_path = agent_dir / "settings.json"
    installed_settings = json.loads(settings_path.read_text())
    assert len(installed_settings.get("packages") or []) == 1

    code2 = await handle_package_command(
        ["remove", f"{pkg_dir}/"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )
    assert code2 == 0, err.getvalue()
    removed_settings = json.loads(settings_path.read_text())
    assert (removed_settings.get("packages") or []) == []


# ---------------------------------------------------------------------------
# list: untrusted vs. trusted project package settings
# ---------------------------------------------------------------------------


async def test_list_skips_untrusted_project_package_settings(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = tmp_path / "project-pkg"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())
    _write(str(cwd / CONFIG_DIR_NAME / "settings.json"), json.dumps({"packages": [str(pkg_dir)]}))

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["list"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    assert "No packages installed." in out.getvalue()
    assert "Project packages:" not in out.getvalue()


async def test_list_uses_default_project_trust_always(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = tmp_path / "project-pkg"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())
    _write(str(agent_dir / "settings.json"), json.dumps({"defaultProjectTrust": "always"}))
    _write(str(cwd / CONFIG_DIR_NAME / "settings.json"), json.dumps({"packages": [str(pkg_dir)]}))

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["list"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    assert "Project packages:" in out.getvalue()
    assert str(pkg_dir) in out.getvalue()
    assert "No packages installed." not in out.getvalue()


async def test_list_approve_flag_trusts_project_for_this_command(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = tmp_path / "project-pkg"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())
    _write(str(cwd / CONFIG_DIR_NAME / "settings.json"), json.dumps({"packages": [str(pkg_dir)]}))

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["list", "--approve"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    assert "Project packages:" in out.getvalue()
    assert str(pkg_dir) in out.getvalue()
    assert "No packages installed." not in out.getvalue()


# ---------------------------------------------------------------------------
# install -l: trust gating
# ---------------------------------------------------------------------------


async def test_install_local_blocked_when_project_untrusted(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    _write(str(cwd / CONFIG_DIR_NAME / "settings.json"), "{}")

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["install", "-l", "./local-package"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )
    assert code == 1
    assert "Project is not trusted. Use --approve to modify local package config." in err.getvalue()


async def test_install_local_allowed_initializes_fresh_project_settings(tmp_path):
    """TS: 'allows local package install to initialize fresh project settings'.

    Deliberately no `--approve`: a project with no `.pi` resources yet has
    nothing to distrust, so `resolveProjectTrusted` short-circuits to trusted
    and the first `install -l` is allowed to create `.pi/settings.json`.
    """
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = tmp_path / "local-package"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["install", "-l", str(pkg_dir)], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )
    assert code == 0, err.getvalue()

    settings_path = cwd / CONFIG_DIR_NAME / "settings.json"
    settings = json.loads(settings_path.read_text())
    assert len(settings.get("packages") or []) == 1
    stored = settings["packages"][0]
    resolved = os.path.realpath(os.path.join(str(cwd / CONFIG_DIR_NAME), stored))
    assert resolved == os.path.realpath(str(pkg_dir))


async def test_list_uses_remembered_project_trust(tmp_path):
    """TS: 'uses remembered project trust for list'."""
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = tmp_path / "project-pkg"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())
    _write(str(cwd / CONFIG_DIR_NAME / "settings.json"), json.dumps({"packages": [str(pkg_dir)]}))
    ProjectTrustStore(str(agent_dir)).set(str(cwd), True)

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["list"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    assert "Project packages:" in out.getvalue()
    assert str(pkg_dir) in out.getvalue()
    assert "No packages installed." not in out.getvalue()


async def test_trust_json_overrides_default_project_trust(tmp_path):
    """TS: 'lets trust.json override default project trust'."""
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = tmp_path / "project-pkg"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())
    _write(str(agent_dir / "settings.json"), json.dumps({"defaultProjectTrust": "always"}))
    _write(str(cwd / CONFIG_DIR_NAME / "settings.json"), json.dumps({"packages": [str(pkg_dir)]}))
    ProjectTrustStore(str(agent_dir)).set(str(cwd), False)

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["list"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    assert "No packages installed." in out.getvalue()
    assert "Project packages:" not in out.getvalue()


async def test_update_uses_saved_project_trust_and_ignores_the_default(tmp_path):
    """TS: 'uses saved project trust during update' + 'does not prompt or ask
    extensions for project trust during update'.

    TypeScript proves both by watching whether a stubbed npm binary ran. This
    port has no npm source, so the same decision is observed through a project
    package that only a trusted `update` can see: filtering `update <source>`
    against project settings reports "No matching package found" while the
    project is untrusted, and matches once the decision is saved. The
    `defaultProjectTrust: "always"` in global settings must not be enough on
    its own, which is exactly what `useSavedProjectTrustOnly` pins.
    """
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = tmp_path / "project-pkg"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())
    _write(str(agent_dir / "settings.json"), json.dumps({"defaultProjectTrust": "always"}))
    _write(str(cwd / CONFIG_DIR_NAME / "settings.json"), json.dumps({"packages": [str(pkg_dir)]}))

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["update", str(pkg_dir)], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )
    assert code == 1
    assert "No matching package found" in err.getvalue()

    ProjectTrustStore(str(agent_dir)).set(str(cwd), True)
    out2, err2 = io.StringIO(), io.StringIO()
    code2 = await handle_package_command(
        ["update", str(pkg_dir)], cwd=str(cwd), agent_dir=str(agent_dir), out=out2, err=err2
    )
    assert code2 == 0, err2.getvalue()
    assert "No matching package found" not in err2.getvalue()


# ---------------------------------------------------------------------------
# help output / friendly errors
# ---------------------------------------------------------------------------


async def test_install_help_shows_usage(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["install", "--help"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    assert "Usage:" in out.getvalue()
    assert f"{APP_NAME} install <source> [-l]" in out.getvalue()
    assert err.getvalue() == ""


async def test_install_unknown_option_shows_friendly_error(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["install", "--unknown"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )
    assert code == 1
    assert 'Unknown option --unknown for "install".' in err.getvalue()
    # Spelled out rather than routed through get_package_command_usage(): TS
    # asserts the literal usage string, so reusing the function under test here
    # would make the assertion pass whatever that function returns.
    assert f'Use "{APP_NAME} --help" or "{APP_NAME} install <source> [-l] [--approve|--no-approve]".' in err.getvalue()


async def test_install_missing_source_shows_friendly_error(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["install"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 1
    assert "Missing install source." in err.getvalue()
    assert f"Usage: {APP_NAME} install <source> [-l] [--approve|--no-approve]" in err.getvalue()
    assert "at " not in err.getvalue()


async def test_remove_missing_source_reports_no_match(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["remove"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 1
    assert "Missing remove source." in err.getvalue()


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


async def test_update_with_no_matching_source_reports_error(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    _write(str(agent_dir / "settings.json"), json.dumps({"packages": ["git:github.com/user/repo"]}))

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["update", "git:github.com/other/repo"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )
    assert code == 1
    assert "No matching package found" in err.getvalue()


async def test_update_with_no_configured_packages_reports_success_message(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(["update"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    assert "Updated packages" in out.getvalue()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


async def test_config_command_opens_the_resource_selector(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    calls: list[dict] = []

    async def fake_select_config(**kwargs):
        calls.append(kwargs)

    out, err = io.StringIO(), io.StringIO()
    code = await handle_config_command(
        ["config"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err, select_config=fake_select_config
    )

    assert code == 0
    assert len(calls) == 1
    assert calls[0]["write_scope"] == "global"
    assert calls[0]["cwd"] == str(cwd)
    assert calls[0]["agent_dir"] == str(agent_dir)
    assert set(calls[0]["resolved_paths"]) == {"global", "project"}
    assert calls[0]["project_mode_available"] is True


async def test_config_command_local_mode_uses_the_project_write_scope(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    calls: list[dict] = []

    async def fake_select_config(**kwargs):
        calls.append(kwargs)

    out, err = io.StringIO(), io.StringIO()
    code = await handle_config_command(
        ["config", "-l"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err, select_config=fake_select_config
    )

    assert code == 0
    assert calls[0]["write_scope"] == "project"


async def test_config_command_help_flag_returns_zero(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()

    out, err = io.StringIO(), io.StringIO()
    code = await handle_config_command(["config", "--help"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 0
    assert "Usage:" in out.getvalue()


async def test_config_command_local_blocked_when_untrusted(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    _write(str(cwd / CONFIG_DIR_NAME / "settings.json"), "{}")

    out, err = io.StringIO(), io.StringIO()
    code = await handle_config_command(["config", "-l"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err)
    assert code == 1
    assert "Project is not trusted" in err.getvalue()


async def test_handle_package_command_returns_none_for_non_package_args(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    assert await handle_package_command(["run"], cwd=str(cwd), agent_dir=str(agent_dir)) is None
    assert await handle_config_command(["run"], cwd=str(cwd), agent_dir=str(agent_dir)) is None


async def test_list_no_approve_flag_overrides_default_project_trust(tmp_path):
    """TS: 'overrides remembered trust for list with --no-approve'.

    `ProjectTrustStore` is fully ported (see `test_list_uses_remembered_project_trust`
    above), so this uses the real persisted store exactly like TypeScript,
    rather than substituting `defaultProjectTrust`: an explicit `--no-approve`
    must win over a remembered `trusted=True` decision.
    """
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = tmp_path / "project-pkg"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())
    _write(str(cwd / CONFIG_DIR_NAME / "settings.json"), json.dumps({"packages": [str(pkg_dir)]}))
    ProjectTrustStore(str(agent_dir)).set(str(cwd), True)

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["list", "--no-approve"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )
    assert code == 0
    assert "No packages installed." in out.getvalue()
    assert "Project packages:" not in out.getvalue()


# ---------------------------------------------------------------------------
# Through the real CLI entry point
#
# Every TypeScript case in `package-command-paths.test.ts` drives `main([...])`,
# not `handlePackageCommand` directly, so it also covers the pre-parseArgs
# dispatch: cwd/agent-dir resolution from the environment, and the subcommand
# returning its exit code instead of falling through to the agent parser. The
# cases above inject `cwd`/`agent_dir`/`out`/`err`, which is faster and lets
# them assert on captured streams, but leaves that wiring untested -- exactly
# the seam where an unawaited coroutine or a dropped return value hides. These
# two drive `pi_coding_agent.cli.entry.main` for real.
# ---------------------------------------------------------------------------


def test_main_dispatches_list_to_the_package_command(tmp_path, monkeypatch, capsys):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = tmp_path / "project-pkg"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())
    _write(str(cwd / CONFIG_DIR_NAME / "settings.json"), json.dumps({"packages": [str(pkg_dir)]}))
    monkeypatch.setenv(ENV_AGENT_DIR, str(agent_dir))
    monkeypatch.chdir(cwd)

    assert main(["list"]) == 0

    stdout = capsys.readouterr().out
    assert "No packages installed." in stdout
    assert "Project packages:" not in stdout


def test_main_dispatches_install_and_persists_a_relative_package_path(tmp_path, monkeypatch, capsys):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    pkg_dir = cwd / "packages" / "local-package"
    _write(str(pkg_dir / "extensions" / "main.py"), _ext())
    monkeypatch.setenv(ENV_AGENT_DIR, str(agent_dir))
    monkeypatch.chdir(cwd)

    assert main(["install", "./packages/local-package"]) == 0
    capsys.readouterr()

    settings = json.loads((agent_dir / "settings.json").read_text())
    assert len(settings.get("packages") or []) == 1
    stored = settings["packages"][0]
    assert os.path.realpath(os.path.join(str(agent_dir), stored)) == os.path.realpath(str(pkg_dir))


# ---------------------------------------------------------------------------
# TypeScript cases with no Python counterpart
#
# Each placeholder below names every `package-command-paths.test.ts` case it
# stands for, so no case disappears silently. The skip reason is the missing
# surface, not the effort.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "TS 'uses project_trust extensions for package commands' and 'does not prompt or ask extensions "
        "for project trust during update' pass extensionFactories into main() and have a `project_trust` "
        "hook answer {trusted: 'yes'}. This port's package commands never load extensions and so never "
        "pass a `trust_decider` into `resolve_project_trusted`, so there is no handler for the command to "
        "consult. The second case's other half -- that `update` consults neither the extension hook nor "
        "`defaultProjectTrust`, only a saved decision -- is pinned by "
        "test_update_uses_saved_project_trust_and_ignores_the_default."
    )
)
def test_project_trust_extension_hook_cases():
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "TS 'refreshes only model catalogs with update --models' and 'rejects update --models combined "
        "with another update target' need the remote model-catalog refresh (`update --models`), which "
        "this port omits (see package_manager.py's module docstring); `--models` is not an update target "
        "here, so there is neither a refresh to observe nor a conflict to reject."
    )
)
def test_update_models_catalog_refresh_cases():
    raise AssertionError("unreachable")


async def test_cycles_project_package_overrides_in_config_local_mode(tmp_path):
    # TypeScript's "cycles project package overrides in config local mode"
    # drives ConfigSelectorComponent directly (not `main(["config"])`), so no
    # terminal or event loop is involved and the case ports as-is. TS's
    # "npm:pi-tools" becomes "github:pi/pi-tools" only because this port has no
    # npm source; both are remote, so neither gets path-relativized.
    init_theme("dark")
    project_dir = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    package_root = tmp_path / "pkg"
    project_dir.mkdir()
    agent_dir.mkdir()

    storage = InMemorySettingsStorage()
    storage.with_lock("global", lambda _current: json.dumps({"packages": ["github:pi/pi-tools"]}))
    settings_manager = SettingsManager.from_storage(storage, SettingsManagerCreateOptions(project_trusted=True))

    resolved_paths = ResolvedPaths(
        extensions=[
            ResolvedResource(
                path=str(package_root / "extensions" / "bar.py"),
                enabled=True,
                metadata=PathMetadata(
                    source="github:pi/pi-tools", scope="user", origin="package", base_dir=str(package_root)
                ),
            )
        ]
    )

    selector = ConfigSelectorComponent(
        {"global": resolved_paths, "project": resolved_paths},
        settings_manager,
        str(project_dir),
        str(agent_dir),
        lambda: None,
        lambda: None,
        lambda: None,
        24,
        "project",
    )

    selector.get_resource_list().handle_input(" ")
    assert settings_manager.get_project_settings().get("packages") == [
        {"source": "github:pi/pi-tools", "autoload": False, "extensions": ["-extensions/bar.py"]}
    ]

    selector.get_resource_list().handle_input(" ")
    assert settings_manager.get_project_settings().get("packages") == [
        {"source": "github:pi/pi-tools", "autoload": False, "extensions": ["+extensions/bar.py"]}
    ]

    selector.get_resource_list().handle_input(" ")
    assert settings_manager.get_project_settings().get("packages") == []


@pytest.mark.skip(
    reason=(
        "The seven self-update cases -- 'allows explicit self-update checks when automatic version checks "
        "are disabled', 'retries a transient self-update version check', 'uses the update check version "
        "for forced self updates even when current', 'uses the current package name when the update check "
        "omits packageName', 'installs the active package name from the update check during self-update', "
        "'prints a pnpm metadata hint when self-update fails' and 'fails self-update when renamed npm "
        "package installation fails' -- all drive `update --self`/`--force` against the npm registry and a "
        "stubbed PATH npm binary. This port has no self-update at all (see package_manager.py), so there is "
        "no version check and no npm install to fail. The eighth case in the same TS `describe` block, "
        "'suggests the configured source when update input omits the npm prefix', is NOT a self-update case "
        "-- it drives plain `update <source>`, not `update --self`, and exercises "
        "`findSuggestedConfiguredSource`'s prefix-omitted suggestion, which also has a git-source branch in "
        "this port; it is ported for real below as "
        "test_update_suggests_configured_source_when_input_omits_scheme_prefix."
    )
)
def test_self_update_cases():
    raise AssertionError("unreachable")


async def test_update_suggests_configured_source_when_input_omits_scheme_prefix(tmp_path):
    """TS: 'suggests the configured source when update input omits the npm prefix'.

    TS configures `packages: ["npm:pi-formatter"]` and runs
    `update pi-formatter` (the source without its `npm:` prefix), which
    matches `pi-formatter` against the configured npm package's bare name in
    `findSuggestedConfiguredSource` and reports "Did you mean npm:pi-formatter?"
    without touching the stored settings. npm sources have no Python
    equivalent, but the same suggestion function
    (`_find_suggested_configured_source`) has an equivalent git-shorthand
    branch (`f"{host}/{path}"`), so this ports the identical claim -- omitting
    a configured source's scheme prefix on `update` reports "Did you mean
    <full source>?", exits 1, and leaves the configured package untouched --
    using a git source instead of an npm one.
    """
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()
    settings_path = agent_dir / "settings.json"
    _write(str(settings_path), json.dumps({"packages": ["git:github.com/user/pi-formatter"]}, indent=2))

    out, err = io.StringIO(), io.StringIO()
    code = await handle_package_command(
        ["update", "github.com/user/pi-formatter"], cwd=str(cwd), agent_dir=str(agent_dir), out=out, err=err
    )

    assert code == 1
    assert "Did you mean git:github.com/user/pi-formatter?" in err.getvalue()
    assert "Updated github.com/user/pi-formatter" not in out.getvalue()

    settings = json.loads(settings_path.read_text())
    assert settings.get("packages") == ["git:github.com/user/pi-formatter"]
