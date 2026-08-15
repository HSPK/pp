"""Package manager CLI subcommands: install/remove/uninstall/update/list/config.

Python port of `packages/coding-agent/src/package-manager-cli.ts` (889 lines),
narrowed to the parts that have a Python equivalent -- see
`core/package_manager.py`'s module docstring for the underlying scope
decisions this CLI layer inherits:

- **No `npm:` sources.** `install`/`remove`/`update` surface
  `PackageManager.parse_source()`'s `ValueError` for `npm:`-prefixed input
  as a normal command error (exit code 1), rather than special-casing it.
- **No self-update / model-catalog refresh.** The TypeScript `update`
  subcommand's `--self`/`--models`/`--all` flags and the entire
  `getSelfUpdatePlan`/`runSelfUpdate`/`refreshModelCatalogs` machinery (pi
  self-installs itself via whatever package manager installed it; there is
  no Python equivalent story -- see `config.py`'s module docstring) are
  dropped. `update [source]` here only ever updates git-sourced packages
  (optionally filtered to one `source`); with no source it updates every
  configured git package. `--self`/`--models`/`--all`/`--force`/
  `--extension <source>` are consequently not recognized options.
- **`config` opens the resource TUI, like TypeScript.** `handle_config_command`
  resolves the global and (if trusted) project resource sets and hands them to
  `cli/config_selector.select_config`, which shows the same
  `ConfigSelectorComponent` the interactive mode uses. The only narrowing is
  TypeScript's `initTheme(..., enableWatcher = true)`: the theme file watcher
  is not ported.
- **No interactive trust prompt, but the same trust decision otherwise.**
  `create_command_settings_manager()` below mirrors
  `createCommandSettingsManager`: it resolves trust through the shared
  `core.project_trust.resolve_project_trusted`, so an explicit
  `--approve`/`--no-approve` wins, a project with no trust-requiring
  resources under `.pi`/`.agents/skills` is trusted, then the persisted
  `ProjectTrustStore` (`trust.json`) decides, then `defaultProjectTrust`
  ("always"/"never"), and otherwise trust is refused. `update` passes
  `use_saved_project_trust_only`, matching TypeScript, so it never promotes
  an untrusted project through `defaultProjectTrust`. The one narrowing is
  the terminal prompt: TypeScript falls back to an interactive "do you trust
  this project?" selector when stdin and stdout are both TTYs, and this port
  has no TUI (see the top-level README's "Not ported" list), so `has_ui` is
  always false and the "ask" default resolves to *untrusted*. The
  `project_trust` extension hook is likewise not consulted here, because
  package commands in this port do not load extensions.

All of `install`/`remove`/`update`/`list` are async (they call into
`PackageManager`'s async git operations); `handle_package_command`/
`handle_config_command` are the two async entry points `cli/entry.py` wires
in before falling back to the ordinary agent-run argument scanner, mirroring
`main.ts`'s `handlePackageCommand`/`handleConfigCommand` pre-parse dispatch.
"""

from __future__ import annotations

import sys
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import IO, Literal, Protocol

from pi_coding_agent.cli.config_selector import select_config as _select_config
from pi_coding_agent.core.config import APP_NAME, CONFIG_DIR_NAME
from pi_coding_agent.core.package_manager import PackageManager, ResolvedPaths
from pi_coding_agent.core.project_trust import create_project_trust_context, resolve_project_trusted
from pi_coding_agent.core.settings_manager import SettingsManager, SettingsManagerCreateOptions
from pi_coding_agent.core.trust_manager import ProjectTrustStore
from pi_coding_agent.modes.interactive.components.config_selector import ConfigWriteScope


class SelectConfig(Protocol):
    def __call__(
        self,
        *,
        resolved_paths: dict[str, ResolvedPaths],
        settings_manager: SettingsManager,
        cwd: str,
        agent_dir: str,
        write_scope: ConfigWriteScope,
        project_mode_available: bool,
    ) -> Awaitable[None]: ...


PackageCommand = Literal["install", "remove", "update", "list"]

_PACKAGE_COMMAND_NAMES = {"install", "remove", "uninstall", "update", "list"}


@dataclass
class PackageCommandOptions:
    command: PackageCommand
    source: str | None = None
    local: bool = False
    project_trust_override: bool | None = None
    help: bool = False
    invalid_option: str | None = None
    invalid_argument: str | None = None
    conflicting_options: str | None = None


def get_package_command_usage(command: PackageCommand) -> str:
    if command == "install":
        return f"{APP_NAME} install <source> [-l] [--approve|--no-approve]"
    if command == "remove":
        return f"{APP_NAME} remove <source> [-l] [--approve|--no-approve]"
    if command == "update":
        return f"{APP_NAME} update [source] [--approve|--no-approve]"
    return f"{APP_NAME} list [--approve|--no-approve]"


CONFIG_COMMAND_USAGE = f"{APP_NAME} config [-l] [--approve|--no-approve]"


def print_config_command_help(out: IO[str] | None = None) -> None:
    out = sys.stdout if out is None else out
    print(
        f"""Usage:
  {CONFIG_COMMAND_USAGE}

Open the resource configuration TUI to enable or disable package resources.
Without -l, starts in global settings (~/{CONFIG_DIR_NAME}/agent/settings.json).
Press Tab in the TUI to switch between global and project-local modes.

Options:
  -l, --local       Edit project overrides ({CONFIG_DIR_NAME}/settings.json)
  -a, --approve     Trust project-local files for this command with -l
  -na, --no-approve Ignore project-local files for this command with -l
""",
        file=out,
    )


def print_package_command_help(command: PackageCommand, out: IO[str] | None = None) -> None:
    out = sys.stdout if out is None else out
    if command == "install":
        print(
            f"""Usage:
  {get_package_command_usage("install")}

Install a package and add it to settings.

Options:
  -l, --local       Install project-locally ({CONFIG_DIR_NAME}/settings.json)
  -a, --approve     Trust project-local files for this command
  -na, --no-approve Ignore project-local files for this command

Examples:
  {APP_NAME} install git:github.com/user/repo
  {APP_NAME} install git:git@github.com:user/repo
  {APP_NAME} install https://github.com/user/repo
  {APP_NAME} install ssh://git@github.com/user/repo
  {APP_NAME} install ./local/path

Note: npm:<spec> sources are not supported by this port (no Python package
registry equivalent); use a git or local path source instead.
""",
            file=out,
        )
        return
    if command == "remove":
        print(
            f"""Usage:
  {get_package_command_usage("remove")}

Remove a package and its source from settings.
Alias: {APP_NAME} uninstall <source> [-l]

Options:
  -l, --local       Remove from project settings ({CONFIG_DIR_NAME}/settings.json)
  -a, --approve     Trust project-local files for this command
  -na, --no-approve Ignore project-local files for this command
""",
            file=out,
        )
        return
    if command == "update":
        print(
            f"""Usage:
  {get_package_command_usage("update")}

Update installed git-sourced packages.

Options:
  -a, --approve      Trust project-local files for this command
  -na, --no-approve  Ignore project-local files for this command

Examples:
  {APP_NAME} update                          Update every configured git package
  {APP_NAME} update git:github.com/user/repo  Update one package

Note: self-update and model-catalog refresh (the TypeScript `update --self`/
`--models`/`--all`) are not supported by this port; see this module's
docstring.
""",
            file=out,
        )
        return
    print(
        f"""Usage:
  {get_package_command_usage("list")}

List installed packages from user and project settings.

Options:
  -a, --approve      Trust project-local files for this command
  -na, --no-approve  Ignore project-local files for this command
""",
        file=out,
    )


def parse_package_command(args: list[str]) -> PackageCommandOptions | None:
    """Parse `argv` into package-command options, or `None` if it isn't one."""
    if not args:
        return None
    raw_command, rest = args[0], args[1:]
    if raw_command == "uninstall":
        command: PackageCommand = "remove"
    elif raw_command in ("install", "remove", "update", "list"):
        command = raw_command  # type: ignore[assignment]
    else:
        return None

    local = False
    project_trust_override: bool | None = None
    help_flag = False
    invalid_option: str | None = None
    invalid_argument: str | None = None
    source: str | None = None

    for arg in rest:
        if arg in ("-h", "--help"):
            help_flag = True
            continue
        if arg in ("-l", "--local"):
            if command in ("install", "remove"):
                local = True
            else:
                invalid_option = invalid_option or arg
            continue
        if arg in ("--approve", "-a"):
            project_trust_override = True
            continue
        if arg in ("--no-approve", "-na"):
            project_trust_override = False
            continue
        if arg.startswith("-"):
            invalid_option = invalid_option or arg
            continue
        if source is None:
            source = arg
        else:
            invalid_argument = invalid_argument or arg

    return PackageCommandOptions(
        command=command,
        source=source,
        local=local,
        project_trust_override=project_trust_override,
        help=help_flag,
        invalid_option=invalid_option,
        invalid_argument=invalid_argument,
    )


async def resolve_project_trust(
    settings_manager: SettingsManager,
    project_trust_override: bool | None,
    *,
    cwd: str,
    agent_dir: str,
) -> bool:
    """Resolve project trust for a package command; see module docstring.

    Delegates to the shared `resolve_project_trusted`, exactly as
    `createCommandSettingsManager` does: an explicit `--approve`/
    `--no-approve` wins, then a project with no trust-requiring resources is
    trusted (a fresh project has nothing to distrust yet), then the persisted
    `ProjectTrustStore` (`trust.json`), then `defaultProjectTrust`, and
    otherwise untrusted. `has_ui` is always false because package commands
    have no interactive selector to prompt with.
    """
    return await resolve_project_trusted(
        cwd=cwd,
        trust_store=ProjectTrustStore(agent_dir),
        project_trust_context=create_project_trust_context(cwd=cwd, mode="print", has_ui=False),
        trust_override=project_trust_override,
        default_project_trust=settings_manager.get_default_project_trust(),
    )


async def create_command_settings_manager(
    cwd: str,
    agent_dir: str,
    *,
    project_trust_override: bool | None,
    use_saved_project_trust_only: bool = False,
) -> SettingsManager:
    settings_manager = SettingsManager.create(cwd, agent_dir, SettingsManagerCreateOptions(project_trusted=False))
    if use_saved_project_trust_only:
        # `update` must never prompt, ask extensions, or promote an untrusted
        # project through `defaultProjectTrust`: only an explicit flag or a
        # previously saved decision counts.
        saved_project_trusted = ProjectTrustStore(agent_dir).get(cwd) is True
        settings_manager.set_project_trusted(
            saved_project_trusted if project_trust_override is None else project_trust_override
        )
        return settings_manager
    project_trusted = await resolve_project_trust(
        settings_manager, project_trust_override, cwd=cwd, agent_dir=agent_dir
    )
    settings_manager.set_project_trusted(project_trusted)
    return settings_manager


def report_settings_errors(settings_manager: SettingsManager, context: str, err: IO[str] | None = None) -> None:
    err = sys.stderr if err is None else err
    for settings_error in settings_manager.drain_errors():
        print(f"Warning ({context}, {settings_error.scope} settings): {settings_error.error}", file=err)


async def handle_package_command(
    args: list[str],
    *,
    cwd: str,
    agent_dir: str,
    settings_manager: SettingsManager | None = None,
    out: IO[str] | None = None,
    err: IO[str] | None = None,
) -> int | None:
    """Dispatch install/remove/uninstall/update/list.

    Returns the process exit code if this was a package command, or `None`
    if `args` isn't one (so the caller can fall through to the ordinary
    agent-run argument scanner) -- mirrors `handlePackageCommand`'s
    `boolean` "did I handle it" return, but returns the exit code directly
    since Python's CLI entry point doesn't rely on a global `process.exitCode`.

    `out`/`err` default to `sys.stdout`/`sys.stderr` *at call time*, not at
    import time: TypeScript writes through `console.log`, which resolves
    `process.stdout` on every call, so a caller that has replaced the stream
    (a test harness, or `entry.take_over_stdout()`) sees the output.
    """
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    if not args or args[0] not in _PACKAGE_COMMAND_NAMES:
        return None
    options = parse_package_command(args)
    if options is None:
        return None

    if options.help:
        print_package_command_help(options.command, out)
        return 0

    if options.invalid_option:
        print(f'Unknown option {options.invalid_option} for "{options.command}".', file=err)
        print(f'Use "{APP_NAME} --help" or "{get_package_command_usage(options.command)}".', file=err)
        return 1

    if options.invalid_argument:
        print(f"Unexpected argument {options.invalid_argument}.", file=err)
        print(f"Usage: {get_package_command_usage(options.command)}", file=err)
        return 1

    if options.conflicting_options:
        print(options.conflicting_options, file=err)
        print(f"Usage: {get_package_command_usage(options.command)}", file=err)
        return 1

    source = options.source
    if options.command in ("install", "remove") and not source:
        print(f"Missing {options.command} source.", file=err)
        print(f"Usage: {get_package_command_usage(options.command)}", file=err)
        return 1

    writes_project_package_config = options.command in ("install", "remove") and options.local
    owns_settings_manager = settings_manager is None
    if settings_manager is None:
        settings_manager = await create_command_settings_manager(
            cwd,
            agent_dir,
            project_trust_override=options.project_trust_override,
            use_saved_project_trust_only=options.command == "update",
        )
    if owns_settings_manager and writes_project_package_config and not settings_manager.is_project_trusted():
        print("Project is not trusted. Use --approve to modify local package config.", file=err)
        return 1
    report_settings_errors(settings_manager, "package command", err)

    package_manager = PackageManager(cwd, agent_dir, settings_manager)
    package_manager.set_progress_callback(
        lambda event: print(event.message, file=out) if event.type == "start" else None
    )

    try:
        if options.command == "install":
            assert source is not None
            await package_manager.install_and_persist(source, local=options.local)
            print(f"Installed {source}", file=out)
            return 0

        if options.command == "remove":
            assert source is not None
            removed = await package_manager.remove_and_persist(source, local=options.local)
            if not removed:
                print(f"No matching package found for {source}", file=err)
                return 1
            print(f"Removed {source}", file=out)
            return 0

        if options.command == "list":
            configured_packages = package_manager.list_configured_packages()
            user_packages = [p for p in configured_packages if p.scope == "user"]
            project_packages = [p for p in configured_packages if p.scope == "project"]

            if not configured_packages:
                print("No packages installed.", file=out)
                return 0

            def format_package(pkg) -> None:
                display = f"{pkg.source} (filtered)" if pkg.filtered else pkg.source
                print(f"  {display}", file=out)
                if pkg.installed_path:
                    print(f"    {pkg.installed_path}", file=out)

            if user_packages:
                print("User packages:", file=out)
                for pkg in user_packages:
                    format_package(pkg)
            if project_packages:
                if user_packages:
                    print(file=out)
                print("Project packages:", file=out)
                for pkg in project_packages:
                    format_package(pkg)
            return 0

        if options.command == "update":
            await package_manager.update(source)
            if source:
                print(f"Updated {source}", file=out)
            else:
                print("Updated packages", file=out)
            return 0
    except Exception as error:
        print(f"Error: {error}", file=err)
        return 1

    return 1


async def handle_config_command(
    args: list[str],
    *,
    cwd: str,
    agent_dir: str,
    settings_manager: SettingsManager | None = None,
    out: IO[str] | None = None,
    err: IO[str] | None = None,
    select_config: SelectConfig | None = None,
) -> int | None:
    """Handle the `config` subcommand: open the resource-configuration TUI.

    Port of `handleConfigCommand`. `select_config` is a parameter only so the
    caller can observe the TUI without a terminal; it defaults to the real
    selector.
    """
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    if not args or args[0] != "config":
        return None
    rest = args[1:]

    if "-h" in rest or "--help" in rest:
        print_config_command_help(out)
        return 0

    local = False
    project_trust_override: bool | None = None
    for arg in rest:
        if arg in ("-l", "--local"):
            local = True
        elif arg in ("-a", "--approve"):
            project_trust_override = True
        elif arg in ("-na", "--no-approve"):
            project_trust_override = False
        elif arg.startswith("-"):
            print(f'Unknown option {arg} for "config".', file=err)
            print(f'Use "{APP_NAME} --help" or "{CONFIG_COMMAND_USAGE}".', file=err)
            return 1
        else:
            print(f"Unexpected argument {arg}.", file=err)
            print(f"Usage: {CONFIG_COMMAND_USAGE}", file=err)
            return 1

    if settings_manager is None:
        settings_manager = await create_command_settings_manager(
            cwd, agent_dir, project_trust_override=project_trust_override
        )
    if local and not settings_manager.is_project_trusted():
        print("Project is not trusted. Use --approve to modify local resource config.", file=err)
        return 1
    report_settings_errors(settings_manager, "config command", err)

    global_settings_manager = SettingsManager.create(
        cwd, agent_dir, SettingsManagerCreateOptions(project_trusted=False)
    )
    global_resolved_paths = await PackageManager(cwd, agent_dir, global_settings_manager).resolve()
    project_resolved_paths = (
        await PackageManager(cwd, agent_dir, settings_manager).resolve()
        if settings_manager.is_project_trusted()
        else global_resolved_paths
    )

    await (select_config or _select_config)(
        resolved_paths={"global": global_resolved_paths, "project": project_resolved_paths},
        settings_manager=settings_manager,
        cwd=cwd,
        agent_dir=agent_dir,
        write_scope="project" if local else "global",
        project_mode_available=settings_manager.is_project_trusted(),
    )
    return 0


__all__ = [
    "CONFIG_COMMAND_USAGE",
    "PackageCommand",
    "PackageCommandOptions",
    "create_command_settings_manager",
    "get_package_command_usage",
    "handle_config_command",
    "handle_package_command",
    "parse_package_command",
    "print_config_command_help",
    "print_package_command_help",
    "report_settings_errors",
    "resolve_project_trust",
]
