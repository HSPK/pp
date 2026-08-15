"""Coding agent CLI.

Python port of ``packages/coding-agent/src/cli.ts`` and ``main.ts``.

``main()`` first checks whether ``argv`` is a package-manager subcommand
(``install``/``remove``/``uninstall``/``update``/``list``/``config``) or an
``auth`` subcommand and dispatches to those, mirroring ``main.ts``'s handling
before ``parseArgs`` runs.

Otherwise it parses with the TS-parity parser in `cli/args.py` and resolves an
app mode exactly as upstream ``resolveAppMode`` does: ``--mode rpc``/``json``
win, then ``--print`` or a non-TTY stdin/stdout forces print mode, otherwise
the interactive TUI. Print/JSON modes run through
`modes/print_mode.py`; interactive runs
`modes/interactive/interactive_mode.py`.

RPC mode is not ported (superseded by ``pi_server``/``pi_client``); asking for
it reports that instead of silently falling back.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from pi_agent import (
    Agent,
    AgentTool,
    MutableAgentState,
)
from pi_ai import Model, content_text
from pi_ai.providers import all_providers, openai_compatible_provider
from pi_ai.registry import Models

from pi_coding_agent.cli.args import Args, parse_args, print_help
from pi_coding_agent.cli.auth_command import handle_auth_command
from pi_coding_agent.cli.file_processor import (
    FileProcessingError,
    ProcessedFiles,
    process_file_arguments,
)
from pi_coding_agent.cli.initial_message import build_initial_message
from pi_coding_agent.cli.list_models import handle_list_models
from pi_coding_agent.cli.package_manager_cli import handle_config_command, handle_package_command
from pi_coding_agent.cli.session_picker import select_session
from pi_coding_agent.cli.session_selection import (
    SessionSelectionAborted,
    SessionSelectionError,
    create_session_manager,
)
from pi_coding_agent.cli.startup_ui import show_startup_selector
from pi_coding_agent.core.agent_session_runtime import (
    AgentSessionRuntime,
    create_agent_session_runtime,
)
from pi_coding_agent.core.config import VERSION, get_agent_dir
from pi_coding_agent.core.extensions import (
    SessionRuntimeActions,
    discover_and_load_extensions,
    emit_project_trust_event,
)
from pi_coding_agent.core.extensions.types import ProjectTrustEvent
from pi_coding_agent.core.model_resolver import resolve_model_scope
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.output_guard import restore_stdout, take_over_stdout
from pi_coding_agent.core.project_trust import (
    ExtensionTrustDecision,
    create_project_trust_context,
    resolve_project_trusted,
)
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager, get_default_session_dir
from pi_coding_agent.core.settings_manager import SettingsManager, SettingsManagerCreateOptions
from pi_coding_agent.core.timings import print_timings, reset_timings
from pi_coding_agent.core.timings import time as record_timing
from pi_coding_agent.core.trust_manager import ProjectTrustStore
from pi_coding_agent.migrations import run_migrations, show_deprecation_warnings
from pi_coding_agent.modes.interactive.theme.theme import set_custom_theme_discovery_enabled
from pi_coding_agent.modes.print_mode import PrintModeOptions, run_print_mode
from pi_coding_agent.modes.rpc import run_rpc_mode

AppMode = Literal["interactive", "print", "json", "rpc"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a coding agent running in a terminal. Use the provided tools to inspect and "
    "modify the user's project. Prefer small, verifiable steps. Be concise."
)


def build_models(base_url: str | None = None, model_id: str | None = None) -> Models:
    """Build the provider registry, optionally adding a custom endpoint."""
    providers = all_providers()
    if base_url:
        from pi_ai.types import Model as ModelType

        custom_model = ModelType(
            id=model_id or "custom",
            name=model_id or "custom",
            api="openai-completions",
            provider="custom",
            base_url=base_url,
            context_window=128_000,
            max_tokens=8_192,
        )
        providers.append(
            openai_compatible_provider(
                "custom",
                "Custom",
                base_url,
                ["PI_API_KEY", "OPENAI_API_KEY"],
                [custom_model],
            )
        )
    return Models(providers)


def resolve_model(models: Models, reference: str | None) -> Model:
    """Resolve ``provider/model`` or a bare model id, else the first available."""
    if reference:
        model = models.find_model(reference)
        if model is None:
            known = ", ".join(f"{m.provider}/{m.id}" for m in models.get_models())
            raise SystemExit(f"Unknown model: {reference}\nKnown models: {known}")
        return model

    for provider in models.get_providers():
        for env_var in provider.auth.api_key.env_vars if provider.auth.api_key else ():
            if os.environ.get(env_var):
                provider_models = provider.get_models()
                if provider_models:
                    return provider_models[0]
    raise SystemExit(
        "No model selected and no provider API key found in the environment.\n"
        "Set one of OPENAI_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY, DEEPSEEK_API_KEY, "
        "or pass --model."
    )


def format_event(event: Any) -> str | None:
    """Render an agent event as a terminal line, or None to print nothing."""
    if event.type == "message_end":
        message = event.message
        if message.role == "assistant":
            text = content_text([block for block in message.content if block.type == "text"])
            if message.error_message:
                return f"error: {message.error_message}"
            return text or None
        if message.role == "toolResult":
            status = "error" if message.is_error else "ok"
            body = content_text(message.content).strip()
            first_line = body.splitlines()[0] if body else ""
            return f"  [{message.tool_name} {status}] {first_line}"
    elif event.type == "tool_execution_start":
        return f"  -> {event.tool_name}({event.args})"
    return None


# Port of `main.ts:70`. The recovery flag is the entire content of the
# message, so omitting it leaves the user at a dead end.
EXTENSION_LOAD_FAILURE_HINT = 'Hint: Start without extensions using "pp -ne".'


async def run_once(
    prompt: str,
    model: Model,
    models: Models,
    tools: list[AgentTool],
    system_prompt: str,
    quiet: bool = False,
) -> int:
    """Run a single prompt to completion. Returns a process exit code."""
    state = MutableAgentState(system_prompt=system_prompt, model=model)
    state.tools = tools

    async def stream_fn(m: Model, context: Any, options: Any = None) -> Any:
        return await models.stream_simple(m, context, options)

    agent = Agent(stream_fn, initial_state=state)

    failed = False

    async def listener(event: Any, _signal: Any) -> None:
        nonlocal failed
        if event.type == "turn_end" and getattr(event.message, "error_message", None):
            failed = True
        if quiet:
            return
        line = format_event(event)
        if line:
            print(line, flush=True)

    agent.subscribe(listener)
    await agent.prompt(prompt)

    if quiet:
        for message in agent.state.messages:
            if message.role == "assistant" and not message.error_message:
                text = content_text([b for b in message.content if b.type == "text"])
                if text:
                    print(text, flush=True)

    return 1 if failed else 0


def load_default_tools(cwd: str) -> list[AgentTool]:
    """Load every built-in coding tool.

    Imported here rather than at module scope so ``--no-tools`` and
    ``--list-models`` keep working if the tool package is unavailable.
    """
    from pi_coding_agent.tools import create_all_tools

    return list(create_all_tools(cwd).values())


def build_parser() -> argparse.ArgumentParser:
    """Legacy vertical-slice parser, kept for `run_once`-based tests."""
    parser = argparse.ArgumentParser(prog="pp", description="Python port of the pi coding agent")
    parser.add_argument("prompt", nargs="*", help="Prompt to run. Reads stdin when omitted.")
    parser.add_argument("-m", "--model", help="Model as provider/model-id or a bare model id")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL for a custom endpoint")
    parser.add_argument("--system", help="Override the system prompt")
    parser.add_argument("--list-models", action="store_true", help="List known models and exit")
    parser.add_argument("--no-tools", action="store_true", help="Run without the built-in tools")
    parser.add_argument("-q", "--quiet", action="store_true", help="Print only the final answer")
    parser.add_argument("--cwd", help="Working directory for file tools", default=None)
    parser.add_argument("-p", "--print", action="store_true", dest="print_mode", help="Force single-shot print mode")
    parser.add_argument("-i", "--interactive", action="store_true", help="Force the interactive TUI")
    return parser


def resolve_app_mode(parsed: Args, stdin_is_tty: bool, stdout_is_tty: bool) -> AppMode:
    """Port of TS ``resolveAppMode``."""
    if parsed.mode == "rpc":
        return "rpc"
    if parsed.mode == "json":
        return "json"
    if parsed.print or not stdin_is_tty or not stdout_is_tty:
        return "print"
    return "interactive"


def to_print_output_mode(app_mode: AppMode) -> Literal["text", "json"]:
    return "json" if app_mode == "json" else "text"


def _is_plain_runtime_metadata_command(parsed: Args) -> bool:
    """Port of TS ``isPlainRuntimeMetadataCommand``."""
    return not parsed.print and parsed.mode is None and (parsed.help is True or parsed.list_models is not None)


async def dispatch_subcommand(raw_args: list[str]) -> int | None:
    """Run a package-manager/config/auth subcommand, or return `None`.

    Mirrors `main.ts`'s pre-`parseArgs` dispatch: these subcommands have their
    own argument grammars and never reach the agent argument parser.
    """
    cwd = os.getcwd()
    agent_dir = get_agent_dir()

    package_result = await handle_package_command(raw_args, cwd=cwd, agent_dir=agent_dir)
    if package_result is not None:
        return package_result
    config_result = await handle_config_command(raw_args, cwd=cwd, agent_dir=agent_dir)
    if config_result is not None:
        return config_result
    return await handle_auth_command(raw_args, agent_dir=agent_dir)


def _resolve_session_model(runtime: ModelRuntime, parsed: Args) -> Model | None:
    """Pick the model named by `--model`/`--provider`, if any."""
    if parsed.model:
        found = runtime.find_model(parsed.model)
        if found is not None:
            return found
    if parsed.provider:
        candidates = runtime.get_models(parsed.provider)
        if candidates:
            return candidates[0]
    return None


def _resolve_no_tools(parsed: Args) -> str | None:
    if parsed.no_tools:
        return "all"
    return "builtin" if parsed.no_builtin_tools else None


async def resolve_startup_trust(parsed: Args, cwd: str, agent_dir: str, app_mode: AppMode) -> bool:
    """Decide whether this project's `.pi` settings and resources may load.

    A project can ship settings, extensions and packages that execute code, so
    an unfamiliar folder must be answered for before any of it is read. Only
    interactive mode can prompt; every other mode falls back to the remembered
    decision or the `defaultProjectTrust` setting.
    """
    trust_store = ProjectTrustStore(agent_dir)
    bootstrap_settings = SettingsManager.create(cwd, agent_dir, SettingsManagerCreateOptions(project_trusted=False))

    async def select(title: str, options: Sequence[str]) -> str | None:
        return await show_startup_selector(bootstrap_settings, title, [(option, option) for option in options])

    context = create_project_trust_context(
        cwd=cwd,
        mode=app_mode,
        has_ui=app_mode == "interactive" and sys.stdout.isatty(),
        ui=_StartupTrustUI(select),
    )

    async def decide_with_extensions(target_cwd: str) -> ExtensionTrustDecision | None:
        """Give user-level extensions the first word, as upstream does.

        Only extensions from `~/.pi/agent/extensions` are loaded here:
        `project_trusted=False` keeps the project's own `.pi/extensions` out,
        which is the whole point -- running them to decide whether they may
        run would answer the question with the code being questioned.
        """
        loaded = await discover_and_load_extensions(
            parsed.extensions or [],
            target_cwd,
            project_trusted=False,
            agent_dir=agent_dir,
            no_extensions=bool(parsed.no_extensions),
        )
        emitted = await emit_project_trust_event(loaded.extensions, ProjectTrustEvent(cwd=target_cwd), context)
        for failure in emitted.errors:
            print(
                f'Extension "{failure.extension_path}" project_trust error: {failure.error}',
                file=sys.stderr,
            )
        if emitted.result is None or emitted.result.trusted == "undecided":
            return None
        return ExtensionTrustDecision(
            trusted=emitted.result.trusted == "yes",
            remember=bool(emitted.result.remember),
        )

    return await resolve_project_trusted(
        cwd=cwd,
        trust_store=trust_store,
        project_trust_context=context,
        trust_override=parsed.project_trust_override,
        default_project_trust=bootstrap_settings.get_default_project_trust(),
        trust_decider=decide_with_extensions,
    )


class _StartupTrustUI:
    """Adapts the startup selector to the `ProjectTrustUI` protocol."""

    def __init__(self, select: Callable[[str, Sequence[str]], Awaitable[str | None]]) -> None:
        self._select = select

    async def select(self, title: str, options: Sequence[str]) -> str | None:
        return await self._select(title, options)


async def build_session_runtime(
    parsed: Args, cwd: str, agent_dir: str, project_trusted: bool = True
) -> AgentSessionRuntime:
    """Build the `AgentSessionRuntime` both interactive and print mode run on."""
    session_dir = parsed.session_dir or get_default_session_dir(cwd, agent_dir)
    settings_manager = SettingsManager.create(
        cwd, agent_dir, SettingsManagerCreateOptions(project_trusted=project_trusted)
    )
    if parsed.use_theme is not None:
        # In-memory only: --use-theme must not be written back to settings.json.
        settings_manager.apply_overrides({"theme": parsed.use_theme})

    async def pick_session(pick_cwd: str, pick_session_dir: str | None) -> str | None:
        """`--resume`'s interactive picker. Only reachable on a TTY."""
        return await select_session(
            lambda on_progress=None: SessionManager.list(pick_cwd, pick_session_dir, on_progress),
            lambda on_progress=None: SessionManager.list_all(pick_session_dir, on_progress),
            settings_manager,
        )

    session_manager = await create_session_manager(
        parsed,
        cwd,
        session_dir,
        select_session=pick_session if sys.stdin.isatty() and sys.stdout.isatty() else None,
    )

    # TypeScript applies `--name` right after session selection, before any
    # model validation, so a named session records its name even when the run
    # then fails on a missing model.
    if parsed.name is not None:
        startup_name = parsed.name.strip()
        if not startup_name:
            raise SessionSelectionError("Error: --name requires a non-empty value")
        session_manager.append_session_info(startup_name)
    record_timing("createSessionManager")
    model_runtime = await ModelRuntime.create(agent_dir=agent_dir)

    model = _resolve_session_model(model_runtime, parsed)

    if parsed.api_key:
        # `--api-key` authenticates this process only; it must not be written
        # to auth.json, and without a model there is no provider to apply it to.
        if model is None:
            print(
                "--api-key requires a model to be specified via --model, --provider/--model, or --models",
                file=sys.stderr,
            )
        else:
            await model_runtime.set_runtime_api_key(model.provider, parsed.api_key)
    scoped_models = await resolve_model_scope(parsed.models, model_runtime) if parsed.models else None
    record_timing("resolveModelScope")

    # TypeScript loads extensions through `DefaultResourceLoader`
    # (`main.ts:769` -> `resource-loader.ts:451`). This port keeps extensions
    # out of `ResourceLoader`, so the CLI is the caller that must load them and
    # hand them to the session -- without this, `--extension` and
    # `--no-extensions` parse but do nothing and a project's
    # `.pi/extensions/*.py` never loads in the real binary.
    # `pi.*` runtime actions are baked into each extension's `pi` object as the
    # file loads, which is before any session exists, so they are bound to a
    # holder that `create_runtime` fills in below. Without this the CLI loads
    # extensions with the default no-ops and `pi.send_user_message()` and
    # friends silently do nothing.
    runtime_actions = SessionRuntimeActions()
    extensions_result = await discover_and_load_extensions(
        parsed.extensions or [],
        cwd,
        project_trusted=project_trusted,
        agent_dir=agent_dir,
        actions=runtime_actions.actions,
        no_extensions=bool(parsed.no_extensions),
    )
    if extensions_result.errors:
        for load_error in extensions_result.errors:
            print(
                f'Failed to load extension "{load_error.get("path")}": {load_error.get("error")}',
                file=sys.stderr,
            )
        # TypeScript treats a failed extension load as a fatal startup
        # diagnostic (`main.ts:894-899`): it prints the hint and exits 1.
        # Continuing would hand the user a running agent that silently lacks
        # the extension they asked for -- the failure mode they cannot see.
        # This port has no startup diagnostics pipeline, so the check lives
        # here, where the errors are already reported.
        print(EXTENSION_LOAD_FAILURE_HINT, file=sys.stderr)
        raise SystemExit(1)

    async def create_runtime(**kwargs: Any) -> Any:
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=kwargs.get("cwd", cwd),
                agent_dir=kwargs.get("agent_dir", agent_dir),
                session_manager=kwargs.get("session_manager", session_manager),
                settings_manager=settings_manager,
                model_runtime=model_runtime,
                model=model,
                thinking_level=parsed.thinking,
                scoped_models=scoped_models,
                no_tools=_resolve_no_tools(parsed),
                tools=parsed.tools,
                exclude_tools=parsed.exclude_tools,
                extensions=extensions_result.extensions,
            )
        )
        # Re-bound on every replacement (`/new`, `/import`, `/clone`), so the
        # actions never point at a disposed session.
        runtime_actions.bind(result.session)
        return result

    return await create_agent_session_runtime(
        create_runtime, cwd=cwd, agent_dir=agent_dir, session_manager=session_manager
    )


async def run_app(
    parsed: Args,
    app_mode: AppMode,
    processed_files: ProcessedFiles | None = None,
    stdin_content: str | None = None,
) -> int:
    processed_files = processed_files or ProcessedFiles()
    cwd = os.getcwd()
    agent_dir = get_agent_dir()

    # `--no-themes` was parsed and never consumed, so the flag was accepted and
    # silently ignored. TypeScript gates theme discovery on it in
    # `DefaultResourceLoader` (`resource-loader.ts:501`).
    if parsed.no_themes:
        set_custom_theme_discovery_enabled(False)

    if app_mode in ("print", "json", "rpc"):
        # The protocol owns stdout from here on; stray writes go to stderr.
        take_over_stdout()

    try:
        try:
            project_trusted = await resolve_startup_trust(parsed, cwd, agent_dir, app_mode)
            runtime = await build_session_runtime(parsed, cwd, agent_dir, project_trusted)
            record_timing("createAgentSessionRuntime")
        except SessionSelectionAborted as aborted:
            print(str(aborted), file=sys.stderr)
            return 0
        except SessionSelectionError as error:
            print(str(error), file=sys.stderr)
            return 1
        except Exception as error:
            print(f"Failed to start: {error}", file=sys.stderr)
            return 1

        if app_mode == "interactive":
            return await run_interactive_mode(runtime, parsed, processed_files)

        if app_mode == "rpc":
            print_timings()
            return await run_rpc_mode(runtime)

        initial = build_initial_message(
            parsed,
            file_text=processed_files.text,
            file_images=processed_files.images,
            stdin_content=stdin_content,
        )
        record_timing("prepareInitialMessage")
        print_timings()
        return await run_print_mode(
            runtime,
            PrintModeOptions(
                mode=to_print_output_mode(app_mode),
                messages=list(parsed.messages),
                initial_message=initial.initial_message,
                initial_images=initial.initial_images or [],
            ),
        )
    finally:
        if app_mode in ("print", "json", "rpc"):
            restore_stdout()


async def run_interactive_mode(runtime: AgentSessionRuntime, parsed: Args, processed_files: ProcessedFiles) -> int:
    from pi_coding_agent.modes.interactive.interactive_mode import (
        InteractiveMode,
        InteractiveModeOptions,
    )

    initial = build_initial_message(parsed, file_text=processed_files.text, file_images=processed_files.images)
    record_timing("prepareInitialMessage")
    mode = InteractiveMode(
        runtime,
        InteractiveModeOptions(
            model_fallback_message=runtime.model_fallback_message,
            initial_message=initial.initial_message,
            initial_images=initial.initial_images or [],
            initial_messages=list(parsed.messages),
            verbose=bool(parsed.verbose),
            tui_mode=parsed.tui_mode,
            initial_theme_setting=parsed.use_theme,
        ),
    )
    print_timings()
    try:
        await mode.run()
    finally:
        await mode.shutdown()
    return 0


def read_piped_stdin(stdin_is_tty: bool) -> str | None:
    """Read everything piped into stdin, or `None` on a TTY.

    Port of TS ``readPipedStdin``. A closed or unreadable stdin is treated as
    "nothing piped" rather than a startup failure, so `pi` still runs when it
    is spawned without a usable stdin.
    """
    if stdin_is_tty:
        return None
    try:
        data = sys.stdin.read()
    except (OSError, ValueError):
        return None
    return data.strip() or None


def main(argv: list[str] | None = None) -> int:
    reset_timings()
    raw_args = list(sys.argv[1:]) if argv is None else list(argv)

    dispatched = asyncio.run(dispatch_subcommand(raw_args))
    if dispatched is not None:
        return dispatched

    parsed = parse_args(raw_args)
    for diagnostic in parsed.diagnostics:
        print(diagnostic.message, file=sys.stderr)
    record_timing("parseArgs")

    if parsed.version:
        print(VERSION)
        return 0
    if parsed.export:
        # TypeScript exits here via `exportFromFile` (`main.ts:625-637`). That
        # path needs `generateHtml`, the document assembly that embeds vendored
        # `marked`/`highlight.js` browser bundles -- the one piece of the HTML
        # exporter this port does not carry (see README). Failing loudly beats
        # the previous behaviour, where the flag parsed, did nothing, and the
        # run continued on to report "No prompt given."
        print(
            "--export is not available in this Python port: it needs the HTML "
            "document assembly (vendored marked/highlight.js), which is not ported. "
            "The session JSONL is portable -- export it with the TypeScript pi, "
            "or use /share for a hosted copy.",
            file=sys.stderr,
        )
        return 1
    if parsed.offline:
        os.environ["PI_OFFLINE"] = "1"

    stdin_is_tty = sys.stdin.isatty()
    stdout_is_tty = sys.stdout.isatty()
    app_mode = resolve_app_mode(parsed, stdin_is_tty, stdout_is_tty)

    # TS `shouldTakeOverStdout`: a non-interactive run owns stdout for its
    # protocol stream, so anything else it prints (including `--help` and
    # `--list-models`) has to go to stderr. A *plain* metadata command --
    # `--help`/`--list-models` with neither `-p` nor `--mode` -- is exempt and
    # keeps printing to stdout.
    if app_mode != "interactive" and not _is_plain_runtime_metadata_command(parsed):
        take_over_stdout()

    if parsed.help:
        print_help()
        return 0
    if parsed.list_models is not None and parsed.list_models is not False:
        search = parsed.list_models if isinstance(parsed.list_models, str) else None
        return asyncio.run(handle_list_models(search, agent_dir=get_agent_dir()))

    migrations = run_migrations(os.getcwd(), get_agent_dir())
    if migrations.migrated_auth_providers:
        providers = ", ".join(migrations.migrated_auth_providers)
        print(f"Migrated credentials to auth.json: {providers}", file=sys.stderr)
    show_deprecation_warnings(migrations.deprecation_warnings)
    record_timing("runMigrations")

    try:
        processed_files = process_file_arguments(parsed.file_args)
    except FileProcessingError as error:
        print(str(error), file=sys.stderr)
        return 1

    if app_mode == "rpc" and parsed.file_args:
        print("Error: @file arguments are not supported in RPC mode", file=sys.stderr)
        return 1

    # Piped input becomes the head of the first prompt, before @file contents.
    # Content on stdin means there is nothing to interact with, so an
    # otherwise-interactive run degrades to print mode (TS does the same).
    # RPC mode owns stdin as its command channel, so draining it here would
    # swallow the host's first commands and then report "No prompt given".
    stdin_content = None if app_mode == "rpc" else read_piped_stdin(stdin_is_tty)
    if stdin_content is not None and app_mode == "interactive":
        app_mode = "print"
    record_timing("readPipedStdin")

    has_prompt = bool(parsed.messages or stdin_content or processed_files.text)
    if app_mode not in ("interactive", "rpc") and not has_prompt:
        print("No prompt given.", file=sys.stderr)
        return 2

    return asyncio.run(run_app(parsed, app_mode, processed_files, stdin_content))


if __name__ == "__main__":
    raise SystemExit(main())
