"""Extension discovery and loading.

Port of `packages/coding-agent/src/core/extensions/loader.ts` (737 lines).

**The JS-module substitution.** TypeScript extensions are `.ts`/`.js` files
loaded at runtime through `jiti` (an on-the-fly TypeScript-to-JS transpiler),
with a bundled set of packages (`@earendil-works/pi-agent-core`, `pi-ai`,
`pi-tui`, `typebox`, this package itself) made available to the extension
module via import aliasing/virtual modules. There is no JavaScript runtime
here and this port does not attempt to execute JavaScript: a Python extension
is instead a **plain Python file (or package directory) loaded from disk via
`importlib.util.spec_from_file_location`**, exactly the mechanism CPython
itself uses to import a module given a filesystem path rather than a
dotted name on `sys.path`. This is the same category of substitution as
`extensions/loader.py` calling `importlib` where TypeScript called `jiti`:
both dynamically load a module given a path and hand the loaded module's
top-level callable a capability object.

**What an extension author writes.** A Python extension file must define a
module-level, plain-importable value named `pi_extension` (the direct
equivalent of TypeScript's `export default function(pi) { ... }`): a
callable taking one `ExtensionAPI` argument, either a plain function or an
`async def`. For example::

    # my_extension.py
    from pi_coding_agent.core.extensions.types import ToolDefinition

    async def _echo(tool_call_id, params, signal, on_update, ctx):
        from pi_agent.types import AgentToolResult
        from pi_ai.types import TextContent
        return AgentToolResult(content=[TextContent(text=params.get("text", ""))])

    def pi_extension(pi):
        pi.register_tool(ToolDefinition(
            name="echo", label="Echo", description="Echo text back.",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            execute=_echo,
        ))
        pi.on("session_start", lambda event, ctx: ctx.ui.notify("loaded"))

`pi_extension` may also be a class instance or any other callable; only
`callable(value)` is required. A directory extension provides the same
contract from its `__init__.py` (the "index" entry point, mirroring
TypeScript's `index.ts`/`index.js` directory-entry rule).

**Directory manifests.** TypeScript directories can declare non-default entry
points via a `package.json` `"pi"."extensions"` array
(`readPiManifest`/`resolveExtensionEntries`). Python packages have no
`package.json`, so a directory declares the same thing in a top-level
`pi.json` (see `core/pi_manifest.py`) -- the substitution `core/package_manager.py`
already uses for package resource discovery. The precedence is TypeScript's:
a manifest that names at least one existing extension file wins, otherwise
discovery falls back to `subdir/__init__.py` (TypeScript's `index.ts`).

**Trust gating.** Loading an extension executes arbitrary Python code from
disk on import. This port reproduces the TypeScript trust boundary exactly:
`discover_and_load_extensions()` only walks the **project-local** extensions
directory (`cwd/.pi/extensions/`) when `project_trusted=True`; the
**global**/user extensions directory (`agent_dir/extensions/`) and any
explicitly configured paths are always discovered, matching
`resource-loader.ts`'s `loadProjectTrustExtensions()` (which forces
`projectTrusted=False` for the bootstrap trust-decision pass so only
user/global/CLI extensions run before the user has decided whether to trust
the project). Callers that already have a `ResourceLoader`/`SettingsManager`
should pass its `is_project_trusted()` value through unchanged, exactly as
`resource_loader.py` already gates project-local skills/prompts/system-prompt
files on the same flag.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pi_coding_agent.core.config import CONFIG_DIR_NAME, get_agent_dir
from pi_coding_agent.core.event_bus import EventBus, EventBusHandler, create_event_bus
from pi_coding_agent.core.exec import ExecOptions, ExecResult, exec_command
from pi_coding_agent.core.extensions.types import (
    Extension,
    ExtensionCommandContext,
    ExtensionFactory,
    ExtensionHandler,
    RegisteredCommand,
    RegisteredTool,
    ToolDefinition,
)
from pi_coding_agent.core.pi_manifest import manifest_path_for_package_root, read_pi_manifest
from pi_coding_agent.core.source_info import create_synthetic_source_info
from pi_coding_agent.core.timings import time as record_timing
from pi_coding_agent.utils.paths import canonicalize_path, resolve_path

_EXTENSION_ENTRY_POINT_NAME = "pi_extension"
"""Attribute name a Python extension module/package must export. See module
docstring for the full contract."""


@dataclass
class LoadExtensionsResult:
    extensions: list[Extension] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    """Extensions that failed to load. A caller may treat this as fatal."""
    conflicts: list[dict[str, str]] = field(default_factory=list)
    """Advisory diagnostics from `detect_extension_conflicts`.

    Kept separate from `errors` because upstream keeps them separate: conflict
    detection lives in `resource-loader.ts`'s `detectConflicts()`, which
    returns its own list, while `discoverAndLoadExtensions` reports only load
    failures. Folding conflicts into `errors` made two extensions registering
    the same tool name look like a load failure, even though both extensions
    load fine and the first registration simply wins.
    """


@dataclass(frozen=True)
class ExtensionCacheToken:
    """Proof that a cached load started while the cache held a given cwd.

    Port of `ExtensionCacheToken`. A token minted before the loop is compared
    against the module state again on every module load, so a
    `clear_extension_cache()` (or a switch to another cwd) that happens
    *during* a load invalidates the rest of that load instead of mixing
    pre- and post-clear modules.
    """

    cwd: str
    generation: int


_extension_cache_cwd: str | None = None
_extension_cache_generation = 0
_extension_cache: dict[str, ExtensionFactory] = {}


def clear_extension_cache() -> None:
    """Drop every cached extension module. Port of `clearExtensionCache()`."""
    global _extension_cache_cwd, _extension_cache_generation
    _extension_cache.clear()
    _extension_cache_cwd = None
    _extension_cache_generation += 1


def _use_extension_cache_cwd(cwd: str) -> ExtensionCacheToken:
    global _extension_cache_cwd
    resolved_cwd = resolve_path(cwd)
    if _extension_cache_cwd is not None and _extension_cache_cwd != resolved_cwd:
        clear_extension_cache()
    _extension_cache_cwd = resolved_cwd
    return ExtensionCacheToken(cwd=resolved_cwd, generation=_extension_cache_generation)


def _is_current_cache_token(token: ExtensionCacheToken | None) -> bool:
    return token is not None and _extension_cache_cwd == token.cwd and _extension_cache_generation == token.generation


class ExtensionAPI:
    """The `pi` object passed to an extension's `pi_extension(pi)` entry point.

    Registration methods write directly onto the `Extension` being built.
    Port of `createExtensionAPI()` narrowed to the registration/action
    surface this port kept (see `types.py`'s module docstring for what was
    dropped and why: shortcuts, message/markdown/entry renderers, provider
    registration, CLI flags).
    """

    def __init__(self, extension: Extension, actions: ExtensionRuntimeActions, cwd: str = ".") -> None:
        self._extension = extension
        self._actions = actions
        self._cwd = cwd

    def on(self, event: str, handler: ExtensionHandler) -> None:
        self._extension.handlers.setdefault(event, []).append(handler)

    def register_tool(self, tool: ToolDefinition) -> None:
        self._extension.tools[tool.name] = RegisteredTool(definition=tool, source_info=self._extension.source_info)
        if self._extension.on_tools_changed is not None:
            self._extension.on_tools_changed()

    def register_command(
        self,
        name: str,
        *,
        handler: Callable[[str, ExtensionCommandContext], Awaitable[None]],
        description: str | None = None,
        get_argument_completions: Callable[[str], object] | None = None,
    ) -> None:
        self._extension.commands[name] = RegisteredCommand(
            name=name,
            handler=handler,
            source_info=self._extension.source_info,
            description=description,
            get_argument_completions=get_argument_completions,
        )

    def send_message(self, *args: object, **kwargs: object) -> None:
        self._actions.send_message(*args, **kwargs)

    def send_user_message(self, *args: object, **kwargs: object) -> None:
        self._actions.send_user_message(*args, **kwargs)

    def append_entry(self, custom_type: str, data: object = None) -> None:
        self._actions.append_entry(custom_type, data)

    def set_session_name(self, name: str) -> None:
        self._actions.set_session_name(name)

    def get_session_name(self) -> str | None:
        return self._actions.get_session_name()

    async def exec(self, command: str, args: list[str], options: ExecOptions | None = None) -> ExecResult:
        """Run a command, defaulting the working directory to the extension's cwd."""
        return await exec_command(command, args, (options.cwd if options else None) or self._cwd, options)

    def set_active_tools(self, tool_names: list[str]) -> None:
        self._actions.set_active_tools(tool_names)

    def get_active_tools(self) -> list[str]:
        return self._actions.get_active_tools()

    @property
    def events(self) -> ExtensionEventBusAPI:
        """`pi.events` -- the cross-extension pub/sub channel.

        Port of `createExtensionAPI()`'s `events` member. Subscriptions made
        through here are recorded on the owning `Extension` so that disposing
        the session removes them from the shared bus; a bus that outlives the
        session (the common case -- callers create one bus and reuse it) would
        otherwise keep calling handlers belonging to a dead session.
        """
        return ExtensionEventBusAPI(self._extension, self._actions.event_bus)


class ExtensionEventBusAPI:
    """`pi.events`. Emits on the shared bus; tracks its own subscriptions."""

    def __init__(self, extension: Extension, event_bus: EventBus | None) -> None:
        self._extension = extension
        self._event_bus = event_bus

    def emit(self, channel: str, data: object = None) -> None:
        if self._event_bus is not None:
            self._event_bus.emit(channel, data)

    def on(self, channel: str, handler: EventBusHandler) -> Callable[[], None]:
        if self._event_bus is None:
            return lambda: None
        unsubscribe = self._event_bus.on(channel, handler)
        state = {"active": True}

        def tracked_unsubscribe() -> None:
            if not state["active"]:
                return
            state["active"] = False
            if tracked_unsubscribe in self._extension.event_bus_unsubscribers:
                self._extension.event_bus_unsubscribers.remove(tracked_unsubscribe)
            unsubscribe()

        self._extension.event_bus_unsubscribers.append(tracked_unsubscribe)
        return tracked_unsubscribe


@dataclass
class ExtensionRuntimeActions:
    """Action implementations bound to a real session.

    Port of the throwing-stub half of TypeScript's `createExtensionRuntime()`
    plus the action-copying half of `ExtensionRunner.bindCore()`, merged into
    one object since this port's `ExtensionRunner` binds actions eagerly at
    construction rather than lazily patching a shared mutable `runtime`
    object post-hoc (Python extensions load and execute during
    `AgentSession.__init__`, after the session already exists, so there is no
    "extension loaded before actions are available" ordering problem to work
    around).
    """

    send_message: Callable[..., None] = lambda *a, **k: None
    send_user_message: Callable[..., None] = lambda *a, **k: None
    append_entry: Callable[[str, object], None] = lambda custom_type, data=None: None
    set_session_name: Callable[[str], None] = lambda name: None
    get_session_name: Callable[[], str | None] = lambda: None
    set_active_tools: Callable[[list[str]], None] = lambda tool_names: None
    get_active_tools: Callable[[], list[str]] = list
    event_bus: EventBus | None = None
    """Shared cross-extension pub/sub bus backing `pi.events`.

    TypeScript threads the bus as a separate `eventBus` parameter through
    `loadExtensions`/`loadExtension`/`createExtensionAPI`; this port carries it
    alongside the other runtime collaborators so every existing loader entry
    point picks it up without a signature change. `None` means "no bus wired",
    in which case `pi.events.emit` is a no-op and `pi.events.on` returns an
    inert unsubscribe -- an extension that uses the bus must not crash a host
    that never created one.
    """


def _default_runtime_actions() -> ExtensionRuntimeActions:
    return ExtensionRuntimeActions()


class SessionRuntimeActions:
    """`pi.*` runtime actions bound to whichever session is current.

    The actions are baked into each extension's `pi` object when the file is
    loaded, but extensions load *before* the session exists, and the session is
    replaced on `/new`, `/import` and `/clone`. So the bindings close over this
    holder rather than a session, and the host calls :meth:`bind` once the
    session is available and again after every replacement.

    Without it the CLI loads extensions with the default no-op actions, so
    `pi.send_user_message()`, `pi.send_message()`, `pi.append_entry()`,
    `pi.set_session_name()` and `pi.set_active_tools()` silently do nothing --
    the extension runs, reports no error, and has no effect.
    """

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._session: Any = None
        # Upstream resolves this with `eventBus ?? createEventBus()`, so
        # `pi.events` always works. Leaving it `None` would make every
        # `emit`/`on` an inert no-op in the shipped CLI, which is a silent
        # failure for any extension that coordinates through the bus.
        self._event_bus = event_bus if event_bus is not None else create_event_bus()

    def bind(self, session: Any) -> None:
        self._session = session

    @property
    def actions(self) -> ExtensionRuntimeActions:
        return ExtensionRuntimeActions(
            send_message=self._send_message,
            send_user_message=self._send_user_message,
            append_entry=self._append_entry,
            set_session_name=self._set_session_name,
            get_session_name=self._get_session_name,
            set_active_tools=self._set_active_tools,
            get_active_tools=self._get_active_tools,
            event_bus=self._event_bus,
        )

    # -- individual actions -------------------------------------------------

    def _spawn(self, coro: Awaitable[None]) -> None:
        # Fire-and-forget, matching TypeScript's `void (async () => ...)()`.
        # The task is kept referenced until it settles so it is not collected
        # mid-flight.
        task = asyncio.ensure_future(coro)
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)

    def _send_message(self, message: Any, options: Any = None, **kwargs: Any) -> None:
        session = self._session
        if session is None:
            return
        deliver_as = kwargs.get("deliver_as")
        trigger_turn = bool(kwargs.get("trigger_turn", False))
        if isinstance(options, dict):
            # TypeScript spells the options camelCase; Python extensions in
            # this port use keyword arguments. Accept both.
            deliver_as = options.get("deliverAs", options.get("deliver_as", deliver_as))
            trigger_turn = bool(options.get("triggerTurn", options.get("trigger_turn", trigger_turn)))
        is_dict = isinstance(message, dict)
        self._spawn(
            session.send_custom_message(
                message["customType"] if is_dict else message.custom_type,
                message["content"] if is_dict else message.content,
                message["display"] if is_dict else message.display,
                message.get("details") if is_dict else getattr(message, "details", None),
                trigger_turn=trigger_turn,
                deliver_as=deliver_as,
            )
        )

    def _send_user_message(self, content: Any, options: Any = None, **kwargs: Any) -> None:
        session = self._session
        if session is None:
            return
        deliver_as = kwargs.get("deliver_as")
        if isinstance(options, dict):
            deliver_as = options.get("deliverAs", options.get("deliver_as", deliver_as))
        self._spawn(session.send_user_message(content, deliver_as=deliver_as))

    def _append_entry(self, custom_type: str, data: object = None) -> None:
        session = self._session
        if session is None:
            return
        session.session_manager.append_custom_entry(custom_type, data)

    def _set_session_name(self, name: str) -> None:
        session = self._session
        if session is None:
            return
        session.set_session_name(name)

    def _get_session_name(self) -> str | None:
        session = self._session
        return session.session_name if session is not None else None

    def _set_active_tools(self, tool_names: list[str]) -> None:
        session = self._session
        if session is None:
            return
        session.set_active_tools_by_name(tool_names)

    def _get_active_tools(self) -> list[str]:
        session = self._session
        return session.get_active_tool_names() if session is not None else []


_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def _is_extension_file(name: str) -> bool:
    return name.endswith(".py")


def _resolve_extension_entries(directory: str) -> list[str] | None:
    """Resolve entry points for a subdirectory: `pi.json` manifest, then `__init__.py`.

    Port of TypeScript's `resolveExtensionEntries()`. The manifest wins over
    the directory convention, but only if it declares at least one extension
    path that actually exists -- otherwise discovery falls back to
    `__init__.py`, exactly as TypeScript falls back to `index.ts`/`index.js`.

    `package.json`'s nested `"pi"` field maps to a top-level `pi.json` here;
    see `core/pi_manifest.py`'s module docstring.
    """
    manifest = read_pi_manifest(manifest_path_for_package_root(directory))
    if manifest and manifest.extensions:
        entries = [
            resolved
            for ext_path in manifest.extensions
            if os.path.isfile(resolved := os.path.abspath(os.path.join(directory, ext_path)))
        ]
        if entries:
            return entries

    init_path = os.path.join(directory, "__init__.py")
    if os.path.isfile(init_path):
        return [init_path]
    return None


def discover_extensions_in_dir(directory: str) -> list[str]:
    """Discover extension entry points directly under `directory`.

    Discovery rules (port of `discoverExtensionsInDir`):
    1. Direct files: `extensions/*.py` -> load.
    2. Subdirectory with a `pi.json` manifest declaring extensions -> load what
       it declares.
    3. Subdirectory with `__init__.py`: `extensions/*/__init__.py` -> load.
    No recursion beyond one level.
    """
    if not os.path.isdir(directory):
        return []

    discovered: list[str] = []
    try:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except OSError:
        return []

    for entry in entries:
        entry_path = entry.path
        try:
            is_file = entry.is_file(follow_symlinks=True)
            is_dir = entry.is_dir(follow_symlinks=True)
        except OSError:
            continue

        if is_file and _is_extension_file(entry.name):
            discovered.append(entry_path)
            continue

        if is_dir:
            resolved = _resolve_extension_entries(entry_path)
            if resolved:
                discovered.extend(resolved)

    return discovered


def _load_extension_module(resolved_path: str):
    """Load a Python module from an absolute file path.

    This is the JS-module substitution: `importlib.util.spec_from_file_location`
    is the standard-library mechanism for importing a module given a
    filesystem path (rather than a dotted name resolved via `sys.path`),
    exactly analogous to how `jiti.import(extensionPath)` loads a TypeScript
    file given a path in the original. Each extension gets a synthetic
    `sys.modules` entry keyed by a random UUID so that two extensions with
    the same filename (e.g. two different `__init__.py` directory
    extensions) never collide in the module cache, and so reloading (a fresh
    `discover_and_load_extensions()` call) always re-executes the file
    instead of returning a stale cached module.
    """
    module_name = f"_pi_extension_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create a module spec for {resolved_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _load_extension_factory(resolved_path: str, cache_token: ExtensionCacheToken | None):
    """Return the `pi_extension` callable for a path, using the module cache.

    Only the *module* is cached, never the `Extension`/`ExtensionAPI` built
    from it, so a cached load still reruns the factory and produces a fresh
    extension object -- exactly `loadExtensionModule()`'s contract.
    """
    if _is_current_cache_token(cache_token):
        cached = _extension_cache.get(resolved_path)
        if cached is not None:
            return cached

    module = _load_extension_module(resolved_path)
    factory = getattr(module, _EXTENSION_ENTRY_POINT_NAME, None)
    if not callable(factory):
        return None
    if _is_current_cache_token(cache_token):
        _extension_cache[resolved_path] = factory
    return factory


async def _invoke_factory(factory: ExtensionFactory, api: ExtensionAPI) -> None:
    result = factory(api)
    if hasattr(result, "__await__"):
        await result  # type: ignore[func-returns-value]


def _create_extension(extension_path: str, resolved_path: str) -> Extension:
    source = (
        (extension_path[1:-1].split(":")[0] or "temporary")
        if extension_path.startswith("<") and extension_path.endswith(">")
        else "local"
    )
    base_dir = None if extension_path.startswith("<") else os.path.dirname(resolved_path)

    return Extension(
        path=extension_path,
        resolved_path=resolved_path,
        source_info=create_synthetic_source_info(extension_path, source, base_dir=base_dir),
    )


@dataclass
class NamedInlineExtension:
    """A named in-process extension factory.

    Port of the object arm of TypeScript's `InlineExtension` union
    (`{ name, factory, hidden? }`). `name` is shown as `<inline:name>` in the
    startup Extensions list; `hidden` omits it from that list entirely.
    """

    name: str
    factory: ExtensionFactory
    hidden: bool = False


InlineExtension = ExtensionFactory | NamedInlineExtension
"""Port of TypeScript's `InlineExtension`: a bare factory or a named wrapper."""


async def load_extension_from_factory(
    factory: ExtensionFactory,
    cwd: str,
    actions: ExtensionRuntimeActions | None = None,
    extension_path: str = "<inline>",
) -> Extension:
    """Build an `Extension` by running an in-process factory.

    Port of `loadExtensionFromFactory()`. Unlike `load_extension()` this does
    not swallow factory errors -- callers decide (see
    `load_extension_factories`).
    """
    extension = _create_extension(extension_path, extension_path)
    api = ExtensionAPI(extension, actions or _default_runtime_actions(), resolve_path(cwd))
    await _invoke_factory(factory, api)
    record_timing(f"{extension_path} factory", "extensions")
    return extension


async def load_extension_factories(
    factories: list[InlineExtension],
    cwd: str,
    actions: ExtensionRuntimeActions | None = None,
) -> LoadExtensionsResult:
    """Load inline extension factories. Port of `loadExtensionFactories()`.

    Bare factories are displayed as `<inline:N>` using their **1-based
    position in the full list** (so a named factory in the middle does not
    renumber the bare ones after it); named wrappers are displayed as
    `<inline:name>` and carry their `hidden` flag onto the `Extension`.
    """
    resolved_cwd = resolve_path(cwd)
    resolved_actions = actions or _default_runtime_actions()
    extensions: list[Extension] = []
    errors: list[dict[str, str]] = []

    for index, entry in enumerate(factories):
        named = isinstance(entry, NamedInlineExtension)
        factory = entry.factory if isinstance(entry, NamedInlineExtension) else entry
        extension_path = f"<inline:{entry.name if isinstance(entry, NamedInlineExtension) else index + 1}>"
        try:
            extension = await load_extension_from_factory(factory, resolved_cwd, resolved_actions, extension_path)
        except Exception as err:
            errors.append({"path": extension_path, "error": str(err) or "failed to load extension"})
            continue
        extension.hidden = named and entry.hidden  # type: ignore[union-attr]
        extensions.append(extension)

    return LoadExtensionsResult(extensions=extensions, errors=errors)


async def load_extension(
    extension_path: str,
    cwd: str,
    actions: ExtensionRuntimeActions | None = None,
    cache_token: ExtensionCacheToken | None = None,
) -> tuple[Extension | None, str | None]:
    """Load one extension. Returns `(extension, None)` or `(None, error)`.

    Port of `loadExtension()`: never raises for extension-authored problems
    (missing entry point, import errors, exceptions from the factory) --
    those become the returned error string so a caller can isolate one bad
    extension from the rest, matching `loadExtensionsInternal`'s per-path
    try/except-and-continue.
    """
    resolved_path = resolve_path(extension_path, cwd)

    try:
        factory = _load_extension_factory(resolved_path, cache_token)
    except Exception as err:
        return None, f"Failed to load extension: {err}"

    if factory is None:
        return (
            None,
            f"Extension does not export a valid `{_EXTENSION_ENTRY_POINT_NAME}` callable: {extension_path}",
        )

    extension = _create_extension(extension_path, resolved_path)
    api = ExtensionAPI(extension, actions or _default_runtime_actions(), cwd)
    try:
        await _invoke_factory(factory, api)
    except Exception as err:
        return None, f"Failed to load extension: {err}"
    record_timing(f"{extension_path} factory", "extensions")

    return extension, None


async def _load_extensions_internal(
    paths: list[str],
    cwd: str,
    actions: ExtensionRuntimeActions | None,
    use_cache: bool,
) -> LoadExtensionsResult:
    cache_token = _use_extension_cache_cwd(cwd) if use_cache else None
    resolved_cwd = cache_token.cwd if cache_token else resolve_path(cwd)
    resolved_actions = actions or _default_runtime_actions()
    extensions: list[Extension] = []
    errors: list[dict[str, str]] = []

    for path in paths:
        extension, error = await load_extension(path, resolved_cwd, resolved_actions, cache_token)
        if error:
            errors.append({"path": path, "error": error})
            continue
        if extension:
            extensions.append(extension)

    return LoadExtensionsResult(extensions=extensions, errors=errors)


async def load_extensions(
    paths: list[str],
    cwd: str,
    actions: ExtensionRuntimeActions | None = None,
) -> LoadExtensionsResult:
    """Load extensions from explicit paths. Port of `loadExtensions()`.

    Always re-executes each extension module; `load_extensions_cached()` is
    the caching variant.
    """
    return await _load_extensions_internal(paths, cwd, actions, False)


async def load_extensions_cached(
    paths: list[str],
    cwd: str,
    actions: ExtensionRuntimeActions | None = None,
) -> LoadExtensionsResult:
    """Load extensions, reusing modules loaded earlier for the same cwd.

    Port of `loadExtensionsCached()`. Extension *modules* (their top-level
    code) run once per cwd until `clear_extension_cache()`; the factories
    themselves rerun on every call, so each load still yields fresh
    `Extension` objects with their own handlers, tools and commands.
    """
    return await _load_extensions_internal(paths, cwd, actions, True)


async def discover_and_load_extensions(
    configured_paths: list[str],
    cwd: str,
    *,
    project_trusted: bool = True,
    agent_dir: str | None = None,
    actions: ExtensionRuntimeActions | None = None,
    extension_factories: list[InlineExtension] | None = None,
    no_extensions: bool = False,
) -> LoadExtensionsResult:
    """Discover and load extensions from standard locations.

    Port of `discoverAndLoadExtensions()`, with the trust gate `resource-
    loader.ts` applies via `SettingsManager.isProjectTrusted()` inlined as an
    explicit `project_trusted` parameter (this module has no
    `SettingsManager`/`ResourceLoader` dependency of its own, keeping the
    loader unit-testable without constructing either). See the module
    docstring's "Trust gating" section for exactly what is and is not gated.
    """
    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir if agent_dir is not None else get_agent_dir())

    all_paths: list[str] = []
    seen: set[str] = set()

    def add_paths(paths: list[str]) -> None:
        for p in paths:
            # `DefaultResourceLoader.mergePaths()` dedups on `canonicalizePath`
            # (realpath), so a shared extension dir symlinked into both
            # `agent_dir/extensions` and `cwd/.pi/extensions` loads once, keeping
            # the first (project) alias as the extension's path.
            canonical = canonicalize_path(os.path.abspath(p))
            if canonical not in seen:
                seen.add(canonical)
                all_paths.append(p)

    # `--no-extensions` suppresses *discovery* only. TypeScript keeps the
    # CLI-supplied paths either way (`resource-loader.ts:451` and `:555`:
    # `noExtensions ? cliEnabledExtensions : mergePaths(cli, discovered)`), so
    # `pi --no-extensions -e ./one.py` still loads `one.py`.
    if not no_extensions:
        # 1. Project-local extensions: cwd/.pi/extensions/ -- gated on trust.
        if project_trusted:
            local_ext_dir = os.path.join(resolved_cwd, CONFIG_DIR_NAME, "extensions")
            add_paths(discover_extensions_in_dir(local_ext_dir))

        # 2. Global/user extensions: agent_dir/extensions/ -- always discovered.
        global_ext_dir = os.path.join(resolved_agent_dir, "extensions")
        add_paths(discover_extensions_in_dir(global_ext_dir))

    # 3. Explicitly configured paths -- always discovered (matches TypeScript:
    # CLI-provided/inline extension paths bypass project trust entirely).
    missing_configured: list[str] = []
    for p in configured_paths:
        resolved = resolve_path(p, resolved_cwd)
        if os.path.isdir(resolved):
            entries = _resolve_extension_entries(resolved)
            if entries:
                add_paths(entries)
                continue
            add_paths(discover_extensions_in_dir(resolved))
            continue
        if not os.path.exists(resolved):
            # `resource-loader.ts:455-461` pre-checks explicitly requested
            # paths and reports the path itself. Without this the loader's
            # raw errno surfaces instead ("[Errno 2] No such file or
            # directory"), which reads like an internal fault rather than
            # "you asked for a file that isn't there".
            missing_configured.append(resolved)
            continue
        add_paths([resolved])

    result = await load_extensions(all_paths, resolved_cwd, actions)
    for resolved in missing_configured:
        result.errors.append({"path": resolved, "error": f"Extension path does not exist: {resolved}"})
    if extension_factories:
        # `resource-loader.ts` appends inline factories after path-loaded
        # extensions, so they always sort last in the startup list.
        inline = await load_extension_factories(extension_factories, resolved_cwd, actions)
        result.extensions.extend(inline.extensions)
        result.errors.extend(inline.errors)
    result.conflicts.extend(detect_extension_conflicts(result.extensions))
    return result


def detect_extension_conflicts(extensions: list[Extension]) -> list[dict[str, str]]:
    """Report tools registered under the same name by two different extensions.

    Port of `DefaultResourceLoader.detectExtensionConflicts()`. TypeScript runs
    this from the resource loader right after `discoverAndLoadExtensions()`;
    this port has no `ResourceLoader` extension stage (see
    `core/resource_loader.py`'s module docstring), so it runs at the end of
    `discover_and_load_extensions()` instead -- the same set of extensions,
    discovered plus explicitly configured.

    Both extensions stay loaded; the conflict is only a diagnostic reported
    on `LoadExtensionsResult.conflicts` (not `errors`), and precedence keeps
    following load order. TypeScript also conflicts on flag
    names, which this port has no equivalent for (`pi.register_flag` is not
    part of this extension API), and deliberately does *not* conflict on
    command names -- those are disambiguated by `ExtensionRunner` into
    `name:1`/`name:2` invocation names instead.
    """
    conflicts: list[dict[str, str]] = []
    tool_owners: dict[str, str] = {}
    for extension in extensions:
        for tool_name in extension.tools:
            existing_owner = tool_owners.get(tool_name)
            if existing_owner is not None and existing_owner != extension.path:
                conflicts.append(
                    {"path": extension.path, "error": f'Tool "{tool_name}" conflicts with {existing_owner}'}
                )
            else:
                tool_owners[tool_name] = extension.path
    return conflicts
