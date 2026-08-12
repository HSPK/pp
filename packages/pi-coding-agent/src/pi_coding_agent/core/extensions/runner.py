"""Extension runner: invokes hooks, isolates errors, orders handlers.

Port of `packages/coding-agent/src/core/extensions/runner.ts` (1236 lines) and
`wrapper.ts` (45 lines, folded into this module since `wrap_registered_tool`
only exists to adapt a `RegisteredTool` into an `AgentTool` using the runner's
`create_context()`). `wrap_registered_tool`'s context-defaulting also inlines
what `core/tools/tool-definition-wrapper.ts`'s `wrapToolDefinition` does
(the reusable, ctx-factory-taking version of that same step lives in
`pi_coding_agent.tools.tool_definition_wrapper` instead).

**Scope narrowing** (see `types.py`'s module docstring for the full list):
no keybinding/shortcut resolution (`getShortcuts`/`buildBuiltinKeybindings`),
no message/markdown/entry renderer registries, no provider
registration/`ModelRegistry`, no CLI flags. `resources_discover` contributes
skill and prompt paths (`emit_resources_discover` below, consumed by
`AgentSession`); the `themePaths` half of its result is collected but
unused, because this port has no theme loading.

**Error isolation.** Every `emit_*` method mirrors TypeScript's per-handler
try/except: one handler raising never stops later handlers (same extension or
a different one) from running, and never crashes the caller. The one
documented exception, matching TypeScript exactly, is `emit_tool_call`
(`before_tool_call`'s hook): its handler exceptions propagate, because
`AgentSession._install_agent_tool_hooks`'s `before_tool_call` callback
re-raises around it (`_installAgentToolHooks`'s `beforeToolCall` in the
original does not catch either) -- blocking tool execution on an extension
bug is the deliberate, documented behavior, not an oversight.

**Ordering.** Handlers run in extension-list order, then per-extension
registration order (`list.append` order, matching TypeScript's `Array.push`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pi_agent.types import AgentMessage
from pi_ai.types import ImageContent
from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.core.extensions.types import (
    BeforeAgentStartEvent,
    BeforeAgentStartEventResult,
    BeforeProviderHeadersEvent,
    BeforeProviderRequestEvent,
    ContextEvent,
    ContextEventResult,
    ContextUsage,
    Extension,
    ExtensionCommandContext,
    ExtensionContext,
    ExtensionError,
    ExtensionMode,
    ExtensionUIContext,
    InputEvent,
    InputEventResult,
    InputSource,
    MessageEndEvent,
    MessageEndEventResult,
    NullExtensionUIContext,
    ProjectTrustContext,
    ProjectTrustEvent,
    ProjectTrustEventResult,
    RegisteredCommand,
    RegisteredTool,
    ResourcesDiscoverEvent,
    ResourcesDiscoverResult,
    SessionBeforeCompactResult,
    SessionBeforeForkResult,
    SessionBeforeSwitchResult,
    SessionBeforeTreeResult,
    ToolCallEvent,
    ToolCallEventResult,
    ToolDefinition,
    ToolResultEvent,
    ToolResultEventResult,
    UserBashEvent,
    UserBashEventResult,
)
from pi_coding_agent.core.system_prompt import BuildSystemPromptOptions
from pi_coding_agent.tools.tool_definition_wrapper import wrap_tool_definition

ExtensionErrorListener = Callable[[ExtensionError], None]

_SESSION_BEFORE_EVENT_TYPES = frozenset(
    {"session_before_switch", "session_before_fork", "session_before_compact", "session_before_tree"}
)

_SessionBeforeResult = (
    SessionBeforeSwitchResult | SessionBeforeForkResult | SessionBeforeCompactResult | SessionBeforeTreeResult
)


@dataclass
class ProjectTrustEmitResult:
    """What `emit_project_trust_event` decided, plus any handler failures."""

    result: ProjectTrustEventResult | None = None
    errors: list[ExtensionError] = field(default_factory=list)


async def emit_project_trust_event(
    extensions: list[Extension],
    event: ProjectTrustEvent,
    ctx: ProjectTrustContext,
) -> ProjectTrustEmitResult:
    """Ask the loaded extensions whether this project should be trusted.

    Port of `emitProjectTrustEvent` in `runner.ts`. This runs *before* an
    `ExtensionRunner` exists -- trust decides whether the project's own
    extensions may load at all -- so it is a module-level function over the
    already-loaded extension list rather than a runner method.

    A single extension may register several `project_trust` handlers. The
    first one that answers `"yes"` or `"no"` wins; `"undecided"` falls through
    to the next handler, and a raising handler is recorded and skipped rather
    than aborting the scan.
    """
    errors: list[ExtensionError] = []
    for ext in extensions:
        handlers = ext.handlers.get("project_trust")
        if not handlers:
            continue
        for handler in list(handlers):
            try:
                handler_result = await _maybe_await(handler(event, ctx))
                if handler_result is None or handler_result.trusted == "undecided":
                    continue
                return ProjectTrustEmitResult(result=handler_result, errors=errors)
            except Exception as err:
                errors.append(ExtensionError(extension_path=ext.path, event=event.type, error=str(err)))
    return ProjectTrustEmitResult(errors=errors)


@dataclass
class ExtensionContextActions:
    """Bound accessors backing `ExtensionContext`. Port of TypeScript's
    `ExtensionContextActions`, narrowed to what this port's `agent_session.py`
    can supply (no `getSystemPromptOptions` split out here -- folded into
    `ExtensionCommandContext` construction directly, see `create_command_context`).
    """

    get_model: Callable[[], Any] = lambda: None
    get_scoped_models: Callable[[], tuple[Any, ...]] = lambda: ()
    is_idle: Callable[[], bool] = lambda: True
    is_project_trusted: Callable[[], bool] = lambda: True
    get_signal: Callable[[], AbortSignal | None] = lambda: None
    abort: Callable[[], None] = lambda: None
    has_pending_messages: Callable[[], bool] = lambda: False
    shutdown: Callable[[], None] = lambda: None
    get_context_usage: Callable[[], ContextUsage | None] = lambda: None
    compact: Callable[..., None] = lambda **k: None
    get_system_prompt: Callable[[], str] = lambda: ""
    get_system_prompt_options: Callable[[], BuildSystemPromptOptions] = field(
        default=lambda: BuildSystemPromptOptions()
    )
    get_thinking_level: Callable[[], Any] = lambda: None
    get_active_tool_names: Callable[[], list[str]] = lambda: []
    wait_for_idle: Callable[[], Awaitable[None]] | None = None
    refresh_tools: Callable[[], None] | None = None
    """Rebuild the owning session's tool registry.

    Port of `ExtensionRuntime.refreshTools`. `pi.register_tool()` calls it so a
    tool registered *after* the session was constructed (typically from a
    `session_start` handler) still reaches `agent.state.tools` and the system
    prompt.
    """


@dataclass(frozen=True)
class DiscoveredResourcePath:
    """A resource path an extension contributed, plus which extension contributed it."""

    path: str
    extension_path: str


@dataclass(frozen=True)
class DiscoveredResourcePaths:
    """Everything `resources_discover` handlers returned, grouped by resource kind."""

    skill_paths: list[DiscoveredResourcePath] = field(default_factory=list)
    prompt_paths: list[DiscoveredResourcePath] = field(default_factory=list)
    theme_paths: list[DiscoveredResourcePath] = field(default_factory=list)


class ExtensionRunner:
    """Executes extensions and manages their lifecycle.

    Constructed with the extensions to run plus the mode-independent pieces
    every mode has available (`cwd`, `session_manager`); the mode-specific
    pieces (current model, idle state, abort, ...) are supplied afterwards
    via `bind_core()`, mirroring TypeScript's `bindCore()` two-phase
    construction (extensions load before the owning `AgentSession` has
    finished wiring itself up).
    """

    def __init__(self, extensions: list[Extension], *, cwd: str, session_manager: Any = None) -> None:
        self.extensions = list(extensions)
        self.cwd = cwd
        self.session_manager = session_manager
        self._ui: ExtensionUIContext = NullExtensionUIContext()
        self._mode: ExtensionMode = "print"
        self._error_listeners: set[ExtensionErrorListener] = set()
        self.errors: list[ExtensionError] = []
        self._actions = ExtensionContextActions()
        self._command_wait_for_idle: Callable[[], Awaitable[None]] | None = None

    # ------------------------------------------------------------------
    # Binding
    # ------------------------------------------------------------------

    def bind_core(self, actions: ExtensionContextActions) -> None:
        self._actions = actions
        # Port of `bindCore()`'s `this.runtime.refreshTools = actions.refreshTools`.
        # This port has no shared per-load runtime object, so the hook is installed
        # on each extension (see `Extension.on_tools_changed`).
        for ext in self.extensions:
            ext.on_tools_changed = actions.refresh_tools

    def set_ui_context(self, ui: ExtensionUIContext | None = None, mode: ExtensionMode = "print") -> None:
        self._ui = ui if ui is not None else NullExtensionUIContext()
        self._mode = mode

    def get_ui_context(self) -> ExtensionUIContext:
        return self._ui

    def has_ui(self) -> bool:
        return not isinstance(self._ui, NullExtensionUIContext)

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def on_error(self, listener: ExtensionErrorListener) -> Callable[[], None]:
        self._error_listeners.add(listener)

        def unsubscribe() -> None:
            self._error_listeners.discard(listener)

        return unsubscribe

    def emit_error(self, error: ExtensionError) -> None:
        self.errors.append(error)
        for listener in list(self._error_listeners):
            listener(error)

    def unsubscribe_events(self) -> None:
        """Drop every `pi.events.on()` subscription these extensions made.

        Port of the event-bus half of TypeScript's `runtime.invalidate()`. The
        bus is owned by the host and typically outlives the session, so a
        disposed session must take its extensions' handlers off it; otherwise
        a later `eventBus.emit()` still runs handlers that close over the dead
        session (issue #7193).
        """
        for ext in self.extensions:
            for unsubscribe in list(ext.event_bus_unsubscribers):
                unsubscribe()
            ext.event_bus_unsubscribers.clear()

    def _catch_handler_error(self, ext: Extension, event_type: str, err: Exception) -> None:
        self.emit_error(ExtensionError(extension_path=ext.path, event=event_type, error=str(err)))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def has_handlers(self, event_type: str) -> bool:
        return any(ext.handlers.get(event_type) for ext in self.extensions)

    def get_all_registered_tools(self) -> list[RegisteredTool]:
        """First registration per name wins (matches TypeScript)."""
        by_name: dict[str, RegisteredTool] = {}
        for ext in self.extensions:
            for tool in ext.tools.values():
                by_name.setdefault(tool.definition.name, tool)
        return list(by_name.values())

    def get_tool_definition(self, tool_name: str) -> ToolDefinition | None:
        for ext in self.extensions:
            tool = ext.tools.get(tool_name)
            if tool:
                return tool.definition
        return None

    def _resolve_registered_commands(self) -> list[tuple[str, RegisteredCommand]]:
        """Resolve `(invocation_name, command)` pairs, port of
        `resolveRegisteredCommands()`: commands with a name collision across
        extensions get suffixed `name:2`, `name:3`, ... in extension order.
        """
        commands: list[RegisteredCommand] = []
        counts: dict[str, int] = {}
        for ext in self.extensions:
            for command in ext.commands.values():
                commands.append(command)
                counts[command.name] = counts.get(command.name, 0) + 1

        seen: dict[str, int] = {}
        taken: set[str] = set()
        resolved: list[tuple[str, RegisteredCommand]] = []
        for command in commands:
            occurrence = seen.get(command.name, 0) + 1
            seen[command.name] = occurrence
            invocation_name = f"{command.name}:{occurrence}" if counts[command.name] > 1 else command.name
            suffix = occurrence
            while invocation_name in taken:
                suffix += 1
                invocation_name = f"{command.name}:{suffix}"
            taken.add(invocation_name)
            resolved.append((invocation_name, command))
        return resolved

    def get_registered_commands(self) -> list[tuple[str, RegisteredCommand]]:
        return self._resolve_registered_commands()

    def get_command(self, name: str) -> RegisteredCommand | None:
        for invocation_name, command in self._resolve_registered_commands():
            if invocation_name == name:
                return command
        return None

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def create_context(self) -> ExtensionContext:
        """Build a fresh `ExtensionContext` reflecting current bound state.

        See `types.py`'s `ExtensionContext` docstring: this port rebuilds a
        new snapshot on every call instead of TypeScript's live-getter
        object, so callers should not hold a `ctx` across an `await` and
        expect it to reflect state changed during that `await` -- call
        `create_context()` again instead.
        """
        actions = self._actions
        return ExtensionContext(
            ui=self._ui,
            mode=self._mode,
            has_ui=self.has_ui(),
            cwd=self.cwd,
            session_manager=self.session_manager,
            model=actions.get_model(),
            scoped_models=tuple(actions.get_scoped_models()),
            is_idle=actions.is_idle,
            is_project_trusted=actions.is_project_trusted,
            signal=actions.get_signal(),
            abort=actions.abort,
            has_pending_messages=actions.has_pending_messages,
            shutdown=actions.shutdown,
            get_context_usage=actions.get_context_usage,
            compact=actions.compact,
            get_system_prompt=actions.get_system_prompt,
            thinking_level=actions.get_thinking_level(),
        )

    def create_command_context(self) -> ExtensionCommandContext:
        base = self.create_context()
        actions = self._actions
        wait_for_idle = actions.wait_for_idle or (lambda: _immediate())
        return ExtensionCommandContext(
            **{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
            get_system_prompt_options=actions.get_system_prompt_options,
            wait_for_idle=wait_for_idle,
        )

    def get_active_tool_names(self) -> list[str]:
        return self._actions.get_active_tool_names()

    # ------------------------------------------------------------------
    # Generic emit()
    # ------------------------------------------------------------------

    async def emit(self, event: Any) -> Any:
        """Generic dispatcher for events without a dedicated `emit_*` method.

        `session_before_*` events short-circuit on `cancel=True` and
        otherwise keep the last non-`None` handler result (matching
        TypeScript's `emit()`); every other event type's return value is
        discarded (there is nothing for callers to combine).
        """
        ctx = self.create_context()
        event_type = event.type
        result: _SessionBeforeResult | None = None
        is_session_before = event_type in _SESSION_BEFORE_EVENT_TYPES

        for ext in self.extensions:
            handlers = ext.handlers.get(event_type)
            if not handlers:
                continue
            for handler in handlers:
                try:
                    handler_result = await _maybe_await(handler(event, ctx))
                except Exception as err:
                    self._catch_handler_error(ext, event_type, err)
                    continue
                if is_session_before and handler_result is not None:
                    result = handler_result
                    if getattr(result, "cancel", False):
                        return result
        return result

    async def emit_message_end(self, event: MessageEndEvent) -> Any | None:
        ctx = self.create_context()
        current_message = event.message
        modified = False

        for ext in self.extensions:
            handlers = ext.handlers.get("message_end")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    current_event = MessageEndEvent(message=current_message)
                    handler_result: MessageEndEventResult | None = await _maybe_await(handler(current_event, ctx))
                except Exception as err:
                    self._catch_handler_error(ext, "message_end", err)
                    continue
                if not handler_result or handler_result.message is None:
                    continue
                new_role = getattr(handler_result.message, "role", None)
                old_role = getattr(current_message, "role", None)
                if new_role != old_role:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="message_end",
                            error="message_end handlers must return a message with the same role",
                        )
                    )
                    continue
                current_message = handler_result.message
                modified = True

        return current_message if modified else None

    async def emit_tool_result(self, event: ToolResultEvent) -> ToolResultEventResult | None:
        ctx = self.create_context()
        current = ToolResultEvent(
            tool_call_id=event.tool_call_id,
            tool_name=event.tool_name,
            input=event.input,
            content=event.content,
            is_error=event.is_error,
            details=event.details,
            usage=event.usage,
        )
        modified = False

        for ext in self.extensions:
            handlers = ext.handlers.get("tool_result")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    handler_result: ToolResultEventResult | None = await _maybe_await(handler(current, ctx))
                except Exception as err:
                    self._catch_handler_error(ext, "tool_result", err)
                    continue
                if not handler_result:
                    continue
                if handler_result.content is not None:
                    current.content = handler_result.content
                    modified = True
                if handler_result.details is not None:
                    current.details = handler_result.details
                    modified = True
                if handler_result.is_error is not None:
                    current.is_error = handler_result.is_error
                    modified = True
                if handler_result.usage is not None:
                    current.usage = handler_result.usage
                    modified = True

        if not modified:
            return None
        return ToolResultEventResult(
            content=current.content, details=current.details, is_error=current.is_error, usage=current.usage
        )

    async def emit_tool_call(self, event: ToolCallEvent) -> ToolCallEventResult | None:
        """Not error-isolated: see module docstring's "Error isolation" section."""
        ctx = self.create_context()
        result: ToolCallEventResult | None = None

        for ext in self.extensions:
            handlers = ext.handlers.get("tool_call")
            if not handlers:
                continue
            for handler in handlers:
                handler_result: ToolCallEventResult | None = await _maybe_await(handler(event, ctx))
                if handler_result:
                    result = handler_result
                    if result.block:
                        return result
        return result

    async def emit_before_agent_start(
        self,
        prompt: str,
        images: list[ImageContent] | None,
        system_prompt: str,
        system_prompt_options: BuildSystemPromptOptions,
    ) -> tuple[list[Any], str | None] | None:
        """Returns `(messages, system_prompt_override)` or `None` if nothing changed."""
        current_system_prompt = system_prompt
        messages: list[Any] = []
        system_prompt_modified = False

        for ext in self.extensions:
            handlers = ext.handlers.get("before_agent_start")
            if not handlers:
                continue
            for handler in handlers:
                ctx = self.create_context()
                # Default arg binds the loop variable's *current* value at lambda-creation time
                # rather than a late-binding reference to whatever it is when the lambda runs.
                ctx.get_system_prompt = lambda prompt=current_system_prompt: prompt
                try:
                    event = BeforeAgentStartEvent(
                        prompt=prompt,
                        images=images,
                        system_prompt=current_system_prompt,
                        system_prompt_options=system_prompt_options,
                    )
                    handler_result: BeforeAgentStartEventResult | None = await _maybe_await(handler(event, ctx))
                except Exception as err:
                    self._catch_handler_error(ext, "before_agent_start", err)
                    continue
                if not handler_result:
                    continue
                if handler_result.message is not None:
                    messages.append(handler_result.message)
                if handler_result.system_prompt is not None:
                    current_system_prompt = handler_result.system_prompt
                    system_prompt_modified = True

        if messages or system_prompt_modified:
            return messages, (current_system_prompt if system_prompt_modified else None)
        return None

    async def emit_user_bash(self, event: UserBashEvent) -> UserBashEventResult | None:
        """Port of `emitUserBash`. The first handler returning a result wins.

        A handler may return `operations` (a replacement execution backend) or a
        complete `result`, in which case the caller skips execution entirely.
        """
        ctx = self.create_context()

        for ext in self.extensions:
            handlers = ext.handlers.get("user_bash")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    handler_result: UserBashEventResult | None = await _maybe_await(handler(event, ctx))
                except Exception as err:
                    self._catch_handler_error(ext, "user_bash", err)
                    continue
                if handler_result:
                    return handler_result

        return None

    async def emit_context(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        """Port of `emitContext`. Chains each handler's replacement message list.

        TypeScript deep-clones the incoming list with `structuredClone`; this
        port copies the list (handlers get the same message objects) because
        `AgentMessage` dataclasses are not uniformly deep-copyable here and no
        caller relies on the clone.
        """
        ctx = self.create_context()
        current_messages = list(messages)

        for ext in self.extensions:
            handlers = ext.handlers.get("context")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    result: ContextEventResult | None = await _maybe_await(
                        handler(ContextEvent(messages=current_messages), ctx)
                    )
                except Exception as err:
                    self._catch_handler_error(ext, "context", err)
                    continue
                if result is not None and result.messages is not None:
                    current_messages = result.messages

        return current_messages

    async def emit_before_provider_request(self, payload: Any) -> Any:
        """Port of `emitBeforeProviderRequest`. Chains each handler's replacement payload.

        A handler returning `None` (TypeScript: `undefined`) leaves the payload
        untouched, so a handler that only inspects the request does not have to
        return it.
        """
        ctx = self.create_context()
        current_payload = payload

        for ext in self.extensions:
            handlers = ext.handlers.get("before_provider_request")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    handler_result = await _maybe_await(
                        handler(BeforeProviderRequestEvent(payload=current_payload), ctx)
                    )
                except Exception as err:
                    self._catch_handler_error(ext, "before_provider_request", err)
                    continue
                if handler_result is not None:
                    current_payload = handler_result

        return current_payload

    async def emit_before_provider_headers(self, headers: dict[str, str | None]) -> dict[str, str | None]:
        """Port of `emitBeforeProviderHeaders`.

        Handlers mutate `headers` in place; return values are ignored, and the
        same dict is handed back so callers can chain. A raising handler is
        isolated and reported through `on_error`, exactly as in TypeScript.
        """
        ctx = self.create_context()

        for ext in self.extensions:
            handlers = ext.handlers.get("before_provider_headers")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    await _maybe_await(handler(BeforeProviderHeadersEvent(headers=headers), ctx))
                except Exception as err:
                    self._catch_handler_error(ext, "before_provider_headers", err)

        return headers

    async def emit_resources_discover(self, cwd: str, reason: Literal["startup", "reload"]) -> DiscoveredResourcePaths:
        """Collect extension-contributed skill/prompt paths.

        Each returned path is paired with the path of the extension that
        contributed it, so the caller can label the resulting skills and prompts
        with that extension as their source.
        """
        ctx = self.create_context()
        skill_paths: list[DiscoveredResourcePath] = []
        prompt_paths: list[DiscoveredResourcePath] = []
        theme_paths: list[DiscoveredResourcePath] = []

        for ext in self.extensions:
            handlers = ext.handlers.get("resources_discover")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    event = ResourcesDiscoverEvent(cwd=cwd, reason=reason)
                    result: ResourcesDiscoverResult | None = await _maybe_await(handler(event, ctx))
                except Exception as err:
                    self._catch_handler_error(ext, "resources_discover", err)
                    continue
                if result is None:
                    continue
                skill_paths.extend(DiscoveredResourcePath(p, ext.path) for p in result.skill_paths)
                prompt_paths.extend(DiscoveredResourcePath(p, ext.path) for p in result.prompt_paths)
                theme_paths.extend(DiscoveredResourcePath(p, ext.path) for p in result.theme_paths)

        return DiscoveredResourcePaths(skill_paths, prompt_paths, theme_paths)

    async def emit_input(
        self,
        text: str,
        images: list[ImageContent] | None,
        source: InputSource,
        streaming_behavior: str | None = None,
    ) -> InputEventResult:
        """Chains `transform` results; `handled` short-circuits."""
        ctx = self.create_context()
        current_text = text
        current_images = images

        for ext in self.extensions:
            handlers = ext.handlers.get("input")
            if not handlers:
                continue
            for handler in handlers:
                try:
                    event = InputEvent(
                        text=current_text, images=current_images, source=source, streaming_behavior=streaming_behavior
                    )
                    result: InputEventResult | None = await _maybe_await(handler(event, ctx))
                except Exception as err:
                    self._catch_handler_error(ext, "input", err)
                    continue
                if result is None:
                    continue
                if result.action == "handled":
                    return result
                if result.action == "transform":
                    current_text = result.text if result.text is not None else current_text
                    current_images = result.images if result.images is not None else current_images

        if current_text != text or current_images != images:
            return InputEventResult(action="transform", text=current_text, images=current_images)
        return InputEventResult(action="continue")


def _immediate() -> Awaitable[None]:
    async def _noop() -> None:
        return None

    return _noop()


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


# ============================================================================
# Tool wrapping (port of `wrapper.ts`)
# ============================================================================


def wrap_registered_tool(registered_tool: RegisteredTool, runner: ExtensionRunner) -> Any:
    """Wrap a `RegisteredTool` into an `AgentTool` for the agent loop.

    Port of `wrapRegisteredTool()`. The agent loop (`pi_agent.agent_loop`)
    calls `tool.execute(tool_call_id, params, signal, on_update)` with four
    positional arguments; the returned `AgentTool.execute` closes over
    `runner.create_context()` to supply the fifth `ExtensionContext` argument
    the extension's `ToolDefinition.execute` expects, exactly as TypeScript's
    wrapper defaults `ctx` from `ctxFactory()` when the caller does not pass
    one explicitly.

    The context-defaulting itself is `tools.tool_definition_wrapper`'s
    `wrap_tool_definition`; this function only adds the tool-activation
    diffing from ``core/extensions/wrapper.ts`` on top.

    `added_tool_names` diffing (TypeScript: comparing `getActiveTools()`
    before/after execution so a tool that activates other tools via
    `pi.setActiveTools()` reports them) is preserved when the runner's
    `ExtensionContextActions.get_active_tool_names` is bound to a real
    accessor; it degrades to a no-op when left at its default (`lambda: []`).
    """
    base = wrap_tool_definition(registered_tool.definition, runner.create_context)

    async def execute(tool_call_id: str, params: Any, signal: AbortSignal | None, on_update: Any = None) -> Any:
        active_before = runner.get_active_tool_names()
        result = await base.execute(tool_call_id, params, signal, on_update)
        active_after = runner.get_active_tool_names()
        if all(name in active_after for name in active_before):
            before_set = set(active_before)
            added = [name for name in active_after if name not in before_set]
            if added:
                existing = list(result.added_tool_names or [])
                result.added_tool_names = list(dict.fromkeys(existing + added))
        return result

    return replace(base, execute=execute)


def wrap_registered_tools(registered_tools: list[RegisteredTool], runner: ExtensionRunner) -> list[Any]:
    return [wrap_registered_tool(tool, runner) for tool in registered_tools]
