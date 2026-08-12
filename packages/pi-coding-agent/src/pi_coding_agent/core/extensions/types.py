"""Extension system types.

Port of `packages/coding-agent/src/core/extensions/types.ts` (1727 lines): the
extension contract -- lifecycle-event dataclasses, tool/command registration
shapes, and the `ExtensionContext`/`ExtensionCommandContext` objects handed to
event handlers, tools, and commands.

**Scope narrowing versus the TypeScript original** (see also
`extensions/loader.py`'s module docstring for the JS-module substitution):
this port has no ported TUI (`pi_tui`), no interactive-mode themes
(`modes/interactive/theme/theme.ts`), no `KeybindingsManager`, and no
`ModelRegistry` (see `model_runtime.py`'s own documented boundary). Every
extension-API surface that only exists to plug into those subsystems is
dropped, not stubbed:

- **`ExtensionUIContext`** keeps only the headless-safe subset: `select`,
  `confirm`, `input`, `notify`, `set_status`, `set_title`,
  `get_tools_expanded`/`set_tools_expanded`. Dropped: `onTerminalInput`,
  `setWorkingMessage`/`setWorkingVisible`/`setWorkingIndicator`,
  `setHiddenThinkingLabel`, `setWidget`/`setFooter`/`setHeader` (Component
  factories), `pasteToEditor`/`setEditorText`/`getEditorText`/`editor`,
  `addAutocompleteProvider`, `setEditorComponent`/`getEditorComponent`, `theme`
  and the theme accessors -- all `pi_tui`/theme-shaped.
- **No shortcut/message-renderer/markdown-transformer/entry-renderer
  registration** (`registerShortcut`, `registerMessageRenderer`,
  `registerMarkdownTransformer`, `registerEntryRenderer`): these render into
  the interactive TUI transcript, which this port does not have.
- **No provider registration** (`registerProvider`/`registerNativeProvider`/
  `unregisterProvider`): there is no `ModelRegistry` to register into.
- **No CLI flags** (`registerFlag`/`getFlag`): flag parsing belongs to
  `cli/args.py`, not wired to extensions in this port.
- **`ModelSelectEvent`/`ThinkingLevelSelectEvent`** are defined (part of the
  event contract) but `agent_session.py` does not emit them -- see that
  module's own documented boundary.

Everything else -- the full session/agent/tool/input event lifecycle, tool
definitions, and command registration -- is ported faithfully.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pi_agent.types import AgentToolResult, AgentToolUpdateCallback, ThinkingLevel, ToolExecutionMode
from pi_ai.types import AssistantMessageEvent, ImageContent, Model, TextContent, Usage
from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.core.compaction import CompactionResult
from pi_coding_agent.core.system_prompt import BuildSystemPromptOptions

__all__ = [
    "AfterProviderResponseEvent",
    "AgentEndEvent",
    "AgentSettledEvent",
    "AgentStartEvent",
    "BeforeAgentStartEvent",
    "BeforeAgentStartEventResult",
    "BeforeProviderHeadersEvent",
    "BeforeProviderRequestEvent",
    "CompactOptions",
    "ContextEvent",
    "ContextEventResult",
    "ContextUsage",
    "Extension",
    "ExtensionCommandContext",
    "ExtensionContext",
    "ExtensionError",
    "ExtensionEvent",
    "ExtensionFactory",
    "ExtensionMode",
    "ExtensionUIContext",
    "InputEvent",
    "InputEventResult",
    "InputSource",
    "MessageEndEvent",
    "MessageEndEventResult",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "ModelSelectEvent",
    "ModelSelectSource",
    "NullExtensionUIContext",
    "ProjectTrustContext",
    "ProjectTrustEvent",
    "ProjectTrustEventDecision",
    "ProjectTrustEventResult",
    "ProjectTrustHandler",
    "RegisteredCommand",
    "RegisteredTool",
    "ReplacedSessionContext",
    "ResourcesDiscoverEvent",
    "ResourcesDiscoverResult",
    "SessionBeforeCompactEvent",
    "SessionBeforeCompactResult",
    "SessionBeforeForkEvent",
    "SessionBeforeForkResult",
    "SessionBeforeSwitchEvent",
    "SessionBeforeSwitchResult",
    "SessionBeforeTreeEvent",
    "SessionBeforeTreeResult",
    "SessionCompactEvent",
    "SessionEvent",
    "SessionInfoChangedEvent",
    "SessionShutdownEvent",
    "SessionStartEvent",
    "SessionTreeEvent",
    "ThinkingLevelSelectEvent",
    "ToolCallEvent",
    "ToolCallEventResult",
    "ToolDefinition",
    "ToolExecutionEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolResultEvent",
    "ToolResultEventResult",
    "TreePreparation",
    "TurnEndEvent",
    "TurnStartEvent",
    "UserBashEvent",
    "UserBashEventResult",
    "define_tool",
]


# ============================================================================
# UI Context
# ============================================================================


class ExtensionUIContext(Protocol):
    """Headless-safe subset of extension UI. See module docstring for scope."""

    async def select(self, title: str, options: list[str]) -> str | None: ...

    async def confirm(self, title: str, message: str) -> bool: ...

    async def input(self, title: str, placeholder: str | None = None) -> str | None: ...

    def notify(self, message: str, type: Literal["info", "warning", "error"] = "info") -> None: ...

    def set_status(self, key: str, text: str | None) -> None: ...

    def set_title(self, title: str) -> None: ...

    def get_tools_expanded(self) -> bool: ...

    def set_tools_expanded(self, expanded: bool) -> None: ...


class NullExtensionUIContext:
    """No-op `ExtensionUIContext`, used when a mode supplies no real UI.

    Port of `noOpUIContext` in `runner.ts`, narrowed to the methods this port
    kept (see the module docstring).
    """

    async def select(self, title: str, options: list[str]) -> str | None:
        return None

    async def confirm(self, title: str, message: str) -> bool:
        return False

    async def input(self, title: str, placeholder: str | None = None) -> str | None:
        return None

    def notify(self, message: str, type: Literal["info", "warning", "error"] = "info") -> None:
        pass

    def set_status(self, key: str, text: str | None) -> None:
        pass

    def set_title(self, title: str) -> None:
        pass

    def get_tools_expanded(self) -> bool:
        return False

    def set_tools_expanded(self, expanded: bool) -> None:
        pass


# ============================================================================
# Extension Context
# ============================================================================


@dataclass
class ContextUsage:
    tokens: int | None
    context_window: int
    percent: float | None


@dataclass
class CompactOptions:
    custom_instructions: str | None = None
    on_complete: Callable[[CompactionResult], None] | None = None
    on_error: Callable[[Exception], None] | None = None


ExtensionMode = Literal["tui", "rpc", "json", "print"]


@dataclass
class ExtensionContext:
    """Context passed to extension event handlers and tools.

    Fields are plain values captured at emit time (not lazy property getters
    like TypeScript's `runner.createContext()`), since Python has no
    equivalent to per-property getter descriptors that stay live after
    `Object.defineProperties` copies; `ExtensionRunner.create_context()`
    rebuilds a fresh `ExtensionContext` on every emit instead, which yields
    the same "always current" behavior for callers that don't cache `ctx`
    across an `await`.
    """

    ui: ExtensionUIContext
    mode: ExtensionMode
    has_ui: bool
    cwd: str
    session_manager: Any
    model: Model | None
    scoped_models: tuple[Any, ...]
    is_idle: Callable[[], bool]
    is_project_trusted: Callable[[], bool]
    signal: AbortSignal | None
    abort: Callable[[], None]
    has_pending_messages: Callable[[], bool]
    shutdown: Callable[[], None]
    get_context_usage: Callable[[], ContextUsage | None]
    compact: Callable[..., None]
    get_system_prompt: Callable[[], str]
    thinking_level: ThinkingLevel | None = None


@dataclass
class ExtensionCommandContext(ExtensionContext):
    """Extended context for command handlers.

    `new_session`/`fork`/`switch_session`/`reload` are dropped: they require
    the multi-session-file/reload machinery `agent_session.py`'s own module
    docstring documents as out of scope (no `ResourceLoader.reload()`
    consumer, no session-replacement flow). `wait_for_idle` is kept since it
    only needs an `asyncio.Event`.
    """

    get_system_prompt_options: Callable[[], BuildSystemPromptOptions] = field(
        default=lambda: BuildSystemPromptOptions()
    )
    wait_for_idle: Callable[[], Awaitable[None]] = field(default=None)  # type: ignore[assignment]


@dataclass
class ReplacedSessionContext(ExtensionCommandContext):
    """Command context bound to a post-replacement session.

    Kept for type-contract parity with TypeScript; nothing in this port
    constructs one (no `newSession`/`fork`/`switchSession`, see above).
    """

    send_message: Callable[..., Awaitable[None]] = field(default=None)  # type: ignore[assignment]
    send_user_message: Callable[..., Awaitable[None]] = field(default=None)  # type: ignore[assignment]


# ============================================================================
# Tool Types
# ============================================================================


@dataclass
class ToolDefinition:
    """Tool definition for `pi.register_tool()`.

    `parameters` is a plain JSON Schema `dict` (matching `pi_ai.types.Tool`),
    not TypeBox's `TSchema`/`Static` -- this port has no TypeBox dependency
    (see README: "Prefer the standard library"; `jsonschema` is already the
    validation dependency used elsewhere). `render_call`/`render_result`
    (TUI `Component` renderers) are dropped -- no `pi_tui` consumer.
    """

    name: str
    label: str
    description: str
    execute: Callable[
        [str, Any, AbortSignal | None, AgentToolUpdateCallback | None, ExtensionContext], Awaitable[AgentToolResult]
    ]
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    prompt_snippet: str | None = None
    prompt_guidelines: list[str] = field(default_factory=list)
    constrained_sampling: Any = None
    prepare_arguments: Callable[[Any], Any] | None = None
    execution_mode: ToolExecutionMode | None = None


def define_tool(tool: ToolDefinition) -> ToolDefinition:
    """Identity helper mirroring TypeScript's `defineTool()`.

    TypeScript's version exists purely to preserve generic parameter
    inference when a tool is assigned to a variable; Python has no
    equivalent inference-widening problem, so this is a plain passthrough
    kept only so ported extension code reads the same.
    """
    return tool


# ============================================================================
# Startup/Resource Events
# ============================================================================


@dataclass
class ProjectTrustEvent:
    cwd: str
    type: Literal["project_trust"] = "project_trust"


ProjectTrustEventDecision = Literal["yes", "no", "undecided"]


@dataclass
class ProjectTrustEventResult:
    trusted: ProjectTrustEventDecision
    remember: bool | None = None


@dataclass
class ProjectTrustContext:
    cwd: str
    mode: ExtensionMode
    has_ui: bool
    ui: ExtensionUIContext


ProjectTrustHandler = Callable[
    [ProjectTrustEvent, ProjectTrustContext], Awaitable[ProjectTrustEventResult] | ProjectTrustEventResult
]


@dataclass
class ResourcesDiscoverEvent:
    cwd: str
    reason: Literal["startup", "reload"]
    type: Literal["resources_discover"] = "resources_discover"


@dataclass
class ResourcesDiscoverResult:
    skill_paths: list[str] = field(default_factory=list)
    prompt_paths: list[str] = field(default_factory=list)
    theme_paths: list[str] = field(default_factory=list)


# ============================================================================
# Session Events
# ============================================================================


@dataclass
class SessionStartEvent:
    reason: Literal["startup", "reload", "new", "resume", "fork"]
    previous_session_file: str | None = None
    type: Literal["session_start"] = "session_start"


@dataclass
class SessionInfoChangedEvent:
    name: str | None
    type: Literal["session_info_changed"] = "session_info_changed"


@dataclass
class SessionBeforeSwitchEvent:
    reason: Literal["new", "resume"]
    target_session_file: str | None = None
    type: Literal["session_before_switch"] = "session_before_switch"


@dataclass
class SessionBeforeForkEvent:
    entry_id: str
    position: Literal["before", "at"]
    type: Literal["session_before_fork"] = "session_before_fork"


@dataclass
class SessionBeforeCompactEvent:
    preparation: Any
    branch_entries: list[Any]
    reason: Literal["manual", "threshold", "overflow"]
    will_retry: bool
    signal: AbortSignal
    custom_instructions: str | None = None
    type: Literal["session_before_compact"] = "session_before_compact"


@dataclass
class SessionCompactEvent:
    compaction_entry: Any
    from_extension: bool
    reason: Literal["manual", "threshold", "overflow"]
    will_retry: bool
    type: Literal["session_compact"] = "session_compact"


@dataclass
class SessionShutdownEvent:
    reason: Literal["quit", "reload", "new", "resume", "fork"]
    target_session_file: str | None = None
    type: Literal["session_shutdown"] = "session_shutdown"


@dataclass
class TreePreparation:
    target_id: str
    old_leaf_id: str | None
    common_ancestor_id: str | None
    entries_to_summarize: list[Any]
    user_wants_summary: bool
    custom_instructions: str | None = None
    replace_instructions: bool | None = None
    label: str | None = None


@dataclass
class SessionBeforeTreeEvent:
    preparation: TreePreparation
    signal: AbortSignal
    type: Literal["session_before_tree"] = "session_before_tree"


@dataclass
class SessionTreeEvent:
    new_leaf_id: str | None
    old_leaf_id: str | None
    summary_entry: Any = None
    from_extension: bool | None = None
    type: Literal["session_tree"] = "session_tree"


SessionEvent = (
    SessionStartEvent
    | SessionInfoChangedEvent
    | SessionBeforeSwitchEvent
    | SessionBeforeForkEvent
    | SessionBeforeCompactEvent
    | SessionCompactEvent
    | SessionShutdownEvent
    | SessionBeforeTreeEvent
    | SessionTreeEvent
)


# ============================================================================
# Agent Events
# ============================================================================


@dataclass
class ContextEvent:
    messages: list[Any]
    type: Literal["context"] = "context"


@dataclass
class ContextEventResult:
    messages: list[Any] | None = None


@dataclass
class BeforeProviderRequestEvent:
    payload: Any
    type: Literal["before_provider_request"] = "before_provider_request"


@dataclass
class BeforeProviderHeadersEvent:
    headers: dict[str, str | None]
    type: Literal["before_provider_headers"] = "before_provider_headers"


@dataclass
class AfterProviderResponseEvent:
    status: int
    headers: dict[str, str]
    type: Literal["after_provider_response"] = "after_provider_response"


@dataclass
class BeforeAgentStartEvent:
    prompt: str
    system_prompt: str
    system_prompt_options: BuildSystemPromptOptions
    images: list[ImageContent] | None = None
    type: Literal["before_agent_start"] = "before_agent_start"


@dataclass
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"


@dataclass
class AgentEndEvent:
    messages: list[Any] = field(default_factory=list)
    type: Literal["agent_end"] = "agent_end"


@dataclass
class AgentSettledEvent:
    type: Literal["agent_settled"] = "agent_settled"


@dataclass
class TurnStartEvent:
    turn_index: int
    timestamp: float
    type: Literal["turn_start"] = "turn_start"


@dataclass
class TurnEndEvent:
    turn_index: int
    message: Any
    tool_results: list[Any] = field(default_factory=list)
    type: Literal["turn_end"] = "turn_end"


@dataclass
class MessageStartEvent:
    message: Any
    type: Literal["message_start"] = "message_start"


@dataclass
class MessageUpdateEvent:
    message: Any
    assistant_message_event: AssistantMessageEvent
    type: Literal["message_update"] = "message_update"


@dataclass
class MessageEndEvent:
    message: Any
    type: Literal["message_end"] = "message_end"


@dataclass
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    type: Literal["tool_execution_start"] = "tool_execution_start"


@dataclass
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any
    type: Literal["tool_execution_update"] = "tool_execution_update"


@dataclass
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool
    type: Literal["tool_execution_end"] = "tool_execution_end"


# ============================================================================
# Model Events
# ============================================================================

ModelSelectSource = Literal["set", "cycle", "restore"]


@dataclass
class ModelSelectEvent:
    model: Model
    previous_model: Model | None
    source: ModelSelectSource
    type: Literal["model_select"] = "model_select"


@dataclass
class ThinkingLevelSelectEvent:
    level: ThinkingLevel
    previous_level: ThinkingLevel
    type: Literal["thinking_level_select"] = "thinking_level_select"


# ============================================================================
# User Bash Events
# ============================================================================


@dataclass
class UserBashEvent:
    command: str
    exclude_from_context: bool
    cwd: str
    type: Literal["user_bash"] = "user_bash"


@dataclass
class UserBashEventResult:
    operations: Any = None
    result: Any = None


# ============================================================================
# Input Events
# ============================================================================

InputSource = Literal["interactive", "rpc", "extension"]


@dataclass
class InputEvent:
    text: str
    source: InputSource
    images: list[ImageContent] | None = None
    streaming_behavior: Literal["steer", "followUp"] | None = None
    type: Literal["input"] = "input"


@dataclass
class InputEventResult:
    action: Literal["continue", "transform", "handled"]
    text: str | None = None
    images: list[ImageContent] | None = None


# ============================================================================
# Tool Events
# ============================================================================


@dataclass
class ToolCallEvent:
    """Fired before a tool executes. `input` is mutated in place to patch args."""

    tool_call_id: str
    tool_name: str
    input: dict[str, Any]
    type: Literal["tool_call"] = "tool_call"


@dataclass
class ToolCallEventResult:
    block: bool | None = None
    reason: str | None = None
    terminate: bool | None = None


@dataclass
class ToolResultEvent:
    tool_call_id: str
    tool_name: str
    input: dict[str, Any]
    content: list[TextContent | ImageContent]
    is_error: bool
    details: Any = None
    usage: Usage | None = None
    type: Literal["tool_result"] = "tool_result"


@dataclass
class ToolResultEventResult:
    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None
    usage: Usage | None = None


ExtensionEvent = (
    SessionEvent
    | ProjectTrustEvent
    | ResourcesDiscoverEvent
    | ContextEvent
    | BeforeProviderRequestEvent
    | BeforeProviderHeadersEvent
    | AfterProviderResponseEvent
    | BeforeAgentStartEvent
    | AgentStartEvent
    | AgentEndEvent
    | AgentSettledEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | ModelSelectEvent
    | ThinkingLevelSelectEvent
    | UserBashEvent
    | InputEvent
    | ToolCallEvent
    | ToolResultEvent
)


@dataclass
class MessageEndEventResult:
    message: Any = None


@dataclass
class BeforeAgentStartEventResult:
    message: Any = None
    system_prompt: str | None = None


@dataclass
class SessionBeforeSwitchResult:
    cancel: bool | None = None


@dataclass
class SessionBeforeForkResult:
    cancel: bool | None = None
    skip_conversation_restore: bool | None = None


@dataclass
class SessionBeforeCompactResult:
    cancel: bool | None = None
    compaction: CompactionResult | None = None


@dataclass
class SessionBeforeTreeResult:
    cancel: bool | None = None
    summary: dict[str, Any] | None = None
    custom_instructions: str | None = None
    replace_instructions: bool | None = None
    label: str | None = None


# ============================================================================
# Markdown Transformers
# ============================================================================


MarkdownMessageType = Literal["user", "assistant", "assistant-thinking"]


@dataclass
class MarkdownTransformContext:
    message_type: MarkdownMessageType
    is_streaming: bool
    available_width: int


MarkdownTransformer = Callable[[str, MarkdownTransformContext], str]


# ============================================================================
# Command Registration
# ============================================================================


@dataclass
class RegisteredCommand:
    name: str
    handler: Callable[[str, ExtensionCommandContext], Awaitable[None]]
    source_info: Any = None
    description: str | None = None
    get_argument_completions: Callable[[str], Any] | None = None


# ============================================================================
# Extension API / Loaded Extension
# ============================================================================

ExtensionHandler = Callable[..., Any]
"""`(event, ctx) -> result | Awaitable[result]`. Kept loosely typed (matching
TypeScript's overloaded `on()` signature collapsed to one type) since Python
has no per-event-type overload resolution for a single `on()` method."""


@dataclass
class RegisteredTool:
    definition: ToolDefinition
    source_info: Any = None


@dataclass
class Extension:
    """A loaded extension with all registered items.

    `handlers` maps event type name -> ordered list of handler callables
    (registration order is preserved, matching TS's `Map<string, Handler[]>`
    plus array `push`).
    """

    path: str
    resolved_path: str
    handlers: dict[str, list[ExtensionHandler]] = field(default_factory=dict)
    tools: dict[str, RegisteredTool] = field(default_factory=dict)
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    source_info: Any = None
    hidden: bool = False
    event_bus_unsubscribers: list[Callable[[], None]] = field(default_factory=list)
    """Unsubscribes for every `pi.events.on()` this extension made.

    TypeScript keeps these on the shared `ExtensionRuntime` and drops them in
    `runtime.invalidate()`; this port has no per-load runtime object, so the
    subscriptions live on the extension that made them and are dropped by
    `ExtensionRunner.unsubscribe_events()` when the session is disposed.
    """
    on_tools_changed: Callable[[], None] | None = None
    """Called by `pi.register_tool()` once the extension is bound to a session.

    Port of TypeScript's shared `runtime.refreshTools`, which
    `ExtensionRunner.bindCore()` patches with the session's
    `_refreshToolRegistry`. This port has no per-load runtime object (see
    `event_bus_unsubscribers` above), so the hook lives on the extension and is
    installed by `ExtensionRunner.bind_core()`. `None` means "not bound yet":
    registrations made while the extension module is loading are picked up by
    the session's initial registry build, so no refresh is needed.
    """
    markdown_transformer: MarkdownTransformer | None = None
    """Set by `pi.register_markdown_transformer()`; at most one per extension.

    Matches TypeScript's `extension.markdownTransformer`: a second call
    replaces the first rather than appending.
    """


@dataclass
class ExtensionError:
    extension_path: str
    event: str
    error: str
    stack: str | None = None


ExtensionFactory = Callable[[Any], Awaitable[None] | None]
"""`(pi: ExtensionAPI) -> None | Awaitable[None]`. See `extensions/loader.py`
for how a Python extension file provides one."""
