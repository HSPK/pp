"""`AgentSession` -- core abstraction for agent lifecycle and session management.

Port of `packages/coding-agent/src/core/agent-session.ts` (3342 lines), the
largest and most central file in the coding-agent package. It wires an
`Agent` (the low-level agent loop, already ported in `pi_agent.agent`) to a
`SessionManager` (transcript persistence, already ported), a
`SettingsManager` (already ported), a `ResourceLoader` (skills/prompts/system
prompt, already ported), a `ModelRuntime` (auth + model catalog, already
ported), and the builtin tool set (`pi_coding_agent.tools`).

This class is shared between all run modes (interactive, print, rpc) in the
TypeScript original. Only the mode-independent session layer is ported here;
the modes themselves (and their I/O layers) are out of scope.

**Documented boundaries** (subsystems intentionally not pulled in, matching
`README.md`'s "port only the narrow interface required" rule):

- **Extension system: implemented, with a bounded wiring scope.** The
  extension contract, disk loader, and hook-dispatch runner are fully ported
  (see `extensions/types.py`, `extensions/loader.py`, `extensions/runner.py`).
  `AgentSession.__init__` accepts an `extensions: list[Extension] | None`
  parameter, always constructs an `ExtensionRunner` (empty list when none
  given -- see below for why this matters), binds its live accessors via
  `_bind_extension_core()`, and merges extension-registered tools into
  `self._custom_tools` (explicit caller tools win on name collision) before
  the tool registry is built. The following hook points are wired, matching
  TypeScript's call sites in `agent-session.ts`:
    - `tool_call`/`tool_result` -- `_install_agent_tool_hooks()`'s
      `before_tool_call`/`after_tool_call` callbacks dispatch
      `emit_tool_call`/`emit_tool_result` (see that method's docstring for the
      one behavioral nuance: `emit_tool_call` is not error-isolated in the
      runner, matching TypeScript, but `agent_loop.py`'s pre-existing
      try/except around `before_tool_call` turns a raised exception into an
      error tool result for that call rather than propagating out of
      `prompt()`, unlike TypeScript's session-level throw -- an artifact of
      `agent_loop.py`'s existing structure, out of scope to change here).
    - `agent_start`/`agent_end`/`agent_settled`/`turn_start`/`turn_end`/
      `message_start`/`message_update`/`message_end`/`tool_execution_start`/
      `tool_execution_update`/`tool_execution_end` -- re-emitted verbatim by
      `_emit_extension_event()`, called from `_handle_agent_event()` before
      any session-level persistence/state-tracking runs on the same event, so
      a `message_end` handler's replacement message (applied in place via
      `_replace_message_in_place`) is what actually gets persisted.
    - `input` -- emitted in `prompt()` before skill/template expansion; a
      `transform` result changes the text/images used for the rest of the
      call, a `handled` result short-circuits (nothing is sent to the LLM).
    - `before_agent_start` -- emitted in `prompt()` right before dispatching
      to the agent; extension-supplied custom messages
      (`_extension_message_to_custom_message`) are appended and a returned
      system-prompt override is applied for that turn only (reset next
      `prompt()` call, matching TypeScript).
    - Extension commands (`/name args...`) -- `prompt()` dispatches them via
      `_try_execute_extension_command()` (port of
      `_tryExecuteExtensionCommand`) before anything else, even while
      streaming; `steer()`/`follow_up()` raise via
      `_throw_if_extension_command()` (port of `_throwIfExtensionCommand`) if
      given a registered command name, since queued messages cannot execute
      commands immediately.
    - `session_before_compact`/`session_compact` -- wired in `compact()`
      (manual reason) and in `_run_auto_compaction()` (the `threshold` and
      `overflow` reasons), matching TypeScript, which emits them from both.
      An extension may cancel the compaction or supply its own
      `CompactionResult`, skipping `compaction_compact()`/LLM summarization
      entirely; `session_compact` fires once the compaction entry is
      persisted, referencing the saved `CompactionEntry`.

  A no-extensions session is unaffected: `ExtensionRunner([])` has no
  handlers, so every `has_handlers(...)` check the above call sites make is
  `False`, `emit_*` return `None`/no-ops, and the tool registry, prompt flow,
  and compaction flow all run exactly as they did before this wiring existed
  (see `test_agent_session.py`, unmodified and still green, plus
  `test_extensions_runner.py`'s hook-ordering/isolation cases).

  **Deliberately still out of scope** (no wiring, no call site added), each
  because it has no reachable consumer or no equivalent subsystem in this
  port:
    - `session_before_tree`/`session_tree` -- **now wired** in
      `navigate_tree()`: an extension may cancel the navigation, supply its
      own branch summary, or override the instructions/label, and
      `session_tree` reports the resulting leaf.
    - `model_select`/`thinking_level_select` -- **now wired**: `set_model()`,
      `_cycle_scoped_model()`, `_cycle_available_model()` and
      `set_thinking_level()` emit them, matching TypeScript.
    - `resources_discover` -- **now wired**: `bind_extensions()` collects the
      skill and prompt paths handlers return and hands them to
      `ResourceLoader.extend_resources()`, then rebuilds the system prompt so
      a contributed skill is visible to the model. The `theme_paths` half of
      the result is ignored: this port has no theme loading.
    - `entry_appended` -- extension-only event, no consumer without the
      `appendEntry` core binding TypeScript exposes to extensions (this port
      does not expose an equivalent write-access binding).
    - `session_start`/`session_shutdown` -- **now wired**: `bind_extensions()`
      emits `session_start` (with the `reason`/`previous_session_file` the
      constructing caller passed as `session_start_event`), and
      `AgentSessionRuntime` emits `session_shutdown` while tearing a session
      down. `reload()` still has no counterpart here (no resource-reload
      machinery), so the `"reload"` reason is never produced.

- **No themes/export-html.** `export_to_html()` raises `NotImplementedError`
  (see its docstring); `export_to_jsonl()` and `get_last_assistant_text()`
  have no such dependency and are ported faithfully.
- **No OAuth / remote model catalog / locked `ModelsStore`.** Inherited from
  `model_runtime.py`'s documented boundary; `is_using_oauth` is always
  `False`, so the `isOAuth` branches in `_ensure_model_auth`/`set_model`
  degrade to the plain "no API key" message.
- **Simplified summarization auth-threading.** TypeScript's
  `_getRequiredRequestAuth`/`_getSummarizationRequestAuth` manually resolve
  and thread `apiKey`/`headers`/`env`/a `baseUrl`-overridden `model` through
  every compaction/branch-summary call (this also compensates for
  `ResolvedAuth.baseUrl`, a field the Python `pi_ai.auth.types.ResolvedAuth`
  does not have). This port's `compaction.py` already dropped those
  parameters -- its `compact()`/`generate_branch_summary()` take a `stream_fn`
  that carries whatever auth resolution the caller configured -- so
  `compact()`/`_run_auto_compaction()`/`navigate_tree()` call them directly
  with `self.agent.stream_function`, `self.model`, and no separate auth
  threading. The one auth check this port keeps (because it is behaviorally
  tested and raises user-visible errors) is `_ensure_model_auth()`, mirroring
  the inline check `prompt()` used to do before dispatching a turn.
- **Image resizing needs Pillow.** `after_tool_call` calls
  `normalize_tool_result_images`, which resizes through `image_resize.py`
  (Pillow standing in for the TS Photon/WASM codec). Without Pillow installed
  the resize returns `None` and the original image block passes through, the
  same fallback the TypeScript takes when `loadPhoton()` returns `null`.
- **No `cleanupSessionResources`.** TypeScript's `dispose()` ends with
  `cleanupSessionResources(this.sessionId)`, an HTTP-transport-connection
  cleanup hook owned by `pi_ai`'s transport layer. No Python equivalent
  exists anywhere in `pi_ai`/`pi_agent`/`pi_coding_agent` (grepped for it);
  `dispose()` omits the call as a documented no-op.
- **`AgentSessionEvent` union scope.** `entry_appended` (used only by the
  extension `appendEntry` core binding) and `model_select`/
  `thinking_level_select` (both extension-only events, distinct from the
  `thinking_level_changed` session event this port keeps) are dropped
  entirely -- there is no consumer for them without the extension system.
- **`AgentSession.model` returns `None` for the placeholder model.**
  `pi_agent.agent.MutableAgentState.model` is never `None` (it defaults to
  `pi_agent.agent.DEFAULT_MODEL`, a sentinel with `provider == id ==
  "unknown"`), unlike TypeScript's genuinely optional `AgentState.model`.
  `AgentSession.model` treats that sentinel as "no model selected" so
  `format_no_model_selected_message()` still fires the same way TypeScript's
  `!this.model` check did. Construct the initial `Agent` with a real `Model`
  for a session that starts with one already selected.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pi_agent.agent import DEFAULT_MODEL, Agent
from pi_agent.harness.messages import BashExecutionMessage, CustomMessage
from pi_agent.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentEvent,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    MessageEndEvent,
    MessageStartEvent,
    PrepareNextTurnContext,
    ThinkingLevel,
)
from pi_ai.models import clamp_thinking_level, get_supported_thinking_levels, models_are_equal
from pi_ai.types import (
    AssistantMessage,
    ImageContent,
    Model,
    TextContent,
    Usage,
    UserMessage,
    now_ms,
)
from pi_ai.utils.abort import AbortController, AbortError, AbortSignal
from pi_ai.utils.overflow import is_context_overflow, is_recoverable_length
from pi_ai.utils.retry import RetryCallbacks, RetryPolicy, is_retryable_assistant_error
from pi_ai.utils.text import content_text

from pi_coding_agent.core.auth_guidance import (
    format_no_api_key_found_message,
    format_no_model_selected_message,
)
from pi_coding_agent.core.bash_executor import (
    BashOperations,
    BashResult,
    create_local_bash_operations,
    execute_bash_with_operations,
)
from pi_coding_agent.core.compaction import (
    CompactionResult,
    CompactionSettings,
    GenerateBranchSummaryOptions,
    calculate_context_tokens,
    collect_entries_for_branch_summary,
    estimate_context_tokens,
    estimate_tokens,
    generate_branch_summary,
    prepare_compaction,
    should_compact,
)
from pi_coding_agent.core.compaction import (
    compact as compaction_compact,
)
from pi_coding_agent.core.extensions.runner import (
    DiscoveredResourcePath,
    ExtensionContextActions,
    ExtensionRunner,
    wrap_registered_tools,
)
from pi_coding_agent.core.extensions.types import (
    AgentEndEvent as ExtAgentEndEvent,
)
from pi_coding_agent.core.extensions.types import (
    AgentSettledEvent as ExtAgentSettledEvent,
)
from pi_coding_agent.core.extensions.types import (
    AgentStartEvent as ExtAgentStartEvent,
)
from pi_coding_agent.core.extensions.types import (
    Extension,
    ExtensionError,
    InputSource,
    ModelSelectEvent,
    ModelSelectSource,
    SessionBeforeCompactEvent,
    SessionBeforeCompactResult,
    SessionBeforeTreeEvent,
    SessionBeforeTreeResult,
    SessionCompactEvent,
    SessionStartEvent,
    SessionTreeEvent,
    ThinkingLevelSelectEvent,
    ToolCallEvent,
    ToolResultEvent,
    TreePreparation,
)
from pi_coding_agent.core.extensions.types import (
    MessageEndEvent as ExtMessageEndEvent,
)
from pi_coding_agent.core.extensions.types import (
    MessageStartEvent as ExtMessageStartEvent,
)
from pi_coding_agent.core.extensions.types import (
    MessageUpdateEvent as ExtMessageUpdateEvent,
)
from pi_coding_agent.core.extensions.types import (
    SessionInfoChangedEvent as ExtSessionInfoChangedEvent,
)
from pi_coding_agent.core.extensions.types import (
    ToolExecutionEndEvent as ExtToolExecutionEndEvent,
)
from pi_coding_agent.core.extensions.types import (
    ToolExecutionStartEvent as ExtToolExecutionStartEvent,
)
from pi_coding_agent.core.extensions.types import (
    ToolExecutionUpdateEvent as ExtToolExecutionUpdateEvent,
)
from pi_coding_agent.core.extensions.types import (
    TurnEndEvent as ExtTurnEndEvent,
)
from pi_coding_agent.core.extensions.types import (
    TurnStartEvent as ExtTurnStartEvent,
)
from pi_coding_agent.core.model_resolver import DEFAULT_THINKING_LEVEL, ScopedModel
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import (
    ExtensionResourcePath,
    PromptTemplate,
    ResourceLoader,
    ResourcePathMetadata,
    expand_prompt_template,
    strip_frontmatter,
)
from pi_coding_agent.core.session_manager import (  # narrow, same-package reuse
    CURRENT_SESSION_VERSION,
    BranchSummaryEntry,
    CompactionEntry,
    CustomMessageEntry,
    SessionEntry,
    SessionManager,
    SessionMessageEntry,
    _entry_to_raw,
    _now_iso,
    get_latest_compaction_entry,
)
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.core.source_info import create_synthetic_source_info
from pi_coding_agent.core.system_prompt import BuildSystemPromptOptions, ContextFile, build_system_prompt
from pi_coding_agent.core.tool_result_images import (
    NormalizeToolResultImagesOptions,
    normalize_tool_result_images,
)
from pi_coding_agent.core.usage_totals import add_usage_to_totals, create_usage_totals
from pi_coding_agent.tools import ALL_TOOL_NAMES, create_tool
from pi_coding_agent.utils.paths import resolve_path

# ============================================================================
# Skill Block Parsing
# ============================================================================

# `\Z`, not `$`: JavaScript's non-multiline `$` anchors at the very end of the
# string, while Python's `$` also matches just before a trailing newline. With
# `$` a message ending in "</skill>\n" parsed as a skill block here but not
# upstream, and "...\n\nhello\n" lost its trailing newline before `.strip()`.
_SKILL_BLOCK_RE = re.compile(r'^<skill name="([^"]+)" location="([^"]+)">\n(.*?)\n</skill>(?:\n\n(.+))?\Z', re.DOTALL)


def _is_abort_error(error: BaseException) -> bool:
    """Whether an exception means "the user cancelled", not "compaction failed".

    Summarization reports cancellation as ``CompactionError(code="aborted")``
    rather than raising an ``AbortError``, so checking the exception type alone
    surfaces a spurious failure to the user when they press Esc.
    """
    if isinstance(error, AbortError):
        return True
    if getattr(error, "code", None) == "aborted":
        return True
    return str(error) == "Compaction cancelled"


@dataclass
class ParsedSkillBlock:
    """Parsed skill block from a user message. Port of TS `ParsedSkillBlock`."""

    name: str
    location: str
    content: str
    user_message: str | None = None


def parse_skill_block(text: str) -> ParsedSkillBlock | None:
    """Parse a skill block from message text. Returns `None` if absent."""
    match = _SKILL_BLOCK_RE.match(text)
    if not match:
        return None
    trailing = match.group(4)
    return ParsedSkillBlock(
        name=match.group(1),
        location=match.group(2),
        content=match.group(3),
        user_message=trailing.strip() if trailing and trailing.strip() else None,
    )


# ============================================================================
# Event Types
# ============================================================================

CompactionReason = Literal["manual", "threshold", "overflow"]
SummarizationSource = Literal["branchSummary", "compaction"]


@dataclass
class AgentEndEvent:
    """Session-level `agent_end` event: adds `will_retry` to the core `AgentEvent` variant."""

    messages: list[AgentMessage] = field(default_factory=list)
    will_retry: bool = False
    type: Literal["agent_end"] = "agent_end"


@dataclass
class AgentSettledEvent:
    type: Literal["agent_settled"] = "agent_settled"


@dataclass
class QueueUpdateEvent:
    steering: list[str] = field(default_factory=list)
    follow_up: list[str] = field(default_factory=list)
    type: Literal["queue_update"] = "queue_update"


@dataclass
class CompactionStartEvent:
    reason: CompactionReason
    type: Literal["compaction_start"] = "compaction_start"


@dataclass
class SessionInfoChangedEvent:
    name: str | None
    type: Literal["session_info_changed"] = "session_info_changed"


@dataclass
class ThinkingLevelChangedEvent:
    level: ThinkingLevel
    type: Literal["thinking_level_changed"] = "thinking_level_changed"


@dataclass
class CompactionEndEvent:
    reason: CompactionReason
    result: CompactionResult | None
    aborted: bool
    will_retry: bool
    error_message: str | None = None
    type: Literal["compaction_end"] = "compaction_end"


@dataclass
class AutoRetryStartEvent:
    attempt: int
    max_attempts: int
    delay_ms: float
    error_message: str
    type: Literal["auto_retry_start"] = "auto_retry_start"


@dataclass
class AutoRetryEndEvent:
    success: bool
    attempt: int
    final_error: str | None = None
    type: Literal["auto_retry_end"] = "auto_retry_end"


@dataclass
class SummarizationRetryScheduledEvent:
    attempt: int
    max_attempts: int
    delay_ms: float
    error_message: str
    type: Literal["summarization_retry_scheduled"] = "summarization_retry_scheduled"


@dataclass
class SummarizationRetryAttemptStartEvent:
    source: SummarizationSource
    reason: CompactionReason | None = None
    type: Literal["summarization_retry_attempt_start"] = "summarization_retry_attempt_start"


@dataclass
class SummarizationRetryFinishedEvent:
    type: Literal["summarization_retry_finished"] = "summarization_retry_finished"


@dataclass
class BashExecutionUpdateEvent:
    delta: str
    id: str | None = None
    type: Literal["bash_execution_update"] = "bash_execution_update"


AgentSessionEvent = (
    AgentEvent
    | AgentEndEvent
    | AgentSettledEvent
    | QueueUpdateEvent
    | CompactionStartEvent
    | SessionInfoChangedEvent
    | ThinkingLevelChangedEvent
    | CompactionEndEvent
    | AutoRetryStartEvent
    | AutoRetryEndEvent
    | SummarizationRetryScheduledEvent
    | SummarizationRetryAttemptStartEvent
    | SummarizationRetryFinishedEvent
    | BashExecutionUpdateEvent
)

AgentSessionEventListener = Callable[[AgentSessionEvent], Any]


# ============================================================================
# Other Types
# ============================================================================


@dataclass
class ModelCycleResult:
    model: Model
    thinking_level: ThinkingLevel
    is_scoped: bool


@dataclass
class TokenStats:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total: int = 0


@dataclass
class ContextUsage:
    """Port of TS `ContextUsage`. `tokens`/`percent` are `None` when unknown."""

    tokens: int | None
    context_window: int
    percent: float | None


@dataclass
class SessionStats:
    session_file: str | None
    session_id: str
    user_messages: int
    assistant_messages: int
    tool_calls: int
    tool_results: int
    total_messages: int
    tokens: TokenStats
    cost: float
    context_usage: ContextUsage | None = None


@dataclass
class ToolInfo:
    name: str
    description: str
    parameters: dict[str, Any]
    prompt_guidelines: list[str] = field(default_factory=list)
    source_info: Any = None
    """Provenance of the tool: `<builtin:name>`, `<sdk:name>` or the
    extension's own `SourceInfo`. Port of TS `ToolInfo.sourceInfo`."""


@dataclass
class NavigateTreeResult:
    cancelled: bool
    aborted: bool = False
    editor_text: str | None = None
    summary_entry: BranchSummaryEntry | None = None


@dataclass
class QueueSnapshot:
    steering: list[str]
    follow_up: list[str]


# ============================================================================
# Constants
# ============================================================================

# Fallback thinking levels used only when no model is selected yet -- distinct
# from `pi_ai.types.THINKING_LEVELS` (7 items incl. xhigh/max), matching TS's
# own local `THINKING_LEVELS` const exactly (5 items).
_FALLBACK_THINKING_LEVELS: list[ThinkingLevel] = ["off", "minimal", "low", "medium", "high"]


def _normalize_prompt_snippet(text: str | None) -> str | None:
    """Port of `AgentSession._normalizePromptSnippet`.

    Collapses a tool's `promptSnippet` to a single whitespace-normalized line;
    the system prompt renders snippets one per line, so an embedded newline
    would break the tool list.
    """
    if not text:
        return None
    one_line = re.sub(r"\s+", " ", re.sub(r"[\r\n]+", " ", text)).strip()
    return one_line or None


def _normalize_prompt_guidelines(guidelines: Sequence[str] | None) -> list[str]:
    """Port of `AgentSession._normalizePromptGuidelines`: trim, drop empties, dedupe."""
    if not guidelines:
        return []
    unique: dict[str, None] = {}
    for guideline in guidelines:
        normalized = guideline.strip()
        if normalized:
            unique[normalized] = None
    return list(unique)


# Verbatim system-prompt contributions per builtin tool, ported from each
# tool's `*SystemPromptContribution` constant in
# `packages/coding-agent/src/core/tools/{bash,edit,find,grep,ls,read,write}.ts`.
TOOL_PROMPT_CONTRIBUTIONS: dict[str, tuple[str, list[str]]] = {
    "read": ("Read file contents", ["Use read to examine files instead of cat or sed."]),
    "bash": (
        "Execute bash commands (ls, grep, find, etc.)",
        ["You can inspect PI_* environment variables for current model and session details."],
    ),
    "edit": (
        "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
        [
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with multiple entries in "
            "edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier edits are applied. "
            "Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with "
            "large unchanged regions.",
        ],
    ),
    "write": ("Create or overwrite files", ["Use write only for new files or complete rewrites."]),
    "grep": ("Search file contents for patterns (respects .gitignore)", []),
    "find": ("Find files by glob pattern (respects .gitignore)", []),
    "ls": ("List directory contents", []),
}


def _is_placeholder_model(model: Model | None) -> bool:
    """Whether `model` is the `pi_agent.agent.DEFAULT_MODEL` sentinel ("no model selected")."""
    return model is None or (model.provider == DEFAULT_MODEL.provider and model.id == DEFAULT_MODEL.id)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _replace_message_in_place(target: AgentMessage, replacement: Any) -> None:
    """Replace `target`'s fields with `replacement`'s in place.

    Port of `_replaceMessageInPlace`: agent state stores the finalized message
    object before emitting `message_end`, and `_handle_agent_event`'s persistence
    logic (which follows `_emit_extension_event` in the same call) reads that same
    object -- mutating it in place keeps agent state, later events, and eventual
    `SessionManager.append_message()` persistence in sync. `emit_message_end`
    already validated `replacement.role == target.role`, so both are always the
    same dataclass type and safely support a raw `__dict__` swap.
    """
    if target is replacement:
        return
    target.__dict__.clear()
    target.__dict__.update(replacement.__dict__)


def _extension_message_to_custom_message(message: Any) -> CustomMessage:
    """Convert an extension-supplied `BeforeAgentStartEventResult.message` into a `CustomMessage`.

    TypeScript's `message` field is an untyped object literal
    (`{customType, content, display, details}`); this port's extension contract
    expects a `pi_agent.harness.messages.CustomMessage`-shaped value (or any
    object exposing the same attributes) since Python extensions are Python code
    and can construct `CustomMessage` directly. `getattr` with defaults keeps this
    tolerant of minimally-populated stand-ins.
    """
    if isinstance(message, CustomMessage):
        return message
    return CustomMessage(
        custom_type=getattr(message, "custom_type", "extension"),
        content=getattr(message, "content", None) or [],
        display=bool(getattr(message, "display", False)),
        details=getattr(message, "details", None),
        timestamp=now_ms(),
    )


async def _sleep_abortable(delay_ms: float, signal: AbortSignal | None) -> None:
    """Port of `packages/coding-agent/src/utils/sleep.ts`.

    Resolves after `delay_ms` milliseconds unless `signal` fires first, in
    which case it raises `AbortError` (mirroring the TS promise's rejection).
    """
    if signal is not None and signal.aborted:
        raise AbortError("Aborted")
    delay_s = max(delay_ms, 0) / 1000
    if signal is None:
        await asyncio.sleep(delay_s)
        return
    try:
        await asyncio.wait_for(signal.wait(), timeout=delay_s)
    except TimeoutError:
        return
    raise AbortError("Aborted")


def _estimate_messages_tokens(messages: list[Any]) -> int:
    return sum(estimate_tokens(message) for message in messages)


def _iso_to_epoch_ms(timestamp: str) -> int:
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _last_index_by_id(entries: list[SessionEntry], target: SessionEntry) -> int:
    for i in range(len(entries) - 1, -1, -1):
        if entries[i] is target or entries[i].id == target.id:
            return i
    return -1


def _extension_source_label(extension_path: str) -> str:
    """`extension:<name>` for a resource an extension contributed.

    Inline extensions are identified by a bracketed pseudo-path such as
    `<inline:demo>`; file-backed ones by their path, whose basename minus the
    module suffix is the name.
    """
    if extension_path.startswith("<"):
        return f"extension:{extension_path.replace('<', '').replace('>', '')}"
    name = re.sub(r"\.(py|ts|js)$", "", os.path.basename(extension_path))
    return f"extension:{name}"


# ============================================================================
# AgentSession
# ============================================================================


class AgentSession:
    """Wires an `Agent` to session persistence, settings, tools and models.

    Constructed with direct keyword arguments (matching `SessionManager`'s
    and `SettingsManager`'s own constructor conventions), rather than TS's
    single `AgentSessionConfig` object -- Python keyword arguments already
    give named-parameter construction, so no separate config dataclass is
    needed.
    """

    def __init__(
        self,
        *,
        agent: Agent,
        session_manager: SessionManager,
        settings_manager: SettingsManager,
        cwd: str,
        resource_loader: ResourceLoader,
        model_runtime: ModelRuntime,
        scoped_models: list[ScopedModel] | None = None,
        custom_tools: dict[str, AgentTool] | None = None,
        initial_active_tool_names: list[str] | None = None,
        allowed_tool_names: list[str] | None = None,
        excluded_tool_names: list[str] | None = None,
        base_tools_override: dict[str, AgentTool] | None = None,
        extensions: list[Extension] | None = None,
        session_start_event: SessionStartEvent | None = None,
    ) -> None:
        self.agent = agent
        self.session_manager = session_manager
        self.settings_manager = settings_manager
        self._model_runtime = model_runtime
        self._cwd = cwd
        self._resource_loader = resource_loader
        self._scoped_models: list[ScopedModel] = list(scoped_models or [])
        self._sdk_custom_tools: dict[str, AgentTool] = dict(custom_tools or {})
        self._custom_tools: dict[str, AgentTool] = dict(self._sdk_custom_tools)
        self._initial_active_tool_names = initial_active_tool_names
        self._allowed_tool_names = set(allowed_tool_names) if allowed_tool_names is not None else None
        self._excluded_tool_names = set(excluded_tool_names) if excluded_tool_names is not None else None
        self._base_tools_override = base_tools_override
        self._session_start_event = session_start_event or SessionStartEvent(reason="startup")

        # Extension runner: always constructed, even with an empty extension list, so
        # `has_handlers()`/`emit()` are unconditional no-ops rather than an optional branch --
        # a session with no extensions behaves exactly as it did before this system existed.
        self._extension_runner = ExtensionRunner(list(extensions or []), cwd=cwd, session_manager=session_manager)
        self._turn_index = 0
        # Extension-registered tools merge in as custom tools; an explicit `custom_tools` entry
        # of the same name wins (matches `_buildRuntime`'s `includeAllExtensionTools` merge order).
        # Their prompt contributions are kept aside because the wrapped `AgentTool` has no
        # `prompt_snippet`/`prompt_guidelines` fields (TS reads them off the `ToolDefinition`
        # when building `_toolPromptSnippets`/`_toolPromptGuidelines`).
        self._custom_tool_prompt_contributions: dict[str, tuple[str | None, list[str]]] = {}
        self._custom_tool_source_info: dict[str, Any] = {}
        self._collect_custom_tools()

        # Event subscription state
        self._event_listeners: list[AgentSessionEventListener] = []
        self._is_agent_run_active = False
        self._idle_event = asyncio.Event()
        self._idle_event.set()
        # Fire-and-forget tasks (e.g. extension-triggered compaction) are held here so they
        # aren't garbage-collected mid-flight; each discards itself once done.
        self._background_tasks: set[asyncio.Task[None]] = set()

        # Queue display state (mirrors what's queued on `agent`, for UI display)
        self._steering_messages: list[str] = []
        self._follow_up_messages: list[str] = []
        self._pending_next_turn_messages: list[CustomMessage] = []

        # Compaction state
        self._compaction_abort_controller: AbortController | None = None
        self._auto_compaction_abort_controller: AbortController | None = None
        self._overflow_recovery_attempted = False

        # Branch summarization state
        self._branch_summary_abort_controller: AbortController | None = None

        # Retry state
        self._retry_abort_controller: AbortController | None = None
        self._retry_attempt = 0

        # Bash execution state
        self._bash_abort_controllers: set[AbortController] = set()
        self._pending_bash_messages: list[BashExecutionMessage] = []

        # Tool registry
        self._base_tools: dict[str, AgentTool] | None = None
        self._bash_options: tuple[str | None, str | None] = (None, None)
        self._tool_registry: dict[str, AgentTool] = {}
        self._tool_source_info: dict[str, Any] = {}
        self._tool_prompt_snippets: dict[str, str] = {}
        self._tool_prompt_guidelines: dict[str, list[str]] = {}

        # Base system prompt (without per-turn overrides)
        self._base_system_prompt = ""
        self._base_system_prompt_options: BuildSystemPromptOptions | None = None
        self._system_prompt_override: str | None = None

        # Track last assistant message for auto-compaction/retry checks
        self._last_assistant_message: AssistantMessage | None = None

        self._unsubscribe_agent = self.agent.subscribe(self._handle_agent_event)
        self._install_agent_tool_hooks()
        self._install_agent_next_turn_refresh()
        self._refresh_tool_registry(initial=True)
        self._bind_extension_core()

    @property
    def model_runtime(self) -> ModelRuntime:
        return self._model_runtime

    # =========================================================================
    # Construction Helpers
    # =========================================================================

    def _bind_extension_core(self) -> None:
        """Bind live accessors the extension runner needs to build `ExtensionContext`.

        Port of the `bindCore()` half of TypeScript's `_applyExtensionBindings` (the
        registration-action half -- `sendMessage`/`appendEntry`/`setSessionName`/... --
        has no equivalent here: those actions are only reachable through
        `ExtensionCommandContext`/`pi.*` calls made from a loaded extension's own
        module-level code, and this port's `ExtensionRuntimeActions` already defaults
        every one of them to a no-op when a caller doesn't supply its own bindings at
        load time, see `loader.py`).
        """
        self._extension_runner.bind_core(
            ExtensionContextActions(
                get_model=lambda: self.model,
                get_scoped_models=lambda: tuple(self._scoped_models),
                is_idle=lambda: self.is_idle,
                is_project_trusted=self.settings_manager.is_project_trusted,
                get_signal=lambda: self.agent.signal,
                abort=lambda: asyncio.ensure_future(self.abort()),
                has_pending_messages=lambda: self.pending_message_count > 0,
                shutdown=lambda: None,
                get_context_usage=lambda: self.get_context_usage(),
                compact=self._extension_compact_action,
                get_system_prompt=lambda: self.system_prompt,
                get_system_prompt_options=lambda: self._base_system_prompt_options or BuildSystemPromptOptions(),
                get_thinking_level=lambda: self.thinking_level,
                get_active_tool_names=self.get_active_tool_names,
                refresh_tools=self._refresh_tool_registry,
                wait_for_idle=self._wait_for_idle,
            )
        )

    def _extension_compact_action(
        self,
        custom_instructions: str | None = None,
        on_complete: Callable[[CompactionResult], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """`pi.compact()` action: fire-and-forget, matching TS's `void (async () => ...)()`."""

        async def _run() -> None:
            try:
                result = await self.compact(custom_instructions)
                if on_complete:
                    on_complete(result)
            except Exception as error:
                if on_error:
                    on_error(error)

        # Keep a reference so the task isn't garbage-collected mid-flight; discard once done.
        task = asyncio.ensure_future(_run())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _wait_for_idle(self) -> None:
        await self._idle_event.wait()

    def _install_agent_tool_hooks(self) -> None:
        """Install `before_tool_call`/`after_tool_call` for extension interception plus
        image-result normalization.

        Port of `_installAgentToolHooks`. `before_tool_call` dispatches the `tool_call`
        extension hook (not error-isolated -- an extension exception blocks tool
        execution, matching TypeScript exactly, see `runner.py`'s module docstring).
        `after_tool_call` dispatches the `tool_result` extension hook first, then
        normalizes images in whatever content the hook left behind, so extension-
        injected/replaced images are normalized too (matching TypeScript's ordering).
        """

        async def before_tool_call(
            context: BeforeToolCallContext, _signal: AbortSignal | None
        ) -> BeforeToolCallResult | None:
            runner = self._extension_runner
            if not runner.has_handlers("tool_call"):
                return None
            result = await runner.emit_tool_call(
                ToolCallEvent(
                    tool_call_id=context.tool_call.id,
                    tool_name=context.tool_call.name,
                    input=context.args,
                )
            )
            if result is None:
                return None
            return BeforeToolCallResult(block=result.block, reason=result.reason, terminate=result.terminate)

        async def after_tool_call(
            context: AfterToolCallContext, _signal: AbortSignal | None
        ) -> AfterToolCallResult | None:
            runner = self._extension_runner
            hook_result = None
            if runner.has_handlers("tool_result"):
                hook_result = await runner.emit_tool_result(
                    ToolResultEvent(
                        tool_call_id=context.tool_call.id,
                        tool_name=context.tool_call.name,
                        input=context.args,
                        content=context.result.content,
                        is_error=context.is_error,
                        details=context.result.details,
                        usage=context.result.usage,
                    )
                )

            content = hook_result.content if hook_result and hook_result.content is not None else context.result.content
            normalized_content = await normalize_tool_result_images(
                content,
                NormalizeToolResultImagesOptions(auto_resize_images=self.settings_manager.get_image_auto_resize()),
            )

            if hook_result is None and normalized_content is content:
                return None

            return AfterToolCallResult(
                content=normalized_content,
                details=hook_result.details if hook_result else None,
                is_error=(
                    hook_result.is_error if hook_result and hook_result.is_error is not None else context.is_error
                ),
                usage=hook_result.usage if hook_result else None,
            )

        self.agent.before_tool_call = before_tool_call
        self.agent.after_tool_call = after_tool_call

    def _install_agent_next_turn_refresh(self) -> None:
        """Keep `context.system_prompt`/`tools`/`model`/`thinking_level` in sync every turn.

        Port of `_installAgentNextTurnRefresh`: wraps any existing
        `prepare_next_turn_with_context` hook (there is none by default) so
        the session's current system prompt override, active tools, model and
        thinking level are always applied to the next turn's context, even if
        they changed mid-run via `set_model`/`set_thinking_level`/
        `set_active_tools_by_name`.
        """
        previous_hook = self.agent.prepare_next_turn_with_context

        async def refresh(turn: PrepareNextTurnContext, signal: AbortSignal | None) -> AgentLoopTurnUpdate:
            previous_snapshot = await _maybe_await(previous_hook(turn, signal)) if previous_hook is not None else None
            previous_context = (
                previous_snapshot.context
                if previous_snapshot is not None and previous_snapshot.context
                else turn.context
            )
            new_context = replace(
                previous_context,
                system_prompt=(
                    self._system_prompt_override
                    if self._system_prompt_override is not None
                    else self._base_system_prompt
                ),
                tools=list(self.agent.state.tools),
            )
            return AgentLoopTurnUpdate(
                context=new_context,
                model=self.agent.state.model,
                thinking_level=self.agent.state.thinking_level,
            )

        self.agent.prepare_next_turn_with_context = refresh

    def _build_session_environment(self) -> dict[str, str]:
        """`PI_*` variables the bash tool exposes to spawned commands.

        Port of `resolveSpawnContext`'s `exposeSessionEnvironment` branch,
        which reads the same five values off the live `ExtensionContext`.
        """
        env: dict[str, str] = {"PI_SESSION_ID": self.session_id}
        session_file = self.session_file
        if session_file:
            env["PI_SESSION_FILE"] = session_file
        model = self.model
        if model is not None:
            env["PI_PROVIDER"] = model.provider
            env["PI_MODEL"] = model.id
        thinking_level = self.agent.state.thinking_level
        if thinking_level:
            env["PI_REASONING_LEVEL"] = thinking_level
        return env

    def _collect_custom_tools(self) -> None:
        """Merge SDK-supplied and extension-registered tools into `_custom_tools`.

        Split out of `__init__` so `_refresh_tool_registry()` can re-run it:
        `pi.register_tool()` may fire long after construction (from a
        `session_start` handler, for example), and TypeScript's
        `_refreshToolRegistry` re-reads `getAllRegisteredTools()` on every call.
        """
        self._custom_tools = dict(self._sdk_custom_tools)
        self._custom_tool_prompt_contributions = {}
        self._custom_tool_source_info = {}
        sdk_tool_names = set(self._sdk_custom_tools)
        registered_tools = self._extension_runner.get_all_registered_tools()
        for registered in registered_tools:
            self._custom_tool_prompt_contributions.setdefault(
                registered.definition.name,
                (
                    _normalize_prompt_snippet(registered.definition.prompt_snippet),
                    _normalize_prompt_guidelines(registered.definition.prompt_guidelines),
                ),
            )
            # An SDK tool of the same name wins the registry slot, so it must
            # win the provenance too -- otherwise `get_all_tools()` would
            # attribute an SDK tool to the extension it shadowed.
            if registered.source_info is not None and registered.definition.name not in sdk_tool_names:
                self._custom_tool_source_info.setdefault(registered.definition.name, registered.source_info)
        for wrapped_tool in wrap_registered_tools(registered_tools, self._extension_runner):
            self._custom_tools.setdefault(wrapped_tool.name, wrapped_tool)

    def _refresh_tool_registry(self, *, initial: bool = False) -> None:
        """Build the tool registry from builtins/overrides + custom tools.

        Simplified stand-in for TS's `_buildRuntime`/`_refreshToolRegistry`
        (almost entirely extension-plumbing there): no `ToolDefinition`
        wrapper layer. Tool provenance *is* tracked, in `_tool_source_info`,
        so `get_all_tools()` can report the same `<builtin:name>` /
        `<sdk:name>` / extension `SourceInfo` values as TypeScript.

        `initial=True` is the construction pass, which seeds the active set
        from `initial_active_tool_names` (TS's `_buildRuntime({activeToolNames,
        includeAllExtensionTools: true})`). Later calls come from
        `pi.register_tool()` and keep the current active set, appending any
        tool name that was not in the registry before -- exactly what TS's
        option-less `_refreshToolRegistry()` does.
        """
        allowed = self._allowed_tool_names
        excluded = self._excluded_tool_names

        def is_allowed(name: str) -> bool:
            return (allowed is None or name in allowed) and not (excluded is not None and name in excluded)

        previous_registry_names = set(self._tool_registry)
        previous_active_names = self.get_active_tool_names()
        self._collect_custom_tools()

        # TypeScript re-reads these settings on every `updateToolDefinitions()`
        # call, so changing `shellCommandPrefix`/`shellPath` mid-session takes
        # effect on the next update. Rebuilding only when they change keeps the
        # tool objects stable in the common case where they do not.
        bash_command_prefix = self.settings_manager.get_shell_command_prefix()
        bash_shell_path = self.settings_manager.get_shell_path()
        bash_options = (bash_command_prefix, bash_shell_path)
        if self._base_tools is not None and self._base_tools_override is None and self._bash_options != bash_options:
            self._base_tools = None

        if self._base_tools is None:
            self._bash_options = bash_options
            if self._base_tools_override is not None:
                self._base_tools = dict(self._base_tools_override)
            else:
                self._base_tools = {
                    name: create_tool(
                        name,
                        self._cwd,
                        session_environment=self._build_session_environment,
                        bash_command_prefix=bash_command_prefix,
                        bash_shell_path=bash_shell_path,
                    )
                    for name in ALL_TOOL_NAMES
                }
        base_tools = self._base_tools

        tool_registry: dict[str, AgentTool] = {name: tool for name, tool in base_tools.items() if is_allowed(name)}
        source_info: dict[str, Any] = {
            name: create_synthetic_source_info(f"<builtin:{name}>", "builtin") for name in tool_registry
        }
        for name, tool in self._custom_tools.items():
            if is_allowed(name):
                tool_registry[name] = tool
                # Extension tools carry their own extension's SourceInfo; anything
                # else in `_custom_tools` came in through the SDK `custom_tools` arg.
                source_info[name] = self._custom_tool_source_info.get(
                    name, create_synthetic_source_info(f"<sdk:{name}>", "sdk")
                )
        self._tool_registry = tool_registry
        self._tool_source_info = source_info

        self._tool_prompt_snippets = {}
        self._tool_prompt_guidelines = {}
        for name in tool_registry:
            contribution = TOOL_PROMPT_CONTRIBUTIONS.get(name)
            if contribution is not None:
                snippet, guidelines = contribution
                # TS reads promptSnippet/promptGuidelines off each ToolDefinition, so a
                # built-in constructed with its guidelines suppressed (bash built with
                # `exposeSessionEnvironment: false` has `promptGuidelines: undefined`)
                # contributes none. This port keys contributions by tool name, so such a
                # tool carries its own guidelines and they win here.
                instance_guidelines = getattr(tool_registry[name], "prompt_guidelines", None)
                if instance_guidelines is not None:
                    guidelines = instance_guidelines
                self._tool_prompt_snippets[name] = snippet
                if guidelines:
                    self._tool_prompt_guidelines[name] = list(guidelines)
            custom_contribution = self._custom_tool_prompt_contributions.get(name)
            if custom_contribution is not None:
                custom_snippet, custom_guidelines = custom_contribution
                if custom_snippet:
                    self._tool_prompt_snippets[name] = custom_snippet
                if custom_guidelines:
                    self._tool_prompt_guidelines[name] = list(custom_guidelines)

        if initial:
            default_active = (
                list(base_tools.keys()) if self._base_tools_override is not None else ["read", "bash", "edit", "write"]
            )
            active_names = (
                list(self._initial_active_tool_names) if self._initial_active_tool_names is not None else default_active
            )
        else:
            active_names = list(previous_active_names)
        active_names = [name for name in active_names if is_allowed(name)]
        if allowed is not None:
            for name in tool_registry:
                if name in allowed and name not in active_names:
                    active_names.append(name)
        elif initial:
            # TS constructs the runtime with `includeAllExtensionTools: true`,
            # which appends every extension-registered and SDK-supplied custom
            # tool to the active set regardless of `activeToolNames`.
            for name in self._custom_tools:
                if is_allowed(name) and name not in active_names:
                    active_names.append(name)
        else:
            for name in tool_registry:
                if name not in previous_registry_names and name not in active_names:
                    active_names.append(name)
        self.set_active_tools_by_name(list(dict.fromkeys(active_names)))

    def _rebuild_system_prompt(self, tool_names: list[str]) -> str:
        valid_tool_names = [name for name in tool_names if name in self._tool_registry]
        tool_snippets: dict[str, str] = {}
        prompt_guidelines: list[str] = []
        for name in valid_tool_names:
            snippet = self._tool_prompt_snippets.get(name)
            if snippet:
                tool_snippets[name] = snippet
            guidelines = self._tool_prompt_guidelines.get(name)
            if guidelines:
                prompt_guidelines.extend(guidelines)

        loader_system_prompt = self._resource_loader.get_system_prompt()
        loader_append_system_prompt = self._resource_loader.get_append_system_prompt()
        append_system_prompt = "\n\n".join(loader_append_system_prompt) if loader_append_system_prompt else None
        loaded_skills = self._resource_loader.get_skills().skills
        loaded_context_files = [
            ContextFile(path=item["path"], content=item["content"]) for item in self._resource_loader.get_agents_files()
        ]

        self._base_system_prompt_options = BuildSystemPromptOptions(
            cwd=self._cwd,
            skills=loaded_skills,
            context_files=loaded_context_files,
            custom_prompt=loader_system_prompt,
            append_system_prompt=append_system_prompt,
            selected_tools=valid_tool_names,
            tool_snippets=tool_snippets,
            prompt_guidelines=prompt_guidelines,
        )
        return build_system_prompt(self._base_system_prompt_options)

    # =========================================================================
    # Event Subscription
    # =========================================================================

    def _emit(self, event: AgentSessionEvent) -> None:
        for listener in list(self._event_listeners):
            listener(event)

    def _emit_queue_update(self) -> None:
        self._emit(QueueUpdateEvent(steering=list(self._steering_messages), follow_up=list(self._follow_up_messages)))

    async def _emit_agent_settled(self) -> None:
        self._is_agent_run_active = False
        try:
            if self._extension_runner.has_handlers("agent_settled"):
                await self._extension_runner.emit(ExtAgentSettledEvent())
            self._emit(AgentSettledEvent())
        finally:
            self._idle_event.set()

    async def _emit_extension_event(self, event: AgentEvent) -> None:
        """Re-emit agent-loop lifecycle events to extensions.

        Port of `_emitExtensionEvent`. Every event this dispatches maps 1:1 onto
        `pi_agent.types.AgentEvent`'s variants; `message_end` additionally applies an
        extension's message replacement in place before returning, so the
        persistence/state-tracking logic in `_handle_agent_event` that runs
        immediately after this call already sees the replaced message.
        """
        if event.type == "agent_start":
            self._turn_index = 0
            if self._extension_runner.has_handlers("agent_start"):
                await self._extension_runner.emit(ExtAgentStartEvent())
        elif event.type == "agent_end":
            if self._extension_runner.has_handlers("agent_end"):
                await self._extension_runner.emit(ExtAgentEndEvent(messages=event.messages))
        elif event.type == "turn_start":
            if self._extension_runner.has_handlers("turn_start"):
                await self._extension_runner.emit(ExtTurnStartEvent(turn_index=self._turn_index, timestamp=now_ms()))
        elif event.type == "turn_end":
            if self._extension_runner.has_handlers("turn_end"):
                await self._extension_runner.emit(
                    ExtTurnEndEvent(turn_index=self._turn_index, message=event.message, tool_results=event.tool_results)
                )
            self._turn_index += 1
        elif event.type == "message_start":
            if self._extension_runner.has_handlers("message_start"):
                await self._extension_runner.emit(ExtMessageStartEvent(message=event.message))
        elif event.type == "message_update":
            if self._extension_runner.has_handlers("message_update"):
                await self._extension_runner.emit(
                    ExtMessageUpdateEvent(message=event.message, assistant_message_event=event.assistant_message_event)
                )
        elif event.type == "message_end":
            if self._extension_runner.has_handlers("message_end"):
                replacement = await self._extension_runner.emit_message_end(ExtMessageEndEvent(message=event.message))
                if replacement is not None:
                    # Untyped extension handlers can return messages with null
                    # content; normalize so it never enters agent state or
                    # session history.
                    if (
                        getattr(replacement, "role", None) in ("user", "assistant", "toolResult", "custom")
                        and getattr(replacement, "content", "sentinel") is None
                    ):
                        replacement = replace(replacement, content=[])
                    _replace_message_in_place(event.message, replacement)
        elif event.type == "tool_execution_start":
            if self._extension_runner.has_handlers("tool_execution_start"):
                await self._extension_runner.emit(
                    ExtToolExecutionStartEvent(
                        tool_call_id=event.tool_call_id, tool_name=event.tool_name, args=event.args
                    )
                )
        elif event.type == "tool_execution_update":
            if self._extension_runner.has_handlers("tool_execution_update"):
                await self._extension_runner.emit(
                    ExtToolExecutionUpdateEvent(
                        tool_call_id=event.tool_call_id,
                        tool_name=event.tool_name,
                        args=event.args,
                        partial_result=event.partial_result,
                    )
                )
        elif event.type == "tool_execution_end":
            if self._extension_runner.has_handlers("tool_execution_end"):
                await self._extension_runner.emit(
                    ExtToolExecutionEndEvent(
                        tool_call_id=event.tool_call_id,
                        tool_name=event.tool_name,
                        result=event.result,
                        is_error=event.is_error,
                    )
                )

    async def _handle_agent_event(self, event: AgentEvent, _signal: AbortSignal | None) -> None:
        """Shared handler installed on `agent.subscribe`: persistence + re-emission."""
        await self._emit_extension_event(event)

        if event.type == "message_start" and getattr(event.message, "role", None) == "user":
            self._overflow_recovery_attempted = False
            message_text = content_text(event.message.content, "")
            if message_text:
                if message_text in self._steering_messages:
                    self._steering_messages.remove(message_text)
                    self._emit_queue_update()
                elif message_text in self._follow_up_messages:
                    self._follow_up_messages.remove(message_text)
                    self._emit_queue_update()

        if event.type == "agent_end":
            self._emit(AgentEndEvent(messages=event.messages, will_retry=self._will_retry_after_agent_end(event)))
        else:
            self._emit(event)

        if event.type == "message_end":
            message = event.message
            role = getattr(message, "role", None)
            if role == "custom":
                self.session_manager.append_custom_message_entry(
                    message.custom_type, message.content, message.display, message.details
                )
            elif role in ("user", "assistant", "toolResult"):
                self.session_manager.append_message(message)

            if role == "assistant":
                self._last_assistant_message = message
                if message.stop_reason not in ("error", "length"):
                    self._overflow_recovery_attempted = False
                if message.stop_reason != "error" and self._retry_attempt > 0:
                    self._emit(AutoRetryEndEvent(success=True, attempt=self._retry_attempt))
                    self._retry_attempt = 0

    def _will_retry_after_agent_end(self, event: AgentEvent) -> bool:
        settings = self.settings_manager.get_retry_settings()
        if not settings["enabled"] or self._retry_attempt >= settings["maxRetries"]:
            return False
        for message in reversed(event.messages):
            if getattr(message, "role", None) == "assistant":
                return self._is_retryable_error(message)
        return False

    def _find_last_assistant_message(self) -> AssistantMessage | None:
        for message in reversed(self.agent.state.messages):
            if getattr(message, "role", None) == "assistant":
                return message
        return None

    def subscribe(self, listener: AgentSessionEventListener) -> Callable[[], None]:
        """Subscribe to session events. Session persistence happens internally regardless."""
        self._event_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._event_listeners:
                self._event_listeners.remove(listener)

        return unsubscribe

    def dispose(self) -> None:
        """Remove all listeners and disconnect from the agent.

        Omits TS's trailing `cleanupSessionResources(this.sessionId)` call (see
        module docstring) and the *staleness* half of the extension-runner
        `invalidate()` call: that half exists in TypeScript only to make a
        captured `ctx`/`pi` object raise if used after
        `ctx.newSession()`/`fork()`/`switchSession()`/`reload()` replace the
        session -- since this port has none of that session-replacement
        machinery (see module docstring), there is nothing for it to guard
        against, and the extension runner has no "stale" state to enter. The
        event-bus half *is* ported (`unsubscribe_events()` below): the bus is
        owned by the host and outlives the session, so leaving extension
        handlers on it keeps a disposed session's handlers running (#7193).
        """
        try:
            self.abort_retry()
            self.abort_compaction()
            self.abort_branch_summary()
            self.abort_bash()
            self.agent.abort()
        except Exception:
            pass  # Dispose must succeed even if an abort hook raises.

        # Dispose must succeed even if an unsubscribe hook raises.
        with contextlib.suppress(Exception):
            self._extension_runner.unsubscribe_events()

        if self._unsubscribe_agent is not None:
            self._unsubscribe_agent()
            self._unsubscribe_agent = None
        self._event_listeners = []

    # =========================================================================
    # Read-only State Access
    # =========================================================================

    @property
    def state(self) -> Any:
        return self.agent.state

    @property
    def model(self) -> Model | None:
        model = self.agent.state.model
        return None if _is_placeholder_model(model) else model

    @property
    def thinking_level(self) -> ThinkingLevel:
        return self.agent.state.thinking_level

    @property
    def is_streaming(self) -> bool:
        return self._is_agent_run_active

    @property
    def is_idle(self) -> bool:
        return not self._is_agent_run_active

    @property
    def system_prompt(self) -> str:
        return self.agent.state.system_prompt

    @property
    def retry_attempt(self) -> int:
        return self._retry_attempt

    def get_active_tool_names(self) -> list[str]:
        return [tool.name for tool in self.agent.state.tools]

    def get_all_tools(self) -> list[ToolInfo]:
        return [
            ToolInfo(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                prompt_guidelines=list(self._tool_prompt_guidelines.get(name, [])),
                source_info=self._tool_source_info.get(name),
            )
            for name, tool in self._tool_registry.items()
        ]

    def set_active_tools_by_name(self, tool_names: list[str]) -> None:
        """Set active tools by name. Unknown tool names are ignored. Rebuilds the system prompt."""
        tools: list[AgentTool] = []
        valid_tool_names: list[str] = []
        for name in tool_names:
            tool = self._tool_registry.get(name)
            if tool is not None:
                tools.append(tool)
                valid_tool_names.append(name)
        self.agent.state.tools = tools

        self._base_system_prompt = self._rebuild_system_prompt(valid_tool_names)
        self.agent.state.system_prompt = (
            self._system_prompt_override if self._system_prompt_override is not None else self._base_system_prompt
        )

    @property
    def extension_runner(self) -> ExtensionRunner:
        return self._extension_runner

    async def bind_extensions(self) -> None:
        """Emit this session's `session_start` extension event.

        Port of the tail of TypeScript's `bindExtensions()`. The binding half
        of the TS method wires the extension UI host into the runner; here the
        interactive mode installs its own `ExtensionUIContext` on the runner
        just before calling this, and the core context bindings are already
        applied eagerly in `__init__`, so only the event emission remains.
        `AgentSessionRuntime` calls this after every session replacement so
        extensions see the same `session_start` sequence they do upstream.
        """
        if self._extension_runner.has_handlers("session_start"):
            await self._extension_runner.emit(self._session_start_event)
        await self._extend_resources_from_extensions(
            "reload" if self._session_start_event.reason == "reload" else "startup"
        )

    async def _extend_resources_from_extensions(self, reason: Literal["startup", "reload"]) -> None:
        """Let `resources_discover` handlers add skill and prompt paths, then reload them.

        Port of `extendResourcesFromExtensions`. The system prompt is rebuilt
        afterwards because it embeds the skill list, so a skill contributed here
        would otherwise be invisible to the model until the next rebuild.
        """
        if not self._extension_runner.has_handlers("resources_discover"):
            return

        discovered = await self._extension_runner.emit_resources_discover(self._cwd, reason)
        # TypeScript also checks themePaths here; this port has no theme loading.
        if not discovered.skill_paths and not discovered.prompt_paths:
            return

        self._resource_loader.extend_resources(
            skill_paths=self._build_extension_resource_paths(discovered.skill_paths),
            prompt_paths=self._build_extension_resource_paths(discovered.prompt_paths),
        )
        self._base_system_prompt = self._rebuild_system_prompt(self.get_active_tool_names())
        if self._system_prompt_override is None:
            self.agent.state.system_prompt = self._base_system_prompt

    def _build_extension_resource_paths(self, entries: list[DiscoveredResourcePath]) -> list[ExtensionResourcePath]:
        return [
            ExtensionResourcePath(
                path=entry.path,
                metadata=ResourcePathMetadata(
                    source=_extension_source_label(entry.extension_path),
                    scope="temporary",
                    origin="top-level",
                    base_dir=None if entry.extension_path.startswith("<") else os.path.dirname(entry.extension_path),
                ),
            )
            for entry in entries
        ]

    @property
    def is_compacting(self) -> bool:
        return (
            self._auto_compaction_abort_controller is not None
            or self._compaction_abort_controller is not None
            or self._branch_summary_abort_controller is not None
        )

    @property
    def messages(self) -> list[AgentMessage]:
        return self.agent.state.messages

    @property
    def steering_mode(self) -> Literal["all", "one-at-a-time"]:
        return self.agent.steering_mode

    @property
    def follow_up_mode(self) -> Literal["all", "one-at-a-time"]:
        return self.agent.follow_up_mode

    @property
    def session_file(self) -> str | None:
        return self.session_manager.get_session_file()

    @property
    def session_id(self) -> str:
        return self.session_manager.get_session_id()

    @property
    def session_name(self) -> str | None:
        return self.session_manager.get_session_name()

    @property
    def scoped_models(self) -> list[ScopedModel]:
        return self._scoped_models

    def set_scoped_models(self, scoped_models: list[ScopedModel]) -> None:
        self._scoped_models = list(scoped_models)

    @property
    def prompt_templates(self) -> list[PromptTemplate]:
        prompts, _diagnostics = self._resource_loader.get_prompts()
        return prompts

    @property
    def resource_loader(self) -> ResourceLoader:
        return self._resource_loader

    @property
    def pending_message_count(self) -> int:
        return len(self._steering_messages) + len(self._follow_up_messages)

    def get_steering_messages(self) -> list[str]:
        return list(self._steering_messages)

    def get_follow_up_messages(self) -> list[str]:
        return list(self._follow_up_messages)

    # =========================================================================
    # Prompting
    # =========================================================================

    async def _run_agent_prompt(self, messages: AgentMessage | list[AgentMessage]) -> None:
        self._is_agent_run_active = True
        self._idle_event.clear()
        try:
            await self.agent.prompt(messages)
            while await self._handle_post_agent_run():
                await self.agent.continue_()
        finally:
            self._system_prompt_override = None
            self._flush_pending_bash_messages()
            await self._emit_agent_settled()

    async def _handle_post_agent_run(self) -> bool:
        message = self._last_assistant_message
        self._last_assistant_message = None
        if message is None:
            return False

        if self._is_retryable_error(message) and await self._prepare_retry(message):
            return True

        if message.stop_reason == "error" and self._retry_attempt > 0:
            self._emit(AutoRetryEndEvent(success=False, attempt=self._retry_attempt, final_error=message.error_message))
            self._retry_attempt = 0

        if await self._check_compaction(message):
            return True

        # The agent loop drains both queues before emitting agent_end. Any messages
        # queued during that draining need a continuation.
        return self.agent.has_queued_messages()

    def _expand_skill_command(self, text: str) -> str:
        if not text.startswith("/skill:"):
            return text

        space_index = text.find(" ")
        skill_name = text[7:] if space_index == -1 else text[7:space_index]
        args = "" if space_index == -1 else text[space_index + 1 :].strip()

        skill = next((s for s in self._resource_loader.get_skills().skills if s.name == skill_name), None)
        if skill is None:
            return text  # Unknown skill, pass through.

        try:
            content = Path(skill.file_path).read_text(encoding="utf-8")
        except OSError as err:
            # Report like extension commands do, then pass the text through unchanged.
            self._extension_runner.emit_error(
                ExtensionError(extension_path=skill.file_path, event="skill_expansion", error=str(err))
            )
            return text

        body = strip_frontmatter(content).strip()
        skill_block = (
            f'<skill name="{skill.name}" location="{skill.file_path}">\n'
            f"References are relative to {skill.base_dir}.\n\n{body}\n</skill>"
        )
        return f"{skill_block}\n\n{args}" if args else skill_block

    async def _ensure_model_auth(self, model: Model) -> None:
        """Raise if `model`'s provider has no configured auth. Mirrors TS's inline `prompt()` check."""
        has_configured_auth = self._model_runtime.has_configured_auth(model.provider) or (
            await self._model_runtime.check_auth(model.provider) is not None
        )
        if has_configured_auth:
            return
        if self._model_runtime.is_using_oauth(model.provider):
            raise RuntimeError(
                f'Authentication failed for "{model.provider}". Credentials may have expired or network is '
                f"unavailable. Run '/login {model.provider}' to re-authenticate."
            )
        raise RuntimeError(format_no_api_key_found_message(model.provider))

    async def _ensure_summarization_auth(self, model: Model) -> None:
        """Port of `_getSummarizationRequestAuth`'s strict/lenient split.

        TypeScript only demands resolvable credentials up front when the agent
        streams through the plain provider dispatch
        (`this.agent.streamFunction === streamSimple`); with any custom stream
        function, auth resolution is best effort and whatever the stream
        function itself does decides the outcome. The Python analogue of that
        plain dispatch is `ModelRuntime.stream_simple` bound to this session's
        runtime, so the same comparison identifies the strict case (the actual
        `apiKey`/`headers`/`env` threading TypeScript does here is unnecessary:
        `ModelRuntime.stream_simple` resolves auth itself, see this module's
        docstring on simplified summarization auth-threading).
        """
        stream_fn = self.agent.stream_function
        uses_provider_dispatch = (
            getattr(stream_fn, "__func__", None) is ModelRuntime.stream_simple
            and getattr(stream_fn, "__self__", None) is self._model_runtime
        )
        if uses_provider_dispatch:
            await self._ensure_model_auth(model)

    async def prompt(
        self,
        text: str,
        *,
        expand_prompt_templates: bool = True,
        images: list[ImageContent] | None = None,
        streaming_behavior: Literal["steer", "followUp"] | None = None,
        preflight_result: Callable[[bool], None] | None = None,
        source: InputSource = "interactive",
    ) -> None:
        """Send a prompt to the agent.

        - Extension commands (`/name ...`) execute immediately, even while streaming,
          and never reach the LLM (`_try_execute_extension_command`).
        - Emits the `input` extension event before skill/template expansion, letting an
          extension transform or fully handle the input.
        - Expands `/skill:name` skill blocks and file-based prompt templates by default.
        - During streaming, queues via `steer()`/`follow_up()` based on `streaming_behavior`.
        - Validates model and auth before sending (when not streaming).
        - Emits `before_agent_start` right before dispatching to the agent, letting
          extensions inject custom messages and override the system prompt for this turn.
        """
        messages: list[AgentMessage] | None = None
        try:
            if expand_prompt_templates and text.startswith("/"):
                handled = await self._try_execute_extension_command(text)
                if handled:
                    if preflight_result:
                        preflight_result(True)
                    return

            if self._compaction_abort_controller is not None:
                raise RuntimeError(
                    "Cannot submit a prompt while compaction is in progress. Wait for compaction to finish and retry."
                )

            current_text = text
            current_images = images
            if self._extension_runner.has_handlers("input"):
                input_result = await self._extension_runner.emit_input(
                    current_text,
                    current_images,
                    source,
                    streaming_behavior if self.is_streaming else None,
                )
                if input_result.action == "handled":
                    if preflight_result:
                        preflight_result(True)
                    return
                if input_result.action == "transform":
                    current_text = input_result.text if input_result.text is not None else current_text
                    current_images = input_result.images if input_result.images is not None else current_images

            expanded_text = current_text
            if expand_prompt_templates:
                expanded_text = self._expand_skill_command(expanded_text)
                expanded_text = expand_prompt_template(expanded_text, list(self.prompt_templates))

            if self.is_streaming:
                if streaming_behavior is None:
                    raise RuntimeError(
                        "Agent is already processing. Specify streaming_behavior ('steer' or 'followUp') "
                        "to queue the message."
                    )
                if streaming_behavior == "followUp":
                    await self._queue_follow_up(expanded_text, current_images)
                else:
                    await self._queue_steer(expanded_text, current_images)
                if preflight_result:
                    preflight_result(True)
                return

            self._flush_pending_bash_messages()

            model = self.model
            if model is None:
                raise RuntimeError(format_no_model_selected_message())
            await self._ensure_model_auth(model)

            last_assistant = self._find_last_assistant_message()
            if last_assistant is not None:
                await self._check_compaction(last_assistant, skip_aborted_check=False)

            messages = []
            user_content: list[TextContent | ImageContent] = [TextContent(text=expanded_text)]
            if current_images:
                user_content.extend(current_images)
            messages.append(UserMessage(content=user_content, timestamp=now_ms()))

            messages.extend(self._pending_next_turn_messages)
            self._pending_next_turn_messages = []

            before_agent_start = await self._extension_runner.emit_before_agent_start(
                expanded_text,
                current_images,
                self._base_system_prompt,
                self._base_system_prompt_options or BuildSystemPromptOptions(),
            )
            if before_agent_start is not None:
                extension_messages, system_prompt_override = before_agent_start
                for extension_message in extension_messages:
                    messages.append(_extension_message_to_custom_message(extension_message))
                if system_prompt_override is not None:
                    self._system_prompt_override = system_prompt_override
                    self.agent.state.system_prompt = system_prompt_override
                else:
                    self._system_prompt_override = None
                    self.agent.state.system_prompt = self._base_system_prompt
            else:
                self._system_prompt_override = None
                self.agent.state.system_prompt = self._base_system_prompt
        except Exception:
            if preflight_result:
                preflight_result(False)
            raise

        if not messages:
            return

        if preflight_result:
            preflight_result(True)
        await self._run_agent_prompt(messages)

    async def _try_execute_extension_command(self, text: str) -> bool:
        """Execute a `/name args...` extension command. Returns whether one was found.

        Port of `_tryExecuteExtensionCommand`: a handler exception is reported through
        `ExtensionRunner.emit_error` rather than propagated, but the command still
        counts as "handled" (no prompt is sent to the LLM) -- matches TypeScript
        exactly.
        """
        space_index = text.find(" ")
        command_name = text[1:] if space_index == -1 else text[1:space_index]
        command = self._extension_runner.get_command(command_name)
        if command is None:
            return False

        ctx = self._extension_runner.create_command_context()
        args = "" if space_index == -1 else text[space_index + 1 :]
        try:
            await _maybe_await(command.handler(args, ctx))
        except Exception as err:
            self._extension_runner.emit_error(
                ExtensionError(extension_path=f"command:{command_name}", event="command", error=str(err))
            )
        return True

    def _throw_if_extension_command(self, text: str) -> None:
        """Raise if `text` is a registered extension command (cannot be queued).

        Port of `_throwIfExtensionCommand`, used by `steer()`/`follow_up()`.
        """
        space_index = text.find(" ")
        command_name = text[1:] if space_index == -1 else text[1:space_index]
        if self._extension_runner.get_command(command_name) is not None:
            raise RuntimeError(
                f'Extension command "/{command_name}" cannot be queued. '
                "Use prompt() or execute the command when not streaming."
            )

    async def _queue_steer(self, text: str, images: list[ImageContent] | None = None) -> None:
        self._steering_messages.append(text)
        self._emit_queue_update()
        content: list[TextContent | ImageContent] = [TextContent(text=text)]
        if images:
            content.extend(images)
        self.agent.steer(UserMessage(content=content, timestamp=now_ms()))

    async def _queue_follow_up(self, text: str, images: list[ImageContent] | None = None) -> None:
        self._follow_up_messages.append(text)
        self._emit_queue_update()
        content: list[TextContent | ImageContent] = [TextContent(text=text)]
        if images:
            content.extend(images)
        self.agent.follow_up(UserMessage(content=content, timestamp=now_ms()))

    async def steer(self, text: str, images: list[ImageContent] | None = None) -> None:
        """Queue a steering message while the agent is running.

        Raises if `text` is a registered extension command (`_throwIfExtensionCommand`):
        extension commands execute immediately and cannot be queued.
        """
        if text.startswith("/"):
            self._throw_if_extension_command(text)
        expanded_text = self._expand_skill_command(text)
        expanded_text = expand_prompt_template(expanded_text, list(self.prompt_templates))
        await self._queue_steer(expanded_text, images)

    async def follow_up(self, text: str, images: list[ImageContent] | None = None) -> None:
        """Queue a follow-up message, delivered once the agent would otherwise stop.

        Raises if `text` is a registered extension command (see `steer()`).
        """
        if text.startswith("/"):
            self._throw_if_extension_command(text)
        expanded_text = self._expand_skill_command(text)
        expanded_text = expand_prompt_template(expanded_text, list(self.prompt_templates))
        await self._queue_follow_up(expanded_text, images)

    async def send_custom_message(
        self,
        custom_type: str,
        content: str | list[TextContent | ImageContent] | None,
        display: bool,
        details: Any = None,
        *,
        trigger_turn: bool = False,
        deliver_as: Literal["steer", "followUp", "nextTurn"] | None = None,
    ) -> None:
        message = CustomMessage(
            custom_type=custom_type,
            content=content if content is not None else [],
            display=display,
            details=details,
            timestamp=now_ms(),
        )
        if deliver_as == "nextTurn":
            self._pending_next_turn_messages.append(message)
        elif self.is_streaming:
            if deliver_as == "followUp":
                self.agent.follow_up(message)
            else:
                self.agent.steer(message)
        elif trigger_turn:
            await self._run_agent_prompt(message)
        else:
            self.agent.state.messages.append(message)
            self.session_manager.append_custom_message_entry(custom_type, message.content, display, details)
            self._emit(MessageStartEvent(message=message))
            self._emit(MessageEndEvent(message=message))

    async def send_user_message(
        self,
        content: str | list[TextContent | ImageContent],
        *,
        deliver_as: Literal["steer", "followUp"] | None = None,
    ) -> None:
        if isinstance(content, str):
            text = content
            images: list[ImageContent] | None = None
        else:
            text_parts: list[str] = []
            image_parts: list[ImageContent] = []
            for part in content:
                if part.type == "text":
                    text_parts.append(part.text)
                else:
                    image_parts.append(part)
            text = "\n".join(text_parts)
            images = image_parts or None

        await self.prompt(
            text,
            expand_prompt_templates=False,
            streaming_behavior=deliver_as,
            images=images,
            source="extension",
        )

    def clear_queue(self) -> QueueSnapshot:
        steering = list(self._steering_messages)
        follow_up = list(self._follow_up_messages)
        self._steering_messages = []
        self._follow_up_messages = []
        self.agent.clear_all_queues()
        self._emit_queue_update()
        return QueueSnapshot(steering=steering, follow_up=follow_up)

    async def abort(self) -> None:
        """Abort current operation and wait for the agent to become idle."""
        self.abort_retry()
        self.agent.abort()
        await self.wait_for_idle()

    async def wait_for_idle(self) -> None:
        await self._idle_event.wait()

    # =========================================================================
    # Model Management
    # =========================================================================

    async def _emit_model_select(
        self, next_model: Model, previous_model: Model | None, source: ModelSelectSource
    ) -> None:
        """Port of `_emitModelSelect`."""
        if models_are_equal(previous_model, next_model):
            return
        await self._extension_runner.emit(
            ModelSelectEvent(model=next_model, previous_model=previous_model, source=source)
        )

    async def set_model(self, model: Model) -> None:
        """Set model directly. Raises if no auth is configured for the model."""
        if not await self._model_runtime.check_auth(model.provider):
            raise RuntimeError(f"No API key for {model.provider}/{model.id}")

        previous_model = self.model
        thinking_level = self._get_thinking_level_for_model_switch()
        self.agent.state.model = model
        self.session_manager.append_model_change(model.provider, model.id)
        self.settings_manager.set_default_model_and_provider(model.provider, model.id)
        self.set_thinking_level(thinking_level)
        await self._emit_model_select(model, previous_model, "set")

    async def cycle_model(self, direction: Literal["forward", "backward"] = "forward") -> ModelCycleResult | None:
        """Cycle to next/previous model. Uses scoped models (--models flag) if available."""
        if self._scoped_models:
            return await self._cycle_scoped_model(direction)
        return await self._cycle_available_model(direction)

    async def _cycle_scoped_model(self, direction: Literal["forward", "backward"]) -> ModelCycleResult | None:
        available_ids = {(m.provider, m.id) for m in self._model_runtime.get_available_snapshot()}
        scoped_models = [sm for sm in self._scoped_models if (sm.model.provider, sm.model.id) in available_ids]
        if len(scoped_models) <= 1:
            return None

        current_model = self.model
        current_index = next((i for i, sm in enumerate(scoped_models) if models_are_equal(sm.model, current_model)), -1)
        if current_index == -1:
            current_index = 0
        length = len(scoped_models)
        next_index = (current_index + 1) % length if direction == "forward" else (current_index - 1) % length
        next_scoped = scoped_models[next_index]
        thinking_level = self._get_thinking_level_for_model_switch(next_scoped.thinking_level)

        self.agent.state.model = next_scoped.model
        self.session_manager.append_model_change(next_scoped.model.provider, next_scoped.model.id)
        self.settings_manager.set_default_model_and_provider(next_scoped.model.provider, next_scoped.model.id)
        self.set_thinking_level(thinking_level)
        await self._emit_model_select(next_scoped.model, current_model, "cycle")

        return ModelCycleResult(model=next_scoped.model, thinking_level=self.thinking_level, is_scoped=True)

    async def _cycle_available_model(self, direction: Literal["forward", "backward"]) -> ModelCycleResult | None:
        available_models = self._model_runtime.get_available_snapshot()
        if len(available_models) <= 1:
            return None

        current_model = self.model
        current_index = next((i for i, m in enumerate(available_models) if models_are_equal(m, current_model)), -1)
        if current_index == -1:
            current_index = 0
        length = len(available_models)
        next_index = (current_index + 1) % length if direction == "forward" else (current_index - 1) % length
        next_model = available_models[next_index]

        thinking_level = self._get_thinking_level_for_model_switch()
        self.agent.state.model = next_model
        self.session_manager.append_model_change(next_model.provider, next_model.id)
        self.settings_manager.set_default_model_and_provider(next_model.provider, next_model.id)
        self.set_thinking_level(thinking_level)
        await self._emit_model_select(next_model, current_model, "cycle")

        return ModelCycleResult(model=next_model, thinking_level=self.thinking_level, is_scoped=False)

    # =========================================================================
    # Thinking Level Management
    # =========================================================================

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        available_levels = self.get_available_thinking_levels()
        effective_level = level if level in available_levels else self._clamp_thinking_level(level)

        previous_level = self.agent.state.thinking_level
        is_changing = effective_level != previous_level
        self.agent.state.thinking_level = effective_level

        if is_changing:
            self.session_manager.append_thinking_level_change(effective_level)
            if self.supports_thinking() or effective_level != "off":
                self.settings_manager.set_default_thinking_level(effective_level)
            self._emit(ThinkingLevelChangedEvent(level=effective_level))
            # TS uses `void this._extensionRunner.emit(...)` here (a floating
            # promise from a synchronous method); the Python equivalent is a
            # tracked fire-and-forget task, only schedulable when a loop runs.
            if self._extension_runner.has_handlers("thinking_level_select"):
                try:
                    task = asyncio.ensure_future(
                        self._extension_runner.emit(
                            ThinkingLevelSelectEvent(level=effective_level, previous_level=previous_level)
                        )
                    )
                except RuntimeError:
                    pass
                else:
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)

    def cycle_thinking_level(self) -> ThinkingLevel | None:
        if not self.supports_thinking():
            return None
        levels = self.get_available_thinking_levels()
        # A level the current model does not support behaves like TS's
        # `indexOf(...) === -1`: `(-1 + 1) % len` restarts the cycle at `levels[0]`.
        current_index = levels.index(self.thinking_level) if self.thinking_level in levels else -1
        next_level = levels[(current_index + 1) % len(levels)]
        self.set_thinking_level(next_level)
        return next_level

    def get_available_thinking_levels(self) -> list[ThinkingLevel]:
        if self.model is None:
            return list(_FALLBACK_THINKING_LEVELS)
        return get_supported_thinking_levels(self.model)

    def supports_thinking(self) -> bool:
        return bool(self.model and self.model.reasoning)

    def _get_thinking_level_for_model_switch(self, explicit_level: ThinkingLevel | None = None) -> ThinkingLevel:
        if explicit_level is not None:
            return explicit_level
        if not self.supports_thinking():
            return self.settings_manager.get_default_thinking_level() or DEFAULT_THINKING_LEVEL
        return self.thinking_level

    def _clamp_thinking_level(self, level: ThinkingLevel) -> ThinkingLevel:
        return clamp_thinking_level(self.model, level) if self.model else "off"

    # =========================================================================
    # Queue Mode Management
    # =========================================================================

    def sync_queue_modes_from_settings(self) -> None:
        self.agent.steering_mode = self.settings_manager.get_steering_mode()
        self.agent.follow_up_mode = self.settings_manager.get_follow_up_mode()

    def set_steering_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self.agent.steering_mode = mode
        self.settings_manager.set_steering_mode(mode)

    def set_follow_up_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self.agent.follow_up_mode = mode
        self.settings_manager.set_follow_up_mode(mode)

    # =========================================================================
    # Compaction
    # =========================================================================

    def _compaction_settings(self) -> CompactionSettings:
        settings = self.settings_manager.get_compaction_settings()
        return CompactionSettings(
            enabled=settings["enabled"],
            reserve_tokens=settings["reserveTokens"],
            keep_recent_tokens=settings["keepRecentTokens"],
        )

    def _retry_policy(self) -> RetryPolicy:
        settings = self.settings_manager.get_retry_settings()
        return RetryPolicy(
            enabled=settings["enabled"], max_retries=settings["maxRetries"], base_delay_ms=settings["baseDelayMs"]
        )

    def _summarization_retry_callbacks(
        self, source: SummarizationSource, reason: CompactionReason | None = None
    ) -> RetryCallbacks:
        """Retry policy + callbacks shared by compaction and branch-summary summarization calls."""

        async def on_retry_scheduled(attempt: int, max_attempts: int, delay_ms: float, error_message: str) -> None:
            self._emit(
                SummarizationRetryScheduledEvent(
                    attempt=attempt, max_attempts=max_attempts, delay_ms=delay_ms, error_message=error_message
                )
            )

        async def on_retry_attempt_start() -> None:
            self._emit(SummarizationRetryAttemptStartEvent(source=source, reason=reason))

        async def on_retry_finished(_success: bool, _attempt: int, _error_message: str | None) -> None:
            self._emit(SummarizationRetryFinishedEvent())

        return RetryCallbacks(
            on_retry_scheduled=on_retry_scheduled,
            on_retry_attempt_start=on_retry_attempt_start,
            on_retry_finished=on_retry_finished,
        )

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        """Manually compact the session context. Aborts current agent operation first.

        Emits `session_before_compact` (an extension may cancel or supply its own
        `CompactionResult`, skipping `compaction_compact`/LLM summarization entirely)
        and, after the compaction entry is saved, `session_compact` -- both with
        `reason="manual"` since this method is only reachable for manual compaction
        (the `threshold`/`overflow` reasons used by `_check_compaction`'s automatic
        path are not wired, see module docstring).
        """
        await self.abort()
        self._compaction_abort_controller = AbortController()
        self._emit(CompactionStartEvent(reason="manual"))

        try:
            model = self.model
            if model is None:
                raise RuntimeError(format_no_model_selected_message())
            await self._ensure_summarization_auth(model)

            path_entries = self.session_manager.get_branch()
            settings = self._compaction_settings()

            preparation = prepare_compaction(path_entries, settings)
            if preparation is None:
                last_entry = path_entries[-1] if path_entries else None
                if isinstance(last_entry, CompactionEntry):
                    raise RuntimeError("Already compacted")
                raise RuntimeError("Nothing to compact (session too small)")

            extension_compaction: CompactionResult | None = None
            from_extension = False
            if self._extension_runner.has_handlers("session_before_compact"):
                before_result = await self._extension_runner.emit(
                    SessionBeforeCompactEvent(
                        preparation=preparation,
                        branch_entries=path_entries,
                        reason="manual",
                        will_retry=False,
                        signal=self._compaction_abort_controller.signal,
                        custom_instructions=custom_instructions,
                    )
                )
                if isinstance(before_result, SessionBeforeCompactResult):
                    if before_result.cancel:
                        raise RuntimeError("Compaction cancelled")
                    if before_result.compaction is not None:
                        extension_compaction = before_result.compaction
                        from_extension = True

            if extension_compaction is not None:
                result = extension_compaction
            else:
                result = await compaction_compact(
                    preparation,
                    self.agent.stream_function,
                    model,
                    custom_instructions,
                    self._compaction_abort_controller.signal,
                    self.thinking_level,
                    self._retry_policy(),
                    self._summarization_retry_callbacks("compaction", "manual"),
                )

            if self._compaction_abort_controller.signal.aborted:
                raise RuntimeError("Compaction cancelled")

            self.session_manager.append_compaction(
                result.summary,
                result.first_kept_entry_id,
                result.tokens_before,
                result.details,
                from_extension,
                result.usage,
            )
            session_context = self.session_manager.build_session_context()
            self.agent.state.messages = session_context.messages
            estimated_tokens_after = _estimate_messages_tokens(session_context.messages)

            if self._extension_runner.has_handlers("session_compact"):
                saved_entry = next(
                    (
                        entry
                        for entry in reversed(self.session_manager.get_entries())
                        if isinstance(entry, CompactionEntry) and entry.summary == result.summary
                    ),
                    None,
                )
                if saved_entry is not None:
                    await self._extension_runner.emit(
                        SessionCompactEvent(
                            compaction_entry=saved_entry,
                            from_extension=from_extension,
                            reason="manual",
                            will_retry=False,
                        )
                    )

            compaction_result = replace(result, estimated_tokens_after=estimated_tokens_after)
            # compaction_end listeners may submit queued prompts, so expose idle state before notifying them.
            self._compaction_abort_controller = None
            self._emit(CompactionEndEvent(reason="manual", result=compaction_result, aborted=False, will_retry=False))
            return compaction_result
        except Exception as error:
            message = str(error)
            aborted = _is_abort_error(error)
            self._compaction_abort_controller = None
            self._emit(
                CompactionEndEvent(
                    reason="manual",
                    result=None,
                    aborted=aborted,
                    will_retry=False,
                    error_message=None if aborted else f"Compaction failed: {message}",
                )
            )
            raise
        finally:
            self._compaction_abort_controller = None

    def abort_compaction(self) -> None:
        if self._compaction_abort_controller is not None:
            self._compaction_abort_controller.abort()
        if self._auto_compaction_abort_controller is not None:
            self._auto_compaction_abort_controller.abort()

    def abort_branch_summary(self) -> None:
        if self._branch_summary_abort_controller is not None:
            self._branch_summary_abort_controller.abort()

    async def _check_compaction(self, assistant_message: AssistantMessage, skip_aborted_check: bool = True) -> bool:
        """Check if compaction is needed and run it. Called after agent_end and before prompt submission.

        Two cases:
        1. Recoverable failure: LLM returned context overflow or stopped below its desired output
           limit; remove the assistant message, compact, and auto-retry once.
        2. Threshold: context is over threshold, compact, no auto-retry (user continues manually).
        """
        settings = self._compaction_settings()
        if not settings.enabled:
            return False

        if skip_aborted_check and assistant_message.stop_reason == "aborted":
            return False

        model = self.model
        context_window = model.context_window if model else 0

        same_model = bool(
            model and assistant_message.provider == model.provider and assistant_message.model == model.id
        )

        compaction_entry = get_latest_compaction_entry(self.session_manager.get_branch())
        assistant_is_from_before_compaction = (
            compaction_entry is not None and assistant_message.timestamp <= _iso_to_epoch_ms(compaction_entry.timestamp)
        )
        if assistant_is_from_before_compaction:
            return False

        recoverable_length = same_model and is_recoverable_length(assistant_message, model.max_tokens if model else 0)
        if same_model and (is_context_overflow(assistant_message, context_window) or recoverable_length):
            will_retry = assistant_message.stop_reason != "stop"

            if not will_retry:
                return await self._run_auto_compaction("overflow", False)

            if self._overflow_recovery_attempted:
                self._emit(
                    CompactionEndEvent(
                        reason="overflow",
                        result=None,
                        aborted=False,
                        will_retry=False,
                        error_message=(
                            "Context overflow recovery failed after one compact-and-retry attempt. "
                            "Try reducing context or switching to a larger-context model."
                        ),
                    )
                )
                return False

            self._overflow_recovery_attempted = True
            messages = self.agent.state.messages
            if messages and getattr(messages[-1], "role", None) == "assistant":
                self.agent.state.messages = messages[:-1]
            return await self._run_auto_compaction("overflow", will_retry)

        direct_context_tokens = calculate_context_tokens(assistant_message.usage) if assistant_message.usage else 0
        if assistant_message.stop_reason == "error" or direct_context_tokens == 0:
            messages = self.agent.state.messages
            estimate = estimate_context_tokens(messages)
            if estimate.last_usage_index is None:
                return False
            usage_message = messages[estimate.last_usage_index]
            if (
                compaction_entry is not None
                and getattr(usage_message, "role", None) == "assistant"
                and usage_message.timestamp <= _iso_to_epoch_ms(compaction_entry.timestamp)
            ):
                return False
            context_tokens = estimate.tokens
        else:
            context_tokens = direct_context_tokens

        if should_compact(context_tokens, context_window, settings):
            return await self._run_auto_compaction("threshold", False)
        return False

    async def _run_auto_compaction(self, reason: Literal["overflow", "threshold"], will_retry: bool) -> bool:
        started = False
        try:
            model = self.model
            if model is None:
                return False
            await self._ensure_summarization_auth(model)

            path_entries = self.session_manager.get_branch()
            settings = self._compaction_settings()
            preparation = prepare_compaction(path_entries, settings)
            if preparation is None:
                return False

            self._emit(CompactionStartEvent(reason=reason))
            self._auto_compaction_abort_controller = AbortController()
            started = True

            extension_compaction: CompactionResult | None = None
            from_extension = False
            if self._extension_runner.has_handlers("session_before_compact"):
                before_result = await self._extension_runner.emit(
                    SessionBeforeCompactEvent(
                        preparation=preparation,
                        branch_entries=path_entries,
                        reason=reason,
                        will_retry=will_retry,
                        signal=self._auto_compaction_abort_controller.signal,
                        custom_instructions=None,
                    )
                )
                if isinstance(before_result, SessionBeforeCompactResult):
                    if before_result.cancel:
                        self._emit(CompactionEndEvent(reason=reason, result=None, aborted=True, will_retry=False))
                        return False
                    if before_result.compaction is not None:
                        extension_compaction = before_result.compaction
                        from_extension = True

            if extension_compaction is not None:
                result = extension_compaction
            else:
                result = await compaction_compact(
                    preparation,
                    self.agent.stream_function,
                    model,
                    None,
                    self._auto_compaction_abort_controller.signal,
                    self.thinking_level,
                    self._retry_policy(),
                    self._summarization_retry_callbacks("compaction", reason),
                )

            if self._auto_compaction_abort_controller.signal.aborted:
                self._emit(CompactionEndEvent(reason=reason, result=None, aborted=True, will_retry=False))
                return False

            self.session_manager.append_compaction(
                result.summary,
                result.first_kept_entry_id,
                result.tokens_before,
                result.details,
                from_extension,
                result.usage,
            )
            session_context = self.session_manager.build_session_context()
            self.agent.state.messages = session_context.messages
            estimated_tokens_after = _estimate_messages_tokens(session_context.messages)

            if self._extension_runner.has_handlers("session_compact"):
                saved_entry = next(
                    (
                        entry
                        for entry in reversed(self.session_manager.get_entries())
                        if isinstance(entry, CompactionEntry) and entry.summary == result.summary
                    ),
                    None,
                )
                if saved_entry is not None:
                    await self._extension_runner.emit(
                        SessionCompactEvent(
                            compaction_entry=saved_entry,
                            from_extension=from_extension,
                            reason=reason,
                            will_retry=will_retry,
                        )
                    )

            compaction_result = replace(result, estimated_tokens_after=estimated_tokens_after)
            self._emit(
                CompactionEndEvent(reason=reason, result=compaction_result, aborted=False, will_retry=will_retry)
            )

            if will_retry:
                messages = self.agent.state.messages
                last_message = messages[-1] if messages else None
                if (
                    last_message is not None
                    and getattr(last_message, "role", None) == "assistant"
                    and last_message.stop_reason in ("error", "length")
                ):
                    self.agent.state.messages = messages[:-1]
                return True

            # Auto-compaction can complete while follow-up/steering/custom messages are waiting.
            return self.agent.has_queued_messages()
        except Exception as error:
            error_message = str(error) or "compaction failed"
            aborted = _is_abort_error(error)
            if started:
                self._emit(
                    CompactionEndEvent(
                        reason=reason,
                        result=None,
                        aborted=aborted,
                        will_retry=False,
                        error_message=(
                            None
                            if aborted
                            else (
                                f"Context overflow recovery failed: {error_message}"
                                if reason == "overflow"
                                else f"Auto-compaction failed: {error_message}"
                            )
                        ),
                    )
                )
            return False
        finally:
            self._auto_compaction_abort_controller = None

    def set_auto_compaction_enabled(self, enabled: bool) -> None:
        self.settings_manager.set_compaction_enabled(enabled)

    @property
    def auto_compaction_enabled(self) -> bool:
        return self.settings_manager.get_compaction_enabled()

    # =========================================================================
    # Retry
    # =========================================================================

    def _is_retryable_error(self, message: AssistantMessage) -> bool:
        """Context overflow is handled by compaction, not retry."""
        if is_context_overflow(message, self.model.context_window if self.model else 0):
            return False
        return is_retryable_assistant_error(message)

    async def _prepare_retry(self, message: AssistantMessage) -> bool:
        """Prepare a retryable error for continuation with exponential backoff.

        Returns `True` if the caller should continue the agent, `False` otherwise.
        """
        settings = self.settings_manager.get_retry_settings()
        if not settings["enabled"]:
            return False

        self._retry_attempt += 1
        if self._retry_attempt > settings["maxRetries"]:
            # Preserve the completed attempt count so post-run handling can emit the final failure.
            self._retry_attempt -= 1
            return False

        delay_ms = settings["baseDelayMs"] * (2 ** (self._retry_attempt - 1))

        self._emit(
            AutoRetryStartEvent(
                attempt=self._retry_attempt,
                max_attempts=settings["maxRetries"],
                delay_ms=delay_ms,
                error_message=message.error_message or "Unknown error",
            )
        )

        messages = self.agent.state.messages
        if messages and getattr(messages[-1], "role", None) == "assistant":
            self.agent.state.messages = messages[:-1]

        self._retry_abort_controller = AbortController()
        try:
            await _sleep_abortable(delay_ms, self._retry_abort_controller.signal)
        except AbortError:
            attempt = self._retry_attempt
            self._retry_attempt = 0
            self._emit(AutoRetryEndEvent(success=False, attempt=attempt, final_error="Retry cancelled"))
            return False
        finally:
            self._retry_abort_controller = None

        return True

    def abort_retry(self) -> None:
        if self._retry_abort_controller is not None:
            self._retry_abort_controller.abort()

    @property
    def is_retrying(self) -> bool:
        return self._retry_abort_controller is not None

    @property
    def auto_retry_enabled(self) -> bool:
        return self.settings_manager.get_retry_enabled()

    def set_auto_retry_enabled(self, enabled: bool) -> None:
        self.settings_manager.set_retry_enabled(enabled)

    # =========================================================================
    # Bash Execution
    # =========================================================================

    async def execute_bash(
        self,
        command: str,
        on_chunk: Callable[[str], None] | None = None,
        *,
        exclude_from_context: bool = False,
        id: str | None = None,
        operations: BashOperations | None = None,
    ) -> BashResult:
        abort_controller = AbortController()
        self._bash_abort_controllers.add(abort_controller)

        prefix = self.settings_manager.get_shell_command_prefix()
        shell_path = self.settings_manager.get_shell_path()
        resolved_command = f"{prefix}\n{command}" if prefix else command

        def on_chunk_wrapper(delta: str) -> None:
            if on_chunk:
                on_chunk(delta)
            self._emit(BashExecutionUpdateEvent(id=id, delta=delta))

        try:
            result = await execute_bash_with_operations(
                resolved_command,
                self.session_manager.get_cwd(),
                operations if operations is not None else create_local_bash_operations(shell_path),
                on_chunk_wrapper,
                abort_controller.signal,
            )
            self.record_bash_result(command, result, exclude_from_context=exclude_from_context)
            return result
        finally:
            self._bash_abort_controllers.discard(abort_controller)

    def record_bash_result(self, command: str, result: BashResult, *, exclude_from_context: bool = False) -> None:
        """Record a bash execution result in session history.

        Used by `execute_bash` and by callers that run bash execution themselves.
        """
        bash_message = BashExecutionMessage(
            command=command,
            output=result.output,
            exit_code=result.exit_code,
            cancelled=result.cancelled,
            truncated=result.truncated,
            full_output_path=result.full_output_path,
            timestamp=now_ms(),
            exclude_from_context=exclude_from_context,
        )

        if self.is_streaming:
            # Defer adding to avoid breaking tool_call/tool_result ordering.
            self._pending_bash_messages.append(bash_message)
        else:
            self.agent.state.messages.append(bash_message)
            self.session_manager.append_message(bash_message)

    def abort_bash(self) -> None:
        for controller in list(self._bash_abort_controllers):
            controller.abort()

    @property
    def is_bash_running(self) -> bool:
        return len(self._bash_abort_controllers) > 0

    @property
    def has_pending_bash_messages(self) -> bool:
        return len(self._pending_bash_messages) > 0

    def _flush_pending_bash_messages(self) -> None:
        if not self._pending_bash_messages:
            return
        for bash_message in self._pending_bash_messages:
            self.agent.state.messages.append(bash_message)
            self.session_manager.append_message(bash_message)
        self._pending_bash_messages = []

    # =========================================================================
    # Session Management
    # =========================================================================

    def set_session_name(self, name: str) -> None:
        self.session_manager.append_session_info(name)
        event = SessionInfoChangedEvent(name=self.session_manager.get_session_name())
        self._emit(event)
        # TS uses `void this._extensionRunner.emit(event)` from this synchronous
        # method; the Python equivalent is a tracked fire-and-forget task, only
        # schedulable when a loop is running.
        if self._extension_runner.has_handlers("session_info_changed"):
            try:
                task = asyncio.ensure_future(
                    self._extension_runner.emit(
                        ExtSessionInfoChangedEvent(name=self.session_manager.get_session_name())
                    )
                )
            except RuntimeError:
                pass
            else:
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

    # =========================================================================
    # Tree Navigation
    # =========================================================================

    async def navigate_tree(
        self,
        target_id: str,
        *,
        summarize: bool = False,
        custom_instructions: str | None = None,
        replace_instructions: bool = False,
        label: str | None = None,
    ) -> NavigateTreeResult:
        """Navigate to a different node in the session tree, staying in the same session file."""
        if self.is_streaming:
            raise RuntimeError("Wait for the current response to finish before navigating the session tree.")

        old_leaf_id = self.session_manager.get_leaf_id()
        if target_id == old_leaf_id:
            return NavigateTreeResult(cancelled=False)

        if summarize and self.model is None:
            raise RuntimeError("No model available for summarization")

        target_entry = self.session_manager.get_entry(target_id)
        if target_entry is None:
            raise ValueError(f"Entry {target_id} not found")

        collected = collect_entries_for_branch_summary(self.session_manager, old_leaf_id, target_id)

        self._branch_summary_abort_controller = AbortController()
        try:
            summary_text: str | None = None
            summary_details: Any = None
            summary_usage: Usage | None = None
            extension_summary: dict[str, Any] | None = None
            from_extension = False

            if self._extension_runner.has_handlers("session_before_tree"):
                preparation = TreePreparation(
                    target_id=target_id,
                    old_leaf_id=old_leaf_id,
                    common_ancestor_id=collected.common_ancestor_id,
                    entries_to_summarize=list(collected.entries),
                    user_wants_summary=summarize,
                    custom_instructions=custom_instructions,
                    replace_instructions=replace_instructions,
                    label=label,
                )
                before_result = await self._extension_runner.emit(
                    SessionBeforeTreeEvent(
                        preparation=preparation,
                        signal=self._branch_summary_abort_controller.signal,
                    )
                )
                if isinstance(before_result, SessionBeforeTreeResult):
                    if before_result.cancel:
                        return NavigateTreeResult(cancelled=True)
                    if before_result.summary is not None and summarize:
                        extension_summary = before_result.summary
                        from_extension = True
                    if before_result.custom_instructions is not None:
                        custom_instructions = before_result.custom_instructions
                    if before_result.replace_instructions is not None:
                        replace_instructions = before_result.replace_instructions
                    if before_result.label is not None:
                        label = before_result.label

            if summarize and collected.entries and extension_summary is None:
                model = self.model
                assert model is not None
                await self._ensure_summarization_auth(model)
                branch_summary_settings = self.settings_manager.get_branch_summary_settings()
                summary_result = await generate_branch_summary(
                    collected.entries,
                    GenerateBranchSummaryOptions(
                        model=model,
                        signal=self._branch_summary_abort_controller.signal,
                        custom_instructions=custom_instructions,
                        replace_instructions=replace_instructions,
                        reserve_tokens=branch_summary_settings["reserveTokens"],
                        stream_fn=self.agent.stream_function,
                        retry=self._retry_policy(),
                        callbacks=self._summarization_retry_callbacks("branchSummary"),
                    ),
                )
                if summary_result.aborted:
                    return NavigateTreeResult(cancelled=True, aborted=True)
                if summary_result.error:
                    raise RuntimeError(summary_result.error)
                summary_text = summary_result.summary
                summary_usage = summary_result.usage
                summary_details = {
                    "readFiles": summary_result.read_files or [],
                    "modifiedFiles": summary_result.modified_files or [],
                }
            elif extension_summary is not None:
                summary_text = extension_summary.get("summary")
                summary_details = extension_summary.get("details")
                summary_usage = extension_summary.get("usage")

            new_leaf_id: str | None
            editor_text: str | None = None
            if isinstance(target_entry, SessionMessageEntry) and getattr(target_entry.message, "role", None) == "user":
                new_leaf_id = target_entry.parent_id
                editor_text = content_text(target_entry.message.content, "")
            elif isinstance(target_entry, CustomMessageEntry):
                new_leaf_id = target_entry.parent_id
                editor_text = content_text(target_entry.content, "")
            else:
                new_leaf_id = target_id

            summary_entry: BranchSummaryEntry | None = None
            if summary_text:
                summary_id = self.session_manager.branch_with_summary(
                    new_leaf_id, summary_text, summary_details, from_extension, summary_usage
                )
                entry = self.session_manager.get_entry(summary_id)
                assert isinstance(entry, BranchSummaryEntry)
                summary_entry = entry
                if label:
                    self.session_manager.append_label_change(summary_id, label)
            elif new_leaf_id is None:
                self.session_manager.reset_leaf()
            else:
                self.session_manager.branch(new_leaf_id)

            if label and not summary_text:
                self.session_manager.append_label_change(target_id, label)

            session_context = self.session_manager.build_session_context()
            self.agent.state.messages = session_context.messages

            await self._extension_runner.emit(
                SessionTreeEvent(
                    new_leaf_id=self.session_manager.get_leaf_id(),
                    old_leaf_id=old_leaf_id,
                    summary_entry=summary_entry,
                    from_extension=from_extension if summary_text else None,
                )
            )

            return NavigateTreeResult(editor_text=editor_text, cancelled=False, summary_entry=summary_entry)
        finally:
            self._branch_summary_abort_controller = None

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_user_messages_for_forking(self) -> list[dict[str, str]]:
        """All user messages from the session, for a fork/history selector."""
        result: list[dict[str, str]] = []
        for entry in self.session_manager.get_entries():
            if not isinstance(entry, SessionMessageEntry):
                continue
            if getattr(entry.message, "role", None) != "user":
                continue
            text = content_text(entry.message.content, "")
            if text:
                result.append({"entryId": entry.id, "text": text})
        return result

    def get_session_stats(self) -> SessionStats:
        """Aggregate over ALL session entries, including history compacted away.

        This makes token/cost totals reflect what was actually billed across
        the session, not just the current (post-compaction) branch.
        """
        user_messages = 0
        assistant_messages = 0
        tool_results = 0
        total_messages = 0
        tool_calls = 0
        usage_totals = create_usage_totals()

        for entry in self.session_manager.get_entries():
            if isinstance(entry, (BranchSummaryEntry, CompactionEntry)) and entry.usage:
                add_usage_to_totals(usage_totals, entry.usage)
            if not isinstance(entry, SessionMessageEntry):
                continue
            total_messages += 1
            message = entry.message
            role = getattr(message, "role", None)
            if role == "user":
                user_messages += 1
            elif role == "toolResult":
                tool_results += 1
                if message.usage:
                    add_usage_to_totals(usage_totals, message.usage)
            elif role == "assistant":
                assistant_messages += 1
                tool_calls += sum(1 for block in message.content if getattr(block, "type", None) == "toolCall")
                add_usage_to_totals(usage_totals, message.usage)

        return SessionStats(
            session_file=self.session_file,
            session_id=self.session_id,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            tool_calls=tool_calls,
            tool_results=tool_results,
            total_messages=total_messages,
            tokens=TokenStats(
                input=usage_totals.input,
                output=usage_totals.output,
                cache_read=usage_totals.cache_read,
                cache_write=usage_totals.cache_write,
                total=usage_totals.input + usage_totals.output + usage_totals.cache_read + usage_totals.cache_write,
            ),
            cost=usage_totals.cost,
            context_usage=self.get_context_usage(),
        )

    def get_context_usage(self) -> ContextUsage | None:
        model = self.model
        if model is None:
            return None
        context_window = model.context_window or 0
        if context_window <= 0:
            return None

        # After compaction, the last assistant usage reflects pre-compaction context size. We can
        # only trust usage from an assistant that responded after the latest compaction boundary --
        # if none exists yet, context token count is unknown until the next LLM response.
        branch_entries = self.session_manager.get_branch()
        latest_compaction = get_latest_compaction_entry(branch_entries)

        if latest_compaction is not None:
            compaction_index = _last_index_by_id(branch_entries, latest_compaction)
            has_post_compaction_usage = False
            for entry in reversed(branch_entries[compaction_index + 1 :]):
                if isinstance(entry, SessionMessageEntry) and getattr(entry.message, "role", None) == "assistant":
                    assistant = entry.message
                    if assistant.stop_reason not in ("aborted", "error"):
                        context_tokens = calculate_context_tokens(assistant.usage)
                        if context_tokens > 0:
                            has_post_compaction_usage = True
                            break
            if not has_post_compaction_usage:
                return ContextUsage(tokens=None, context_window=context_window, percent=None)

        estimate = estimate_context_tokens(self.messages)
        percent = (estimate.tokens / context_window) * 100
        return ContextUsage(tokens=estimate.tokens, context_window=context_window, percent=percent)

    async def export_to_html(self, output_path: str | None = None) -> str:
        """Not ported: depends on `export-html/index.ts`'s document assembly.

        The rest of `core/export-html/` *is* ported (`ansi_to_html`, `colors`,
        `tool_renderer`); what is missing is `exportSessionToHtml`, which
        stitches the transcript into `template.html`/`template.css`/
        `template.js` around vendored `marked` and `highlight.js` browser
        bundles. See the README's "Not ported, by decision" section.
        """
        del output_path
        raise NotImplementedError(
            "export_to_html is not ported: `exportSessionToHtml` assembles the HTML document from "
            "templates and vendored marked/highlight.js browser bundles. The ANSI-to-HTML converter, "
            "colour maths and tool renderer under core/export_html are ported."
        )

    def export_to_jsonl(self, output_path: str | None = None) -> str:
        """Export the current branch to a JSONL file, re-chaining `parent_id` linearly."""
        default_name = f"session-{_now_iso().replace(':', '-').replace('.', '-')}.jsonl"
        file_path = resolve_path(output_path or default_name, str(Path.cwd()))
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        header = {
            # The loader dispatches on "type"; without it the header is dropped
            # and the exported file reloads as an empty session.
            "type": "session",
            "id": self.session_manager.get_session_id(),
            "timestamp": _now_iso(),
            "cwd": self.session_manager.get_cwd(),
            "version": CURRENT_SESSION_VERSION,
        }

        lines = [json.dumps(header)]
        prev_id: str | None = None
        for entry in self.session_manager.get_branch():
            linear_entry = replace(entry, parent_id=prev_id)
            lines.append(json.dumps(_entry_to_raw(linear_entry)))
            prev_id = entry.id

        Path(file_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return file_path

    def get_last_assistant_text(self) -> str | None:
        last_assistant: AssistantMessage | None = None
        for message in reversed(self.messages):
            if getattr(message, "role", None) != "assistant":
                continue
            if message.stop_reason == "aborted" and not message.content:
                continue
            last_assistant = message
            break
        if last_assistant is None:
            return None
        text = "".join(block.text for block in last_assistant.content if getattr(block, "type", None) == "text")
        return text.strip() or None
