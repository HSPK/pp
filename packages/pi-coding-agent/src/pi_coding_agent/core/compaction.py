"""Coding-agent-specific compaction: cut-point detection and prepare/compact
orchestration operating on `pi_coding_agent.core.session_manager.SessionEntry`
(the coding agent's tree-based session, distinct from
`pi_agent.harness.session.types.Entry`'s linear one).

Port of `packages/coding-agent/src/core/compaction/compaction.ts` (969 lines),
`packages/coding-agent/src/core/compaction/utils.ts` (158 lines), and
`packages/coding-agent/src/core/compaction/branch-summarization.ts` (376
lines, ported at the bottom of this file since it reuses the same primitives
and is compaction-adjacent -- summarizing an abandoned tree branch on
navigation rather than compacting linear history).

Reuses `pi_agent.harness.compaction.compaction`/`.utils` wherever the underlying
logic operates only on `HarnessMessage`/`Usage` (types shared with
`session_manager.py`, since that module deliberately reuses
`pi_agent.harness.messages.HarnessMessage` for its own entries) rather than on
harness's own `Entry` tree:
  - `combine_usage`, `calculate_context_tokens`, `estimate_tokens`,
    `estimate_context_tokens`, `should_compact`, `CompactionSettings`,
    `DEFAULT_COMPACTION_SETTINGS`, `ContextUsageEstimate` -- pure functions over
    `Usage`/`list[HarnessMessage]`, reused unmodified (this module still exports
    aliases for the same names, matching the TS module's public surface).
  - `complete_simple_with_retries` (TS `completeSummarization`) -- the retry +
    cache-isolation choke point wrapping one LLM summarization call, reused
    unmodified.
  - `SUMMARIZATION_SYSTEM_PROMPT`, `create_file_ops`, `extract_file_ops_from_message`,
    `compute_file_lists`, `format_file_operations`, `serialize_conversation` from
    `.utils` -- reused unmodified.
  - `generate_summary`/`generate_summary_with_usage` delegate to the harness
    implementation. The harness version returns a `Result[..., CompactionError]`
    (its callers expect that), but the coding-agent TS `generateSummary`/`compact`
    always `throw` on failure -- so failures are re-raised here as the underlying
    `CompactionError` (itself an `Exception` subclass) instead of being returned,
    restoring the original throwing behavior instead of swallowing it.

Everything that walks `SessionEntry`'s tree (cut-point detection, turn-boundary
detection, file-operation extraction across a compaction boundary, and
prepare/compact orchestration) is a fresh, `session_manager.py`-specific port,
since harness's `Entry` is a different (linear-history) session representation
that these functions cannot operate on directly.

`apiKey`/`headers`/`env` parameters present in the TS signatures existed because
`compact()`/`generateSummary()` called `completeSimple(model, context, options)`
directly (the `@earendil-works/pi-ai/compat` helper). This port always goes
through a caller-supplied `stream_fn` (matching `pi_agent.agent_loop` and
`model_runtime.py`'s pattern), which already carries whatever auth/headers were
resolved by `ModelRuntime`/`provider_composer.py` -- so those parameters are
dropped here (documented boundary: no second round of apiKey/headers/env
threading).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pi_agent.harness.compaction.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    SUMMARIZATION_SYSTEM_PROMPT,
    CompactionSettings,
    ContextUsageEstimate,
    calculate_context_tokens,
    combine_usage,
    complete_simple_with_retries,
    estimate_context_tokens,
    estimate_tokens,
    should_compact,
)
from pi_agent.harness.compaction.compaction import generate_summary_with_usage as _harness_generate_summary_with_usage
from pi_agent.harness.compaction.utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)
from pi_agent.harness.messages import HarnessMessage, convert_to_llm
from pi_agent.types import StreamFn, ThinkingLevel
from pi_ai.types import Model, Usage
from pi_ai.utils.abort import AbortSignal
from pi_ai.utils.retry import RetryCallbacks, RetryPolicy

from .session_manager import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomMessageEntry,
    ReadonlySessionManager,
    SessionEntry,
    SessionMessageEntry,
    build_session_context,
    session_entry_to_context_messages,
)

__all__ = [
    "DEFAULT_COMPACTION_SETTINGS",
    "SUMMARIZATION_SYSTEM_PROMPT",
    "BranchPreparation",
    "BranchSummaryDetails",
    "BranchSummaryResult",
    "CollectEntriesResult",
    "CompactionDetails",
    "CompactionPreparation",
    "CompactionResult",
    "CompactionSettings",
    "ContextUsageEstimate",
    "FileOperations",
    "GenerateBranchSummaryOptions",
    "calculate_context_tokens",
    "collect_entries_for_branch_summary",
    "combine_usage",
    "compact",
    "compute_file_lists",
    "create_file_ops",
    "estimate_context_tokens",
    "estimate_tokens",
    "extract_file_operations",
    "find_cut_point",
    "find_turn_start_index",
    "generate_branch_summary",
    "generate_summary",
    "generate_summary_with_usage",
    "get_last_assistant_usage",
    "get_message_from_entry_for_compaction",
    "prepare_branch_entries",
    "prepare_compaction",
    "should_compact",
]


# ---------------------------------------------------------------------------
# File operation tracking (session-entry aware; delegates to harness `.utils`
# for the per-message extraction step)
# ---------------------------------------------------------------------------


@dataclass
class CompactionDetails:
    """Details stored in `CompactionEntry.details` for file tracking. Port of TS `CompactionDetails`."""

    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)


def extract_file_operations(
    messages: list[HarnessMessage], entries: list[SessionEntry], prev_compaction_index: int
) -> FileOperations:
    """Port of `extractFileOperations` -- seeds from the previous compaction's
    recorded file ops (unless it came from a hook) then extracts fresh ones
    from `messages`' tool calls."""
    file_ops = create_file_ops()

    if prev_compaction_index >= 0:
        prev_compaction = entries[prev_compaction_index]
        if isinstance(prev_compaction, CompactionEntry) and not prev_compaction.from_hook and prev_compaction.details:
            details = prev_compaction.details
            read_files = details.get("readFiles") if isinstance(details, dict) else None
            modified_files = details.get("modifiedFiles") if isinstance(details, dict) else None
            if isinstance(read_files, list):
                file_ops.read.update(f for f in read_files if isinstance(f, str))
            if isinstance(modified_files, list):
                file_ops.edited.update(f for f in modified_files if isinstance(f, str))

    for msg in messages:
        extract_file_ops_from_message(msg, file_ops)

    return file_ops


# ---------------------------------------------------------------------------
# Message extraction
# ---------------------------------------------------------------------------


def get_message_from_entry_for_compaction(entry: SessionEntry) -> HarnessMessage | None:
    """Port of `getMessageFromEntryForCompaction`. Compaction entries never
    contribute to LLM context (they're replaced by summaries), everything
    else defers to `session_entry_to_context_messages`."""
    if isinstance(entry, CompactionEntry):
        return None
    messages = session_entry_to_context_messages(entry)
    return messages[0] if messages else None


@dataclass
class CompactionResult:
    """Result from `compact()`. `SessionManager` adds id/parentId when saving. Port of TS `CompactionResult`."""

    summary: str
    first_kept_entry_id: str
    tokens_before: int
    estimated_tokens_after: int | None = None
    usage: Usage | None = None
    details: Any = None


def get_last_assistant_usage(entries: list[SessionEntry]) -> Usage | None:
    """Port of `getLastAssistantUsage` -- find the last valid assistant usage in session entries."""
    for entry in reversed(entries):
        from .session_manager import SessionMessageEntry

        if isinstance(entry, SessionMessageEntry):
            message = entry.message
            if (
                getattr(message, "role", None) == "assistant"
                and message.stop_reason not in ("aborted", "error")
                and message.usage
                and calculate_context_tokens(message.usage) > 0
            ):
                return message.usage
    return None


# ---------------------------------------------------------------------------
# Cut point detection
# ---------------------------------------------------------------------------


def _is_cut_point_message(message: HarnessMessage) -> bool:
    return message.role in ("user", "assistant", "bashExecution", "custom", "branchSummary", "compactionSummary")


def _is_turn_start_message(message: HarnessMessage) -> bool:
    return message.role in ("user", "bashExecution", "custom", "branchSummary", "compactionSummary")


def _is_turn_start_entry(entry: SessionEntry) -> bool:
    if isinstance(entry, CompactionEntry):
        return False
    return any(_is_turn_start_message(m) for m in session_entry_to_context_messages(entry))


def _find_valid_cut_points(entries: list[SessionEntry], start_index: int, end_index: int) -> list[int]:
    """Port of `findValidCutPoints` -- never cut at tool results (they must follow their tool call)."""
    cut_points: list[int] = []
    for i in range(start_index, end_index):
        entry = entries[i]
        if isinstance(entry, CompactionEntry):
            continue
        if any(_is_cut_point_message(m) for m in session_entry_to_context_messages(entry)):
            cut_points.append(i)
    return cut_points


def find_turn_start_index(entries: list[SessionEntry], entry_index: int, start_index: int) -> int:
    """Port of `findTurnStartIndex` -- find the turn-start entry containing `entry_index`, or -1."""
    for i in range(entry_index, start_index - 1, -1):
        if _is_turn_start_entry(entries[i]):
            return i
    return -1


@dataclass
class CutPointResult:
    first_kept_entry_index: int
    """Index of first entry to keep."""
    turn_start_index: int
    """Index of the turn-start entry for a split turn, or -1 if not splitting."""
    is_split_turn: bool
    """Whether this cut splits a turn (cut point is not itself a turn start)."""


def find_cut_point(
    entries: list[SessionEntry], start_index: int, end_index: int, keep_recent_tokens: int
) -> CutPointResult:
    """Port of `findCutPoint`.

    Walk backwards from newest, accumulating estimated message sizes, and cut
    at the closest valid cut point once `keep_recent_tokens` has been reached.
    Only considers entries between `start_index` and `end_index` (exclusive).
    """
    cut_points = _find_valid_cut_points(entries, start_index, end_index)

    if not cut_points:
        return CutPointResult(first_kept_entry_index=start_index, turn_start_index=-1, is_split_turn=False)

    accumulated_tokens = 0
    cut_index = cut_points[0]

    for i in range(end_index - 1, start_index - 1, -1):
        entry = entries[i]
        message_tokens = sum(estimate_tokens(m) for m in session_entry_to_context_messages(entry))
        if message_tokens == 0:
            continue
        accumulated_tokens += message_tokens

        if accumulated_tokens >= keep_recent_tokens:
            for candidate in cut_points:
                if candidate >= i:
                    cut_index = candidate
                    break
            break

    while cut_index > start_index:
        prev_entry = entries[cut_index - 1]
        if isinstance(prev_entry, CompactionEntry) or session_entry_to_context_messages(prev_entry):
            break
        cut_index -= 1

    cut_entry = entries[cut_index]
    starts_turn = _is_turn_start_entry(cut_entry)
    turn_start_index = -1 if starts_turn else find_turn_start_index(entries, cut_index, start_index)

    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start_index,
        is_split_turn=(not starts_turn) and turn_start_index != -1,
    )


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


async def generate_summary(
    current_messages: list[HarnessMessage],
    stream_fn: StreamFn,
    model: Model,
    reserve_tokens: int,
    signal: AbortSignal | None = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: ThinkingLevel | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> str:
    """Port of `generateSummary`. Raises on failure (matches TS `throw`)."""
    text, _usage = await generate_summary_with_usage(
        current_messages,
        stream_fn,
        model,
        reserve_tokens,
        signal,
        custom_instructions,
        previous_summary,
        thinking_level,
        retry,
        callbacks,
    )
    return text


async def generate_summary_with_usage(
    current_messages: list[HarnessMessage],
    stream_fn: StreamFn,
    model: Model,
    reserve_tokens: int,
    signal: AbortSignal | None = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: ThinkingLevel | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> tuple[str, Usage]:
    """Port of `generateSummaryWithUsage`. Raises on failure (matches TS `throw`);
    delegates the actual LLM call to `pi_agent.harness.compaction.compaction.generate_summary_with_usage`."""
    result = await _harness_generate_summary_with_usage(
        current_messages,
        stream_fn,
        model,
        reserve_tokens,
        signal,
        custom_instructions,
        previous_summary,
        thinking_level,
        retry,
        callbacks,
    )
    if not result.ok:
        raise result.error
    return result.value


_TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


async def _generate_turn_prefix_summary(
    messages: list[HarnessMessage],
    stream_fn: StreamFn,
    model: Model,
    reserve_tokens: int,
    signal: AbortSignal | None = None,
    thinking_level: ThinkingLevel | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> tuple[str, Usage]:
    """Port of `generateTurnPrefixSummary`."""
    from pi_ai.types import Context, SimpleStreamOptions, TextContent, UserMessage
    from pi_ai.utils.text import content_text

    max_tokens = int(0.5 * reserve_tokens)
    if model.max_tokens > 0:
        max_tokens = min(max_tokens, model.max_tokens)

    llm_messages = convert_to_llm(messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{_TURN_PREFIX_SUMMARIZATION_PROMPT}"
    summarization_messages = [UserMessage(content=[TextContent(text=prompt_text)])]

    options_kwargs: dict[str, Any] = {"max_tokens": max_tokens, "signal": signal}
    if model.reasoning and thinking_level and thinking_level != "off":
        options_kwargs["reasoning"] = thinking_level

    response = await complete_simple_with_retries(
        stream_fn,
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        SimpleStreamOptions(**options_kwargs),
        retry,
        callbacks,
    )
    if response.stop_reason == "error":
        raise RuntimeError(f"Turn prefix summarization failed: {response.error_message or 'Unknown error'}")

    return content_text(response.content), response.usage


# ---------------------------------------------------------------------------
# Compaction preparation (for extensions)
# ---------------------------------------------------------------------------


@dataclass
class CompactionPreparation:
    """Prepared inputs for a compaction run. Port of TS `CompactionPreparation`."""

    first_kept_entry_id: str
    """Id of first entry to keep."""
    messages_to_summarize: list[HarnessMessage]
    """Messages that will be summarized and discarded."""
    turn_prefix_messages: list[HarnessMessage]
    """Messages that will be turned into a turn prefix summary (if splitting)."""
    is_split_turn: bool
    tokens_before: int
    previous_summary: str | None
    """Summary from previous compaction, for iterative update."""
    file_ops: FileOperations
    """File operations extracted from `messages_to_summarize`."""
    settings: CompactionSettings
    """Compaction settings from settings.jsonl."""


def prepare_compaction(path_entries: list[SessionEntry], settings: CompactionSettings) -> CompactionPreparation | None:
    """Port of `prepareCompaction`. Returns `None` when there's nothing to compact
    (already compacted at the tip, or no history within the cut range)."""
    if path_entries and isinstance(path_entries[-1], CompactionEntry):
        return None

    prev_compaction_index = -1
    for i in range(len(path_entries) - 1, -1, -1):
        if isinstance(path_entries[i], CompactionEntry):
            prev_compaction_index = i
            break

    previous_summary: str | None = None
    boundary_start = 0
    if prev_compaction_index >= 0:
        prev_compaction = path_entries[prev_compaction_index]
        assert isinstance(prev_compaction, CompactionEntry)
        previous_summary = prev_compaction.summary
        first_kept_entry_index = next(
            (i for i, e in enumerate(path_entries) if e.id == prev_compaction.first_kept_entry_id), -1
        )
        boundary_start = first_kept_entry_index if first_kept_entry_index >= 0 else prev_compaction_index + 1
    boundary_end = len(path_entries)

    tokens_before = estimate_context_tokens(build_session_context(path_entries).messages).tokens

    cut_point = find_cut_point(path_entries, boundary_start, boundary_end, settings.keep_recent_tokens)

    first_kept_entry = path_entries[cut_point.first_kept_entry_index] if path_entries else None
    if first_kept_entry is None or not first_kept_entry.id:
        return None  # Session needs migration
    first_kept_entry_id = first_kept_entry.id

    history_end = cut_point.turn_start_index if cut_point.is_split_turn else cut_point.first_kept_entry_index

    messages_to_summarize: list[HarnessMessage] = []
    for i in range(boundary_start, history_end):
        msg = get_message_from_entry_for_compaction(path_entries[i])
        if msg is not None:
            messages_to_summarize.append(msg)

    turn_prefix_messages: list[HarnessMessage] = []
    if cut_point.is_split_turn:
        for i in range(cut_point.turn_start_index, cut_point.first_kept_entry_index):
            msg = get_message_from_entry_for_compaction(path_entries[i])
            if msg is not None:
                turn_prefix_messages.append(msg)

    if not messages_to_summarize and not turn_prefix_messages:
        return None

    file_ops = extract_file_operations(messages_to_summarize, path_entries, prev_compaction_index)

    if cut_point.is_split_turn:
        for msg in turn_prefix_messages:
            extract_file_ops_from_message(msg, file_ops)

    return CompactionPreparation(
        first_kept_entry_id=first_kept_entry_id,
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix_messages,
        is_split_turn=cut_point.is_split_turn,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        file_ops=file_ops,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Main compaction function
# ---------------------------------------------------------------------------


async def compact(
    preparation: CompactionPreparation,
    stream_fn: StreamFn,
    model: Model,
    custom_instructions: str | None = None,
    signal: AbortSignal | None = None,
    thinking_level: ThinkingLevel | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> CompactionResult:
    """Port of `compact`. Generate summaries for compaction using prepared data.

    `SessionManager` adds id/parentId when persisting the resulting entry.
    """
    if preparation.is_split_turn and preparation.turn_prefix_messages:
        history_text = "No prior history."
        history_usage: Usage | None = None
        if preparation.messages_to_summarize:
            history_text, history_usage = await generate_summary_with_usage(
                preparation.messages_to_summarize,
                stream_fn,
                model,
                preparation.settings.reserve_tokens,
                signal,
                custom_instructions,
                preparation.previous_summary,
                thinking_level,
                retry,
                callbacks,
            )
        prefix_text, prefix_usage = await _generate_turn_prefix_summary(
            preparation.turn_prefix_messages,
            stream_fn,
            model,
            preparation.settings.reserve_tokens,
            signal,
            thinking_level,
            retry,
            callbacks,
        )
        summary = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_text}"
        summary_usage = combine_usage(history_usage, prefix_usage) if history_usage is not None else prefix_usage
    else:
        summary, summary_usage = await generate_summary_with_usage(
            preparation.messages_to_summarize,
            stream_fn,
            model,
            preparation.settings.reserve_tokens,
            signal,
            custom_instructions,
            preparation.previous_summary,
            thinking_level,
            retry,
            callbacks,
        )

    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)

    if not preparation.first_kept_entry_id:
        raise RuntimeError("First kept entry has no id - session may need migration")

    return CompactionResult(
        summary=summary,
        first_kept_entry_id=preparation.first_kept_entry_id,
        tokens_before=preparation.tokens_before,
        usage=summary_usage,
        # A plain camelCase dict, matching the TypeScript on-disk shape: this
        # value is JSON-serialised straight into the session file, and the
        # readers above look up "readFiles"/"modifiedFiles".
        details={"readFiles": read_files, "modifiedFiles": modified_files},
    )


# ---------------------------------------------------------------------------
# Branch summarization (for tree navigation)
#
# Port of `packages/coding-agent/src/core/compaction/branch-summarization.ts`
# (376 lines). When navigating to a different point in the session tree, this
# generates a summary of the branch being left so context isn't lost.
# ---------------------------------------------------------------------------


@dataclass
class BranchSummaryResult:
    summary: str | None = None
    usage: Usage | None = None
    read_files: list[str] | None = None
    modified_files: list[str] | None = None
    aborted: bool = False
    error: str | None = None


@dataclass
class BranchSummaryDetails:
    """Details stored in `BranchSummaryEntry.details` for file tracking."""

    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)


@dataclass
class BranchPreparation:
    messages: list[HarnessMessage]
    """Messages extracted for summarization, in chronological order."""
    file_ops: FileOperations
    """File operations extracted from tool calls."""
    total_tokens: int
    """Total estimated tokens in messages."""


@dataclass
class CollectEntriesResult:
    entries: list[SessionEntry]
    """Entries to summarize, in chronological order."""
    common_ancestor_id: str | None
    """Common ancestor between old and new position, if any."""


@dataclass
class GenerateBranchSummaryOptions:
    """Port of TS `GenerateBranchSummaryOptions`.

    `apiKey`/`headers`/`env` are dropped -- see the module docstring's
    documented boundary on `stream_fn` carrying resolved auth/headers.
    """

    model: Model
    signal: AbortSignal | None = None
    custom_instructions: str | None = None
    replace_instructions: bool = False
    """If true, `custom_instructions` replaces the default prompt instead of being appended."""
    reserve_tokens: int = 16384
    """Tokens reserved for prompt + LLM response."""
    stream_fn: StreamFn | None = None
    retry: RetryPolicy | None = None
    callbacks: RetryCallbacks | None = None


def collect_entries_for_branch_summary(
    session: ReadonlySessionManager, old_leaf_id: str | None, target_id: str
) -> CollectEntriesResult:
    """Port of `collectEntriesForBranchSummary`.

    Walks from `old_leaf_id` back to the common ancestor with `target_id`,
    collecting entries along the way. Does NOT stop at compaction boundaries
    -- those are included and their summaries become context.
    """
    if not old_leaf_id:
        return CollectEntriesResult(entries=[], common_ancestor_id=None)

    old_path = {e.id for e in session.get_branch(old_leaf_id)}
    target_path = session.get_branch(target_id)

    common_ancestor_id: str | None = None
    for i in range(len(target_path) - 1, -1, -1):
        if target_path[i].id in old_path:
            common_ancestor_id = target_path[i].id
            break

    entries: list[SessionEntry] = []
    current: str | None = old_leaf_id
    while current and current != common_ancestor_id:
        entry = session.get_entry(current)
        if entry is None:
            break
        entries.append(entry)
        current = entry.parent_id

    entries.reverse()
    return CollectEntriesResult(entries=entries, common_ancestor_id=common_ancestor_id)


def _get_message_from_entry_for_branch_summary(entry: SessionEntry) -> HarnessMessage | None:
    """Port of branch-summarization.ts's local `getMessageFromEntry`.

    Unlike `get_message_from_entry_for_compaction`, this explicitly skips
    `toolResult` messages (their context lives in the assistant's tool call)
    and treats compaction/branch-summary entries as regular content (not
    excluded), matching the TS switch statement exactly.
    """
    if isinstance(entry, SessionMessageEntry):
        message = entry.message
        if getattr(message, "role", None) == "toolResult":
            return None
        return message
    if isinstance(entry, CustomMessageEntry | BranchSummaryEntry | CompactionEntry):
        messages = session_entry_to_context_messages(entry)
        return messages[0] if messages else None
    return None


def prepare_branch_entries(entries: list[SessionEntry], token_budget: int = 0) -> BranchPreparation:
    """Port of `prepareBranchEntries`.

    Walks entries from NEWEST to OLDEST, adding messages until the token
    budget is hit, so the most recent context is kept when the branch is too
    long. Also collects file operations from tool calls and from existing
    `branch_summary` entries' details (cumulative tracking across nested
    branch summaries) -- only from pi-generated summaries (`from_hook` is not
    `True`), not extension-generated ones.
    """
    messages: list[HarnessMessage] = []
    file_ops = create_file_ops()
    total_tokens = 0

    for entry in entries:
        if isinstance(entry, BranchSummaryEntry) and not entry.from_hook and entry.details:
            details = entry.details
            read_files = details.get("readFiles") if isinstance(details, dict) else None
            modified_files = details.get("modifiedFiles") if isinstance(details, dict) else None
            if isinstance(read_files, list):
                file_ops.read.update(f for f in read_files if isinstance(f, str))
            if isinstance(modified_files, list):
                file_ops.edited.update(f for f in modified_files if isinstance(f, str))

    for entry in reversed(entries):
        message = _get_message_from_entry_for_branch_summary(entry)
        if message is None:
            continue

        extract_file_ops_from_message(message, file_ops)
        tokens = estimate_tokens(message)

        if token_budget > 0 and total_tokens + tokens > token_budget:
            if isinstance(entry, CompactionEntry | BranchSummaryEntry) and total_tokens < token_budget * 0.9:
                messages.insert(0, message)
                total_tokens += tokens
            break

        messages.insert(0, message)
        total_tokens += tokens

    return BranchPreparation(messages=messages, file_ops=file_ops, total_tokens=total_tokens)


_BRANCH_SUMMARY_PREAMBLE = """The user explored a different conversation branch before returning here.
Summary of that exploration:

"""

_BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


async def generate_branch_summary(
    entries: list[SessionEntry], options: GenerateBranchSummaryOptions
) -> BranchSummaryResult:
    """Port of `generateBranchSummary`. Generate a summary of abandoned branch entries."""
    from pi_ai.types import Context, SimpleStreamOptions, TextContent, UserMessage
    from pi_ai.utils.text import content_text

    context_window = options.model.context_window or 128000
    token_budget = context_window - options.reserve_tokens

    preparation = prepare_branch_entries(entries, token_budget)

    if not preparation.messages:
        return BranchSummaryResult(summary="No content to summarize")

    llm_messages = convert_to_llm(preparation.messages)
    conversation_text = serialize_conversation(llm_messages)

    if options.replace_instructions and options.custom_instructions:
        instructions = options.custom_instructions
    elif options.custom_instructions:
        instructions = f"{_BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {options.custom_instructions}"
    else:
        instructions = _BRANCH_SUMMARY_PROMPT

    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}"
    summarization_messages = [UserMessage(content=[TextContent(text=prompt_text)])]

    if options.stream_fn is None:
        raise ValueError("generate_branch_summary requires a stream_fn")

    response = await complete_simple_with_retries(
        options.stream_fn,
        options.model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        SimpleStreamOptions(signal=options.signal, max_tokens=2048),
        options.retry,
        options.callbacks,
    )

    if response.stop_reason == "aborted":
        return BranchSummaryResult(aborted=True)
    if response.stop_reason == "error":
        return BranchSummaryResult(error=response.error_message or "Summarization failed")

    summary = content_text(response.content)
    summary = _BRANCH_SUMMARY_PREAMBLE + summary

    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)

    return BranchSummaryResult(
        summary=summary or "No summary generated",
        usage=response.usage,
        read_files=read_files,
        modified_files=modified_files,
    )
