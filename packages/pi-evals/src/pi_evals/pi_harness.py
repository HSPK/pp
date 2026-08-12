"""The Pi `AgentSession` adapter: a harness that runs a real coding-agent session.

Python port of `packages/evals/src/pi-harness.ts`.

Each run creates an isolated temporary project and agent directory, builds a
real `pi_coding_agent` session against them, executes the eval's prompt (or
prompt/reload sequence), converts the session transcript to harness transcript
events, snapshots the native Pi session JSONL as an artifact, and deletes the
temporary tree.

**Differences from the TypeScript, forced by this port's coding-agent API.**

- TypeScript calls `createAgentSessionServices(...)` then
  `createAgentSessionFromServices(...)`. Neither is ported: as
  `pi_coding_agent.core.agent_session_runtime`'s docstring records, the
  per-cwd `AgentSessionServices` bundle was dropped because
  `create_agent_session` already takes the model runtime, settings manager,
  resource loader and session manager as plain arguments. This module calls
  `create_agent_session` directly with exactly those pieces, which is the
  same construction path.
- TypeScript passes `resourceLoaderOptions: { systemPromptOverride }` so the
  loader can rewrite the custom system prompt. `ResourceLoaderOptions` here
  has no override callback, so `_EvalResourceLoader` subclasses
  `ResourceLoader` and overrides `get_system_prompt()`. The effect is
  identical: the override feeds `build_system_prompt`'s `custom_prompt`,
  which is what `transform_system_prompt` is expected to rewrite.
- `AgentSession.reload()` is **not ported** (extension reloading, session
  restart events and `resetApiProviders()` all live in the parts of
  `agent-session.ts` this port left out). A `{"type": "reload"}` step here
  reloads settings and resources and rebuilds the system prompt and tool
  registry -- see `reload_eval_session`. It does **not** pick up extensions
  the model just wrote, because `create_agent_session` never loads extensions
  at all (a documented boundary of `pi_coding_agent.core.sdk`).
- For the same reason the TypeScript isolation assertion
  (`extensionRunner.getExtensionPaths().length !== 0`) is expressed here as
  "the isolated agent and project directories contain no extension files",
  checked with `discover_extensions_in_dir`.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import shutil
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pi_ai import content_text
from pi_ai.types import Model
from pi_ai.utils.abort import AbortSignal
from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.extensions.loader import discover_extensions_in_dir
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager

from pi_evals.harness import (
    Harness,
    HarnessTimings,
    HarnessUsage,
    JsonValue,
    SimpleHarnessResult,
    TranscriptEvent,
    TranscriptMessageEvent,
    TranscriptToolCallEvent,
    TranscriptToolResultError,
    TranscriptToolResultEvent,
    create_harness,
    normalize_record,
    to_json_value,
)
from pi_evals.vitest_evals.artifacts import PI_SESSION_SNAPSHOT_ARTIFACT


@dataclass
class PromptStep:
    content: str
    type: Literal["prompt"] = "prompt"


@dataclass
class ReloadStep:
    type: Literal["reload"] = "reload"


PiCodingAgentStep = PromptStep | ReloadStep
PiCodingAgentInput = str | Sequence[PiCodingAgentStep]
"""Port of `PiCodingAgentInput`: one prompt, or a sequence of prompt/reload steps."""


@dataclass
class PiCodingAgentModelSelection:
    provider: str
    id: str


@dataclass
class PiCodingAgentOutputContext:
    """What an `output` transform receives.

    TypeScript destructures `{ response, session }`. `cwd` and `agent_dir` are
    added here because this port's `AgentSession` exposes neither the agent
    directory nor a loaded-extension list, and an eval that inspects what the
    model wrote into the isolated workspace needs both.
    """

    response: str
    session: AgentSession
    cwd: str
    agent_dir: str


OutputTransform = Callable[[PiCodingAgentOutputContext], JsonValue | Awaitable[JsonValue]]


@dataclass
class PiCodingAgentHarnessOptions:
    """Port of `PiCodingAgentHarnessOptions` merged with `PiCodingAgentHarnessWithOutput`."""

    name: str | None = None
    model: PiCodingAgentModelSelection | None = None
    no_tools: str | None = None
    """Pi's tool-disable configuration: `"all"` or `"builtin"`."""
    transform_system_prompt: Callable[[str], str] | None = None
    output: OutputTransform | None = None


def resolve_model_selection(
    explicit_model: PiCodingAgentModelSelection | None = None,
    environment: Mapping[str, str] | None = None,
) -> PiCodingAgentModelSelection:
    """Port of `resolveModelSelection`.

    An explicit harness model wins over the runner's `PI_PROVIDER`/`PI_MODEL`
    defaults; both halves of a selection must be present.
    """
    env = os.environ if environment is None else environment
    provider_source = explicit_model.provider if explicit_model is not None else env.get("PI_PROVIDER")
    id_source = explicit_model.id if explicit_model is not None else env.get("PI_MODEL")
    provider = provider_source.strip() if provider_source else None
    model_id = id_source.strip() if id_source else None
    if not provider or not model_id:
        raise ValueError("Select a harness model explicitly or set both PI_PROVIDER and PI_MODEL as defaults.")
    return PiCodingAgentModelSelection(provider=provider, id=model_id)


class _EvalResourceLoader(ResourceLoader):
    """A `ResourceLoader` whose custom system prompt can be overridden per run.

    Stands in for TypeScript's `DefaultResourceLoaderOptions.systemPromptOverride`
    callback, which this port's `ResourceLoaderOptions` does not have.
    """

    def __init__(self, options: ResourceLoaderOptions) -> None:
        super().__init__(options)
        self._system_prompt_override: str | None = None

    def set_system_prompt_override(self, prompt: str | None) -> None:
        self._system_prompt_override = prompt

    def get_system_prompt(self) -> str | None:
        if self._system_prompt_override is not None:
            return self._system_prompt_override
        return super().get_system_prompt()


def to_transcript_events(messages: Sequence[object]) -> list[TranscriptEvent]:
    """Port of `toTranscriptEvents`: session messages to normalized trace events."""
    events: list[TranscriptEvent] = []
    for message in messages:
        role = getattr(message, "role", None)
        if role == "user":
            events.append(TranscriptMessageEvent(role="user", content=content_text(message.content)))
        elif role == "assistant":
            text = content_text(message.content)
            if text:
                events.append(TranscriptMessageEvent(role="assistant", content=text))
            for part in message.content:
                if part.type == "toolCall":
                    events.append(
                        TranscriptToolCallEvent(
                            id=part.id,
                            name=part.name,
                            arguments=normalize_record(part.arguments),
                        )
                    )
        elif role == "toolResult":
            text = content_text(message.content)
            all_text = all(part.type == "text" for part in message.content)
            events.append(
                TranscriptToolResultEvent(
                    tool_call_id=message.tool_call_id,
                    name=message.tool_name,
                    content=text if all_text else to_json_value(message.content),
                    error=TranscriptToolResultError(message=text or "Tool failed") if message.is_error else None,
                )
            )
    return events


async def reload_eval_session(session: AgentSession) -> None:
    """The `{"type": "reload"}` step, standing in for `AgentSession.reload()`.

    Reloads settings and resources and rebuilds the system prompt and tool
    registry from them. See the module docstring: extension reloading has no
    counterpart in this port, so a reload cannot activate an extension the
    model just wrote.
    """
    await session.settings_manager.reload()
    session.sync_queue_modes_from_settings()
    session.resource_loader.reload()
    session.set_active_tools_by_name(session.get_active_tool_names())


async def prompt_agent(session: AgentSession, text: str, signal: AbortSignal | None) -> str:
    """Port of `promptAgent`: run one prompt and return its assistant text."""
    if signal is not None:
        signal.throw_if_aborted()
    previous_message_count = len(session.messages)
    await session.prompt(text)
    assistant = next(
        (
            message
            for message in reversed(session.messages[previous_message_count:])
            if getattr(message, "role", None) == "assistant"
        ),
        None,
    )
    if assistant is None:
        raise RuntimeError("Agent run completed without an assistant message.")
    if assistant.stop_reason != "stop":
        raise RuntimeError(
            assistant.error_message or f"Agent run ended with unexpected stop reason: {assistant.stop_reason}."
        )
    output = session.get_last_assistant_text()
    if not output:
        raise RuntimeError("Agent run produced no assistant text.")
    return output


def _has_pricing(model: Model) -> bool:
    cost = model.cost
    for rates in [cost, *(cost.tiers or [])]:
        if rates.input > 0 or rates.output > 0 or rates.cache_read > 0 or rates.cache_write > 0:
            return True
    return False


def _steps(input: PiCodingAgentInput) -> list[PiCodingAgentStep]:
    return [PromptStep(content=input)] if isinstance(input, str) else list(input)


async def _run_pi_coding_agent(
    input: PiCodingAgentInput,
    signal: AbortSignal | None,
    set_artifact: Callable[[str, JsonValue], None],
    options: PiCodingAgentHarnessOptions,
) -> SimpleHarnessResult:
    started_at = time.perf_counter()
    if signal is not None:
        signal.throw_if_aborted()
    selection = resolve_model_selection(options.model)

    # No `agent_dir` override: authentication deliberately comes from Pi's
    # normal agent directory (subscription credentials, provider API keys),
    # not from the isolated one the session below runs in.
    model_runtime = await ModelRuntime.create()
    model = model_runtime.get_model(selection.provider, selection.id)
    if model is None:
        raise RuntimeError(f"Eval model not found: {selection.provider}/{selection.id}")

    root = tempfile.mkdtemp(prefix="pi-eval-")
    cwd = os.path.join(root, "workspace")
    agent_dir = os.path.join(root, "agent")

    session: AgentSession | None = None
    session_manager: SessionManager | None = None
    result: SimpleHarnessResult | None = None
    failure: BaseException | None = None
    try:
        os.mkdir(cwd)
        os.mkdir(agent_dir)
        resource_loader = _EvalResourceLoader(ResourceLoaderOptions(cwd=cwd, agent_dir=agent_dir))
        resource_loader.reload()
        if signal is not None:
            signal.throw_if_aborted()
        session_manager = SessionManager.create(cwd, os.path.join(root, "sessions"))
        set_artifact("runId", session_manager.get_session_id())
        created = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=cwd,
                agent_dir=agent_dir,
                model_runtime=model_runtime,
                settings_manager=SettingsManager.in_memory(),
                resource_loader=resource_loader,
                session_manager=session_manager,
                model=model,
                thinking_level="off",
                no_tools=options.no_tools,
            )
        )
        session = created.session

        if options.transform_system_prompt is not None:
            transformed = options.transform_system_prompt(session.system_prompt)
            if not transformed.strip():
                raise ValueError("Transformed eval system prompt must not be empty.")
            resource_loader.set_system_prompt_override(transformed)
            await reload_eval_session(session)

        if signal is not None:
            signal.throw_if_aborted()
        if _discovered_extension_paths(cwd, agent_dir):
            raise RuntimeError("Expected an isolated eval session to start without extensions.")

        abort_task = _abort_session_when_signalled(signal, session)
        try:
            response: str | None = None
            for step in _steps(input):
                if step.type == "prompt":
                    response = await prompt_agent(session, step.content, signal)
                else:
                    await reload_eval_session(session)
            if response is None:
                raise ValueError("Pi eval input must include at least one prompt step.")
        finally:
            if abort_task is not None:
                await _settle_abort_task(abort_task)

        output: JsonValue | str = response
        if options.output is not None:
            transformed_output = options.output(
                PiCodingAgentOutputContext(response=response, session=session, cwd=cwd, agent_dir=agent_dir)
            )
            output = await transformed_output if inspect.isawaitable(transformed_output) else transformed_output

        stats = session.get_session_stats()
        metadata: dict[str, JsonValue] = {
            "cacheReadTokens": stats.tokens.cache_read,
            "cacheWriteTokens": stats.tokens.cache_write,
        }
        if _has_pricing(model):
            metadata["estimatedCostUsd"] = stats.cost
        result = SimpleHarnessResult(
            output=output,
            events=to_transcript_events(session.messages),
            usage=HarnessUsage(
                provider=model.provider,
                model=model.id,
                input_tokens=stats.tokens.input,
                output_tokens=stats.tokens.output,
                total_tokens=stats.tokens.total,
                tool_calls=stats.tool_calls,
                metadata=metadata,
            ),
        )
    except BaseException as error:
        failure = error

    cleanup_errors: list[BaseException] = []
    if session_manager is not None:
        try:
            session_path = session_manager.get_session_file()
            if session_path and os.path.exists(session_path):
                set_artifact(PI_SESSION_SNAPSHOT_ARTIFACT, Path(session_path).read_text(encoding="utf-8"))
        except OSError as error:
            cleanup_errors.append(error)
    if session is not None:
        try:
            session.dispose()
        except Exception as error:
            cleanup_errors.append(error)
    try:
        shutil.rmtree(root, ignore_errors=False)
    except OSError as error:
        cleanup_errors.append(error)

    if failure is not None:
        if cleanup_errors:
            raise ExceptionGroup("Agent run failed and cleanup also failed.", [failure, *cleanup_errors])
        raise failure
    if len(cleanup_errors) == 1:
        raise cleanup_errors[0]
    if len(cleanup_errors) > 1:
        raise ExceptionGroup("Agent cleanup failed.", cleanup_errors)

    assert result is not None
    result.timings = HarnessTimings(total_ms=(time.perf_counter() - started_at) * 1000)
    return result


def _abort_session_when_signalled(signal: AbortSignal | None, session: AgentSession) -> asyncio.Task[None] | None:
    """TypeScript adds an `abort` listener that calls `session.abort()` once.

    `AbortSignal` here has no listener registration, so the equivalent is a
    task waiting on the signal; `_settle_abort_task` awaits an abort that was
    already started and cancels the waiter otherwise.
    """
    if signal is None:
        return None

    async def abort_when_signalled() -> None:
        await signal.wait()
        await session.abort()

    return asyncio.ensure_future(abort_when_signalled())


async def _settle_abort_task(task: asyncio.Task[None]) -> None:
    if task.done():
        await task
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


def _discovered_extension_paths(cwd: str, agent_dir: str) -> list[str]:
    """The isolation check: an eval session must start with no extensions in scope."""
    return [
        *discover_extensions_in_dir(os.path.join(cwd, ".pi", "extensions")),
        *discover_extensions_in_dir(os.path.join(agent_dir, "extensions")),
    ]


def create_pi_coding_agent_harness(options: PiCodingAgentHarnessOptions | None = None) -> Harness:
    """Port of `createPiCodingAgentHarness`.

    TypeScript's two overloads (with and without an `output` transform) are one
    function here; `PiCodingAgentHarnessOptions.output` decides whether the
    harness output is the raw response string or a domain result.
    """
    resolved = options or PiCodingAgentHarnessOptions()

    async def run(
        *,
        input: PiCodingAgentInput,
        signal: AbortSignal | None,
        set_artifact: Callable[[str, JsonValue], None],
    ) -> SimpleHarnessResult:
        return await _run_pi_coding_agent(input, signal, set_artifact, resolved)

    return create_harness(name=resolved.name or "pi-coding-agent", run=run)


__all__ = [
    "PiCodingAgentHarnessOptions",
    "PiCodingAgentInput",
    "PiCodingAgentModelSelection",
    "PiCodingAgentOutputContext",
    "PromptStep",
    "ReloadStep",
    "create_pi_coding_agent_harness",
    "prompt_agent",
    "reload_eval_session",
    "resolve_model_selection",
    "to_transcript_events",
]
