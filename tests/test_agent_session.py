"""Tests for `pi_coding_agent.core.agent_session`.

Ported from `packages/coding-agent/test/suite/agent-session-prompt.test.ts` and
`packages/coding-agent/test/agent-session-stats.test.ts`, adapted to this
port's narrower surface (see `agent_session.py`'s module docstring for the
documented extension/theme/OAuth boundaries).

Every case in `agent-session-prompt.test.ts` has a counterpart here. The four
that previously carried a "needs the extension system" justification are now
ported for real, since `core/extensions/` was ported after that note was
written:
`test_dispatches_extension_commands_without_consuming_a_provider_response`,
`test_does_not_report_streaming_behavior_to_input_handlers_while_idle`,
`test_reports_streaming_behavior_to_input_handlers_while_streaming` and
`test_throws_when_prompted_during_manual_compaction`.

This module drives `AgentSession` with a scripted stream function (mirroring
`packages/pi-agent/tests/conftest.py`'s `scripted_stream_fn` pattern) rather
than any real provider, and a `ModelRuntime` built over an in-process fake
`openai_compatible_provider` with credentials written to a sandboxed
`tmp_path` (never the real `$HOME`). No test performs real network I/O, and
every `await` that could otherwise hang is wrapped in `asyncio.wait_for`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from pi_agent.agent import Agent, MutableAgentState
from pi_agent.harness.messages import CustomMessage
from pi_ai.providers import openai_compatible_provider
from pi_ai.types import (
    AssistantMessage,
    Cost,
    DoneEvent,
    ErrorEvent,
    ImageContent,
    Model,
    ModelCost,
    StartEvent,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ToolCall,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Usage,
    UserMessage,
    now_ms,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream

from pi_coding_agent.core.agent_session import AgentSession, NavigateTreeResult, parse_skill_block
from pi_coding_agent.core.compaction import CompactionResult
from pi_coding_agent.core.extensions.types import (
    BeforeAgentStartEventResult,
    Extension,
    InputEventResult,
    MessageEndEventResult,
    RegisteredCommand,
    SessionBeforeCompactResult,
    ToolCallEventResult,
    ToolResultEventResult,
)
from pi_coding_agent.core.model_resolver import ScopedModel
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import (
    PromptTemplate,
    ResourceLoader,
    ResourceLoaderOptions,
    Skill,
    create_synthetic_source_info,
)
from pi_coding_agent.core.session_manager import (
    CompactionEntry,
    CustomMessageEntry,
    ModelChangeEntry,
    SessionManager,
    SessionMessageEntry,
    ThinkingLevelChangeEntry,
)
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.core.usage_totals import UsageCostBreakdownEntry, get_usage_cost_breakdown

TEST_MODEL = Model(
    id="test-model",
    name="Test Model",
    api="openai-completions",
    provider="test",
    base_url="https://fake.example.com",
    context_window=1000,
    max_tokens=100,
)


# ---------------------------------------------------------------------------
# Scripted stream helpers (mirrors packages/pi-agent/tests/conftest.py).
# ---------------------------------------------------------------------------


def make_assistant_message(
    content: list, stop_reason: str = "stop", error_message: str | None = None
) -> AssistantMessage:
    return AssistantMessage(
        api=TEST_MODEL.api,
        provider=TEST_MODEL.provider,
        model=TEST_MODEL.id,
        content=content,
        usage=Usage(),
        stop_reason=stop_reason,
        error_message=error_message,
    )


def text_response(text: str) -> AssistantMessage:
    return make_assistant_message([TextContent(text=text)], stop_reason="stop")


def tool_call_response(*tool_calls: ToolCall) -> AssistantMessage:
    return make_assistant_message(list(tool_calls), stop_reason="toolUse")


def replay_stream(message: AssistantMessage) -> AssistantMessageEventStream:
    """Emit the protocol event sequence that produces ``message``."""
    stream = AssistantMessageEventStream()
    partial = message
    stream.push(StartEvent(partial=partial))

    for index, block in enumerate(message.content):
        if block.type == "text":
            stream.push(TextStartEvent(content_index=index, partial=partial))
            stream.push(TextDeltaEvent(content_index=index, delta=block.text, partial=partial))
            stream.push(TextEndEvent(content_index=index, content=block.text, partial=partial))
        elif block.type == "toolCall":
            stream.push(ToolCallStartEvent(content_index=index, partial=partial))
            stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=partial))

    if message.stop_reason in ("error", "aborted"):
        stream.push(ErrorEvent(reason=message.stop_reason, error=message))
    else:
        stream.push(DoneEvent(reason=message.stop_reason, message=message))
    stream.end()
    return stream


def scripted_stream_fn(responses: list[AssistantMessage | Callable]) -> Callable:
    """A ``StreamFn`` that replays ``responses`` one per call.

    Each entry is either an `AssistantMessage` or a callable taking the
    `(model, context, options)` call and returning an `AssistantMessage`
    (used when a response needs to inspect the context, e.g. counting tool
    results).
    """
    remaining = list(responses)
    calls: list = []

    def stream_fn(model, context, options=None):
        calls.append({"model": model, "context": context, "options": options})
        if not remaining:
            raise AssertionError("stream_fn called more times than there are scripted responses")
        entry = remaining.pop(0)
        message = entry(context) if callable(entry) else entry
        return replay_stream(message)

    stream_fn.calls = calls  # type: ignore[attr-defined]
    return stream_fn


def _fake_provider(extra_models: list[Model] | None = None) -> object:
    return openai_compatible_provider(
        provider_id="test",
        name="Fake Test Provider",
        base_url="https://fake.example.com",
        env_vars=["FAKE_TEST_API_KEY"],
        models=[
            Model(
                id="test-model",
                name="Test Model",
                api="openai-completions",
                context_window=1000,
                max_tokens=100,
                cost=ModelCost(input=0, output=0),
            ),
            *(extra_models or []),
        ],
    )


async def build_session(
    tmp_path: Path,
    *,
    responses: list[AssistantMessage | Callable] | None = None,
    custom_tools: dict | None = None,
    allowed_tool_names: list[str] | None = None,
    excluded_tool_names: list[str] | None = None,
    initial_active_tool_names: list[str] | None = None,
    settings=None,
    with_configured_auth: bool = True,
    persist_to_disk: bool = False,
    extensions: list | None = None,
    scoped_models: list | None = None,
    extra_provider_models: list[Model] | None = None,
    use_builtin_tools: bool = False,
) -> tuple[AgentSession, SessionManager, SettingsManager, Callable]:
    """Build an `AgentSession` wired to a scripted stream_fn, sandboxed entirely under `tmp_path`."""
    # `ModelRuntime.check_auth`/`has_configured_auth` only report "unconfigured"
    # (rather than "unknown provider") when the provider is actually registered;
    # an unregistered provider always resolves as unconfigured. So the
    # "no configured auth" case omits the fake provider entirely, matching how
    # `packages/coding-agent/test/suite/harness.ts`'s `withConfiguredAuth: false`
    # skips `modelRegistry.registerProvider(...)`.
    model_runtime = await asyncio.wait_for(
        ModelRuntime.create(
            agent_dir=tmp_path / "agent",
            providers=[_fake_provider(extra_provider_models)] if with_configured_auth else [],
        ),
        timeout=5,
    )
    if with_configured_auth:
        await asyncio.wait_for(model_runtime.login("test", "fake-key"), timeout=5)

    if persist_to_disk:
        session_manager = SessionManager.create(str(tmp_path), session_dir=str(tmp_path / "sessions"))
    else:
        session_manager = SessionManager.in_memory(str(tmp_path))
    settings_manager = SettingsManager.in_memory(settings)

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    resource_loader = ResourceLoader(ResourceLoaderOptions(cwd=str(tmp_path), agent_dir=str(agent_dir)))
    resource_loader.reload()

    stream_fn = scripted_stream_fn(responses or [])
    agent = Agent(
        stream_fn,
        initial_state=MutableAgentState(model=TEST_MODEL, system_prompt="You are a test assistant."),
    )

    base_tools_override = None if use_builtin_tools else (custom_tools if custom_tools is not None else {})
    session = AgentSession(
        agent=agent,
        session_manager=session_manager,
        settings_manager=settings_manager,
        cwd=str(tmp_path),
        resource_loader=resource_loader,
        model_runtime=model_runtime,
        custom_tools=custom_tools,
        allowed_tool_names=allowed_tool_names,
        excluded_tool_names=excluded_tool_names,
        initial_active_tool_names=initial_active_tool_names,
        base_tools_override=base_tools_override,
        extensions=extensions,
        scoped_models=scoped_models,
    )
    return session, session_manager, settings_manager, stream_fn


def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if getattr(part, "type", None) == "text")


def make_echo_tool():
    from pi_agent.types import AgentTool, AgentToolResult

    tool_runs: list[str] = []

    async def execute(tool_call_id, params, signal=None, on_update=None):
        text = params.get("text", "") if isinstance(params, dict) else ""
        tool_runs.append(text)
        return AgentToolResult(content=[TextContent(text=f"echo:{text}")], details={"text": text})

    tool = AgentTool(
        name="echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        label="Echo",
        execute=execute,
    )
    return tool, tool_runs


# ---------------------------------------------------------------------------
# Prompt characterization tests
# ---------------------------------------------------------------------------


async def test_prompts_while_idle_records_single_text_response(tmp_path):
    session, _sm, _stm, stream_fn = await build_session(tmp_path, responses=[text_response("hello")])
    try:
        await asyncio.wait_for(session.prompt("hi"), timeout=5)

        assert [getattr(m, "role", None) for m in session.messages] == ["user", "assistant"]
        assert message_text(session.messages[0]) == "hi"
        # TS: `expect(harness.getPendingResponseCount()).toBe(0)` -- the single
        # scripted response was consumed by exactly one provider call.
        assert len(stream_fn.calls) == 1
    finally:
        session.dispose()


async def test_tool_call_turn_waits_for_follow_up_response(tmp_path):
    echo_tool, tool_runs = make_echo_tool()
    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[
            tool_call_response(ToolCall(id="call-1", name="echo", arguments={"text": "hello"})),
            text_response("done"),
        ],
        custom_tools={"echo": echo_tool},
        allowed_tool_names=["echo"],
    )
    try:
        await asyncio.wait_for(session.prompt("start"), timeout=5)

        assert tool_runs == ["hello"]
        roles = [getattr(m, "role", None) for m in session.messages]
        assert roles == ["user", "assistant", "toolResult", "assistant"]
    finally:
        session.dispose()


async def test_multiple_tool_calls_from_one_response_continue_with_single_follow_up(tmp_path):
    from pi_agent.types import AgentTool, AgentToolResult

    tool_runs: list[str] = []

    def make_tool(name: str):
        async def execute(tool_call_id, params, signal=None, on_update=None):
            value = params.get("value", "") if isinstance(params, dict) else ""
            tool_runs.append(f"{name}:{value}")
            return AgentToolResult(content=[TextContent(text=f"{name}:{value}")], details={"value": value})

        return AgentTool(
            name=name,
            description=f"{name} tool",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
            label=name,
            execute=execute,
        )

    def follow_up_response(context):
        tool_results = [m for m in context.messages if getattr(m, "role", None) == "toolResult"]
        return text_response(f"tool results: {len(tool_results)}")

    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[
            tool_call_response(
                ToolCall(id="call-slow", name="slow", arguments={"value": "a"}),
                ToolCall(id="call-fast", name="fast", arguments={"value": "b"}),
            ),
            follow_up_response,
        ],
        custom_tools={"slow": make_tool("slow"), "fast": make_tool("fast")},
        allowed_tool_names=["slow", "fast"],
    )
    try:
        await asyncio.wait_for(session.prompt("run tools"), timeout=5)

        assert sorted(tool_runs) == ["fast:b", "slow:a"]
        tool_results = [m for m in session.messages if getattr(m, "role", None) == "toolResult"]
        assert len(tool_results) == 2
        assert getattr(session.messages[-1], "role", None) == "assistant"
    finally:
        session.dispose()


async def test_preserves_image_attachments_in_provider_context(tmp_path):
    saw_image = False

    def capture_response(context):
        nonlocal saw_image
        user = next((m for m in context.messages if getattr(m, "role", None) == "user"), None)
        content = getattr(user, "content", None)
        saw_image = bool(
            user is not None
            and not isinstance(content, str)
            and any(getattr(part, "type", None) == "image" for part in content)
        )
        return text_response("ok")

    session, _sm, _stm, _stream = await build_session(tmp_path, responses=[capture_response])
    try:
        await asyncio.wait_for(
            session.prompt("describe", images=[ImageContent(mime_type="image/png", data="ZmFrZQ==")]),
            timeout=5,
        )
        assert saw_image is True
    finally:
        session.dispose()


async def test_expands_skill_commands_before_sending_prompt(tmp_path):
    # TS spreads `createTestResourceLoader()` and overrides `getSkills`. Here the
    # real `ResourceLoader` discovers the skill from disk instead, so the whole
    # frontmatter-parse + discovery path runs rather than a differently-shaped
    # stub standing in for `LoadSkillsResult`.
    skill_dir = tmp_path / "agent" / "skills" / "test"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test\ndescription: Test skill\n---\n\n# Test Skill\n\nUse the skill body.",
        encoding="utf-8",
    )

    expanded_prompt = ""

    def capture_response(context):
        nonlocal expanded_prompt
        user = next((m for m in context.messages if getattr(m, "role", None) == "user"), None)
        expanded_prompt = message_text(user) if user is not None else ""
        return text_response("ok")

    session, _sm, _stm, _stream = await build_session(tmp_path, responses=[capture_response])
    try:
        assert [skill.name for skill in session._resource_loader.get_skills().skills] == ["test"]

        await asyncio.wait_for(session.prompt("/skill:test explain this"), timeout=5)

        assert '<skill name="test" location="' in expanded_prompt
        assert "Use the skill body." in expanded_prompt
        assert "explain this" in expanded_prompt
    finally:
        session.dispose()


async def test_expands_prompt_templates_before_sending_prompt(tmp_path):
    # As with the skill case above, drive the real `ResourceLoader` off disk
    # rather than substituting a `get_prompts` stub for it.
    prompts_dir = tmp_path / "agent" / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "review.md").write_text(
        "---\ndescription: Review template\n---\nReview this code: $1",
        encoding="utf-8",
    )

    expanded_prompt = ""

    def capture_response(context):
        nonlocal expanded_prompt
        user = next((m for m in context.messages if getattr(m, "role", None) == "user"), None)
        expanded_prompt = message_text(user) if user is not None else ""
        return text_response("ok")

    session, _sm, _stm, _stream = await build_session(tmp_path, responses=[capture_response])
    try:
        templates, _diagnostics = session._resource_loader.get_prompts()
        assert [t.name for t in templates] == ["review"]
        assert [t.description for t in templates] == ["Review template"]

        await asyncio.wait_for(session.prompt("/review src/index.ts"), timeout=5)

        assert expanded_prompt == "Review this code: src/index.ts"
    finally:
        session.dispose()


async def test_send_user_message_while_idle_triggers_turn(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path, responses=[text_response("response")])
    try:
        await asyncio.wait_for(session.send_user_message("from extension"), timeout=5)

        assert [getattr(m, "role", None) for m in session.messages] == ["user", "assistant"]
        assert message_text(session.messages[0]) == "from extension"
    finally:
        session.dispose()


async def test_throws_when_prompted_during_streaming_without_streaming_behavior(tmp_path):
    release_event = asyncio.Event()
    tool_started = asyncio.Event()

    from pi_agent.types import AgentTool, AgentToolResult

    async def execute(tool_call_id, params, signal=None, on_update=None):
        tool_started.set()
        await release_event.wait()
        return AgentToolResult(content=[TextContent(text="released")], details={})

    wait_tool = AgentTool(
        name="wait", description="Wait for release", parameters={"type": "object", "properties": {}}, execute=execute
    )

    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[
            tool_call_response(ToolCall(id="call-1", name="wait", arguments={})),
            text_response("done"),
        ],
        custom_tools={"wait": wait_tool},
        allowed_tool_names=["wait"],
    )
    try:
        prompt_task = asyncio.ensure_future(session.prompt("start"))
        await asyncio.wait_for(tool_started.wait(), timeout=5)

        with pytest.raises(RuntimeError, match="Specify streaming_behavior"):
            await asyncio.wait_for(session.prompt("second"), timeout=5)

        release_event.set()
        await asyncio.wait_for(prompt_task, timeout=5)
    finally:
        session.dispose()


async def test_throws_when_prompting_without_a_model(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path, responses=[text_response("hello")])
    try:
        session.agent.state.model = None
        with pytest.raises(RuntimeError, match="No model selected"):
            await asyncio.wait_for(session.prompt("hi"), timeout=5)
    finally:
        session.dispose()


async def test_throws_when_prompting_without_configured_auth(tmp_path):
    session, _sm, _stm, _stream = await build_session(
        tmp_path, responses=[text_response("hello")], with_configured_auth=False
    )
    try:
        with pytest.raises(RuntimeError, match="No API key found for test"):
            await asyncio.wait_for(session.prompt("hi"), timeout=5)
    finally:
        session.dispose()


async def test_prompt_raises_while_compaction_in_progress(tmp_path):
    """Direct exercise of the compaction-in-progress guard.

    `AgentSession.prompt` raises immediately if `_compaction_abort_controller`
    is set; the TS test races a real `compact()` call against a `prompt()`
    using an extension hook to hold compaction open (see module docstring for
    why that specific race isn't reproduced here).
    """
    from pi_ai.utils.abort import AbortController

    session, _sm, _stm, _stream = await build_session(tmp_path, responses=[])
    try:
        session._compaction_abort_controller = AbortController()
        with pytest.raises(RuntimeError, match="Cannot submit a prompt while compaction is in progress"):
            await asyncio.wait_for(session.prompt("third"), timeout=5)
    finally:
        session._compaction_abort_controller = None
        session.dispose()


async def test_dispatches_extension_commands_without_consuming_a_provider_response(tmp_path):
    command_runs: list[str] = []

    async def handler(args, ctx):
        command_runs.append(args)

    extension = Extension(
        path="testcmd.py",
        resolved_path="testcmd.py",
        commands={"testcmd": RegisteredCommand(name="testcmd", handler=handler, description="Test command")},
    )
    session, _sm, _stm, stream_fn = await build_session(
        tmp_path, responses=[text_response("should stay queued")], extensions=[extension]
    )
    try:
        await asyncio.wait_for(session.prompt("/testcmd hello world"), timeout=5)

        assert command_runs == ["hello world"]
        assert session.messages == []
        # TS asserts `harness.getPendingResponseCount() === 1`; the equivalent
        # here is that the one scripted response was never pulled off the queue.
        assert stream_fn.calls == []
    finally:
        session.dispose()


async def test_does_not_report_streaming_behavior_to_input_handlers_while_idle(tmp_path):
    input_events: list = []

    async def on_input(event, ctx):
        input_events.append(event)
        return None

    session, _sm, _stm, _stream = await build_session(
        tmp_path, responses=[text_response("ok")], extensions=[make_extension(input=on_input)]
    )
    try:
        await asyncio.wait_for(session.prompt("idle", streaming_behavior="followUp"), timeout=5)

        assert len(input_events) == 1
        assert input_events[0].streaming_behavior is None
    finally:
        session.dispose()


async def test_reports_streaming_behavior_to_input_handlers_while_streaming(tmp_path):
    from pi_agent.types import AgentTool, AgentToolResult

    release_event = asyncio.Event()
    tool_started = asyncio.Event()
    input_events: list = []

    async def execute(tool_call_id, params, signal=None, on_update=None):
        tool_started.set()
        await release_event.wait()
        return AgentToolResult(content=[TextContent(text="released")], details={})

    wait_tool = AgentTool(
        name="wait", description="Wait for release", parameters={"type": "object", "properties": {}}, execute=execute
    )

    async def on_input(event, ctx):
        input_events.append(event)
        return None

    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[
            tool_call_response(ToolCall(id="call-1", name="wait", arguments={})),
            text_response("done"),
        ],
        custom_tools={"wait": wait_tool},
        allowed_tool_names=["wait"],
        extensions=[make_extension(input=on_input)],
    )
    try:
        prompt_task = asyncio.ensure_future(session.prompt("start"))
        await asyncio.wait_for(tool_started.wait(), timeout=5)

        await asyncio.wait_for(session.prompt("queued", streaming_behavior="followUp"), timeout=5)

        assert [event.streaming_behavior for event in input_events] == [None, "followUp"]

        release_event.set()
        await asyncio.wait_for(prompt_task, timeout=5)
    finally:
        session.dispose()


async def test_throws_when_prompted_during_manual_compaction(tmp_path):
    """The TS race, reproduced with the (now ported) `session_before_compact` hook.

    `test_prompt_raises_while_compaction_in_progress` above pokes the guard
    directly; this drives the same guard the way TS does, through a real
    `compact()` held open mid-flight by an extension.
    """
    compaction_started = asyncio.Event()
    compaction_released = asyncio.Event()

    async def hold(event, ctx):
        compaction_started.set()
        await compaction_released.wait()
        return SessionBeforeCompactResult(
            compaction=CompactionResult(
                summary="manual compacted",
                first_kept_entry_id=event.preparation.first_kept_entry_id,
                tokens_before=event.preparation.tokens_before,
            )
        )

    extension = Extension(path="hold.py", resolved_path="hold.py", handlers={"session_before_compact": [hold]})
    session, session_manager, _stm, _stream = await build_session(
        tmp_path, settings=COMPACTION_SETTINGS, extensions=[extension]
    )
    try:
        _seed_compactable_history(session_manager, session)

        compact_task = asyncio.ensure_future(session.compact())
        await asyncio.wait_for(compaction_started.wait(), timeout=5)

        try:
            with pytest.raises(RuntimeError, match="Cannot submit a prompt while compaction is in progress"):
                await asyncio.wait_for(session.prompt("third"), timeout=5)
        finally:
            compaction_released.set()
            await asyncio.wait_for(compact_task, timeout=5)
    finally:
        session.dispose()


# ---------------------------------------------------------------------------
# Full cycle: prompt -> tool call -> tool result -> final answer, with
# transcript persistence to a real (tmp_path-sandboxed) session file.
# ---------------------------------------------------------------------------


async def test_full_prompt_tool_call_cycle_persists_transcript_to_disk(tmp_path):
    echo_tool, tool_runs = make_echo_tool()
    session, session_manager, _stm, _stream = await build_session(
        tmp_path,
        responses=[
            tool_call_response(ToolCall(id="call-1", name="echo", arguments={"text": "hello"})),
            text_response("all done"),
        ],
        custom_tools={"echo": echo_tool},
        allowed_tool_names=["echo"],
        persist_to_disk=True,
    )
    try:
        events = []
        session.subscribe(events.append)

        await asyncio.wait_for(session.prompt("please echo hello"), timeout=5)

        assert tool_runs == ["hello"]
        roles = [getattr(m, "role", None) for m in session.messages]
        assert roles == ["user", "assistant", "toolResult", "assistant"]
        assert message_text(session.messages[-1]) == "all done"

        # The transcript must be durably persisted: reload from the on-disk
        # session file (a fresh SessionManager instance, same file) and check
        # the same message sequence comes back.
        session_file = session_manager.get_session_file()
        assert session_file is not None
        assert Path(session_file).exists()

        reopened = SessionManager.open(session_file)
        reloaded_entries = [e for e in reopened.get_branch() if isinstance(e, SessionMessageEntry)]
        reloaded_roles = [getattr(e.message, "role", None) for e in reloaded_entries]
        assert reloaded_roles == ["user", "assistant", "toolResult", "assistant"]
        assert message_text(reloaded_entries[0].message) == "please echo hello"
        assert message_text(reloaded_entries[-1].message) == "all done"

        # Session-level event stream saw the expected lifecycle at least once.
        event_types = [e.type for e in events]
        assert "agent_start" in event_types
        assert "tool_execution_start" in event_types
        assert "tool_execution_end" in event_types
        assert "agent_settled" in event_types
    finally:
        session.dispose()


# ---------------------------------------------------------------------------
# Session stats / context usage (ported from agent-session-stats.test.ts)
# ---------------------------------------------------------------------------


def _usage(total_tokens: int) -> Usage:
    return Usage(input=total_tokens, output=0, cache_read=0, cache_write=0, total_tokens=total_tokens, cost=Cost())


def _assistant(text: str, total_tokens: int, timestamp: int) -> AssistantMessage:
    return AssistantMessage(
        api=TEST_MODEL.api,
        provider=TEST_MODEL.provider,
        model=TEST_MODEL.id,
        content=[TextContent(text=text)],
        usage=_usage(total_tokens),
        stop_reason="stop",
        timestamp=timestamp,
    )


def _user(text: str, timestamp: int) -> UserMessage:
    return UserMessage(content=text, timestamp=timestamp)


def _tool_result(usage: Usage) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id="tool-call-1",
        tool_name="test_tool",
        content=[TextContent(text="tool result")],
        usage=usage,
        is_error=False,
        timestamp=1,
    )


def _sync_agent_messages(session: AgentSession, session_manager: SessionManager) -> None:
    session.agent.state.messages = session_manager.build_session_context().messages


async def test_get_session_stats_exposes_current_context_usage(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path)
    try:
        session_manager.append_message(_user("hello", 1))
        session_manager.append_message(_assistant("hi", 200, 2))
        _sync_agent_messages(session, session_manager)

        stats = session.get_session_stats()
        assert stats.context_usage == session.get_context_usage()
        assert stats.context_usage.tokens == 200
        assert stats.context_usage.context_window == TEST_MODEL.context_window
        assert stats.context_usage.percent == (200 / TEST_MODEL.context_window) * 100
    finally:
        session.dispose()


async def test_reports_unknown_context_usage_immediately_after_compaction(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path)
    try:
        session_manager.append_message(_user("first", 1))
        session_manager.append_message(_assistant("response1", 180_000, 2))
        kept_user_id = session_manager.append_message(_user("second", 3))
        session_manager.append_message(_assistant("response2", 195_000, 4))
        session_manager.append_compaction("summary", kept_user_id, 195_000)
        session_manager.append_message(_user("third", 5))
        _sync_agent_messages(session, session_manager)

        stats = session.get_session_stats()
        # Totals cover ALL entries, including history compacted away (180k + 195k).
        assert stats.tokens.input == 375_000
        assert stats.context_usage is not None
        assert stats.context_usage.tokens is None
        assert stats.context_usage.percent is None
    finally:
        session.dispose()


async def test_uses_post_compaction_usage_for_current_context(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path)
    try:
        session_manager.append_message(_user("first", 1))
        session_manager.append_message(_assistant("response1", 180_000, 2))
        kept_user_id = session_manager.append_message(_user("second", 3))
        session_manager.append_message(_assistant("response2", 195_000, 4))
        session_manager.append_compaction("summary", kept_user_id, 195_000)
        session_manager.append_message(_user("third", 5))
        session_manager.append_message(_assistant("response3", 25_000, 6))
        _sync_agent_messages(session, session_manager)

        stats = session.get_session_stats()
        # Totals cover ALL entries, including history compacted away (180k + 195k + 25k).
        assert stats.tokens.input == 400_000
        assert stats.context_usage is not None
        assert stats.context_usage.tokens == 25_000
        assert stats.context_usage.percent == (25_000 / TEST_MODEL.context_window) * 100
    finally:
        session.dispose()


async def test_includes_branch_summary_usage_in_session_totals(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path)
    try:
        session_manager.branch_with_summary(
            None,
            "summary",
            None,
            False,
            Usage(input=10, output=20, cache_read=30, cache_write=40, total_tokens=100, cost=Cost(total=1)),
        )
        _sync_agent_messages(session, session_manager)

        stats = session.get_session_stats()
        assert stats.tokens.input == 10
        assert stats.tokens.output == 20
        assert stats.tokens.cache_read == 30
        assert stats.tokens.cache_write == 40
        assert stats.tokens.total == 100
        assert stats.cost == 1
    finally:
        session.dispose()


async def test_includes_compaction_usage_in_session_totals(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path)
    try:
        first_kept_entry_id = session_manager.append_message(_user("hello", 1))
        session_manager.append_compaction(
            "summary",
            first_kept_entry_id,
            100,
            None,
            False,
            Usage(input=10, output=20, cache_read=30, cache_write=40, total_tokens=100, cost=Cost(total=1)),
        )
        _sync_agent_messages(session, session_manager)

        stats = session.get_session_stats()
        assert stats.tokens.input == 10
        assert stats.tokens.output == 20
        assert stats.tokens.cache_read == 30
        assert stats.tokens.cache_write == 40
        assert stats.tokens.total == 100
        assert stats.cost == 1
    finally:
        session.dispose()


async def test_includes_tool_result_usage_in_session_totals(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path)
    try:
        session_manager.append_message(
            _tool_result(
                Usage(input=10, output=20, cache_read=30, cache_write=40, total_tokens=100, cost=Cost(total=1))
            )
        )
        _sync_agent_messages(session, session_manager)

        stats = session.get_session_stats()
        assert stats.tokens.input == 10
        assert stats.tokens.output == 20
        assert stats.tokens.cache_read == 30
        assert stats.tokens.cache_write == 40
        assert stats.tokens.total == 100
        assert stats.cost == 1
    finally:
        session.dispose()


def test_groups_tool_and_summary_usage_separately_from_model_attributed_usage(tmp_path):
    session_manager = SessionManager.in_memory(str(tmp_path))
    root_id = session_manager.append_message(_user("hello", 1))
    assistant = _assistant("response", 100, 2)
    assistant.usage.cost.total = 0.5
    session_manager.append_message(assistant)
    session_manager.append_message(
        _tool_result(Usage(input=100, output=0, cache_read=0, cache_write=0, total_tokens=100, cost=Cost(total=1)))
    )
    session_manager.append_compaction(
        "summary", root_id, 100, None, False, Usage(input=100, total_tokens=100, cost=Cost(total=2))
    )
    session_manager.branch_with_summary(
        None, "branch summary", None, False, Usage(input=100, total_tokens=100, cost=Cost(total=3))
    )

    breakdown = get_usage_cost_breakdown(session_manager.get_entries())
    # TS asserts `toEqual([...])`: exactly these two entries, in this order.
    assert breakdown == [
        UsageCostBreakdownEntry(key="Tools/summaries", cost=6, tokens=300),
        UsageCostBreakdownEntry(key=f"{TEST_MODEL.provider}/{TEST_MODEL.id}", cost=0.5, tokens=100),
    ]


async def test_ignores_zero_usage_messages_when_checking_post_compaction_context_usage(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path)
    try:
        session_manager.append_message(_user("first", 1))
        session_manager.append_message(_assistant("response1", 180_000, 2))
        kept_user_id = session_manager.append_message(_user("second", 3))
        session_manager.append_message(_assistant("response2", 195_000, 4))
        session_manager.append_compaction("summary", kept_user_id, 195_000)
        session_manager.append_message(_user("third", 5))
        session_manager.append_message(_assistant("response3", 25_000, 6))
        session_manager.append_message(_user("continue", 7))
        session_manager.append_message(_assistant("partial", 0, 8))
        _sync_agent_messages(session, session_manager)

        stats = session.get_session_stats()
        assert stats.context_usage is not None
        assert stats.context_usage.tokens is not None
        assert stats.context_usage.tokens > 25_000
    finally:
        session.dispose()


# ---------------------------------------------------------------------------
# Model and thinking-level management
# ---------------------------------------------------------------------------

REASONING_MODEL = Model(
    id="test-reasoning",
    name="Test Reasoning Model",
    api="openai-completions",
    provider="test",
    base_url="https://fake.example.com",
    context_window=1000,
    max_tokens=100,
    reasoning=True,
)


async def test_set_model_persists_choice_and_reclamps_thinking_level(tmp_path):
    session, session_manager, settings_manager, _stream = await build_session(tmp_path)
    try:
        events = []
        session.subscribe(events.append)

        await asyncio.wait_for(session.set_model(REASONING_MODEL), timeout=5)

        assert session.model is REASONING_MODEL
        assert settings_manager.get_default_provider() == "test"
        assert settings_manager.get_default_model() == "test-reasoning"
        model_changes = [e for e in session_manager.get_entries() if isinstance(e, ModelChangeEntry)]
        assert [(e.provider, e.model_id) for e in model_changes] == [("test", "test-reasoning")]
        # The previous (non-reasoning) model has no thinking support, so the switch
        # falls back to the settings default (DEFAULT_THINKING_LEVEL = "medium").
        assert session.thinking_level == "medium"
        assert [e.level for e in events if e.type == "thinking_level_changed"] == ["medium"]
    finally:
        session.dispose()


async def test_set_model_raises_when_provider_has_no_auth(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path, with_configured_auth=False)
    try:
        with pytest.raises(RuntimeError, match="No API key for test/test-model"):
            await asyncio.wait_for(session.set_model(TEST_MODEL), timeout=5)
        assert session.model is TEST_MODEL  # unchanged
    finally:
        session.dispose()


async def test_set_thinking_level_clamps_to_model_capabilities(tmp_path):
    session, session_manager, settings_manager, _stream = await build_session(tmp_path)
    try:
        events = []
        session.subscribe(events.append)

        # TEST_MODEL has reasoning=False, so "off" is the only available level:
        # requesting "high" clamps back to "off", which is already the current
        # level, so nothing is persisted and no event fires.
        assert session.get_available_thinking_levels() == ["off"]
        assert session.supports_thinking() is False
        session.set_thinking_level("high")

        assert session.thinking_level == "off"
        assert [e for e in events if e.type == "thinking_level_changed"] == []
        assert [e for e in session_manager.get_entries() if isinstance(e, ThinkingLevelChangeEntry)] == []
        assert settings_manager.get_default_thinking_level() is None
    finally:
        session.dispose()


async def test_set_thinking_level_emits_and_persists_for_reasoning_model(tmp_path):
    session, session_manager, settings_manager, _stream = await build_session(tmp_path)
    try:
        session.agent.state.model = REASONING_MODEL
        events = []
        session.subscribe(events.append)

        assert session.get_available_thinking_levels() == ["off", "minimal", "low", "medium", "high"]
        session.set_thinking_level("high")

        assert session.thinking_level == "high"
        assert [e.level for e in events if e.type == "thinking_level_changed"] == ["high"]
        levels = [e.thinking_level for e in session_manager.get_entries() if isinstance(e, ThinkingLevelChangeEntry)]
        assert levels == ["high"]
        assert settings_manager.get_default_thinking_level() == "high"

        # Setting the same level again is a no-op.
        session.set_thinking_level("high")
        assert [e for e in events if e.type == "thinking_level_changed"] == events[:1]
    finally:
        session.dispose()


async def test_cycle_thinking_level_requires_reasoning_support(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path)
    try:
        assert session.cycle_thinking_level() is None
        assert session.thinking_level == "off"
    finally:
        session.dispose()


async def test_cycle_thinking_level_wraps_around_available_levels(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path)
    try:
        session.agent.state.model = REASONING_MODEL
        levels = [session.cycle_thinking_level() for _ in range(6)]
        assert levels == ["minimal", "low", "medium", "high", "off", "minimal"]
    finally:
        session.dispose()


async def test_cycle_thinking_level_from_an_unavailable_level_starts_at_the_first(tmp_path):
    """Matches TS: `levels.indexOf(current)` is -1, so `(-1 + 1) % len` picks `levels[0]`."""
    session, _sm, _stm, _stream = await build_session(tmp_path)
    try:
        session.agent.state.model = REASONING_MODEL
        session.agent.state.thinking_level = "xhigh"  # not in this model's supported levels
        assert session.cycle_thinking_level() == "off"
    finally:
        session.dispose()


async def test_available_thinking_levels_fall_back_without_a_model(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path)
    try:
        session.agent.state.model = None
        assert session.model is None
        assert session.get_available_thinking_levels() == ["off", "minimal", "low", "medium", "high"]
        assert session._clamp_thinking_level("high") == "off"
    finally:
        session.dispose()


async def test_cycle_model_returns_none_with_a_single_available_model(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path)
    try:
        assert await asyncio.wait_for(session.cycle_model(), timeout=5) is None
    finally:
        session.dispose()


async def test_cycle_model_walks_available_models_forward_and_backward(tmp_path):
    session, session_manager, settings_manager, _stream = await build_session(
        tmp_path, extra_provider_models=[REASONING_MODEL]
    )
    try:
        forward = await asyncio.wait_for(session.cycle_model(), timeout=5)
        assert forward is not None
        assert forward.is_scoped is False
        assert forward.model.id == "test-reasoning"
        assert session.model.id == "test-reasoning"
        assert settings_manager.get_default_model() == "test-reasoning"

        backward = await asyncio.wait_for(session.cycle_model("backward"), timeout=5)
        assert backward is not None
        assert backward.model.id == "test-model"
        model_changes = [e.model_id for e in session_manager.get_entries() if isinstance(e, ModelChangeEntry)]
        assert model_changes == ["test-reasoning", "test-model"]
    finally:
        session.dispose()


async def test_cycle_model_prefers_scoped_models_and_their_thinking_level(tmp_path):
    scoped = [
        ScopedModel(model=TEST_MODEL),
        ScopedModel(model=REASONING_MODEL, thinking_level="low"),
    ]
    session, _sm, _stm, _stream = await build_session(
        tmp_path, extra_provider_models=[REASONING_MODEL], scoped_models=scoped
    )
    try:
        assert session.scoped_models == scoped
        result = await asyncio.wait_for(session.cycle_model(), timeout=5)
        assert result is not None
        assert result.is_scoped is True
        assert result.model.id == "test-reasoning"
        assert result.thinking_level == "low"
        assert session.thinking_level == "low"
    finally:
        session.dispose()


async def test_cycle_model_ignores_scoped_models_that_are_unavailable(tmp_path):
    """Only one of the two scoped models is registered, so there is nothing to cycle to."""
    scoped = [ScopedModel(model=TEST_MODEL), ScopedModel(model=REASONING_MODEL)]
    session, _sm, _stm, _stream = await build_session(tmp_path, scoped_models=scoped)
    try:
        assert await asyncio.wait_for(session.cycle_model(), timeout=5) is None
        session.set_scoped_models([])
        assert session.scoped_models == []
    finally:
        session.dispose()


# ---------------------------------------------------------------------------
# Tool registry / active tools
# ---------------------------------------------------------------------------


async def test_builtin_tool_registry_defaults_to_the_core_four(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path, use_builtin_tools=True)
    try:
        registry_names = {tool.name for tool in session.get_all_tools()}
        assert {"read", "bash", "edit", "write", "grep", "find", "ls"} <= registry_names
        assert session.get_active_tool_names() == ["read", "bash", "edit", "write"]
        # System prompt guidelines come from the per-tool contributions.
        edit_info = next(tool for tool in session.get_all_tools() if tool.name == "edit")
        assert edit_info.prompt_guidelines
        assert "oldText" in edit_info.prompt_guidelines[0]
        assert "Use read to examine files instead of cat or sed." in session.system_prompt
    finally:
        session.dispose()


async def test_excluded_tool_names_are_removed_from_the_registry(tmp_path):
    session, _sm, _stm, _stream = await build_session(
        tmp_path, use_builtin_tools=True, excluded_tool_names=["bash", "write"]
    )
    try:
        registry_names = {tool.name for tool in session.get_all_tools()}
        assert "bash" not in registry_names
        assert "write" not in registry_names
        assert session.get_active_tool_names() == ["read", "edit"]
    finally:
        session.dispose()


async def test_allowed_tool_names_activate_every_allowed_tool(tmp_path):
    session, _sm, _stm, _stream = await build_session(
        tmp_path, use_builtin_tools=True, allowed_tool_names=["read", "grep"], initial_active_tool_names=["read"]
    )
    try:
        assert {tool.name for tool in session.get_all_tools()} == {"read", "grep"}
        # Allowed-but-inactive tools are appended to the active set.
        assert session.get_active_tool_names() == ["read", "grep"]
    finally:
        session.dispose()


async def test_set_active_tools_by_name_ignores_unknown_names_and_rebuilds_prompt(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path, use_builtin_tools=True)
    try:
        session.set_active_tools_by_name(["ls", "nope", "read"])
        assert session.get_active_tool_names() == ["ls", "read"]
        assert "Use read to examine files instead of cat or sed." in session.system_prompt
        assert "Use write only for new files or complete rewrites." not in session.system_prompt
    finally:
        session.dispose()


# ---------------------------------------------------------------------------
# Queue modes, session name, forking helpers, exports
# ---------------------------------------------------------------------------


async def test_queue_modes_are_settable_and_syncable_from_settings(tmp_path):
    session, _sm, settings_manager, _stream = await build_session(tmp_path)
    try:
        session.set_steering_mode("one-at-a-time")
        session.set_follow_up_mode("one-at-a-time")
        assert session.steering_mode == "one-at-a-time"
        assert session.follow_up_mode == "one-at-a-time"
        assert settings_manager.get_steering_mode() == "one-at-a-time"
        assert settings_manager.get_follow_up_mode() == "one-at-a-time"

        session.agent.steering_mode = "all"
        session.agent.follow_up_mode = "all"
        session.sync_queue_modes_from_settings()
        assert session.steering_mode == "one-at-a-time"
        assert session.follow_up_mode == "one-at-a-time"
    finally:
        session.dispose()


async def test_set_session_name_emits_session_info_changed(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path)
    try:
        events = []
        session.subscribe(events.append)
        session.set_session_name("my session")

        assert session.session_name == "my session"
        assert [e.name for e in events if e.type == "session_info_changed"] == ["my session"]
    finally:
        session.dispose()


async def test_get_user_messages_for_forking_lists_user_entries(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path)
    try:
        first_id = session_manager.append_message(_user("first question", 1))
        session_manager.append_message(_assistant("answer", 10, 2))
        second_id = session_manager.append_message(_user("second question", 3))

        forking = session.get_user_messages_for_forking()
        assert forking == [
            {"entryId": first_id, "text": "first question"},
            {"entryId": second_id, "text": "second question"},
        ]
    finally:
        session.dispose()


async def test_export_to_jsonl_rewrites_the_branch_as_a_linear_chain(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path)
    try:
        session_manager.append_message(_user("hello", 1))
        session_manager.append_message(_assistant("hi", 10, 2))

        output = str(tmp_path / "exports" / "session.jsonl")
        written = session.export_to_jsonl(output)
        assert written == output

        lines = [json.loads(line) for line in Path(output).read_text(encoding="utf-8").splitlines()]
        assert lines[0]["type"] == "session"
        assert lines[0]["id"] == session.session_id
        assert [entry["type"] for entry in lines[1:]] == ["message", "message"]
        assert lines[1]["parentId"] is None
        assert lines[2]["parentId"] == lines[1]["id"]

        # The export must reload as a real session.
        reopened = SessionManager.open(output)
        roles = [getattr(e.message, "role", None) for e in reopened.get_branch() if isinstance(e, SessionMessageEntry)]
        assert roles == ["user", "assistant"]
    finally:
        session.dispose()


async def test_export_to_html_is_not_ported(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path)
    try:
        with pytest.raises(NotImplementedError, match="export_to_html is not ported"):
            await asyncio.wait_for(session.export_to_html(), timeout=5)
    finally:
        session.dispose()


async def test_get_last_assistant_text_skips_empty_aborted_messages(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path)
    try:
        assert session.get_last_assistant_text() is None

        aborted = make_assistant_message([], stop_reason="aborted")
        session.agent.state.messages = [_user("hi", 1), _assistant("  real answer  ", 10, 2), aborted]
        assert session.get_last_assistant_text() == "real answer"

        # An assistant message with only whitespace text yields None.
        session.agent.state.messages = [make_assistant_message([TextContent(text="   ")])]
        assert session.get_last_assistant_text() is None
    finally:
        session.dispose()


# ---------------------------------------------------------------------------
# Queueing: steer / follow-up / clear / abort mid-turn
# ---------------------------------------------------------------------------


def make_blocking_tool(name: str = "wait"):
    """A tool that blocks until released, or until the run is aborted."""
    from pi_agent.types import AgentTool, AgentToolResult

    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(tool_call_id, params, signal=None, on_update=None):
        started.set()
        waiters = [asyncio.ensure_future(release.wait())]
        if signal is not None:
            waiters.append(asyncio.ensure_future(signal.wait()))
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()
        return AgentToolResult(content=[TextContent(text="released")], details={})

    tool = AgentTool(
        name=name,
        description="Block until released",
        parameters={"type": "object", "properties": {}},
        label=name,
        execute=execute,
    )
    return tool, started, release


async def test_steering_message_is_queued_then_delivered_and_dequeued(tmp_path):
    wait_tool, started, release = make_blocking_tool()
    seen_users: list[list[str]] = []

    def final_response(context):
        seen_users.append([message_text(m) for m in context.messages if getattr(m, "role", None) == "user"])
        return text_response("done")

    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[tool_call_response(ToolCall(id="call-1", name="wait", arguments={})), final_response],
        custom_tools={"wait": wait_tool},
        allowed_tool_names=["wait"],
    )
    try:
        events = []
        session.subscribe(events.append)

        prompt_task = asyncio.ensure_future(session.prompt("start"))
        await asyncio.wait_for(started.wait(), timeout=5)

        await asyncio.wait_for(session.steer("please stop"), timeout=5)
        assert session.get_steering_messages() == ["please stop"]
        assert session.pending_message_count == 1
        queue_updates = [e for e in events if e.type == "queue_update"]
        assert queue_updates[-1].steering == ["please stop"]

        release.set()
        await asyncio.wait_for(prompt_task, timeout=5)

        assert seen_users == [["start", "please stop"]]
        assert session.get_steering_messages() == []
        assert session.pending_message_count == 0
        assert [e.steering for e in events if e.type == "queue_update"][-1] == []
    finally:
        session.dispose()


async def test_follow_up_message_is_delivered_after_the_turn_settles(tmp_path):
    wait_tool, started, release = make_blocking_tool()
    seen_users: list[list[str]] = []

    def final_response(context):
        seen_users.append([message_text(m) for m in context.messages if getattr(m, "role", None) == "user"])
        return text_response("done")

    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[
            tool_call_response(ToolCall(id="call-1", name="wait", arguments={})),
            text_response("first answer"),
            final_response,
        ],
        custom_tools={"wait": wait_tool},
        allowed_tool_names=["wait"],
    )
    try:
        prompt_task = asyncio.ensure_future(session.prompt("start"))
        await asyncio.wait_for(started.wait(), timeout=5)

        await asyncio.wait_for(session.prompt("and then this", streaming_behavior="followUp"), timeout=5)
        assert session.get_follow_up_messages() == ["and then this"]

        release.set()
        await asyncio.wait_for(prompt_task, timeout=5)

        assert seen_users == [["start", "and then this"]]
        assert session.get_follow_up_messages() == []
    finally:
        session.dispose()


async def test_prompt_during_streaming_queues_as_steering_by_default(tmp_path):
    wait_tool, started, release = make_blocking_tool()
    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[tool_call_response(ToolCall(id="call-1", name="wait", arguments={})), text_response("done")],
        custom_tools={"wait": wait_tool},
        allowed_tool_names=["wait"],
    )
    try:
        prompt_task = asyncio.ensure_future(session.prompt("start"))
        await asyncio.wait_for(started.wait(), timeout=5)

        preflight: list[bool] = []
        await asyncio.wait_for(
            session.prompt("mid-flight", streaming_behavior="steer", preflight_result=preflight.append), timeout=5
        )
        assert preflight == [True]
        assert session.get_steering_messages() == ["mid-flight"]

        release.set()
        await asyncio.wait_for(prompt_task, timeout=5)
    finally:
        session.dispose()


async def test_clear_queue_returns_and_empties_both_queues(tmp_path):
    wait_tool, started, release = make_blocking_tool()
    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[tool_call_response(ToolCall(id="call-1", name="wait", arguments={})), text_response("done")],
        custom_tools={"wait": wait_tool},
        allowed_tool_names=["wait"],
    )
    try:
        prompt_task = asyncio.ensure_future(session.prompt("start"))
        await asyncio.wait_for(started.wait(), timeout=5)

        await asyncio.wait_for(session.steer("steer me"), timeout=5)
        await asyncio.wait_for(session.follow_up("follow me"), timeout=5)

        snapshot = session.clear_queue()
        assert snapshot.steering == ["steer me"]
        assert snapshot.follow_up == ["follow me"]
        assert session.pending_message_count == 0

        release.set()
        await asyncio.wait_for(prompt_task, timeout=5)
    finally:
        session.dispose()


async def test_steer_expands_skills_and_templates(tmp_path):
    template = PromptTemplate(
        name="review",
        description="Review template",
        content="Review this code: $1",
        source_info=create_synthetic_source_info("/virtual/review.md", "local", scope="user"),
        file_path="/virtual/review.md",
    )
    session, _sm, _stm, _stream = await build_session(tmp_path)
    try:
        session._resource_loader.get_prompts = lambda: ([template], [])

        await asyncio.wait_for(session.steer("/review src/index.ts"), timeout=5)
        await asyncio.wait_for(session.follow_up("/review other.ts"), timeout=5)

        assert session.get_steering_messages() == ["Review this code: src/index.ts"]
        assert session.get_follow_up_messages() == ["Review this code: other.ts"]
    finally:
        session.dispose()


async def test_abort_mid_turn_settles_the_session(tmp_path):
    wait_tool, started, _release = make_blocking_tool()
    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[tool_call_response(ToolCall(id="call-1", name="wait", arguments={})), text_response("never")],
        custom_tools={"wait": wait_tool},
        allowed_tool_names=["wait"],
    )
    try:
        events = []
        session.subscribe(events.append)

        prompt_task = asyncio.ensure_future(session.prompt("start"))
        await asyncio.wait_for(started.wait(), timeout=5)
        assert session.is_streaming is True

        await asyncio.wait_for(session.abort(), timeout=5)
        await asyncio.wait_for(prompt_task, timeout=5)

        assert session.is_idle is True
        assert session.is_streaming is False
        assert any(e.type == "agent_settled" for e in events)
        await asyncio.wait_for(session.wait_for_idle(), timeout=5)
    finally:
        session.dispose()


# ---------------------------------------------------------------------------
# Custom / user messages
# ---------------------------------------------------------------------------


async def test_send_custom_message_while_idle_records_and_emits(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path)
    try:
        events = []
        session.subscribe(events.append)

        await asyncio.wait_for(
            session.send_custom_message("note", "a note", True, {"k": "v"}),
            timeout=5,
        )

        assert [getattr(m, "role", None) for m in session.messages] == ["custom"]
        assert session.messages[0].custom_type == "note"
        assert [e.type for e in events] == ["message_start", "message_end"]
        entries = [e for e in session_manager.get_entries() if isinstance(e, CustomMessageEntry)]
        assert [e.custom_type for e in entries] == ["note"]
    finally:
        session.dispose()


async def test_send_custom_message_for_next_turn_is_injected_with_the_prompt(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path, responses=[text_response("ok")])
    try:
        await asyncio.wait_for(
            session.send_custom_message("context", "extra context", False, deliver_as="nextTurn"),
            timeout=5,
        )
        assert session.messages == []

        await asyncio.wait_for(session.prompt("hi"), timeout=5)
        roles = [getattr(m, "role", None) for m in session.messages]
        assert roles == ["user", "custom", "assistant"]
    finally:
        session.dispose()


async def test_send_custom_message_triggers_a_turn_when_requested(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path, responses=[text_response("ok")])
    try:
        await asyncio.wait_for(
            session.send_custom_message("kickoff", "go", True, trigger_turn=True),
            timeout=5,
        )
        assert [getattr(m, "role", None) for m in session.messages] == ["custom", "assistant"]
    finally:
        session.dispose()


async def test_send_custom_message_during_streaming_is_queued(tmp_path):
    wait_tool, started, release = make_blocking_tool()
    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[tool_call_response(ToolCall(id="call-1", name="wait", arguments={})), text_response("done")],
        custom_tools={"wait": wait_tool},
        allowed_tool_names=["wait"],
    )
    try:
        prompt_task = asyncio.ensure_future(session.prompt("start"))
        await asyncio.wait_for(started.wait(), timeout=5)

        await asyncio.wait_for(session.send_custom_message("mid", "steered", True), timeout=5)
        release.set()
        await asyncio.wait_for(prompt_task, timeout=5)

        custom_messages = [m for m in session.messages if getattr(m, "role", None) == "custom"]
        assert [m.custom_type for m in custom_messages] == ["mid"]
    finally:
        session.dispose()


async def test_send_user_message_splits_text_and_images(tmp_path):
    seen: dict = {}

    def capture(context):
        user = next(m for m in context.messages if getattr(m, "role", None) == "user")
        seen["text"] = message_text(user)
        seen["images"] = [p for p in user.content if getattr(p, "type", None) == "image"]
        return text_response("ok")

    session, _sm, _stm, _stream = await build_session(tmp_path, responses=[capture])
    try:
        await asyncio.wait_for(
            session.send_user_message(
                [
                    TextContent(text="line one"),
                    TextContent(text="line two"),
                    ImageContent(mime_type="image/png", data="ZmFrZQ=="),
                ]
            ),
            timeout=5,
        )
        assert seen["text"] == "line one\nline two"
        assert len(seen["images"]) == 1
    finally:
        session.dispose()


# ---------------------------------------------------------------------------
# Bash execution
# ---------------------------------------------------------------------------


class _ScriptedBashOperations:
    """A `BashOperations` stand-in: emits scripted chunks, never spawns a process."""

    def __init__(self, chunks: list[bytes], exit_code: int | None = 0, block: bool = False) -> None:
        self.chunks = chunks
        self.exit_code = exit_code
        self.block = block
        self.commands: list[str] = []
        self.started = asyncio.Event()

    async def exec(self, command, cwd, on_data, signal, timeout, env):
        self.commands.append(command)
        self.started.set()
        for chunk in self.chunks:
            on_data(chunk)
        if self.block and signal is not None:
            await signal.wait()
            return None
        return self.exit_code


async def test_execute_bash_streams_chunks_and_records_the_result(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path)
    try:
        events = []
        session.subscribe(events.append)
        chunks: list[str] = []

        operations = _ScriptedBashOperations([b"hello ", b"world\n"])
        result = await asyncio.wait_for(
            session.execute_bash("echo hello world", chunks.append, id="bash-1", operations=operations),
            timeout=5,
        )

        assert result.output == "hello world\n"
        assert result.exit_code == 0
        assert result.cancelled is False
        assert chunks == ["hello ", "world\n"]
        updates = [e for e in events if e.type == "bash_execution_update"]
        assert [(e.id, e.delta) for e in updates] == [("bash-1", "hello "), ("bash-1", "world\n")]

        bash_messages = [m for m in session.messages if getattr(m, "role", None) == "bashExecution"]
        assert len(bash_messages) == 1
        assert bash_messages[0].command == "echo hello world"
        assert bash_messages[0].output == "hello world\n"
        persisted = [
            e
            for e in session_manager.get_entries()
            if isinstance(e, SessionMessageEntry) and getattr(e.message, "role", None) == "bashExecution"
        ]
        assert len(persisted) == 1
        assert session.is_bash_running is False
    finally:
        session.dispose()


async def test_execute_bash_applies_the_configured_shell_command_prefix(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path, settings={"shellCommandPrefix": "export FOO=1"})
    try:
        operations = _ScriptedBashOperations([b"ok\n"])
        await asyncio.wait_for(session.execute_bash("echo hi", operations=operations), timeout=5)
        assert operations.commands == ["export FOO=1\necho hi"]
    finally:
        session.dispose()


async def test_builtin_bash_tool_receives_the_configured_shell_prefix_and_shell_path(tmp_path):
    """TS `updateToolDefinitions` passes `{commandPrefix, shellPath}` into the bash tool.

    This port previously built the built-in tools with neither, so
    `shellCommandPrefix` applied only to user-run `!` commands and `shellPath`
    was ignored entirely -- the model's own bash calls silently ran under
    /bin/bash with no prefix.
    """
    session, _sm, settings_manager, _stream = await build_session(
        tmp_path,
        settings={"shellCommandPrefix": "export FROM_PREFIX=1", "shellPath": "/nonexistent/shell-xyz"},
        use_builtin_tools=True,
    )
    try:
        bash_tool = session._tool_registry["bash"]
        # `shellPath` reaching the tool is observable as the resolution error.
        with pytest.raises(RuntimeError, match="Custom shell path not found: /nonexistent/shell-xyz"):
            await bash_tool.execute("call-1", {"command": "echo hi"})

        settings_manager.set_shell_path(None)
        session._refresh_tool_registry()
        bash_tool = session._tool_registry["bash"]
        result = await bash_tool.execute("call-2", {"command": "echo $FROM_PREFIX"})
        assert "1" in "\n".join(c.text for c in result.content if c.type == "text")
    finally:
        session.dispose()


async def test_abort_bash_cancels_a_running_command(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path)
    try:
        operations = _ScriptedBashOperations([b"partial"], exit_code=None, block=True)
        task = asyncio.ensure_future(session.execute_bash("sleep 100", operations=operations))
        await asyncio.wait_for(operations.started.wait(), timeout=5)
        assert session.is_bash_running is True

        session.abort_bash()
        result = await asyncio.wait_for(task, timeout=5)
        assert result.cancelled is True
        assert session.is_bash_running is False
    finally:
        session.dispose()


async def test_bash_results_recorded_during_streaming_are_flushed_after_the_turn(tmp_path):
    from pi_coding_agent.core.bash_executor import BashResult

    wait_tool, started, release = make_blocking_tool()
    session, session_manager, _stm, _stream = await build_session(
        tmp_path,
        responses=[tool_call_response(ToolCall(id="call-1", name="wait", arguments={})), text_response("done")],
        custom_tools={"wait": wait_tool},
        allowed_tool_names=["wait"],
    )
    try:
        prompt_task = asyncio.ensure_future(session.prompt("start"))
        await asyncio.wait_for(started.wait(), timeout=5)

        session.record_bash_result("ls", BashResult(output="a\nb\n", exit_code=0, cancelled=False, truncated=False))
        assert session.has_pending_bash_messages is True
        assert [m for m in session.messages if getattr(m, "role", None) == "bashExecution"] == []

        release.set()
        await asyncio.wait_for(prompt_task, timeout=5)

        assert session.has_pending_bash_messages is False
        bash_messages = [m for m in session.messages if getattr(m, "role", None) == "bashExecution"]
        assert [m.command for m in bash_messages] == ["ls"]
        persisted = [
            e
            for e in session_manager.get_entries()
            if isinstance(e, SessionMessageEntry) and getattr(e.message, "role", None) == "bashExecution"
        ]
        assert len(persisted) == 1
    finally:
        session.dispose()


# ---------------------------------------------------------------------------
# Provider errors and auto-retry
# ---------------------------------------------------------------------------


def error_response(error_message: str) -> AssistantMessage:
    return make_assistant_message([], stop_reason="error", error_message=error_message)


async def test_non_retryable_provider_error_surfaces_without_retrying(tmp_path):
    session, _sm, _stm, _stream = await build_session(
        tmp_path, responses=[error_response("401 invalid api key")], settings={"retry": {"baseDelayMs": 0}}
    )
    try:
        events = []
        session.subscribe(events.append)

        await asyncio.wait_for(session.prompt("hi"), timeout=5)

        assistant = session.messages[-1]
        assert assistant.stop_reason == "error"
        assert assistant.error_message == "401 invalid api key"
        agent_end = next(e for e in events if e.type == "agent_end")
        assert agent_end.will_retry is False
        assert [e for e in events if e.type == "auto_retry_start"] == []
    finally:
        session.dispose()


async def test_retryable_provider_error_retries_and_reports_success(tmp_path):
    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[error_response("503 service unavailable"), text_response("recovered")],
        settings={"retry": {"enabled": True, "maxRetries": 2, "baseDelayMs": 0}},
    )
    try:
        events = []
        session.subscribe(events.append)

        await asyncio.wait_for(session.prompt("hi"), timeout=10)

        first_agent_end = next(e for e in events if e.type == "agent_end")
        assert first_agent_end.will_retry is True
        starts = [e for e in events if e.type == "auto_retry_start"]
        assert [(e.attempt, e.max_attempts, e.delay_ms, e.error_message) for e in starts] == [
            (1, 2, 0, "503 service unavailable")
        ]
        ends = [e for e in events if e.type == "auto_retry_end"]
        assert [(e.success, e.attempt) for e in ends] == [(True, 1)]
        # The failed assistant message is dropped before the retry.
        assert [getattr(m, "role", None) for m in session.messages] == ["user", "assistant"]
        assert message_text(session.messages[-1]) == "recovered"
        assert session.retry_attempt == 0
    finally:
        session.dispose()


async def test_retry_gives_up_after_max_retries(tmp_path):
    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[error_response("503 service unavailable"), error_response("503 service unavailable")],
        settings={"retry": {"enabled": True, "maxRetries": 1, "baseDelayMs": 0}},
    )
    try:
        events = []
        session.subscribe(events.append)

        await asyncio.wait_for(session.prompt("hi"), timeout=10)

        ends = [e for e in events if e.type == "auto_retry_end"]
        assert [(e.success, e.attempt, e.final_error) for e in ends] == [(False, 1, "503 service unavailable")]
        assert session.retry_attempt == 0
    finally:
        session.dispose()


async def test_retry_is_skipped_when_disabled_in_settings(tmp_path):
    session, _sm, settings_manager, _stream = await build_session(
        tmp_path,
        responses=[error_response("503 service unavailable")],
        settings={"retry": {"enabled": False}},
    )
    try:
        assert session.auto_retry_enabled is False
        events = []
        session.subscribe(events.append)

        await asyncio.wait_for(session.prompt("hi"), timeout=5)

        assert [e for e in events if e.type == "auto_retry_start"] == []
        assert next(e for e in events if e.type == "agent_end").will_retry is False

        session.set_auto_retry_enabled(True)
        assert settings_manager.get_retry_enabled() is True
    finally:
        session.dispose()


async def test_abort_retry_cancels_the_backoff_wait(tmp_path):
    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[error_response("503 service unavailable"), text_response("never reached")],
        settings={"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 30_000}},
    )
    try:
        events = []
        retry_started = asyncio.Event()

        def listener(event):
            events.append(event)
            if event.type == "auto_retry_start":
                retry_started.set()

        session.subscribe(listener)

        prompt_task = asyncio.ensure_future(session.prompt("hi"))
        await asyncio.wait_for(retry_started.wait(), timeout=5)
        assert session.is_retrying is True

        session.abort_retry()
        await asyncio.wait_for(prompt_task, timeout=5)

        ends = [e for e in events if e.type == "auto_retry_end"]
        assert [(e.success, e.attempt, e.final_error) for e in ends] == [(False, 1, "Retry cancelled")]
        assert session.is_retrying is False
    finally:
        session.dispose()


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

COMPACTION_SETTINGS = {"compaction": {"enabled": True, "keepRecentTokens": 1, "reserveTokens": 100}}


def _seed_compactable_history(session_manager: SessionManager, session: AgentSession) -> None:
    session_manager.append_message(_user("first question " * 20, 1))
    session_manager.append_message(_assistant("first answer " * 20, 500, 2))
    session_manager.append_message(_user("second question " * 20, 3))
    session_manager.append_message(_assistant("second answer " * 20, 600, 4))
    _sync_agent_messages(session, session_manager)


async def test_manual_compact_summarizes_and_rebuilds_context(tmp_path):
    session, session_manager, _stm, _stream = await build_session(
        tmp_path,
        responses=[text_response("SUMMARY OF HISTORY"), text_response("TURN PREFIX")],
        settings=COMPACTION_SETTINGS,
    )
    try:
        _seed_compactable_history(session_manager, session)
        events = []
        session.subscribe(events.append)

        result = await asyncio.wait_for(session.compact(), timeout=10)

        assert "SUMMARY OF HISTORY" in result.summary
        assert result.estimated_tokens_after is not None
        assert [e.reason for e in events if e.type == "compaction_start"] == ["manual"]
        end = next(e for e in events if e.type == "compaction_end")
        assert end.aborted is False
        assert end.error_message is None
        assert end.result is result or end.result.summary == result.summary

        compactions = [e for e in session_manager.get_entries() if isinstance(e, CompactionEntry)]
        assert len(compactions) == 1
        assert compactions[0].from_hook is False
        assert session.is_compacting is False
    finally:
        session.dispose()


async def test_manual_compact_raises_when_there_is_nothing_to_compact(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path, settings=COMPACTION_SETTINGS)
    try:
        events = []
        session.subscribe(events.append)

        with pytest.raises(RuntimeError, match="Nothing to compact"):
            await asyncio.wait_for(session.compact(), timeout=5)

        end = next(e for e in events if e.type == "compaction_end")
        assert end.result is None
        assert end.aborted is False
        assert end.error_message == "Compaction failed: Nothing to compact (session too small)"
    finally:
        session.dispose()


async def test_manual_compact_raises_when_already_compacted(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path, settings=COMPACTION_SETTINGS)
    try:
        kept_id = session_manager.append_message(_user("hello", 1))
        session_manager.append_compaction("previous summary", kept_id, 100)
        _sync_agent_messages(session, session_manager)

        with pytest.raises(RuntimeError, match="Already compacted"):
            await asyncio.wait_for(session.compact(), timeout=5)
    finally:
        session.dispose()


async def test_manual_compact_raises_without_a_model(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path, settings=COMPACTION_SETTINGS)
    try:
        _seed_compactable_history(session_manager, session)
        session.agent.state.model = None

        with pytest.raises(RuntimeError, match="No model selected"):
            await asyncio.wait_for(session.compact(), timeout=5)
    finally:
        session.dispose()


async def test_auto_compaction_enabled_flag_round_trips_through_settings(tmp_path):
    session, _sm, settings_manager, _stream = await build_session(tmp_path)
    try:
        assert session.auto_compaction_enabled is True
        session.set_auto_compaction_enabled(False)
        assert session.auto_compaction_enabled is False
        assert settings_manager.get_compaction_enabled() is False
    finally:
        session.dispose()


async def test_check_compaction_is_a_no_op_when_disabled(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path, settings={"compaction": {"enabled": False}})
    try:
        _seed_compactable_history(session_manager, session)
        message = _assistant("done", 900, 5)
        assert await asyncio.wait_for(session._check_compaction(message), timeout=5) is False
    finally:
        session.dispose()


async def test_check_compaction_skips_aborted_responses(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path, settings=COMPACTION_SETTINGS)
    try:
        _seed_compactable_history(session_manager, session)
        aborted = _assistant("partial", 900, 5)
        aborted.stop_reason = "aborted"
        assert await asyncio.wait_for(session._check_compaction(aborted), timeout=5) is False
    finally:
        session.dispose()


async def test_check_compaction_ignores_messages_from_before_the_last_compaction(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path, settings=COMPACTION_SETTINGS)
    try:
        kept_id = session_manager.append_message(_user("hello", 1))
        session_manager.append_compaction("previous summary", kept_id, 100)
        _sync_agent_messages(session, session_manager)

        # timestamp 1 predates the compaction entry, so the message is stale.
        stale = _assistant("old answer", 900, 1)
        assert await asyncio.wait_for(session._check_compaction(stale), timeout=5) is False
    finally:
        session.dispose()


async def test_repeated_overflow_recovery_reports_a_terminal_failure(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path, settings=COMPACTION_SETTINGS)
    try:
        _seed_compactable_history(session_manager, session)
        session._overflow_recovery_attempted = True
        events = []
        session.subscribe(events.append)

        overflow = make_assistant_message([], stop_reason="error", error_message="prompt is too long")
        overflow.timestamp = now_ms()
        assert await asyncio.wait_for(session._check_compaction(overflow), timeout=5) is False

        end = next(e for e in events if e.type == "compaction_end")
        assert end.reason == "overflow"
        assert end.error_message is not None
        assert "Context overflow recovery failed after one compact-and-retry attempt" in end.error_message
    finally:
        session.dispose()


async def test_auto_compaction_runs_on_threshold_with_extension_events(tmp_path):
    """The automatic path emits `session_before_compact`/`session_compact`, like the manual one."""
    hook_calls: list[str] = []

    async def record(event, ctx):
        hook_calls.append(event.type)
        return None

    extension = Extension(
        path="threshold-ext.py",
        resolved_path="threshold-ext.py",
        handlers={"session_before_compact": [record], "session_compact": [record]},
    )

    big_answer = make_assistant_message([TextContent(text="a long answer " * 30)])
    big_answer.usage = Usage(input=900, output=10, total_tokens=910, cost=Cost())

    session, session_manager, _stm, _stream = await build_session(
        tmp_path,
        responses=[big_answer, text_response("SUMMARY OF HISTORY"), text_response("TURN PREFIX")],
        settings=COMPACTION_SETTINGS,
        extensions=[extension],
    )
    try:
        session_manager.append_message(_user("earlier question " * 20, 1))
        session_manager.append_message(_assistant("earlier answer " * 20, 400, 2))
        _sync_agent_messages(session, session_manager)

        events = []
        session.subscribe(events.append)

        await asyncio.wait_for(session.prompt("another question"), timeout=10)

        assert [e.reason for e in events if e.type == "compaction_start"] == ["threshold"]
        end = next(e for e in events if e.type == "compaction_end")
        assert end.reason == "threshold"
        assert end.will_retry is False
        assert end.result is not None
        assert hook_calls == ["session_before_compact", "session_compact"]
        assert len([e for e in session_manager.get_entries() if isinstance(e, CompactionEntry)]) == 1
    finally:
        session.dispose()


async def test_manual_compact_can_be_cancelled_by_an_extension(tmp_path):
    async def cancel(event, ctx):
        return SessionBeforeCompactResult(cancel=True)

    extension = Extension(
        path="cancel-ext.py", resolved_path="cancel-ext.py", handlers={"session_before_compact": [cancel]}
    )
    session, session_manager, _stm, _stream = await build_session(
        tmp_path, settings=COMPACTION_SETTINGS, extensions=[extension]
    )
    try:
        _seed_compactable_history(session_manager, session)
        events = []
        session.subscribe(events.append)

        with pytest.raises(RuntimeError, match="Compaction cancelled"):
            await asyncio.wait_for(session.compact(), timeout=5)

        end = next(e for e in events if e.type == "compaction_end")
        assert end.aborted is True
        assert end.error_message is None
        assert [e for e in session_manager.get_entries() if isinstance(e, CompactionEntry)] == []
    finally:
        session.dispose()


async def test_extension_supplied_compaction_skips_summarization(tmp_path):
    seen_compact_events: list = []

    async def supply(event, ctx):
        return SessionBeforeCompactResult(
            compaction=CompactionResult(
                summary="EXTENSION SUMMARY",
                first_kept_entry_id=event.preparation.first_kept_entry_id,
                tokens_before=event.preparation.tokens_before,
            )
        )

    async def on_compact(event, ctx):
        seen_compact_events.append(event)
        return None

    extension = Extension(
        path="supply-ext.py",
        resolved_path="supply-ext.py",
        handlers={"session_before_compact": [supply], "session_compact": [on_compact]},
    )
    # No scripted responses at all: an LLM summarization call would raise.
    session, session_manager, _stm, _stream = await build_session(
        tmp_path, settings=COMPACTION_SETTINGS, extensions=[extension]
    )
    try:
        _seed_compactable_history(session_manager, session)

        result = await asyncio.wait_for(session.compact(), timeout=5)

        assert result.summary == "EXTENSION SUMMARY"
        compactions = [e for e in session_manager.get_entries() if isinstance(e, CompactionEntry)]
        assert len(compactions) == 1
        assert compactions[0].from_hook is True
        assert len(seen_compact_events) == 1
        assert seen_compact_events[0].from_extension is True
        assert seen_compact_events[0].reason == "manual"
        assert seen_compact_events[0].compaction_entry.summary == "EXTENSION SUMMARY"
    finally:
        session.dispose()


# ---------------------------------------------------------------------------
# Extension hook dispatch
# ---------------------------------------------------------------------------


def make_extension(path: str = "ext.py", **handlers) -> Extension:
    """Build an in-process `Extension` from `event_type=handler` keyword pairs."""
    return Extension(
        path=path,
        resolved_path=path,
        handlers={event: [handler] for event, handler in handlers.items()},
    )


async def test_extension_lifecycle_events_fire_in_agent_loop_order(tmp_path):
    seen: list[str] = []

    async def record(event, ctx):
        seen.append(event.type)
        return None

    echo_tool, _tool_runs = make_echo_tool()
    extension = Extension(
        path="lifecycle.py",
        resolved_path="lifecycle.py",
        handlers={
            event: [record]
            for event in (
                "agent_start",
                "turn_start",
                "message_start",
                "message_update",
                "message_end",
                "tool_execution_start",
                "tool_execution_end",
                "turn_end",
                "agent_end",
                "agent_settled",
            )
        },
    )
    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[
            tool_call_response(ToolCall(id="call-1", name="echo", arguments={"text": "hi"})),
            text_response("done"),
        ],
        custom_tools={"echo": echo_tool},
        allowed_tool_names=["echo"],
        extensions=[extension],
    )
    try:
        await asyncio.wait_for(session.prompt("go"), timeout=5)

        assert seen[0] == "agent_start"
        assert seen[-1] == "agent_settled"
        assert seen[-2] == "agent_end"
        assert seen.index("tool_execution_start") < seen.index("tool_execution_end")
        assert seen.index("tool_execution_end") < seen.index("turn_end")
        assert "turn_start" in seen
        assert "message_update" in seen
    finally:
        session.dispose()


async def test_tool_call_extension_hook_can_block_a_tool(tmp_path):
    seen_calls: list = []
    seen_results: list = []

    async def on_tool_call(event, ctx):
        seen_calls.append(event)
        return ToolCallEventResult(block=True, reason="not allowed")

    async def on_tool_result(event, ctx):
        seen_results.append(event)
        return None

    echo_tool, tool_runs = make_echo_tool()
    extension = make_extension(tool_call=on_tool_call, tool_result=on_tool_result)
    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[
            tool_call_response(ToolCall(id="call-1", name="echo", arguments={"text": "hi"})),
            text_response("understood"),
        ],
        custom_tools={"echo": echo_tool},
        allowed_tool_names=["echo"],
        extensions=[extension],
    )
    try:
        await asyncio.wait_for(session.prompt("go"), timeout=5)

        assert tool_runs == []  # blocked before execution
        assert [(e.tool_call_id, e.tool_name, e.input) for e in seen_calls] == [("call-1", "echo", {"text": "hi"})]
        tool_result = next(m for m in session.messages if getattr(m, "role", None) == "toolResult")
        assert tool_result.is_error is True
        assert "not allowed" in message_text(tool_result)
        # A blocked call short-circuits to an immediate error result, so the
        # tool_result hook never runs (agent-loop.ts returns kind: "immediate").
        assert seen_results == []
    finally:
        session.dispose()


async def test_tool_result_extension_hook_rewrites_the_result(tmp_path):
    async def on_tool_result(event, ctx):
        assert event.tool_name == "echo"
        assert event.is_error is False
        return ToolResultEventResult(content=[TextContent(text="REWRITTEN")], details={"patched": True})

    echo_tool, tool_runs = make_echo_tool()
    extension = make_extension(tool_result=on_tool_result)
    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[
            tool_call_response(ToolCall(id="call-1", name="echo", arguments={"text": "hi"})),
            text_response("done"),
        ],
        custom_tools={"echo": echo_tool},
        allowed_tool_names=["echo"],
        extensions=[extension],
    )
    try:
        await asyncio.wait_for(session.prompt("go"), timeout=5)

        assert tool_runs == ["hi"]
        tool_result = next(m for m in session.messages if getattr(m, "role", None) == "toolResult")
        assert message_text(tool_result) == "REWRITTEN"
        assert tool_result.details == {"patched": True}
    finally:
        session.dispose()


async def test_input_extension_hook_transforms_the_prompt_text(tmp_path):
    seen_events: list = []

    async def on_input(event, ctx):
        seen_events.append(event)
        return InputEventResult(action="transform", text=f"{event.text} (transformed)")

    sent_text = ""

    def capture(context):
        nonlocal sent_text
        user = next(m for m in context.messages if getattr(m, "role", None) == "user")
        sent_text = message_text(user)
        return text_response("ok")

    session, _sm, _stm, _stream = await build_session(
        tmp_path, responses=[capture], extensions=[make_extension(input=on_input)]
    )
    try:
        await asyncio.wait_for(session.prompt("hello", source="cli"), timeout=5)

        assert sent_text == "hello (transformed)"
        assert [e.source for e in seen_events] == ["cli"]
        assert seen_events[0].streaming_behavior is None
    finally:
        session.dispose()


async def test_input_extension_hook_can_fully_handle_the_prompt(tmp_path):
    async def on_input(event, ctx):
        return InputEventResult(action="handled")

    # No scripted responses: reaching the LLM would raise.
    session, _sm, _stm, _stream = await build_session(tmp_path, extensions=[make_extension(input=on_input)])
    try:
        preflight: list[bool] = []
        await asyncio.wait_for(session.prompt("hello", preflight_result=preflight.append), timeout=5)

        assert preflight == [True]
        assert session.messages == []
    finally:
        session.dispose()


async def test_before_agent_start_injects_a_message_and_overrides_the_system_prompt(tmp_path):
    seen_prompts: list[str] = []

    async def on_before_agent_start(event, ctx):
        seen_prompts.append(event.system_prompt)
        return BeforeAgentStartEventResult(
            message=CustomMessage(
                custom_type="injected",
                content="extra context",
                display=False,
                timestamp=now_ms(),
            ),
            system_prompt="OVERRIDDEN SYSTEM PROMPT",
        )

    seen_system_prompts: list[str] = []

    def capture(context):
        seen_system_prompts.append(context.system_prompt)
        return text_response("ok")

    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[capture, capture],
        extensions=[make_extension(before_agent_start=on_before_agent_start)],
    )
    try:
        await asyncio.wait_for(session.prompt("hi"), timeout=5)

        roles = [getattr(m, "role", None) for m in session.messages]
        assert roles == ["user", "custom", "assistant"]
        assert session.messages[1].custom_type == "injected"
        assert seen_system_prompts == ["OVERRIDDEN SYSTEM PROMPT"]
        assert seen_prompts[0] == session._base_system_prompt
        # `_run_agent_prompt` clears the override flag when the run settles, but
        # `agent.state.system_prompt` keeps the overridden text until the next turn
        # rewrites it (same as agent-session.ts `_runAgentPrompt`).
        assert session._system_prompt_override is None
        assert session.system_prompt == "OVERRIDDEN SYSTEM PROMPT"

        session._extension_runner.extensions[0].handlers["before_agent_start"] = [
            lambda event, ctx: None,
        ]
        await asyncio.wait_for(session.prompt("again"), timeout=5)
        assert session._system_prompt_override is None
        assert session.system_prompt == session._base_system_prompt
        assert seen_system_prompts[1] == session._base_system_prompt
    finally:
        session.dispose()


async def test_before_agent_start_message_stand_ins_are_normalized(tmp_path):
    class MinimalMessage:
        custom_type = "stand-in"

    async def on_before_agent_start(event, ctx):
        return BeforeAgentStartEventResult(message=MinimalMessage())

    session, _sm, _stm, _stream = await build_session(
        tmp_path, responses=[text_response("ok")], extensions=[make_extension(before_agent_start=on_before_agent_start)]
    )
    try:
        await asyncio.wait_for(session.prompt("hi"), timeout=5)

        injected = session.messages[1]
        assert getattr(injected, "role", None) == "custom"
        assert injected.custom_type == "stand-in"
        assert injected.content == []
        assert injected.display is False
    finally:
        session.dispose()


async def test_message_end_extension_hook_replacement_is_persisted(tmp_path):
    async def on_message_end(event, ctx):
        if getattr(event.message, "role", None) != "assistant":
            return None
        return MessageEndEventResult(message=replace(event.message, content=[TextContent(text="REPLACED")]))

    session, session_manager, _stm, _stream = await build_session(
        tmp_path, responses=[text_response("original")], extensions=[make_extension(message_end=on_message_end)]
    )
    try:
        await asyncio.wait_for(session.prompt("hi"), timeout=5)

        assert message_text(session.messages[-1]) == "REPLACED"
        persisted = [
            e.message
            for e in session_manager.get_entries()
            if isinstance(e, SessionMessageEntry) and getattr(e.message, "role", None) == "assistant"
        ]
        assert [message_text(m) for m in persisted] == ["REPLACED"]
    finally:
        session.dispose()


async def test_extension_command_runs_instead_of_prompting_the_model(tmp_path):
    calls: list[str] = []

    async def handler(args, ctx):
        calls.append(args)
        assert ctx.cwd == str(tmp_path)
        assert ctx.is_idle() is True

    extension = Extension(
        path="cmd.py",
        resolved_path="cmd.py",
        commands={"greet": RegisteredCommand(name="greet", handler=handler, description="Greet")},
    )
    # No scripted responses: reaching the LLM would raise.
    session, _sm, _stm, _stream = await build_session(tmp_path, extensions=[extension])
    try:
        preflight: list[bool] = []
        await asyncio.wait_for(session.prompt("/greet world", preflight_result=preflight.append), timeout=5)

        assert calls == ["world"]
        assert preflight == [True]
        assert session.messages == []

        await asyncio.wait_for(session.prompt("/greet"), timeout=5)
        assert calls == ["world", ""]
    finally:
        session.dispose()


async def test_extension_command_errors_are_reported_not_raised(tmp_path):
    async def handler(args, ctx):
        raise RuntimeError("command blew up")

    extension = Extension(
        path="cmd.py",
        resolved_path="cmd.py",
        commands={"boom": RegisteredCommand(name="boom", handler=handler)},
    )
    session, _sm, _stm, _stream = await build_session(tmp_path, extensions=[extension])
    try:
        errors: list = []
        session._extension_runner.on_error(errors.append)

        await asyncio.wait_for(session.prompt("/boom"), timeout=5)

        assert [(e.extension_path, e.event, e.error) for e in errors] == [
            ("command:boom", "command", "command blew up")
        ]
        assert session.messages == []
    finally:
        session.dispose()


async def test_extension_commands_cannot_be_queued(tmp_path):
    async def handler(args, ctx):
        return None

    extension = Extension(
        path="cmd.py",
        resolved_path="cmd.py",
        commands={"greet": RegisteredCommand(name="greet", handler=handler)},
    )
    session, _sm, _stm, _stream = await build_session(tmp_path, extensions=[extension])
    try:
        with pytest.raises(RuntimeError, match='Extension command "/greet" cannot be queued'):
            await asyncio.wait_for(session.steer("/greet hi"), timeout=5)
        with pytest.raises(RuntimeError, match='Extension command "/greet" cannot be queued'):
            await asyncio.wait_for(session.follow_up("/greet"), timeout=5)
        assert session.pending_message_count == 0
    finally:
        session.dispose()


async def test_extension_registered_tools_join_the_tool_registry(tmp_path):
    from pi_agent.types import AgentToolResult

    from pi_coding_agent.core.extensions.types import RegisteredTool, ToolDefinition

    runs: list[dict] = []

    async def execute(tool_call_id, params, signal, on_update, ctx):
        runs.append(params)
        return AgentToolResult(content=[TextContent(text="from extension")], details={})

    extension = Extension(
        path="tool.py",
        resolved_path="tool.py",
        tools={
            "ext_tool": RegisteredTool(
                definition=ToolDefinition(
                    name="ext_tool",
                    label="Ext Tool",
                    description="An extension tool",
                    parameters={"type": "object", "properties": {}},
                    execute=execute,
                )
            )
        },
    )
    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[
            tool_call_response(ToolCall(id="call-1", name="ext_tool", arguments={})),
            text_response("done"),
        ],
        allowed_tool_names=["ext_tool"],
        initial_active_tool_names=["ext_tool"],
        extensions=[extension],
    )
    try:
        assert [tool.name for tool in session.get_all_tools()] == ["ext_tool"]

        await asyncio.wait_for(session.prompt("use it"), timeout=5)

        assert runs == [{}]
        tool_result = next(m for m in session.messages if getattr(m, "role", None) == "toolResult")
        assert message_text(tool_result) == "from extension"
    finally:
        session.dispose()


async def test_extension_context_actions_are_bound_to_the_session(tmp_path):
    captured: dict = {}

    async def on_agent_start(event, ctx):
        captured["model"] = ctx.model
        captured["cwd"] = ctx.cwd
        captured["thinking_level"] = ctx.thinking_level
        captured["context_usage"] = ctx.get_context_usage()
        captured["system_prompt"] = ctx.get_system_prompt()
        captured["is_idle"] = ctx.is_idle()
        captured["has_pending_messages"] = ctx.has_pending_messages()
        return None

    session, _sm, _stm, _stream = await build_session(
        tmp_path, responses=[text_response("ok")], extensions=[make_extension(agent_start=on_agent_start)]
    )
    try:
        await asyncio.wait_for(session.prompt("hi"), timeout=5)

        assert captured["model"] is TEST_MODEL
        assert captured["cwd"] == str(tmp_path)
        assert captured["thinking_level"] == "off"
        assert captured["context_usage"].context_window == TEST_MODEL.context_window
        assert captured["system_prompt"] == session.system_prompt
        assert captured["is_idle"] is False
        assert captured["has_pending_messages"] is False
    finally:
        session.dispose()


async def test_extension_compact_action_reports_completion_and_failure(tmp_path):
    """`pi.compact()` is fire-and-forget: results arrive through the callbacks."""
    session, session_manager, _stm, _stream = await build_session(
        tmp_path,
        responses=[text_response("SUMMARY OF HISTORY"), text_response("TURN PREFIX")],
        settings=COMPACTION_SETTINGS,
    )
    try:
        _seed_compactable_history(session_manager, session)
        completed: list = []
        failed: list = []

        session._extension_compact_action(None, completed.append, failed.append)
        await asyncio.wait_for(_drain_background_tasks(session), timeout=10)

        assert failed == []
        assert "SUMMARY OF HISTORY" in completed[0].summary

        session._extension_compact_action(None, completed.append, failed.append)
        await asyncio.wait_for(_drain_background_tasks(session), timeout=10)

        assert len(completed) == 1
        assert isinstance(failed[0], RuntimeError)
        assert "Already compacted" in str(failed[0])
    finally:
        session.dispose()


async def _drain_background_tasks(session: AgentSession) -> None:
    while session._background_tasks:
        await asyncio.gather(*list(session._background_tasks), return_exceptions=True)


# ---------------------------------------------------------------------------
# Skill expansion failures
# ---------------------------------------------------------------------------


async def test_unreadable_skill_files_report_an_extension_error(tmp_path):
    """Regression: `_expand_skill_command` must report read failures like TS's
    `_expandSkillCommand` (`emitError({event: "skill_expansion"})`), not swallow them."""
    missing_path = tmp_path / "gone.md"

    session, _sm, _stm, _stream = await build_session(tmp_path, responses=[text_response("ok")])
    try:
        skill = Skill(
            name="broken",
            description="Missing file",
            file_path=str(missing_path),
            base_dir=str(tmp_path),
            source_info=create_synthetic_source_info(
                str(missing_path), "local", scope="project", base_dir=str(tmp_path)
            ),
        )
        session._resource_loader.get_skills = lambda: type("R", (), {"skills": [skill], "diagnostics": []})()
        errors: list = []
        session._extension_runner.on_error(errors.append)

        await asyncio.wait_for(session.prompt("/skill:broken please"), timeout=5)

        assert [(e.extension_path, e.event) for e in errors] == [(str(missing_path), "skill_expansion")]
        user = next(m for m in session.messages if getattr(m, "role", None) == "user")
        assert message_text(user) == "/skill:broken please"
    finally:
        session.dispose()


async def test_unknown_skill_commands_pass_through_unchanged(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path, responses=[text_response("ok")])
    try:
        errors: list = []
        session._extension_runner.on_error(errors.append)

        await asyncio.wait_for(session.prompt("/skill:nosuch"), timeout=5)

        assert errors == []
        user = next(m for m in session.messages if getattr(m, "role", None) == "user")
        assert message_text(user) == "/skill:nosuch"
    finally:
        session.dispose()


# ---------------------------------------------------------------------------
# Tree navigation
# ---------------------------------------------------------------------------


async def test_navigate_tree_to_a_user_entry_returns_its_text_and_rewinds(tmp_path):
    session, session_manager, _stm, _stream = await build_session(
        tmp_path, responses=[text_response("first answer"), text_response("second answer")]
    )
    try:
        await asyncio.wait_for(session.prompt("first question"), timeout=5)
        await asyncio.wait_for(session.prompt("second question"), timeout=5)

        second_user = [
            e
            for e in session_manager.get_entries()
            if isinstance(e, SessionMessageEntry) and getattr(e.message, "role", None) == "user"
        ][1]

        result = await asyncio.wait_for(session.navigate_tree(second_user.id), timeout=5)

        assert result.cancelled is False
        assert result.aborted is False
        assert result.summary_entry is None
        assert result.editor_text == "second question"
        # The leaf moved back to the entry's parent, so the second exchange is gone.
        assert [message_text(m) for m in session.messages] == ["first question", "first answer"]
    finally:
        session.dispose()


async def test_navigate_tree_to_the_first_user_entry_resets_the_leaf(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path, responses=[text_response("answer")])
    try:
        await asyncio.wait_for(session.prompt("only question"), timeout=5)
        first_user = next(
            e
            for e in session_manager.get_entries()
            if isinstance(e, SessionMessageEntry) and getattr(e.message, "role", None) == "user"
        )

        result = await asyncio.wait_for(session.navigate_tree(first_user.id), timeout=5)

        assert result.editor_text == "only question"
        # The entry has no parent, so the branch resets to an empty session.
        assert session_manager.get_leaf_id() is None
        assert session.messages == []
    finally:
        session.dispose()


async def test_navigate_tree_to_a_custom_entry_returns_its_text(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path, responses=[text_response("answer")])
    try:
        await asyncio.wait_for(session.send_custom_message("note", "a note to self", True), timeout=5)
        custom_entry = next(e for e in session_manager.get_entries() if isinstance(e, CustomMessageEntry))
        await asyncio.wait_for(session.prompt("question"), timeout=5)

        result = await asyncio.wait_for(session.navigate_tree(custom_entry.id), timeout=5)

        assert result.editor_text == "a note to self"
        # The custom entry had no parent, so its branch rewinds to an empty session.
        assert session.messages == []
        assert session_manager.get_leaf_id() is None
    finally:
        session.dispose()


async def test_navigate_tree_to_an_assistant_entry_keeps_it_as_the_leaf(tmp_path):
    session, session_manager, _stm, _stream = await build_session(
        tmp_path, responses=[text_response("first answer"), text_response("second answer")]
    )
    try:
        await asyncio.wait_for(session.prompt("first question"), timeout=5)
        await asyncio.wait_for(session.prompt("second question"), timeout=5)

        first_assistant = next(
            e
            for e in session_manager.get_entries()
            if isinstance(e, SessionMessageEntry) and getattr(e.message, "role", None) == "assistant"
        )

        result = await asyncio.wait_for(session.navigate_tree(first_assistant.id), timeout=5)

        assert result.editor_text is None
        assert session_manager.get_leaf_id() == first_assistant.id
        assert [message_text(m) for m in session.messages] == ["first question", "first answer"]
    finally:
        session.dispose()


async def test_navigate_tree_returns_early_for_the_current_leaf(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path, responses=[text_response("answer")])
    try:
        await asyncio.wait_for(session.prompt("question"), timeout=5)
        leaf_id = session_manager.get_leaf_id()

        result = await asyncio.wait_for(session.navigate_tree(leaf_id), timeout=5)

        assert result == NavigateTreeResult(cancelled=False)
        assert session_manager.get_leaf_id() == leaf_id
    finally:
        session.dispose()


async def test_navigate_tree_rejects_unknown_entries(tmp_path):
    session, _sm, _stm, _stream = await build_session(tmp_path, responses=[text_response("answer")])
    try:
        await asyncio.wait_for(session.prompt("question"), timeout=5)

        with pytest.raises(ValueError, match="Entry nope not found"):
            await asyncio.wait_for(session.navigate_tree("nope"), timeout=5)
    finally:
        session.dispose()


async def test_navigate_tree_requires_a_model_for_summarization(tmp_path):
    session, session_manager, _stm, _stream = await build_session(tmp_path, responses=[text_response("answer")])
    try:
        await asyncio.wait_for(session.prompt("question"), timeout=5)
        target = next(
            e
            for e in session_manager.get_entries()
            if isinstance(e, SessionMessageEntry) and getattr(e.message, "role", None) == "user"
        )
        session.agent.state.model = None

        with pytest.raises(RuntimeError, match="No model available for summarization"):
            await asyncio.wait_for(session.navigate_tree(target.id, summarize=True), timeout=5)
    finally:
        session.dispose()


async def test_navigate_tree_refuses_while_streaming(tmp_path):
    wait_tool, started, release = make_blocking_tool()
    session, _sm, _stm, _stream = await build_session(
        tmp_path,
        responses=[
            tool_call_response(ToolCall(id="call-1", name="wait", arguments={})),
            text_response("done"),
        ],
        custom_tools={"wait": wait_tool},
        allowed_tool_names=["wait"],
    )
    try:
        run = asyncio.ensure_future(session.prompt("go"))
        await asyncio.wait_for(started.wait(), timeout=5)

        with pytest.raises(RuntimeError, match="Wait for the current response to finish"):
            await asyncio.wait_for(session.navigate_tree("anything"), timeout=5)

        release.set()
        await asyncio.wait_for(run, timeout=5)
    finally:
        session.dispose()


async def test_navigate_tree_with_summarization_records_a_branch_summary(tmp_path):
    session, session_manager, _stm, _stream = await build_session(
        tmp_path,
        responses=[
            text_response("first answer"),
            text_response("second answer"),
            text_response("ABANDONED BRANCH SUMMARY"),
        ],
    )
    try:
        await asyncio.wait_for(session.prompt("first question"), timeout=5)
        await asyncio.wait_for(session.prompt("second question"), timeout=5)

        second_user = [
            e
            for e in session_manager.get_entries()
            if isinstance(e, SessionMessageEntry) and getattr(e.message, "role", None) == "user"
        ][1]

        result = await asyncio.wait_for(
            session.navigate_tree(
                second_user.id,
                summarize=True,
                custom_instructions="focus on the plan",
                label="explored",
            ),
            timeout=10,
        )

        assert result.cancelled is False
        assert result.summary_entry is not None
        assert "ABANDONED BRANCH SUMMARY" in result.summary_entry.summary
        assert result.editor_text == "second question"
        # The label entry is appended after the summary, so it becomes the leaf.
        branch_ids = [e.id for e in session_manager.get_branch()]
        assert result.summary_entry.id in branch_ids
        label_entry = session_manager.get_entry(session_manager.get_leaf_id())
        assert label_entry.label == "explored"
        assert label_entry.target_id == result.summary_entry.id
        # The summary becomes part of the rebuilt context.
        summary_message = next(m for m in session.messages if getattr(m, "role", None) == "branchSummary")
        assert "ABANDONED BRANCH SUMMARY" in summary_message.summary
    finally:
        session.dispose()


async def test_navigate_tree_summarization_abort_cancels_the_navigation(tmp_path):
    aborted_summary = make_assistant_message([], stop_reason="aborted")
    session, session_manager, _stm, _stream = await build_session(
        tmp_path, responses=[text_response("answer"), aborted_summary]
    )
    try:
        await asyncio.wait_for(session.prompt("question"), timeout=5)
        target = next(
            e
            for e in session_manager.get_entries()
            if isinstance(e, SessionMessageEntry) and getattr(e.message, "role", None) == "user"
        )
        leaf_before = session_manager.get_leaf_id()

        result = await asyncio.wait_for(session.navigate_tree(target.id, summarize=True), timeout=10)

        assert result == NavigateTreeResult(cancelled=True, aborted=True)
        assert session_manager.get_leaf_id() == leaf_before
    finally:
        session.dispose()


async def test_navigate_tree_summarization_error_propagates(tmp_path):
    failed_summary = make_assistant_message([], stop_reason="error", error_message="401 invalid api key")
    session, session_manager, _stm, _stream = await build_session(
        tmp_path,
        responses=[text_response("answer"), failed_summary],
        settings={"retry": {"enabled": False}},
    )
    try:
        await asyncio.wait_for(session.prompt("question"), timeout=5)
        target = next(
            e
            for e in session_manager.get_entries()
            if isinstance(e, SessionMessageEntry) and getattr(e.message, "role", None) == "user"
        )

        with pytest.raises(RuntimeError, match="401 invalid api key"):
            await asyncio.wait_for(session.navigate_tree(target.id, summarize=True), timeout=10)
        assert session._branch_summary_abort_controller is None
    finally:
        session.dispose()


async def test_overflow_recovery_compacts_and_asks_for_a_retry(tmp_path):
    session, session_manager, _stm, _stream = await build_session(
        tmp_path,
        responses=[text_response("SUMMARY OF HISTORY"), text_response("TURN PREFIX")],
        settings=COMPACTION_SETTINGS,
    )
    try:
        _seed_compactable_history(session_manager, session)
        overflow = make_assistant_message([], stop_reason="error", error_message="prompt is too long")
        overflow.timestamp = now_ms()
        session_manager.append_message(overflow)
        _sync_agent_messages(session, session_manager)
        events = []
        session.subscribe(events.append)

        assert await asyncio.wait_for(session._check_compaction(overflow), timeout=10) is True

        assert session._overflow_recovery_attempted is True
        end = next(e for e in events if e.type == "compaction_end")
        assert end.reason == "overflow"
        assert end.will_retry is True
        assert "SUMMARY OF HISTORY" in end.result.summary
        # The failed assistant response is dropped so the retry resends the turn.
        assert all(getattr(m, "error_message", None) != "prompt is too long" for m in session.messages)
        assert len([e for e in session_manager.get_entries() if isinstance(e, CompactionEntry)]) == 1
    finally:
        session.dispose()


async def test_auto_compaction_failures_are_reported_as_events(tmp_path):
    failed_summary = make_assistant_message([], stop_reason="error", error_message="401 invalid api key")
    session, session_manager, _stm, _stream = await build_session(
        tmp_path,
        responses=[failed_summary],
        settings={**COMPACTION_SETTINGS, "retry": {"enabled": False}},
    )
    try:
        _seed_compactable_history(session_manager, session)
        trigger = _assistant("over the threshold", 950, now_ms())
        events = []
        session.subscribe(events.append)

        assert await asyncio.wait_for(session._check_compaction(trigger), timeout=10) is False

        end = next(e for e in events if e.type == "compaction_end")
        assert end.reason == "threshold"
        assert end.result is None
        assert end.aborted is False
        assert "Auto-compaction failed" in end.error_message
        assert [e for e in session_manager.get_entries() if isinstance(e, CompactionEntry)] == []
    finally:
        session.dispose()


async def test_parse_skill_block_round_trips_an_expanded_skill_command(tmp_path):
    skill_path = tmp_path / "s.md"
    skill_path.write_text("---\nname: s\n---\nBody line", encoding="utf-8")
    session, _sm, _stm, _stream = await build_session(tmp_path)
    try:
        skill = Skill(
            name="s",
            description="A skill",
            file_path=str(skill_path),
            base_dir=str(tmp_path),
            source_info=create_synthetic_source_info(str(skill_path), "local", scope="project", base_dir=str(tmp_path)),
        )
        session._resource_loader.get_skills = lambda: type("R", (), {"skills": [skill], "diagnostics": []})()

        expanded = session._expand_skill_command("/skill:s do the thing")
        parsed = parse_skill_block(expanded)

        assert parsed is not None
        assert parsed.name == "s"
        assert parsed.location == str(skill_path)
        assert "Body line" in parsed.content
        assert parsed.user_message == "do the thing"

        without_args = parse_skill_block(session._expand_skill_command("/skill:s"))
        assert without_args is not None
        assert without_args.user_message is None
        assert parse_skill_block("just text") is None
    finally:
        session.dispose()
