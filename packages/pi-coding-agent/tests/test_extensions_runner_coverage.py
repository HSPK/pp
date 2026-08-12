"""Additional coverage tests for `pi_coding_agent.core.extensions.runner`.

Targets uncovered lines from the baseline run:
- `emit_error` / error-listener unsubscription (lines 140-141, 144)
- `emit` / session_before_* cancel short-circuit (line 157)
- `emit_message_end` role-mismatch error path and modification tracking (lines 186-188, 212-213)
- `set_ui_context` / `has_ui` (line 294)
- `create_command_context` / `_immediate` (lines 315, 320-322, 328-335)
- `emit_tool_call` block result / result accumulation (lines 357, 361-363, 365, 376-377, 380)
- `emit_before_agent_start` no-change path (lines 393+)
- `emit_input` images transform path, error isolation (lines 431-433, 461, 468-472, 475->462, 481, 485-488)
- `wrap_registered_tool` added_tool_names diffing path (lines 525->531, 529-530)
"""

from __future__ import annotations

from pi_ai.types import ImageContent, TextContent
from pi_coding_agent.core.extensions.runner import (
    ExtensionContextActions,
    ExtensionRunner,
    _immediate,
    wrap_registered_tool,
)
from pi_coding_agent.core.extensions.types import (
    Extension,
    ExtensionError,
    ExtensionUIContext,
    MessageEndEvent,
    NullExtensionUIContext,
    RegisteredTool,
    SessionBeforeCompactResult,
    SessionBeforeSwitchResult,
    ToolCallEvent,
    ToolCallEventResult,
    ToolDefinition,
)
from pi_coding_agent.core.system_prompt import BuildSystemPromptOptions

# ---------------------------------------------------------------------------
# emit_error / on_error unsubscription
# ---------------------------------------------------------------------------


def test_on_error_unsubscribe_stops_listener():
    runner = ExtensionRunner([], cwd="/unused")
    received: list[ExtensionError] = []
    unsubscribe = runner.on_error(received.append)
    unsubscribe()
    runner.emit_error(ExtensionError(extension_path="x", event="ctx", error="boom"))
    assert received == []


def test_emit_error_records_in_errors_list():
    runner = ExtensionRunner([], cwd="/unused")
    err = ExtensionError(extension_path="x", event="ctx", error="boom")
    runner.emit_error(err)
    assert runner.errors == [err]


# ---------------------------------------------------------------------------
# emit: session_before_* cancel short-circuit
# ---------------------------------------------------------------------------


async def test_emit_session_before_switch_cancels_on_cancel_true():
    ext = Extension(path="e", resolved_path="e")

    async def _cancel(event, ctx):
        return SessionBeforeSwitchResult(cancel=True)

    async def _should_not_run(event, ctx):
        raise AssertionError("should not run after cancel")

    ext.handlers["session_before_switch"] = [_cancel, _should_not_run]
    runner = ExtensionRunner([ext], cwd="/unused")
    result = await runner.emit(type("E", (), {"type": "session_before_switch"})())
    assert result is not None
    assert result.cancel is True


async def test_emit_session_before_compact_keeps_last_result():
    ext = Extension(path="e", resolved_path="e")

    first_result = SessionBeforeCompactResult()
    second_result = SessionBeforeCompactResult()

    ext.handlers["session_before_compact"] = [
        lambda event, ctx: first_result,
        lambda event, ctx: second_result,
    ]
    runner = ExtensionRunner([ext], cwd="/unused")
    result = await runner.emit(type("E", (), {"type": "session_before_compact"})())
    assert result is second_result


# ---------------------------------------------------------------------------
# emit_message_end
# ---------------------------------------------------------------------------


async def test_emit_message_end_returns_none_when_unmodified():
    from pi_ai.types import UserMessage

    ext = Extension(path="e", resolved_path="e")
    ext.handlers["message_end"] = [lambda event, ctx: None]
    runner = ExtensionRunner([ext], cwd="/unused")
    msg = UserMessage(content=[TextContent(text="hi")], timestamp=0)
    result = await runner.emit_message_end(MessageEndEvent(message=msg))
    assert result is None


async def test_emit_message_end_role_mismatch_emits_error():
    from pi_ai.types import AssistantMessage, Cost, Usage, UserMessage

    user_msg = UserMessage(content=[TextContent(text="hello")], timestamp=0)
    asst_msg = AssistantMessage(
        api="x",
        provider="x",
        model="m",
        content=[TextContent(text="reply")],
        usage=Usage(cost=Cost()),
        stop_reason="stop",
        timestamp=0,
    )

    ext = Extension(path="e", resolved_path="e")
    from pi_coding_agent.core.extensions.types import MessageEndEventResult

    ext.handlers["message_end"] = [lambda event, ctx: MessageEndEventResult(message=asst_msg)]
    runner = ExtensionRunner([ext], cwd="/unused")
    errors: list[ExtensionError] = []
    runner.on_error(errors.append)
    result = await runner.emit_message_end(MessageEndEvent(message=user_msg))
    assert len(errors) == 1
    assert "same role" in errors[0].error
    # Original message returned unchanged
    assert result is None


async def test_emit_message_end_modifies_message():
    from pi_ai.types import UserMessage
    from pi_coding_agent.core.extensions.types import MessageEndEventResult

    original = UserMessage(content=[TextContent(text="original")], timestamp=0)
    modified = UserMessage(content=[TextContent(text="modified")], timestamp=0)

    ext = Extension(path="e", resolved_path="e")
    ext.handlers["message_end"] = [lambda event, ctx: MessageEndEventResult(message=modified)]
    runner = ExtensionRunner([ext], cwd="/unused")
    result = await runner.emit_message_end(MessageEndEvent(message=original))
    assert result is not None
    assert result is modified


# ---------------------------------------------------------------------------
# set_ui_context / has_ui
# ---------------------------------------------------------------------------


def test_set_ui_context_to_null_resets_mode():
    runner = ExtensionRunner([], cwd="/unused")
    runner.set_ui_context(None, "interactive")
    assert isinstance(runner.get_ui_context(), NullExtensionUIContext)
    assert runner.has_ui() is False
    assert runner._mode == "interactive"


def test_set_ui_context_with_real_ui():
    class _FakeUI(ExtensionUIContext):
        pass

    runner = ExtensionRunner([], cwd="/unused")
    fake_ui = _FakeUI()
    runner.set_ui_context(fake_ui, "print")
    assert runner.get_ui_context() is fake_ui
    assert runner.has_ui() is True


# ---------------------------------------------------------------------------
# create_command_context / _immediate
# ---------------------------------------------------------------------------


async def test_immediate_awaitable_returns_none():
    result = await _immediate()
    assert result is None


def test_create_command_context_reflects_actions():
    runner = ExtensionRunner([], cwd="/proj")
    runner.bind_core(
        ExtensionContextActions(
            is_project_trusted=lambda: True,
            get_model=lambda: "my-model",
            get_system_prompt_options=lambda: BuildSystemPromptOptions(cwd="/proj"),
        )
    )
    ctx = runner.create_command_context()
    assert ctx.cwd == "/proj"
    assert ctx.model == "my-model"
    assert callable(ctx.wait_for_idle)


# ---------------------------------------------------------------------------
# emit_tool_call: block / result accumulation
# ---------------------------------------------------------------------------


async def test_emit_tool_call_returns_none_with_no_handlers():
    runner = ExtensionRunner([], cwd="/unused")
    result = await runner.emit_tool_call(ToolCallEvent(tool_call_id="1", tool_name="t", input={}))
    assert result is None


async def test_emit_tool_call_block_short_circuits():
    ext1 = Extension(path="e1", resolved_path="e1")
    ext2 = Extension(path="e2", resolved_path="e2")

    blocking = ToolCallEventResult(block=True, reason="blocked")
    after_result = ToolCallEventResult(block=False)

    ext1.handlers["tool_call"] = [lambda event, ctx: blocking]
    ext2.handlers["tool_call"] = [lambda event, ctx: after_result]

    runner = ExtensionRunner([ext1, ext2], cwd="/unused")
    result = await runner.emit_tool_call(ToolCallEvent(tool_call_id="1", tool_name="t", input={}))
    assert result is not None
    assert result.block is True


async def test_emit_tool_call_accumulates_non_block_result():
    ext = Extension(path="e", resolved_path="e")
    r1 = ToolCallEventResult(block=False)
    r2 = ToolCallEventResult(block=False)
    ext.handlers["tool_call"] = [lambda event, ctx: r1, lambda event, ctx: r2]
    runner = ExtensionRunner([ext], cwd="/unused")
    result = await runner.emit_tool_call(ToolCallEvent(tool_call_id="1", tool_name="t", input={}))
    assert result is r2


# ---------------------------------------------------------------------------
# emit_before_agent_start: no-change returns None
# ---------------------------------------------------------------------------


async def test_emit_before_agent_start_returns_none_when_no_handlers():
    runner = ExtensionRunner([], cwd="/unused")
    result = await runner.emit_before_agent_start("hello", None, "base", BuildSystemPromptOptions(cwd="/unused"))
    assert result is None


async def test_emit_before_agent_start_error_isolation():
    ext = Extension(path="e", resolved_path="e")

    async def _raises(event, ctx):
        raise RuntimeError("before_agent_start error")

    ext.handlers["before_agent_start"] = [_raises]
    runner = ExtensionRunner([ext], cwd="/unused")
    errors: list[ExtensionError] = []
    runner.on_error(errors.append)
    result = await runner.emit_before_agent_start("prompt", None, "base", BuildSystemPromptOptions(cwd="/unused"))
    assert result is None
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# emit_input: images transform path, no handlers
# ---------------------------------------------------------------------------


async def test_emit_input_returns_continue_when_no_handlers():
    runner = ExtensionRunner([], cwd="/unused")
    result = await runner.emit_input("hello", None, "interactive")
    assert result.action == "continue"


async def test_emit_input_images_transform():
    from pi_coding_agent.core.extensions.types import InputEventResult

    ext = Extension(path="e", resolved_path="e")
    img1 = ImageContent(data="orig", mime_type="image/png")
    img2 = ImageContent(data="new", mime_type="image/png")
    ext.handlers["input"] = [lambda event, ctx: InputEventResult(action="transform", text="new-text", images=[img2])]
    runner = ExtensionRunner([ext], cwd="/unused")
    result = await runner.emit_input("hello", [img1], "interactive")
    assert result.action == "transform"
    assert result.text == "new-text"
    assert result.images == [img2]


async def test_emit_input_error_isolation_continues():
    from pi_coding_agent.core.extensions.types import InputEventResult

    ext1 = Extension(path="e1", resolved_path="e1")
    ext1.handlers["input"] = [lambda event, ctx: (_ for _ in ()).throw(RuntimeError("input error"))]

    ext2 = Extension(path="e2", resolved_path="e2")
    ext2.handlers["input"] = [lambda event, ctx: InputEventResult(action="transform", text="from-ext2")]

    runner = ExtensionRunner([ext1, ext2], cwd="/unused")
    errors: list[ExtensionError] = []
    runner.on_error(errors.append)
    result = await runner.emit_input("hello", None, "interactive")
    assert result.action == "transform"
    assert result.text == "from-ext2"
    assert len(errors) == 1


# ---------------------------------------------------------------------------
# wrap_registered_tool: added_tool_names diffing
# ---------------------------------------------------------------------------


async def test_wrap_registered_tool_records_added_tool_names():
    """When the extension's execute adds a new active tool, the result's added_tool_names is populated."""
    from pi_agent.types import AgentToolResult

    active_tools: list[str] = ["read"]

    async def _execute(tool_call_id, params, signal, on_update, ctx):
        active_tools.append("bash")
        result = AgentToolResult(content=[TextContent(text="ok")])
        return result

    definition = ToolDefinition(
        name="adds_tool",
        label="adds_tool",
        description="adds a tool",
        execute=_execute,
    )
    registered = RegisteredTool(definition=definition)

    runner = ExtensionRunner([], cwd="/unused")
    runner.bind_core(
        ExtensionContextActions(
            get_active_tool_names=lambda: list(active_tools),
        )
    )

    agent_tool = wrap_registered_tool(registered, runner)
    result = await agent_tool.execute("call-1", {}, None, None)
    assert result.added_tool_names is not None
    assert "bash" in result.added_tool_names
