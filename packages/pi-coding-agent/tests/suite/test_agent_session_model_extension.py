"""Python port of `packages/coding-agent/test/suite/agent-session-model-extension.test.ts`."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from harness import create_harness, get_assistant_texts
from pi_agent.harness.messages import CustomMessage
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.providers.faux import FauxModelDefinition, faux_assistant_message, faux_tool_call
from pi_ai.types import Cost, TextContent, Usage
from pi_coding_agent.core.extensions.loader import ExtensionAPI
from pi_coding_agent.core.extensions.types import (
    BeforeAgentStartEventResult,
    ContextEventResult,
    InputEventResult,
    ToolCallEventResult,
    ToolResultEventResult,
)
from pi_coding_agent.core.session_manager import ModelChangeEntry


def _text_of(message: Any) -> str:
    return "\n".join(part.text for part in getattr(message, "content", []) if isinstance(part, TextContent))


async def test_set_model_saves_the_model_and_emits_model_select(tmp_path: Path) -> None:
    model_events: list[str] = []

    def factory(pi: ExtensionAPI) -> None:
        async def on_model_select(event, ctx) -> None:
            previous = event.previous_model.id if event.previous_model else "none"
            model_events.append(f"{previous}->{event.model.id}:{event.source}")

        pi.on("model_select", on_model_select)

    harness = await create_harness(
        tmp_path,
        models=[
            FauxModelDefinition(id="faux-1", name="One", reasoning=True),
            FauxModelDefinition(id="faux-2", name="Two", reasoning=True),
        ],
        extension_factories=[factory],
    )
    try:
        next_model = harness.get_model("faux-2")
        assert next_model is not None

        await harness.session.set_model(next_model)

        assert harness.session.model is not None
        assert harness.session.model.id == "faux-2"
        assert model_events == ["faux-1->faux-2:set"]
        assert [
            f"{entry.provider}/{entry.model_id}"
            for entry in harness.session_manager.get_entries()
            if isinstance(entry, ModelChangeEntry)
        ] == [f"{next_model.provider}/{next_model.id}"]
    finally:
        harness.cleanup()


async def test_cycles_through_scoped_models_and_preserves_the_scoped_thinking_preference(tmp_path: Path) -> None:
    from pi_coding_agent.core.model_resolver import ScopedModel

    harness = await create_harness(
        tmp_path,
        models=[
            FauxModelDefinition(id="faux-1", name="One", reasoning=True),
            FauxModelDefinition(id="faux-2", name="Two", reasoning=False),
        ],
    )
    try:
        model_one = harness.get_model("faux-1")
        model_two = harness.get_model("faux-2")
        assert model_one is not None and model_two is not None
        harness.session.set_scoped_models(
            [ScopedModel(model=model_one, thinking_level="high"), ScopedModel(model=model_two)]
        )
        harness.session.set_thinking_level("high")

        # `_cycle_scoped_model` first drops scoped models missing from the
        # availability snapshot and returns `None` when one or fewer survive
        # (TS `_cycleScopedModel` does the same). If the harness's configured
        # auth stops making the faux models available, that early return makes
        # every cycle below a silent no-op and the failure reads as "expected
        # faux-2, got faux-1" instead of naming the real cause.
        assert {model.id for model in harness.session.model_runtime.get_available_snapshot()} == {"faux-1", "faux-2"}

        await harness.session.cycle_model()
        assert harness.session.model is not None
        assert harness.session.model.id == "faux-2"
        assert harness.session.thinking_level == "off"

        await harness.session.cycle_model()
        assert harness.session.model is not None
        assert harness.session.model.id == "faux-1"
        assert harness.session.thinking_level == "high"
    finally:
        harness.cleanup()


async def test_clamps_thinking_levels_to_model_capabilities_and_cycles_available_levels(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, models=[FauxModelDefinition(id="faux-1", reasoning=False)])
    try:
        harness.session.set_thinking_level("high")
        assert harness.session.thinking_level == "off"
        assert harness.session.cycle_thinking_level() is None
    finally:
        harness.cleanup()


async def test_cycles_xhigh_before_max_when_both_are_supported(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, models=[FauxModelDefinition(id="faux-1", reasoning=True)])
    try:
        model = harness.get_model()
        assert model is not None
        model.thinking_level_map = {"xhigh": "xhigh", "max": "max"}

        assert harness.session.get_available_thinking_levels() == [
            "off",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ]
        harness.session.set_thinking_level("high")
        assert harness.session.cycle_thinking_level() == "xhigh"
        assert harness.session.cycle_thinking_level() == "max"
        assert harness.session.cycle_thinking_level() == "off"
    finally:
        harness.cleanup()


async def test_throws_when_set_model_is_called_without_configured_auth(tmp_path: Path) -> None:
    harness = await create_harness(
        tmp_path,
        models=[
            FauxModelDefinition(id="faux-1", name="One", reasoning=True),
            FauxModelDefinition(id="faux-2", name="Two", reasoning=True),
        ],
        with_configured_auth=False,
    )
    try:
        model = harness.get_model()
        target = harness.get_model("faux-2")
        assert model is not None and target is not None
        with pytest.raises(RuntimeError, match=f"No API key for {model.provider}/faux-2"):
            await harness.session.set_model(target)
    finally:
        harness.cleanup()


def _echo_tool(execute) -> AgentTool:
    return AgentTool(
        name="echo",
        label="Echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        execute=execute,
    )


def _tool_result_text(context) -> str:
    tool_result = next((m for m in context.messages if getattr(m, "role", "") == "toolResult"), None)
    if tool_result is None:
        return ""
    return "\n".join(part.text for part in tool_result.content if isinstance(part, TextContent))


async def test_allows_extension_tool_call_handlers_to_block_tool_execution(tmp_path: Path) -> None:
    async def execute(tool_call_id: str, params, signal=None, on_update=None) -> AgentToolResult:
        raise RuntimeError("tool should have been blocked")

    def factory(pi: ExtensionAPI) -> None:
        async def on_tool_call(event, ctx) -> ToolCallEventResult:
            return ToolCallEventResult(block=True, reason="Blocked by test")

        pi.on("tool_call", on_tool_call)

    harness = await create_harness(tmp_path, tools=[_echo_tool(execute)], extension_factories=[factory])
    try:
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("echo", {"text": "hello"})], stop_reason="toolUse"),
                lambda context, options, state, model: faux_assistant_message(_tool_result_text(context)),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("hi"), timeout=10)

        assert any("Blocked by test" in text for text in get_assistant_texts(harness))
        assert any(
            getattr(message, "role", "") == "toolResult" and getattr(message, "is_error", False)
            for message in harness.session.messages
        )
    finally:
        harness.cleanup()


async def test_allows_extension_tool_result_handlers_to_modify_tool_results(tmp_path: Path) -> None:
    tool_usage = Usage(
        input=1,
        output=2,
        cache_read=3,
        cache_write=4,
        total_tokens=10,
        cost=Cost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
    )
    patched_tool_usage = Usage(
        input=5,
        output=6,
        cache_read=7,
        cache_write=8,
        total_tokens=26,
        cost=Cost(input=0.5, output=0.6, cache_read=0.7, cache_write=0.8, total=2.6),
    )
    observed: list[Usage | None] = []

    async def execute(tool_call_id: str, params, signal=None, on_update=None) -> AgentToolResult:
        text = str(params.get("text", "")) if isinstance(params, dict) else ""
        return AgentToolResult(content=[TextContent(text=text)], details={"text": text}, usage=tool_usage)

    def factory(pi: ExtensionAPI) -> None:
        async def on_tool_result(event, ctx) -> ToolResultEventResult:
            observed.append(event.usage)
            return ToolResultEventResult(
                content=[TextContent(text="patched result")],
                details={"patched": True},
                usage=patched_tool_usage,
            )

        pi.on("tool_result", on_tool_result)

    harness = await create_harness(tmp_path, tools=[_echo_tool(execute)], extension_factories=[factory])
    try:
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("echo", {"text": "hello"})], stop_reason="toolUse"),
                lambda context, options, state, model: faux_assistant_message(_tool_result_text(context)),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("hi"), timeout=10)

        assert any("patched result" in text for text in get_assistant_texts(harness))
        tool_result = next(
            (
                message
                for message in harness.session.messages
                if getattr(message, "role", "") == "toolResult"
                and isinstance(getattr(message, "details", None), dict)
                and message.details.get("patched") is True
            ),
            None,
        )
        assert observed == [tool_usage]
        assert tool_result is not None
        assert tool_result.usage == patched_tool_usage
    finally:
        harness.cleanup()


async def test_allows_extension_context_handlers_to_modify_messages_before_the_llm_call(tmp_path: Path) -> None:
    import dataclasses

    def factory(pi: ExtensionAPI) -> None:
        async def on_context(event, ctx) -> ContextEventResult:
            return ContextEventResult(
                messages=[
                    dataclasses.replace(message, content=[TextContent(text="rewritten")])
                    if getattr(message, "role", "") == "user"
                    else message
                    for message in event.messages
                ]
            )

        pi.on("context", on_context)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        seen: list[str] = []

        def respond(context, options, state, model):
            user = next((m for m in context.messages if getattr(m, "role", "") == "user"), None)
            seen.append(_text_of(user) if user is not None else "")
            return faux_assistant_message("done")

        harness.set_responses([respond])

        await asyncio.wait_for(harness.session.prompt("original"), timeout=10)

        assert seen == ["rewritten"]
        stored = next((m for m in harness.session.messages if getattr(m, "role", "") == "user"), None)
        assert stored is not None
        assert stored.content == [TextContent(text="original")]
    finally:
        harness.cleanup()


async def test_allows_extension_input_handlers_to_transform_or_handle_input(tmp_path: Path) -> None:
    seen_api: list[ExtensionAPI] = []

    def factory(pi: ExtensionAPI) -> None:
        seen_api.append(pi)

        async def on_input(event, ctx) -> InputEventResult:
            if event.text == "ping":
                return InputEventResult(action="handled")
            return InputEventResult(action="transform", text=f"transformed:{event.text}")

        pi.on("input", on_input)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        seen: list[str] = []

        def respond(context, options, state, model):
            user = next((m for m in context.messages if getattr(m, "role", "") == "user"), None)
            seen.append(_text_of(user) if user is not None else "")
            return faux_assistant_message("done")

        harness.set_responses([respond, respond])

        await asyncio.wait_for(harness.session.prompt("hello"), timeout=10)
        await asyncio.wait_for(harness.session.prompt("ping"), timeout=10)

        assert seen == ["transformed:hello"]
        assert len([m for m in harness.session.messages if getattr(m, "role", "") == "user"]) == 1
        assert seen_api
    finally:
        harness.cleanup()


async def test_allows_extension_commands_to_inspect_live_system_prompt_options(tmp_path: Path) -> None:
    seen_options: list[Any] = []

    def factory(pi: ExtensionAPI) -> None:
        async def handler(args: str, ctx) -> None:
            options = ctx.get_system_prompt_options()
            seen_options.append(options)
            if options.selected_tools is not None:
                options.selected_tools.append("mutated_tool")

        pi.register_command("inspect-options", handler=handler, description="Inspect system prompt options")

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        await asyncio.wait_for(harness.session.prompt("/inspect-options"), timeout=10)
        await asyncio.wait_for(harness.session.prompt("/inspect-options"), timeout=10)

        assert len(seen_options) == 2
        assert seen_options[0] is seen_options[1]
        assert seen_options[0].cwd == str(harness.temp_dir)
        assert "read" in (seen_options[0].selected_tools or [])
        assert "mutated_tool" in (seen_options[1].selected_tools or [])
    finally:
        harness.cleanup()


async def test_allows_before_agent_start_handlers_to_inject_messages_and_modify_the_system_prompt(
    tmp_path: Path,
) -> None:
    def factory(pi: ExtensionAPI) -> None:
        async def on_before_agent_start(event, ctx) -> BeforeAgentStartEventResult:
            return BeforeAgentStartEventResult(
                message=CustomMessage(
                    custom_type="before-start",
                    content=[TextContent(text="injected")],
                    display=True,
                    details={"injected": True},
                    timestamp=0,
                ),
                system_prompt=f"{event.system_prompt}\n\nextra instructions",
            )

        pi.on("before_agent_start", on_before_agent_start)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        seen_prompt: list[str] = []
        saw_injected: list[bool] = []

        def respond(context, options, state, model):
            seen_prompt.append(context.system_prompt or "")
            saw_injected.append(
                any(
                    getattr(message, "role", "") == "user"
                    and any(isinstance(part, TextContent) and part.text == "injected" for part in message.content)
                    for message in context.messages
                )
            )
            return faux_assistant_message("done")

        harness.set_responses([respond])

        await asyncio.wait_for(harness.session.prompt("hello"), timeout=10)

        assert "extra instructions" in seen_prompt[0]
        assert saw_injected[0] is True
        assert any(
            getattr(message, "role", "") == "custom" and getattr(message, "custom_type", "") == "before-start"
            for message in harness.session.messages
        )
    finally:
        harness.cleanup()


async def test_bind_extensions_emits_session_start(tmp_path: Path) -> None:
    lifecycle_events: list[str] = []

    def factory(pi: ExtensionAPI) -> None:
        async def on_start(event, ctx) -> None:
            lifecycle_events.append(f"start:{event.reason}")

        async def on_shutdown(event, ctx) -> None:
            lifecycle_events.append(f"shutdown:{event.reason}")

        pi.on("session_start", on_start)
        pi.on("session_shutdown", on_shutdown)

    harness = await create_harness(tmp_path, extension_factories=[factory])
    try:
        # TS: `await harness.session.bindExtensions({ shutdownHandler: () => {} })`.
        # This port's `bind_extensions()` takes no arguments -- the TS options
        # only configure the extension UI host and the process shutdown hook,
        # neither of which is ported.
        await harness.session.bind_extensions()

        # TS then calls `await harness.session.reload()` and asserts the full
        # sequence `["start:startup", "shutdown:reload", "start:reload"]`.
        # `AgentSession.reload()` does not exist in this port (documented
        # omission: no runtime rebuild / session replacement in-place), so the
        # two reload events cannot be produced here. `session_shutdown` itself
        # *is* emitted, but only by `AgentSessionRuntime` when it tears a
        # session down -- covered by the runtime's own tests, not reachable
        # from a bare `AgentSession`.
        assert lifecycle_events == ["start:startup"]
    finally:
        harness.cleanup()
