"""Python port of `packages/coding-agent/test/test-harness.test.ts`.

The TypeScript file tests `test/test-harness.ts`, a bespoke harness with its own
declarative faux stream. This port has no equivalent of that file: its harness is
`tests/suite/harness.py`, which drives `pi_ai.providers.faux` (the port of
`test/suite/harness.ts`). Each case below is therefore run against
`tests/suite/harness.py`, and the two cases that only pin behaviour of the
bespoke TypeScript faux stream are skipped with a note.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.providers.faux import faux_assistant_message, faux_text, faux_thinking, faux_tool_call
from pi_ai.types import Context, Model, TextContent

from pi_coding_agent.core.extensions.loader import load_extensions
from pi_coding_agent.core.extensions.runner import ExtensionRunner

_SUITE_DIR = str(Path(__file__).parent / "suite")
if _SUITE_DIR not in sys.path:
    sys.path.insert(0, _SUITE_DIR)

from harness import Harness, create_harness  # noqa: E402


@pytest.fixture
def harness_cleanup() -> Iterator[list[Harness]]:
    created: list[Harness] = []
    yield created
    for harness in created:
        harness.cleanup()


def _echo_tool(on_execute: Callable[[], None] | None = None) -> AgentTool:
    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        if on_execute is not None:
            on_execute()
        return AgentToolResult(content=[TextContent(text="echoed")], details={})

    return AgentTool(
        name="echo",
        label="Echo",
        description="Echo back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        execute=execute,
    )


def _assistant_messages(harness: Harness) -> list[Any]:
    return [m for m in harness.session.messages if getattr(m, "role", None) == "assistant"]


def _first_text(message: Any) -> str | None:
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return None


async def test_simple_text_response(tmp_path: Path, harness_cleanup: list[Harness]) -> None:
    harness = await create_harness(tmp_path)
    harness_cleanup.append(harness)
    harness.set_responses([faux_assistant_message("hello world")])

    await harness.session.prompt("hi")

    assert harness.faux.state.call_count == 1

    assistant_messages = _assistant_messages(harness)
    assert len(assistant_messages) == 1
    assert assistant_messages[0].content == [TextContent(text="hello world")]
    assert assistant_messages[0].stop_reason == "stop"


async def test_response_sequence(tmp_path: Path, harness_cleanup: list[Harness]) -> None:
    harness = await create_harness(tmp_path)
    harness_cleanup.append(harness)
    harness.set_responses(
        [
            faux_assistant_message("first"),
            faux_assistant_message("second"),
            faux_assistant_message("third"),
        ]
    )

    await harness.session.prompt("a")
    await harness.session.prompt("b")
    await harness.session.prompt("c")

    assert harness.faux.state.call_count == 3
    assert [_first_text(m) for m in _assistant_messages(harness)] == ["first", "second", "third"]


async def test_tool_call_response_triggers_tool_execution(tmp_path: Path, harness_cleanup: list[Harness]) -> None:
    executed: list[bool] = []
    echo_tool = _echo_tool(lambda: executed.append(True))

    harness = await create_harness(tmp_path, tools=[echo_tool])
    harness_cleanup.append(harness)
    harness.set_responses(
        [
            faux_assistant_message(faux_tool_call("echo", {"text": "hi"})),
            faux_assistant_message("done after tool"),
        ]
    )

    await harness.session.prompt("use the tool")

    assert executed == [True]
    assert harness.faux.state.call_count == 2

    tool_results = [m for m in harness.session.messages if getattr(m, "role", None) == "toolResult"]
    assert len(tool_results) == 1


async def test_error_response(tmp_path: Path, harness_cleanup: list[Harness]) -> None:
    harness = await create_harness(tmp_path, settings={"retry": {"enabled": False}})
    harness_cleanup.append(harness)
    harness.set_responses(
        [faux_assistant_message([], stop_reason="error", error_message="something broke")],
    )

    await harness.session.prompt("hi")

    assistant_messages = _assistant_messages(harness)
    assert len(assistant_messages) == 1
    assert assistant_messages[0].stop_reason == "error"
    assert assistant_messages[0].error_message == "something broke"


async def test_turns_a_pending_terminal_response_into_an_error(tmp_path: Path, harness_cleanup: list[Harness]) -> None:
    harness = await create_harness(tmp_path, settings={"retry": {"enabled": False}})
    harness_cleanup.append(harness)
    harness.set_responses([faux_assistant_message("partial", stop_reason="pending")])

    await harness.session.prompt("hi")

    assistant_messages = _assistant_messages(harness)
    assert len(assistant_messages) == 1
    assert assistant_messages[0].stop_reason == "error"
    assert assistant_messages[0].error_message == "Faux response ended without a stop reason"


async def test_retry_on_transient_error(tmp_path: Path, harness_cleanup: list[Harness]) -> None:
    harness = await create_harness(
        tmp_path,
        settings={"retry": {"enabled": True, "maxRetries": 3, "baseDelayMs": 1}},
    )
    harness_cleanup.append(harness)
    harness.set_responses(
        [
            faux_assistant_message([], stop_reason="error", error_message="overloaded_error"),
            faux_assistant_message("recovered"),
        ]
    )

    await harness.session.prompt("hi")

    assert harness.faux.state.call_count == 2

    retry_starts = harness.events_of_type("auto_retry_start")
    assert len(retry_starts) == 1

    retry_ends = harness.events_of_type("auto_retry_end")
    assert len(retry_ends) == 1
    assert retry_ends[0].success is True


@pytest.mark.skip(
    reason="`test-harness.ts`'s bespoke faux stream lets a response declare `usage`. "
    "`pi_ai.providers.faux` (this port's only faux provider) always overwrites usage "
    "with its own token estimate, so there is nothing to assert."
)
async def test_custom_usage_numbers() -> None:  # pragma: no cover - skipped
    raise AssertionError


async def test_event_capture(tmp_path: Path, harness_cleanup: list[Harness]) -> None:
    harness = await create_harness(tmp_path)
    harness_cleanup.append(harness)
    harness.set_responses([faux_assistant_message("hello")])

    await harness.session.prompt("hi")

    assert len(harness.events_of_type("agent_start")) == 1
    assert len(harness.events_of_type("agent_end")) == 1
    # user + assistant
    assert len(harness.events_of_type("message_end")) >= 2


async def test_context_capture(tmp_path: Path, harness_cleanup: list[Harness]) -> None:
    contexts: list[Context] = []

    def record(context: Context, _options: Any, _state: Any, _model: Model) -> Any:
        contexts.append(context)
        return faux_assistant_message("reply")

    harness = await create_harness(tmp_path)
    harness_cleanup.append(harness)
    harness.set_responses([record])

    await harness.session.prompt("my question")

    assert len(contexts) == 1
    assert any(getattr(m, "role", None) == "user" for m in contexts[0].messages)


@pytest.mark.skip(
    reason="`test-harness.ts`'s bespoke faux stream cycles its response list "
    "(`responses[callCount % responses.length]`). `pi_ai.providers.faux` consumes a "
    "queue and errors with 'No more faux responses queued' instead, so there is no "
    "wrap-around behaviour to pin here."
)
async def test_wraps_around_when_more_calls_than_responses() -> None:  # pragma: no cover - skipped
    raise AssertionError


def _stream_events(harness: Harness, event_type: str) -> list[Any]:
    return [
        event
        for event in harness.events_of_type("message_update")
        if getattr(event.assistant_message_event, "type", None) == event_type
    ]


async def test_streams_text_deltas(tmp_path: Path, harness_cleanup: list[Harness]) -> None:
    harness = await create_harness(tmp_path)
    harness_cleanup.append(harness)
    harness.set_responses([faux_assistant_message("hello world")])

    await harness.session.prompt("hi")

    text_deltas = _stream_events(harness, "text_delta")
    assert len(text_deltas) > 0
    assert "".join(e.assistant_message_event.delta for e in text_deltas) == "hello world"


async def test_streams_thinking_deltas(tmp_path: Path, harness_cleanup: list[Harness]) -> None:
    harness = await create_harness(tmp_path)
    harness_cleanup.append(harness)
    harness.set_responses([faux_assistant_message([faux_thinking("let me think about this"), faux_text("answer")])])

    await harness.session.prompt("hi")

    assert len(_stream_events(harness, "thinking_start")) == 1
    thinking_deltas = _stream_events(harness, "thinking_delta")
    assert len(thinking_deltas) > 0
    assert len(_stream_events(harness, "thinking_end")) == 1
    assert "".join(e.assistant_message_event.delta for e in thinking_deltas) == "let me think about this"


async def test_streams_tool_call_deltas(tmp_path: Path, harness_cleanup: list[Harness]) -> None:
    harness = await create_harness(tmp_path, tools=[_echo_tool()])
    harness_cleanup.append(harness)
    harness.set_responses(
        [
            faux_assistant_message(faux_tool_call("echo", {"text": "hi"})),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("use tool")

    assert len(_stream_events(harness, "toolcall_start")) == 1
    assert len(_stream_events(harness, "toolcall_delta")) > 0
    assert len(_stream_events(harness, "toolcall_end")) == 1


async def test_streams_thinking_then_text_then_tool_call_in_order(
    tmp_path: Path, harness_cleanup: list[Harness]
) -> None:
    harness = await create_harness(tmp_path, tools=[_echo_tool()])
    harness_cleanup.append(harness)
    harness.set_responses(
        [
            faux_assistant_message(
                [
                    faux_thinking("hmm"),
                    faux_text("I will call a tool"),
                    faux_tool_call("echo", {"text": "x"}),
                ]
            ),
            faux_assistant_message("final"),
        ]
    )

    await harness.session.prompt("do it")

    stream_types = [
        getattr(event.assistant_message_event, "type", None) for event in harness.events_of_type("message_update")
    ]
    first_thinking = stream_types.index("thinking_start")
    first_text = stream_types.index("text_start")
    first_toolcall = stream_types.index("toolcall_start")

    assert first_thinking < first_text
    assert first_text < first_toolcall


async def test_loads_inline_extension_factories_and_disambiguates_duplicate_commands(
    tmp_path: Path, harness_cleanup: list[Harness]
) -> None:
    calls: list[str] = []

    def alpha(pi: Any) -> None:
        async def handler(args: str, _ctx: Any) -> None:
            calls.append(f"alpha:{args}")

        pi.register_command("shared-cmd", description="Alpha command", handler=handler)

    def beta(pi: Any) -> None:
        async def handler(args: str, _ctx: Any) -> None:
            calls.append(f"beta:{args}")

        pi.register_command("shared-cmd", description="Beta command", handler=handler)

    harness = await create_harness(tmp_path, extension_factories=[alpha, beta])
    harness_cleanup.append(harness)

    runner = harness.session.extension_runner
    assert runner is not None

    assert [
        {
            "name": command.name,
            "invocationName": invocation_name,
            "description": command.description,
            "path": command.source_info.path,
        }
        for invocation_name, command in runner.get_registered_commands()
    ] == [
        {
            "name": "shared-cmd",
            "invocationName": "shared-cmd:1",
            "description": "Alpha command",
            "path": "<inline:1>",
        },
        {
            "name": "shared-cmd",
            "invocationName": "shared-cmd:2",
            "description": "Beta command",
            "path": "<inline:2>",
        },
    ]

    first = runner.get_command("shared-cmd:1")
    second = runner.get_command("shared-cmd:2")
    assert first is not None
    assert second is not None
    await first.handler("first", runner.create_command_context())
    await second.handler("second", runner.create_command_context())

    assert calls == ["alpha:first", "beta:second"]


async def test_loads_extensions_from_disk_and_disambiguates_duplicate_commands(tmp_path: Path) -> None:
    """The same disambiguation, but through the on-disk `load_extensions` path.

    TypeScript only has the inline case above; this keeps the `source_info.path`
    assertion honest for a real file, where the path is the resolved file rather
    than a synthetic `<inline:N>`.
    """
    calls_path = tmp_path / "calls.txt"
    alpha_path = tmp_path / "alpha.py"
    alpha_path.write_text(
        "def pi_extension(pi):\n"
        "    async def handler(args, ctx):\n"
        f"        with open({str(calls_path)!r}, 'a') as f:\n"
        "            f.write(f'alpha:{args}\\n')\n"
        "    pi.register_command('shared-cmd', description='Alpha command', handler=handler)\n",
        encoding="utf-8",
    )
    beta_path = tmp_path / "beta.py"
    beta_path.write_text(
        "def pi_extension(pi):\n"
        "    async def handler(args, ctx):\n"
        f"        with open({str(calls_path)!r}, 'a') as f:\n"
        "            f.write(f'beta:{args}\\n')\n"
        "    pi.register_command('shared-cmd', description='Beta command', handler=handler)\n",
        encoding="utf-8",
    )

    result = await load_extensions([str(alpha_path), str(beta_path)], str(tmp_path))
    assert result.errors == []
    runner = ExtensionRunner(result.extensions, cwd=str(tmp_path))

    commands = runner.get_registered_commands()
    assert [
        {
            "name": command.name,
            "invocationName": invocation_name,
            "description": command.description,
            "path": command.source_info.path,
        }
        for invocation_name, command in commands
    ] == [
        {
            "name": "shared-cmd",
            "invocationName": "shared-cmd:1",
            "description": "Alpha command",
            "path": str(alpha_path),
        },
        {
            "name": "shared-cmd",
            "invocationName": "shared-cmd:2",
            "description": "Beta command",
            "path": str(beta_path),
        },
    ]

    first = runner.get_command("shared-cmd:1")
    second = runner.get_command("shared-cmd:2")
    assert first is not None
    assert second is not None
    await first.handler("first", runner.create_command_context())
    await second.handler("second", runner.create_command_context())

    assert calls_path.read_text(encoding="utf-8").split() == ["alpha:first", "beta:second"]


async def test_session_persistence_works(tmp_path: Path, harness_cleanup: list[Harness]) -> None:
    harness = await create_harness(tmp_path)
    harness_cleanup.append(harness)
    harness.set_responses([faux_assistant_message("persisted")])

    await harness.session.prompt("hi")

    entries = harness.session_manager.get_entries()
    # user + assistant
    assert len([e for e in entries if e.type == "message"]) >= 2
