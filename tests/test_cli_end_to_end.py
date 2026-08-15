"""End-to-end check of the CLI wiring against a fake OpenAI-compatible server.

Runs the real registry, provider, HTTP layer, agent loop and CLI rendering; only
the network endpoint is faked.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pi_ai.providers import openai_compatible_provider
from pi_ai.registry import Models
from pi_ai.types import Model, ModelCost


def sse(chunks: list[dict]) -> str:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


@pytest.fixture
def fake_model() -> Model:
    return Model(
        id="fake-1",
        name="Fake",
        api="openai-completions",
        provider="fake",
        base_url="https://fake.invalid/v1",
        context_window=8000,
        max_tokens=1000,
        cost=ModelCost(input=1.0, output=1.0),
    )


def make_models(fake_model: Model, responses: list[str], capture: list[dict]) -> Models:
    remaining = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        capture.append(json.loads(request.content))
        return httpx.Response(200, text=remaining.pop(0), headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = openai_compatible_provider("fake", "Fake", "https://fake.invalid/v1", ["FAKE_API_KEY"], [fake_model])

    original_stream_simple = provider.api.stream_simple

    class BoundApi:
        @staticmethod
        def stream_simple(model, context, options=None, **kwargs):
            return original_stream_simple(model, context, options, client=client, **kwargs)

        @staticmethod
        def stream(model, context, options=None, **kwargs):
            return provider.api.stream(model, context, options, client=client, **kwargs)

    provider.api = BoundApi
    return Models([provider])


async def test_cli_runs_a_prompt_end_to_end(fake_model, monkeypatch, capsys):
    from pi_coding_agent.cli.entry import run_once

    monkeypatch.setenv("FAKE_API_KEY", "secret")
    capture: list[dict] = []
    models = make_models(
        fake_model,
        [sse([{"id": "r", "choices": [{"delta": {"content": "All done."}, "finish_reason": "stop"}]}])],
        capture,
    )

    exit_code = await run_once("say hi", fake_model, models, [], "be brief")

    assert exit_code == 0
    assert "All done." in capsys.readouterr().out
    assert capture[0]["messages"][0] == {"role": "system", "content": "be brief"}
    assert capture[0]["messages"][1]["content"][0]["text"] == "say hi"


async def test_cli_runs_a_tool_call_round_trip(fake_model, monkeypatch, capsys, tmp_path):
    from pi_agent import AgentTool, AgentToolResult
    from pi_ai import TextContent

    from pi_coding_agent.cli.entry import run_once

    monkeypatch.setenv("FAKE_API_KEY", "secret")

    async def execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"read {params['path']}")], details={})

    tool = AgentTool(
        name="read",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        label="Read",
        execute=execute,
    )

    capture: list[dict] = []
    models = make_models(
        fake_model,
        [
            sse(
                [
                    {
                        "id": "r",
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "c1",
                                            "function": {"name": "read", "arguments": '{"path": "a.txt"}'},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    }
                ]
            ),
            sse([{"id": "r2", "choices": [{"delta": {"content": "Done reading."}, "finish_reason": "stop"}]}]),
        ],
        capture,
    )

    exit_code = await run_once("read a.txt", fake_model, models, [tool], "be brief")
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "-> read" in output
    assert "[read ok] read a.txt" in output
    assert "Done reading." in output

    # The second request replays the tool call and its result.
    second_request_roles = [m["role"] for m in capture[1]["messages"]]
    assert second_request_roles == ["system", "user", "assistant", "tool"]
    assert capture[1]["messages"][3]["content"] == "read a.txt"


async def test_cli_reports_provider_errors(fake_model, monkeypatch, capsys):
    from pi_coding_agent.cli.entry import run_once

    monkeypatch.setenv("FAKE_API_KEY", "secret")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text='{"error": {"message": "rate limited"}}')

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = openai_compatible_provider("fake", "Fake", "https://fake.invalid/v1", ["FAKE_API_KEY"], [fake_model])
    original = provider.api

    class BoundApi:
        @staticmethod
        def stream_simple(model, context, options=None, **kwargs):
            return original.stream_simple(model, context, options, client=client, **kwargs)

    provider.api = BoundApi
    models = Models([provider])

    exit_code = await run_once("hi", fake_model, models, [], "sys")

    assert exit_code == 1
    assert "rate limited" in capsys.readouterr().out


async def test_cli_quiet_mode_skips_non_assistant_and_empty_messages(fake_model, monkeypatch, capsys, tmp_path):
    from pi_agent import AgentTool, AgentToolResult
    from pi_ai import TextContent

    from pi_coding_agent.cli.entry import run_once

    monkeypatch.setenv("FAKE_API_KEY", "secret")

    async def execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"read {params['path']}")], details={})

    tool = AgentTool(
        name="read",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        label="Read",
        execute=execute,
    )

    capture: list[dict] = []
    models = make_models(
        fake_model,
        [
            sse(
                [
                    {
                        "id": "r",
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "c1",
                                            "function": {"name": "read", "arguments": '{"path": "a.txt"}'},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    }
                ]
            ),
            sse([{"id": "r2", "choices": [{"delta": {"content": "final"}, "finish_reason": "stop"}]}]),
        ],
        capture,
    )

    exit_code = await run_once("read a.txt", fake_model, models, [tool], "sys", quiet=True)

    assert exit_code == 0
    # The user, tool-call and tool-result messages are skipped by the
    # quiet-mode summary loop; only the final assistant text is printed.
    assert capsys.readouterr().out.strip() == "final"


async def test_stream_simple_requires_configured_provider(fake_model, monkeypatch):
    from pi_ai.types import Context

    monkeypatch.delenv("FAKE_API_KEY", raising=False)
    provider = openai_compatible_provider("fake", "Fake", "https://fake.invalid/v1", ["FAKE_API_KEY"], [fake_model])
    models = Models([provider])

    # A stream-returning entry point never throws: `lazyStream` reports setup
    # failures in-band, as a single error event plus a `stop_reason="error"`
    # result.
    stream = await models.stream_simple(fake_model, Context(messages=[]))
    events = [event async for event in stream]
    assert [event.type for event in events] == ["error"]
    result = await stream.result()
    assert result.stop_reason == "error"
    assert "fake" in (result.error_message or "").lower()


async def test_cli_quiet_mode_prints_only_the_final_answer(fake_model, monkeypatch, capsys):
    from pi_coding_agent.cli.entry import run_once

    monkeypatch.setenv("FAKE_API_KEY", "secret")
    capture: list[dict] = []
    models = make_models(
        fake_model,
        [sse([{"id": "r", "choices": [{"delta": {"content": "quiet answer"}, "finish_reason": "stop"}]}])],
        capture,
    )

    exit_code = await run_once("say hi", fake_model, models, [], "be brief", quiet=True)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.strip() == "quiet answer"


async def test_cli_quiet_mode_suppresses_output_on_provider_errors(fake_model, monkeypatch, capsys):
    from pi_coding_agent.cli.entry import run_once

    monkeypatch.setenv("FAKE_API_KEY", "secret")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text='{"error": {"message": "rate limited"}}')

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = openai_compatible_provider("fake", "Fake", "https://fake.invalid/v1", ["FAKE_API_KEY"], [fake_model])
    original = provider.api

    class BoundApi:
        @staticmethod
        def stream_simple(model, context, options=None, **kwargs):
            return original.stream_simple(model, context, options, client=client, **kwargs)

    provider.api = BoundApi
    models = Models([provider])

    exit_code = await run_once("hi", fake_model, models, [], "sys", quiet=True)

    assert exit_code == 1
    assert capsys.readouterr().out == ""
