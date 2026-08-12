"""Tests for `pi_coding_agent.core.sdk`.

Ports the applicable cases from `packages/coding-agent/test/sdk-session-manager.test.ts`
and `sdk-stream-options.test.ts` in the TypeScript "pi" monorepo, plus direct
coverage of `create_agent_session`'s model/thinking-level restore-then-default
logic and tool allow/exclude/active-name computation (both load-bearing parts
of `sdk.py` not otherwise exercised by `test_agent_session_runtime.py`).

`sdk-openrouter-attribution.test.ts` is ported: its 13 cases are unit-tested
against `merge_provider_attribution_headers` in `test_provider_attribution.py`,
and `TestCreateAgentSessionStreamOptions.test_attribution_headers_reach_the_provider_through_a_real_session`
and `test_attribution_headers_are_absent_when_telemetry_is_disabled` below pin
the wiring through a real `create_agent_session` the way the TypeScript test
does. Only the `before_provider_headers` case in `sdk-stream-options.test.ts`
is NOT ported: it depends on the extension system's `transformHeaders` hook,
which this port's `sdk.py` deliberately drops (see its module docstring).
`sdk-codex-cache-probe-tool-loop.ts` is a manual repro script (not a real test
file, no `describe`/`it`). `sdk-skills.test.ts`'s three cases are ported at the
end of this file.

No test performs real network I/O: models are resolved from an in-memory
`openai_compatible_provider` fake plus a fake `stream_simple` API object that
records the options it receives, and all filesystem work uses `tmp_path`.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.auth.helpers import env_api_key_auth
from pi_ai.auth.types import ProviderAuth
from pi_ai.providers import openai_compatible_provider
from pi_ai.providers.all import get_builtin_model
from pi_ai.registry import create_provider
from pi_ai.types import (
    AssistantMessage,
    DoneEvent,
    Model,
    ModelCost,
    ProviderResponse,
    SimpleStreamOptions,
    TextContent,
    Usage,
    UserMessage,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pi_coding_agent.core.extensions.types import ContextEventResult, Extension
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.sdk import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.core.session_manager import SessionManager, get_default_session_dir
from pi_coding_agent.core.settings_manager import SettingsManager

TIMEOUT = 5.0


async def _wait(awaitable: Any, timeout: float = TIMEOUT) -> Any:
    return await asyncio.wait_for(awaitable, timeout=timeout)


def _fake_provider(provider_id: str = "test", reasoning: bool = False) -> object:
    return openai_compatible_provider(
        provider_id=provider_id,
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
                reasoning=reasoning,
            )
        ],
    )


async def _runtime_with_logged_in_fake(tmp_path: Path, reasoning: bool = False) -> ModelRuntime:
    runtime = await _wait(
        ModelRuntime.create(agent_dir=tmp_path / "agent", providers=[_fake_provider(reasoning=reasoning)])
    )
    await _wait(runtime.login("test", "fake-key"))
    return runtime


def _scripted_provider(responses: list[AssistantMessage]) -> object:
    """A provider whose `stream_simple` replays `responses` in order (no network I/O),
    used to drive a real `AgentSession.prompt()` turn end-to-end so a session actually
    persists to disk (`SessionManager` only flushes once it has a completed assistant
    message -- see `_persist_entry`'s `has_assistant` gate)."""
    remaining = list(responses)

    class _ScriptedApi:
        def stream(self, model: Model, context: Any, options: Any = None, **kwargs: Any) -> AssistantMessageEventStream:
            raise NotImplementedError

        def stream_simple(
            self, model: Model, context: Any, options: Any = None, **kwargs: Any
        ) -> AssistantMessageEventStream:
            if not remaining:
                raise AssertionError("stream_simple called more times than there are scripted responses")
            message = remaining.pop(0)
            stream = AssistantMessageEventStream()
            stream.push(DoneEvent(reason=message.stop_reason, message=message))
            stream.end()
            return stream

    return create_provider(
        id="test",
        name="Fake Test Provider",
        auth=ProviderAuth(api_key=env_api_key_auth("Fake Test Provider API key", ["FAKE_TEST_API_KEY"])),
        api=_ScriptedApi(),
        base_url="https://fake.example.com",
        models=[
            Model(
                id="test-model",
                name="Test Model",
                api="openai-completions",
                context_window=1000,
                max_tokens=100,
                cost=ModelCost(input=0, output=0),
                # Reasoning enabled so "high" is a supported thinking level and
                # doesn't get clamped down to "off" by `clamp_thinking_level`.
                reasoning=True,
            )
        ],
    )


class _CapturingApi:
    """Fake `ApiModule` recording the `SimpleStreamOptions` it receives, mirroring
    `sdk-stream-options.test.ts`'s `modelRegistry.registerProvider(..., { streamSimple })`."""

    def __init__(self) -> None:
        self.captured_options: SimpleStreamOptions | None = None

    def stream(self, model: Model, context: Any, options: Any = None, **kwargs: Any) -> AssistantMessageEventStream:
        raise NotImplementedError

    def stream_simple(
        self, model: Model, context: Any, options: SimpleStreamOptions | None = None, **kwargs: Any
    ) -> AssistantMessageEventStream:
        self.captured_options = options
        stream = AssistantMessageEventStream()
        message = AssistantMessage(
            api=model.api,
            provider=model.provider,
            model=model.id,
            content=[],
            usage=Usage(),
            stop_reason="stop",
        )
        stream.push(DoneEvent(reason="stop", message=message))
        stream.end()
        return stream


async def _capture_stream_options(
    tmp_path: Path,
    settings: dict[str, Any] | None = None,
    request_options: SimpleStreamOptions | None = None,
    api_id: str = "openai-completions",
    provider_id: str = "capture-provider",
    base_url: str = "https://capture.invalid/v1",
) -> SimpleStreamOptions | None:
    """Builds a real session through `create_agent_session` and drives one `stream_function`
    call, returning the `SimpleStreamOptions` the fake provider's `stream_simple` observed.

    `api_id` mirrors the TypeScript helper's `api` parameter: several cases run the
    OpenAI Codex API specifically, because its timeout handling used to diverge from
    the shared default."""
    api = _CapturingApi()
    model = Model(
        id="capture-model",
        name="Capture Model",
        api=api_id,
        provider=provider_id,
        base_url=base_url,
        context_window=128_000,
        max_tokens=4096,
        cost=ModelCost(input=0, output=0),
        headers={"x-model": "model"},
    )
    provider = create_provider(
        id=provider_id,
        name="Capture Provider",
        auth=ProviderAuth(api_key=env_api_key_auth("Capture Provider API key", ["CAPTURE_PROVIDER_API_KEY"])),
        api=api,
        models=[model],
        headers={"x-provider": "provider"},
    )
    model_runtime = await _wait(ModelRuntime.create(agent_dir=tmp_path / "agent", providers=[provider]))
    await _wait(model_runtime.login(provider_id, "fake-key"))
    settings_manager = SettingsManager.in_memory(settings or {})
    session_manager = SessionManager.in_memory(str(tmp_path / "project"))

    result = await _wait(
        create_agent_session(
            CreateAgentSessionOptions(
                cwd=str(tmp_path / "project"),
                agent_dir=str(tmp_path / "agent"),
                model=model_runtime.find_model(f"{provider_id}/capture-model"),
                model_runtime=model_runtime,
                settings_manager=settings_manager,
                session_manager=session_manager,
            )
        )
    )
    try:
        stream = await _wait(result.session.agent.stream_function(model, {"messages": []}, request_options))
        await _wait(stream.result())
        return api.captured_options
    finally:
        result.session.dispose()


class TestCreateAgentSessionManagerDefaults:
    """TS drives these four cases with `getModel("anthropic", "claude-sonnet-4-5")`, a real
    built-in catalog model, and never passes an explicit `modelRuntime` -- `createAgentSession`
    builds its own from the (real, offline, bundled) provider catalog. None of the four cases
    ever streams from the model, so there is no need for a fake provider or a logged-in
    credential here; using `get_builtin_model` keeps this port on the same real collaborator
    TS uses instead of substituting an invented fake provider."""

    def test_uses_agent_dir_for_default_persisted_session_path(self, tmp_path: Path) -> None:
        cwd = str(tmp_path / "project")
        agent_dir = str(tmp_path / "agent")
        Path(cwd).mkdir(parents=True)
        Path(agent_dir).mkdir(parents=True)

        async def run() -> None:
            model = get_builtin_model("anthropic", "claude-sonnet-4-5")
            assert model is not None
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=cwd,
                        agent_dir=agent_dir,
                        model=model,
                    )
                )
            )
            try:
                # TS computes the expected directory literally rather than calling the
                # implementation, so the assertion cannot be tautological.
                safe_path = "--" + re.sub(r"[/\\:]", "-", re.sub(r"^[/\\]", "", cwd)) + "--"
                expected_session_dir = str(Path(agent_dir) / "sessions" / safe_path)
                assert expected_session_dir == get_default_session_dir(cwd, agent_dir)
                session_dir = result.session.session_manager.get_session_dir()
                session_file = result.session.session_manager.get_session_file()
                assert session_dir == expected_session_dir
                assert session_file is not None
                assert session_file.startswith(f"{expected_session_dir}/")
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_keeps_an_explicit_session_manager_override(self, tmp_path: Path) -> None:
        cwd = str(tmp_path / "project")
        agent_dir = str(tmp_path / "agent")
        Path(cwd).mkdir(parents=True)
        Path(agent_dir).mkdir(parents=True)

        async def run() -> None:
            model = get_builtin_model("anthropic", "claude-sonnet-4-5")
            assert model is not None
            session_manager = SessionManager.in_memory(cwd)
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=cwd,
                        agent_dir=agent_dir,
                        model=model,
                        session_manager=session_manager,
                    )
                )
            )
            try:
                assert result.session.session_manager is session_manager
                assert result.session.session_manager.is_persisted() is False
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_derives_cwd_from_explicit_session_manager_when_cwd_omitted(self, tmp_path: Path) -> None:
        agent_dir = str(tmp_path / "agent")
        session_cwd = str(tmp_path / "session-project")
        Path(agent_dir).mkdir(parents=True)
        Path(session_cwd).mkdir(parents=True)

        async def run() -> None:
            model = get_builtin_model("anthropic", "claude-sonnet-4-5")
            assert model is not None
            session_manager = SessionManager.in_memory(session_cwd)
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        agent_dir=agent_dir,
                        model=model,
                        session_manager=session_manager,
                    )
                )
            )
            try:
                assert result.session.session_manager is session_manager
                assert f"Current working directory: {session_cwd}" in result.session.system_prompt

                bash_tool = next(t for t in result.session.agent.state.tools if t.name == "bash")
                bash_result = await _wait(bash_tool.execute("test", {"command": "pwd"}))
                output = "".join(item.text for item in bash_result.content if getattr(item, "type", None) == "text")
                assert str(Path(output.strip()).resolve()) == str(Path(session_cwd).resolve())
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_exposes_current_session_state_to_the_built_in_bash_tool(self, tmp_path: Path) -> None:
        cwd = str(tmp_path / "project")
        agent_dir = str(tmp_path / "agent")
        Path(cwd).mkdir(parents=True)
        Path(agent_dir).mkdir(parents=True)

        async def run() -> None:
            model = get_builtin_model("anthropic", "claude-sonnet-4-5")
            assert model is not None
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=cwd,
                        agent_dir=agent_dir,
                        model=model,
                        thinking_level="high",
                    )
                )
            )
            try:
                assert result.session.session_file
                assert (
                    "You can inspect PI_* environment variables for current model and session details."
                    in result.session.system_prompt
                )

                bash_tool = next(t for t in result.session.agent.state.tools if t.name == "bash")
                bash_result = await _wait(
                    bash_tool.execute(
                        "test",
                        {
                            "command": (
                                'printf \'%s\\n\' "$PI_SESSION_ID" "$PI_SESSION_FILE" '
                                '"$PI_PROVIDER" "$PI_MODEL" "$PI_REASONING_LEVEL"'
                            )
                        },
                    )
                )
                output = "".join(item.text for item in bash_result.content if getattr(item, "type", None) == "text")
                assert output.strip().split("\n") == [
                    result.session.session_id,
                    result.session.session_file,
                    model.provider,
                    model.id,
                    result.session.thinking_level,
                ]
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))


class TestCreateAgentSessionModelAndThinkingRestore:
    def test_restores_model_and_thinking_level_from_existing_session(self, tmp_path: Path) -> None:
        cwd = str(tmp_path / "project")

        async def run() -> None:
            model_runtime = await _wait(
                ModelRuntime.create(
                    agent_dir=tmp_path / "agent",
                    providers=[
                        _scripted_provider(
                            [
                                AssistantMessage(
                                    api="openai-completions",
                                    provider="test",
                                    model="test-model",
                                    content=[],
                                    usage=Usage(),
                                    stop_reason="stop",
                                )
                            ]
                        )
                    ],
                )
            )
            await _wait(model_runtime.login("test", "fake-key"))
            model = model_runtime.find_model("test/test-model")
            assert model is not None

            first_session_manager = SessionManager.create(cwd, str(tmp_path / "sessions"))
            first = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=cwd,
                        model=model,
                        model_runtime=model_runtime,
                        thinking_level="high",
                        session_manager=first_session_manager,
                    )
                )
            )
            session_file = first.session.session_file
            assert session_file is not None
            # Drive one real turn so the session actually persists to disk
            # (`SessionManager` only flushes once it has a completed assistant
            # message; see the `_persist_entry` `has_assistant` gate).
            await _wait(first.session.prompt("hi"))
            first.session.dispose()
            assert Path(session_file).exists()

            reopened_session_manager = SessionManager.open(session_file, str(tmp_path / "sessions"))
            second = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=cwd,
                        model_runtime=model_runtime,
                        session_manager=reopened_session_manager,
                    )
                )
            )
            try:
                assert second.session.agent.state.model.provider == "test"
                assert second.session.agent.state.model.id == "test-model"
                assert second.session.thinking_level == "high"
            finally:
                second.session.dispose()

        asyncio.run(_wait(run()))

    def test_falls_back_to_default_thinking_level_with_no_existing_session(self, tmp_path: Path) -> None:
        async def run() -> None:
            model_runtime = await _runtime_with_logged_in_fake(tmp_path)
            model = model_runtime.find_model("test/test-model")
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=str(tmp_path / "project"),
                        model=model,
                        model_runtime=model_runtime,
                        session_manager=SessionManager.in_memory(str(tmp_path / "project")),
                    )
                )
            )
            try:
                assert result.session.thinking_level in ("off", "low", "medium", "high")
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_thinking_level_is_off_when_no_model_is_available(self, tmp_path: Path) -> None:
        async def run() -> None:
            empty_runtime = await _wait(ModelRuntime.create(agent_dir=tmp_path / "agent", providers=[]))
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=str(tmp_path / "project"),
                        model_runtime=empty_runtime,
                        session_manager=SessionManager.in_memory(str(tmp_path / "project")),
                    )
                )
            )
            try:
                assert result.session.thinking_level == "off"
                assert result.model_fallback_message is not None
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))


class TestCreateAgentSessionToolSelection:
    def test_default_active_tools_are_the_builtin_defaults(self, tmp_path: Path) -> None:
        async def run() -> None:
            model_runtime = await _runtime_with_logged_in_fake(tmp_path)
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=str(tmp_path / "project"),
                        model=model_runtime.find_model("test/test-model"),
                        model_runtime=model_runtime,
                        session_manager=SessionManager.in_memory(str(tmp_path / "project")),
                    )
                )
            )
            try:
                active_names = {tool.name for tool in result.session.agent.state.tools}
                assert active_names == {"read", "bash", "edit", "write"}
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_no_tools_all_disables_the_full_registry(self, tmp_path: Path) -> None:
        async def run() -> None:
            model_runtime = await _runtime_with_logged_in_fake(tmp_path)
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=str(tmp_path / "project"),
                        model=model_runtime.find_model("test/test-model"),
                        model_runtime=model_runtime,
                        session_manager=SessionManager.in_memory(str(tmp_path / "project")),
                        no_tools="all",
                    )
                )
            )
            try:
                assert result.session.agent.state.tools == []
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_explicit_tools_list_restricts_the_registry_and_active_set(self, tmp_path: Path) -> None:
        async def run() -> None:
            model_runtime = await _runtime_with_logged_in_fake(tmp_path)
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=str(tmp_path / "project"),
                        model=model_runtime.find_model("test/test-model"),
                        model_runtime=model_runtime,
                        session_manager=SessionManager.in_memory(str(tmp_path / "project")),
                        tools=["read", "bash"],
                    )
                )
            )
            try:
                active_names = {tool.name for tool in result.session.agent.state.tools}
                assert active_names == {"read", "bash"}
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_exclude_tools_removes_a_default_active_tool(self, tmp_path: Path) -> None:
        async def run() -> None:
            model_runtime = await _runtime_with_logged_in_fake(tmp_path)
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=str(tmp_path / "project"),
                        model=model_runtime.find_model("test/test-model"),
                        model_runtime=model_runtime,
                        session_manager=SessionManager.in_memory(str(tmp_path / "project")),
                        exclude_tools=["write"],
                    )
                )
            )
            try:
                active_names = {tool.name for tool in result.session.agent.state.tools}
                assert active_names == {"read", "bash", "edit"}
                assert "write" not in {tool.name for tool in result.session.get_all_tools()}
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_custom_tools_are_merged_into_the_registry(self, tmp_path: Path) -> None:
        async def run() -> None:
            model_runtime = await _runtime_with_logged_in_fake(tmp_path)

            async def execute(_call_id: str, _args: dict[str, Any]) -> AgentToolResult:
                return AgentToolResult(content=[])

            custom_tool = AgentTool(
                name="custom_echo",
                description="echoes",
                parameters={"type": "object"},
                label="Custom Echo",
                execute=execute,
            )
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=str(tmp_path / "project"),
                        model=model_runtime.find_model("test/test-model"),
                        model_runtime=model_runtime,
                        session_manager=SessionManager.in_memory(str(tmp_path / "project")),
                        custom_tools={"custom_echo": custom_tool},
                    )
                )
            )
            try:
                assert "custom_echo" in {tool.name for tool in result.session.get_all_tools()}
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))


class TestCreateAgentSessionStreamOptions:
    def test_forwards_http_idle_timeout_ms_as_timeout_ms(self, tmp_path: Path) -> None:
        options = asyncio.run(_wait(_capture_stream_options(tmp_path, {"httpIdleTimeoutMs": 1234})))
        assert options is not None
        assert options.timeout_ms == 1234

    def test_forwards_http_idle_timeout_ms_as_timeout_ms_for_openai_codex(self, tmp_path: Path) -> None:
        options = asyncio.run(
            _wait(_capture_stream_options(tmp_path, {"httpIdleTimeoutMs": 1234}, api_id="openai-codex-responses"))
        )
        assert options is not None
        assert options.timeout_ms == 1234

    def test_request_timeout_ms_overrides_http_idle_timeout_ms(self, tmp_path: Path) -> None:
        options = asyncio.run(
            _wait(
                _capture_stream_options(
                    tmp_path,
                    {"httpIdleTimeoutMs": 1234},
                    SimpleStreamOptions(timeout_ms=0),
                    api_id="openai-codex-responses",
                )
            )
        )
        assert options is not None
        assert options.timeout_ms == 0

    def test_forwards_websocket_connect_timeout_ms_from_settings(self, tmp_path: Path) -> None:
        options = asyncio.run(
            _wait(
                _capture_stream_options(tmp_path, {"websocketConnectTimeoutMs": 1234}, api_id="openai-codex-responses")
            )
        )
        assert options is not None
        assert options.websocket_connect_timeout_ms == 1234

    def test_request_websocket_connect_timeout_ms_overrides_settings(self, tmp_path: Path) -> None:
        options = asyncio.run(
            _wait(
                _capture_stream_options(
                    tmp_path,
                    {"websocketConnectTimeoutMs": 1234},
                    SimpleStreamOptions(websocket_connect_timeout_ms=0),
                    api_id="openai-codex-responses",
                )
            )
        )
        assert options is not None
        assert options.websocket_connect_timeout_ms == 0

    def test_forwards_provider_retry_settings(self, tmp_path: Path) -> None:
        options = asyncio.run(
            _wait(
                _capture_stream_options(tmp_path, {"retry": {"provider": {"maxRetries": 2, "maxRetryDelayMs": 3000}}})
            )
        )
        assert options is not None
        assert options.max_retries == 2
        assert options.max_retry_delay_ms == 3000

    def test_assembles_provider_model_and_request_headers_without_a_transform(self, tmp_path: Path) -> None:
        options = asyncio.run(
            _wait(_capture_stream_options(tmp_path, {}, SimpleStreamOptions(headers={"x-explicit": "explicit"})))
        )
        assert options is not None
        # TS's `x-provider` header comes from `modelRegistry.registerProvider(name, config)`,
        # the dynamic/extension provider registration path. This port has no
        # `ModelRuntime.register_provider(name, config)` and therefore no
        # `extensionProviders` map, so a provider-level configured header has no
        # equivalent entry point here. The model-level and request-level halves of
        # the same assembly are asserted instead.
        assert options.headers["x-model"] == "model"
        assert options.headers["x-explicit"] == "explicit"
        # TS's "x-hook" assertion (a `before_provider_headers` extension handler mutating
        # the assembled headers) cannot hold here: TS threads a `transformHeaders` callback
        # through the stream options so pi-ai can invoke it *after* assembling provider +
        # model + request headers. `pi_ai.SimpleStreamOptions` has no such field -- header
        # assembly happens inside `ModelRegistry._resolve_request`, downstream of anything
        # `sdk.py` can wrap -- so `ExtensionRunner.emit_before_provider_headers` currently
        # has no caller. Adding the callback would mean changing `pi-ai`, which is out of
        # this package's scope. The rest of the TS case (assembly order, explicit-wins, and
        # the absence of a forwarded transform) is asserted above and below.
        assert not hasattr(options, "transform_headers")

    def test_attribution_headers_reach_the_provider_through_a_real_session(self, tmp_path: Path) -> None:
        """TS's `sdk-openrouter-attribution.test.ts` captures the options a stub provider
        receives from a real `createAgentSession`. `test_provider_attribution.py` checks the
        header rules as a unit; this pins the wiring -- that `stream_fn` actually calls
        `merge_provider_attribution_headers` and forwards the result -- which a unit test of
        the rules alone cannot show."""
        options = asyncio.run(
            _wait(
                _capture_stream_options(
                    tmp_path,
                    {},
                    provider_id="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                )
            )
        )
        assert options is not None
        assert options.headers["HTTP-Referer"] == "https://pi.dev"
        assert options.headers["X-OpenRouter-Title"] == "pi"
        assert options.headers["X-OpenRouter-Categories"] == "cli-agent"

    def test_attribution_headers_are_absent_when_telemetry_is_disabled(self, tmp_path: Path) -> None:
        options = asyncio.run(
            _wait(
                _capture_stream_options(
                    tmp_path,
                    {"enableInstallTelemetry": False},
                    provider_id="openrouter",
                    base_url="https://openrouter.ai/api/v1",
                )
            )
        )
        assert options is not None
        assert "HTTP-Referer" not in options.headers
        assert "X-OpenRouter-Title" not in options.headers
        assert "X-OpenRouter-Categories" not in options.headers


# ---------------------------------------------------------------------------
# Port of `packages/coding-agent/test/sdk-skills.test.ts`.
#
# TypeScript builds two hand-written `ResourceLoader` object literals for the
# second and third cases. Python's `ResourceLoader` is a concrete class, not a
# structural interface, so these drive real loaders (over real on-disk skill
# directories, or with `no_skills=True`) instead: a shape-mismatched stub would
# not prove `create_agent_session` wires the loader through.
# ---------------------------------------------------------------------------


def _write_skill(skills_root: Path, name: str, description: str) -> None:
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n\nThis is a test skill.\n",
        encoding="utf-8",
    )


class TestCreateAgentSessionSkills:
    def test_discovers_skills_by_default_and_exposes_them_on_the_session(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / "skills", "test-skill", "A test skill for SDK tests.")

        async def run() -> None:
            model_runtime = await _runtime_with_logged_in_fake(tmp_path)
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=str(tmp_path),
                        agent_dir=str(tmp_path),
                        model=model_runtime.find_model("test/test-model"),
                        model_runtime=model_runtime,
                        session_manager=SessionManager.in_memory(),
                    )
                )
            )
            try:
                skills = result.session.resource_loader.get_skills().skills
                assert len(skills) > 0
                assert any(skill.name == "test-skill" for skill in skills)
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_has_empty_skills_when_the_resource_loader_returns_none(self, tmp_path: Path) -> None:
        _write_skill(tmp_path / "skills", "test-skill", "A test skill for SDK tests.")
        resource_loader = ResourceLoader(
            ResourceLoaderOptions(cwd=str(tmp_path), agent_dir=str(tmp_path), no_skills=True)
        )
        resource_loader.reload()

        async def run() -> None:
            model_runtime = await _runtime_with_logged_in_fake(tmp_path)
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=str(tmp_path),
                        agent_dir=str(tmp_path),
                        model=model_runtime.find_model("test/test-model"),
                        model_runtime=model_runtime,
                        session_manager=SessionManager.in_memory(),
                        resource_loader=resource_loader,
                    )
                )
            )
            try:
                assert result.session.resource_loader is resource_loader
                assert result.session.resource_loader.get_skills().skills == []
                assert result.session.resource_loader.get_skills().diagnostics == []
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_uses_provided_skills_when_the_resource_loader_supplies_them(self, tmp_path: Path) -> None:
        # The default discovery root holds `test-skill`; the injected loader points at a
        # different root holding only `custom-skill`, so the assertion proves the session
        # uses the supplied loader rather than rediscovering skills itself.
        _write_skill(tmp_path / "skills", "test-skill", "A test skill for SDK tests.")
        custom_root = tmp_path / "custom-agent"
        _write_skill(custom_root / "skills", "custom-skill", "A custom skill")
        (custom_root / ".pi" / "skills").mkdir(parents=True)
        resource_loader = ResourceLoader(ResourceLoaderOptions(cwd=str(custom_root), agent_dir=str(custom_root)))
        resource_loader.reload()

        async def run() -> None:
            model_runtime = await _runtime_with_logged_in_fake(tmp_path)
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=str(tmp_path),
                        agent_dir=str(tmp_path),
                        model=model_runtime.find_model("test/test-model"),
                        model_runtime=model_runtime,
                        session_manager=SessionManager.in_memory(),
                        resource_loader=resource_loader,
                    )
                )
            )
            try:
                skills = result.session.resource_loader.get_skills().skills
                assert [skill.name for skill in skills] == ["custom-skill"]
                assert skills[0].description == "A custom skill"
                assert result.session.resource_loader.get_skills().diagnostics == []
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))


class TestCreateAgentSessionExtensionProviderHooks:
    """Pins the `extensionRunnerRef` wiring `createAgentSession` does in
    `sdk.ts` (`onPayload`, `onResponse`, `transformContext`).

    The TypeScript file has no dedicated test for these three, but every
    extension registering `before_provider_request`, `after_provider_response`
    or `context` depends on them: without the wiring the handler is registered
    and simply never fires.
    """

    @staticmethod
    async def _session_with(handlers: dict[str, list[Any]], tmp_path: Path):
        model_runtime = await _runtime_with_logged_in_fake(tmp_path)
        extension = Extension(path="inline.py", resolved_path="inline.py", handlers=handlers)
        return await _wait(
            create_agent_session(
                CreateAgentSessionOptions(
                    cwd=str(tmp_path),
                    agent_dir=str(tmp_path),
                    model=model_runtime.find_model("test/test-model"),
                    model_runtime=model_runtime,
                    session_manager=SessionManager.in_memory(),
                    extensions=[extension],
                )
            )
        )

    def test_on_payload_runs_before_provider_request_handlers(self, tmp_path):
        seen: list[object] = []

        async def handler(event, _ctx):
            seen.append(event.payload)
            return {"model": "rewritten"}

        async def run() -> None:
            result = await self._session_with({"before_provider_request": [handler]}, tmp_path)
            try:
                model = result.session.agent.state.model
                replaced = await result.session.agent.on_payload({"model": "original"}, model)
                assert seen == [{"model": "original"}]
                assert replaced == {"model": "rewritten"}
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_on_payload_is_a_no_op_without_handlers(self, tmp_path):
        async def run() -> None:
            result = await self._session_with({}, tmp_path)
            try:
                model = result.session.agent.state.model
                payload = {"model": "original"}
                assert await result.session.agent.on_payload(payload, model) is payload
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_on_response_emits_after_provider_response_with_status_and_headers(self, tmp_path):
        seen: list[tuple[int, dict[str, str]]] = []

        async def handler(event, _ctx):
            seen.append((event.status, event.headers))

        async def run() -> None:
            result = await self._session_with({"after_provider_response": [handler]}, tmp_path)
            try:
                model = result.session.agent.state.model
                await result.session.agent.on_response(
                    ProviderResponse(status=429, headers={"retry-after": "3"}), model
                )
                assert seen == [(429, {"retry-after": "3"})]
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_transform_context_runs_context_handlers(self, tmp_path):
        replacement = [UserMessage(content=[TextContent(text="replaced")])]

        async def handler(_event, _ctx):
            return ContextEventResult(messages=replacement)

        async def run() -> None:
            result = await self._session_with({"context": [handler]}, tmp_path)
            try:
                original = [UserMessage(content=[TextContent(text="original")])]
                assert await result.session.agent.transform_context(original) == replacement
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_transform_context_returns_messages_unchanged_without_handlers(self, tmp_path):
        async def run() -> None:
            result = await self._session_with({}, tmp_path)
            try:
                messages = [UserMessage(content=[TextContent(text="original")])]
                assert await result.session.agent.transform_context(messages) == messages
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))

    def test_provider_hooks_reach_the_provider_on_a_real_turn(self, tmp_path):
        """The three callbacks must survive down to the options the provider api sees,
        otherwise `pi_ai`'s api modules never invoke them at request time. Driving a real
        `prompt()` is the only way to see this: calling `agent.stream_function` directly
        bypasses the agent loop, which is what merges them into the stream options."""
        seen_payloads: list[object] = []
        seen_contexts: list[int] = []

        async def before_request(event, _ctx):
            seen_payloads.append(event.payload)

        async def context_handler(event, _ctx):
            seen_contexts.append(len(event.messages))
            return None

        async def run() -> None:
            api = _CapturingApi()
            model = Model(
                id="capture-model",
                name="Capture Model",
                api="openai-completions",
                provider="capture-provider",
                base_url="https://capture.invalid/v1",
                context_window=128_000,
                max_tokens=4096,
                cost=ModelCost(input=0, output=0),
            )
            provider = create_provider(
                id="capture-provider",
                name="Capture Provider",
                auth=ProviderAuth(api_key=env_api_key_auth("Capture Provider API key", ["CAPTURE_PROVIDER_API_KEY"])),
                api=api,
                models=[model],
            )
            model_runtime = await _wait(ModelRuntime.create(agent_dir=tmp_path / "agent", providers=[provider]))
            await _wait(model_runtime.login("capture-provider", "fake-key"))
            extension = Extension(
                path="inline.py",
                resolved_path="inline.py",
                handlers={"before_provider_request": [before_request], "context": [context_handler]},
            )
            result = await _wait(
                create_agent_session(
                    CreateAgentSessionOptions(
                        cwd=str(tmp_path / "project"),
                        agent_dir=str(tmp_path / "agent"),
                        model=model_runtime.find_model("capture-provider/capture-model"),
                        model_runtime=model_runtime,
                        session_manager=SessionManager.in_memory(str(tmp_path / "project")),
                        extensions=[extension],
                    )
                )
            )
            try:
                await _wait(result.session.prompt("hello"))
                options = api.captured_options
                assert options is not None
                assert options.on_payload is not None
                assert options.on_response is not None
                # `transform_context` runs before the request is built, so the
                # `context` handler has already seen the outgoing message list.
                assert seen_contexts == [1]
                # `on_payload` is only invoked by the real api modules, which the
                # capturing fake replaces; calling it here proves it routes to the
                # extension rather than being an inert placeholder.
                await options.on_payload({"probe": True}, model)
                assert seen_payloads == [{"probe": True}]
            finally:
                result.session.dispose()

        asyncio.run(_wait(run()))
