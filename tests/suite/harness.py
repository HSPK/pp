"""Python port of `packages/coding-agent/test/suite/harness.ts`.

Builds a real `AgentSession` driven by the scripted `faux` provider: no
network, no real model. Everything the session touches (auth storage, the
session transcript, settings, `models.json`) lives in memory or under the
`tmp_path` the caller passes in.

Differences from the TypeScript harness, all forced by this port's shape:

- TS registers the faux provider into the *global* api-provider registry
  (`registerFauxProvider`) and hands `streamSimple` from `pi-ai/compat` to the
  `Agent`. Here the faux provider is a real `pi_ai.registry.Provider`
  (`faux_provider()`), passed to `ModelRuntime.create(providers=[...])`, and
  the `Agent`'s stream function is `model_runtime.stream_simple`. That removes
  the global registry mutation (and its `unregister()` cleanup step) while
  exercising the same auth-resolution path a real session uses.
- TS threads an `extensionRunnerRef` into the `Agent` for `onPayload` /
  `onResponse` / `transformContext`. This port never wires those three
  provider-level extension hooks (`before_provider_request`,
  `after_provider_response`, `context`), so the harness does not either; see
  `agent_session.py`'s module docstring.
- `extension_factories` receive this port's `ExtensionAPI`
  (`extensions/loader.py`) instead of TS's `pi` object, and the registration
  actions (`send_message`, `send_user_message`, `append_entry`,
  `set_session_name`) are bound to the session that the harness builds
  afterwards, mirroring TS's `_applyExtensionBindings`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from pi_agent.agent import Agent, MutableAgentState
from pi_agent.harness.messages import convert_to_llm as harness_convert_to_llm
from pi_agent.types import AgentMessage, AgentTool
from pi_ai.providers.faux import (
    FauxModelDefinition,
    FauxProviderHandle,
    FauxResponseStep,
    RegisterFauxProviderOptions,
    faux_provider,
)
from pi_ai.registry import Provider
from pi_ai.types import AssistantMessage, Cost, ImageContent, Model, TextContent, Usage, UserMessage, now_ms

from pi_coding_agent.core.agent_session import AgentSession, AgentSessionEvent
from pi_coding_agent.core.auth_storage import AuthStorage
from pi_coding_agent.core.event_bus import EventBus
from pi_coding_agent.core.extensions.loader import (
    ExtensionAPI,
    ExtensionRuntimeActions,
    load_extension_factories,
)
from pi_coding_agent.core.extensions.types import Extension
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.session_manager import SessionManager
from pi_coding_agent.core.settings_manager import SettingsManager

ExtensionFactory = Callable[[ExtensionAPI], Awaitable[None] | None]

_EventT = TypeVar("_EventT")


def get_message_text(message: object) -> str:
    """Port of `getMessageText`: the concatenated text parts of any message."""
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if getattr(part, "type", None) == "text")


def get_user_texts(harness: Harness) -> list[str]:
    return [get_message_text(m) for m in harness.session.messages if getattr(m, "role", None) == "user"]


def get_assistant_texts(harness: Harness) -> list[str]:
    return [get_message_text(m) for m in harness.session.messages if getattr(m, "role", None) == "assistant"]


@dataclass
class Harness:
    session: AgentSession
    session_manager: SessionManager
    settings_manager: SettingsManager
    auth_storage: AuthStorage
    model_runtime: ModelRuntime
    faux: FauxProviderHandle
    models: list[Model]
    events: list[AgentSessionEvent]
    temp_dir: Path
    extensions: list[Extension] = field(default_factory=list)

    def get_model(self, model_id: str | None = None) -> Model | None:
        return self.faux.get_model(model_id)

    def set_responses(self, responses: list[FauxResponseStep]) -> None:
        self.faux.set_responses(responses)

    def append_responses(self, responses: list[FauxResponseStep]) -> None:
        self.faux.append_responses(responses)

    def get_pending_response_count(self) -> int:
        return self.faux.get_pending_response_count()

    def events_of_type(self, event_type: str) -> list[Any]:
        return [event for event in self.events if event.type == event_type]

    def cleanup(self) -> None:
        self.session.dispose()


async def build_extensions(
    factories: list[ExtensionFactory],
    cwd: str,
    actions: ExtensionRuntimeActions,
) -> list[Extension]:
    """Port of `createTestExtensionsResult`, delegating to the real inline loader.

    `load_extension_factories` is the production path that `--extension`-style
    inline factories take, and it is what attaches `source_info` to each
    extension (a hand-built `Extension()` leaves it unset, which silently
    weakens any assertion about a registered command's source). Factory errors
    are re-raised here rather than collected, because a test whose extension
    failed to load should fail loudly.
    """
    result = await load_extension_factories(list(factories), cwd, actions)
    if result.errors:
        raise AssertionError(f"extension factory failed to load: {result.errors}")
    return result.extensions


def create_test_resource_loader(tmp_path: Path) -> ResourceLoader:
    """Port of `createTestResourceLoader`: a loader that discovers nothing.

    TS returns a hand-written object literal implementing the `ResourceLoader`
    interface. Here a real `ResourceLoader` is pointed at two empty directories
    under `tmp_path`, which discovers no skills, prompts, AGENTS.md files or
    system prompt -- the same empty result, through the real code path.
    """
    cwd = tmp_path / "resources-cwd"
    agent_dir = tmp_path / "resources-agent"
    cwd.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)
    loader = ResourceLoader(ResourceLoaderOptions(cwd=str(cwd), agent_dir=str(agent_dir)))
    loader.reload()
    return loader


class _SessionRef:
    """Late-bound session holder so extension actions can reach the session
    that is constructed after the extensions are loaded."""

    session: AgentSession | None = None


def _make_extension_actions(ref: _SessionRef, event_bus: EventBus | None = None) -> ExtensionRuntimeActions:
    def send_message(message: Any, options: Any = None) -> None:
        session = ref.session
        if session is None:
            return
        deliver_as = None
        trigger_turn = False
        if isinstance(options, dict):
            deliver_as = options.get("deliverAs")
            trigger_turn = bool(options.get("triggerTurn", False))
        _spawn(
            session.send_custom_message(
                message["customType"] if isinstance(message, dict) else message.custom_type,
                message["content"] if isinstance(message, dict) else message.content,
                message["display"] if isinstance(message, dict) else message.display,
                message.get("details") if isinstance(message, dict) else message.details,
                trigger_turn=trigger_turn,
                deliver_as=deliver_as,
            )
        )

    def send_user_message(content: str | list[TextContent | ImageContent], options: Any = None) -> None:
        session = ref.session
        if session is None:
            return
        deliver_as = options.get("deliverAs") if isinstance(options, dict) else None
        _spawn(session.send_user_message(content, deliver_as=deliver_as))

    def append_entry(custom_type: str, data: object = None) -> None:
        session = ref.session
        if session is None:
            return
        session.session_manager.append_custom_entry(custom_type, data)

    def set_session_name(name: str) -> None:
        session = ref.session
        if session is None:
            return
        session.set_session_name(name)

    def get_session_name() -> str | None:
        session = ref.session
        return session.session_name if session is not None else None

    def set_active_tools(tool_names: list[str]) -> None:
        session = ref.session
        if session is None:
            return
        session.set_active_tools_by_name(tool_names)

    def get_active_tools() -> list[str]:
        session = ref.session
        return session.get_active_tool_names() if session is not None else []

    return ExtensionRuntimeActions(
        send_message=send_message,
        send_user_message=send_user_message,
        append_entry=append_entry,
        set_session_name=set_session_name,
        get_session_name=get_session_name,
        set_active_tools=set_active_tools,
        get_active_tools=get_active_tools,
        event_bus=event_bus,
    )


_background_tasks: set[asyncio.Task[None]] = set()


def _spawn(coro: Awaitable[None]) -> None:
    """Fire-and-forget an extension action, TS-style.

    TypeScript's bindings call `this.sendUserMessage(...).catch(...)`, and a JS
    async function body runs synchronously up to its first `await`, so a
    message queued from a synchronous extension handler is already in the
    queue when the handler returns. `asyncio.ensure_future` would instead
    defer the whole body to the next loop iteration, which is late enough for
    the agent loop's post-run "any queued messages?" check to miss it. Eager
    tasks (3.12+) reproduce the JS start semantics exactly.
    """
    if isinstance(coro, Coroutine):
        task: asyncio.Task[None] = asyncio.Task(coro, loop=asyncio.get_event_loop(), eager_start=True)
        if task.done():
            task.exception()
            return
    else:
        task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def drain_extension_actions() -> None:
    """Wait for the tasks `pi.send_message`/`pi.send_user_message` spawned.

    TypeScript's bindings call `this.sendUserMessage(...).catch(...)`, and a JS
    async function body runs synchronously up to its first `await`, so the
    session's "am I streaming?" decision has already happened by the time the
    caller continues. Python coroutines do not start until the loop schedules
    them, so tests that fire an extension action and then immediately inspect
    session state must drain the spawned tasks first.
    """
    while _background_tasks:
        await asyncio.gather(*list(_background_tasks), return_exceptions=True)


async def wait_until(predicate: Callable[[], bool], *, what: str, timeout: float = 5.0) -> None:
    """Yield to the loop until `predicate` holds, instead of sleeping a fixed delay.

    A fixed `asyncio.sleep(...)` that is long enough to be reliable on a loaded,
    parallel test run is also long enough to slow the suite down, and if the
    host stalls past it the test fails for a reason unrelated to what it checks.
    This returns on the first loop turn where the condition holds and only
    spends the timeout when the behaviour under test is genuinely broken.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    turns = 0
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        turns += 1
        await asyncio.sleep(0 if turns < 50 else 0.005)


async def drain_session_tasks(session: AgentSession) -> None:
    """Wait for the session's own fire-and-forget extension emits.

    Same JS-vs-Python scheduling gap as `drain_extension_actions`: TypeScript's
    `void this._extensionRunner.emit(event)` inside a synchronous method has
    already run its handlers by the time the caller continues, while a Python
    task does not start until the loop schedules it.
    """
    tasks = getattr(session, "_background_tasks", set())
    while tasks:
        await asyncio.gather(*list(tasks), return_exceptions=True)


async def create_harness(
    tmp_path: Path,
    *,
    models: list[FauxModelDefinition] | None = None,
    settings: dict[str, Any] | None = None,
    system_prompt: str | None = None,
    tools: list[AgentTool] | None = None,
    initial_active_tool_names: list[str] | None = None,
    allowed_tool_names: list[str] | None = None,
    excluded_tool_names: list[str] | None = None,
    resource_loader: ResourceLoader | None = None,
    extension_factories: list[ExtensionFactory] | None = None,
    extensions: list[Extension] | None = None,
    with_configured_auth: bool = True,
    models_json: dict[str, Any] | None = None,
    session_manager: SessionManager | None = None,
    event_bus: EventBus | None = None,
    providers: list[Provider] | None = None,
    provider_override: Callable[[Provider], Provider] | None = None,
) -> Harness:
    """Port of `createHarness`.

    `providers` has no TypeScript counterpart: it replaces the provider list
    handed to `ModelRuntime.create`, standing in for the tests that call
    `modelRuntime.registerNativeProvider(...)` after the fact (this port's
    `ModelRuntime` has no runtime provider-registration API).
    `provider_override` does the same for the common case of re-registering the
    faux provider itself with different auth.
    """
    temp_dir = tmp_path
    temp_dir.mkdir(parents=True, exist_ok=True)

    faux = faux_provider(RegisterFauxProviderOptions(models=models))
    faux.set_responses([])
    model = faux.get_model()
    assert model is not None

    tool_map = {tool.name: tool for tool in tools} if tools is not None else None

    resolved_session_manager = session_manager or SessionManager.in_memory(str(temp_dir))
    settings_manager = SettingsManager.in_memory(settings)

    auth_storage = AuthStorage.in_memory()
    models_path = temp_dir / "models.json" if models_json is not None else None
    if models_path is not None:
        models_path.write_text(json.dumps(models_json), encoding="utf-8")

    # `withConfiguredAuth: false` in TS skips `modelRegistry.registerProvider`,
    # which leaves the provider unknown to the registry; the Python equivalent
    # is not passing the provider at all.
    resolved_providers = providers if providers is not None else ([faux.provider] if with_configured_auth else [])
    if provider_override is not None:
        resolved_providers = [provider_override(faux.provider)]
    model_runtime = await ModelRuntime.create(
        agent_dir=temp_dir / "agent",
        models_path=models_path,
        credentials=auth_storage,
        providers=resolved_providers,
    )
    if with_configured_auth:
        await model_runtime.login(faux.provider.id, "faux-key")
        # TS's harness passes `apiKey: "faux-key"` to `registerProvider`, which
        # is what makes the faux models show up in `getAvailableSnapshot()`.
        # This port's synchronous `has_configured_auth` only sees the runtime
        # (`--api-key`) overlay and `models.json`, not the credential store, so
        # the runtime key is the equivalent registration step.
        await model_runtime.set_runtime_api_key(faux.provider.id, "faux-key")

    ref = _SessionRef()
    resolved_extensions: list[Extension] = list(extensions or [])
    if extension_factories:
        resolved_extensions.extend(
            await build_extensions(extension_factories, str(temp_dir), _make_extension_actions(ref, event_bus))
        )

    async def transform_context(messages: list[AgentMessage], _signal: Any = None) -> list[AgentMessage]:
        # Mirrors `harness.ts`'s `transformContext` wiring: the extension
        # `context` hook runs against the messages the agent is about to send.
        session = ref.session
        if session is None:
            return messages
        return await session._extension_runner.emit_context(messages)

    agent = Agent(
        model_runtime.stream_simple,
        initial_state=MutableAgentState(
            model=model,
            system_prompt=system_prompt or "You are a test assistant.",
        ),
        convert_to_llm=harness_convert_to_llm,
        transform_context=transform_context,
    )

    session = AgentSession(
        agent=agent,
        session_manager=resolved_session_manager,
        settings_manager=settings_manager,
        cwd=str(temp_dir),
        model_runtime=model_runtime,
        resource_loader=resource_loader or create_test_resource_loader(temp_dir),
        base_tools_override=tool_map,
        initial_active_tool_names=initial_active_tool_names,
        allowed_tool_names=allowed_tool_names,
        excluded_tool_names=excluded_tool_names,
        extensions=resolved_extensions,
    )
    ref.session = session

    events: list[AgentSessionEvent] = []
    session.subscribe(events.append)

    return Harness(
        session=session,
        session_manager=resolved_session_manager,
        settings_manager=settings_manager,
        auth_storage=auth_storage,
        model_runtime=model_runtime,
        faux=faux,
        models=faux.models,
        events=events,
        temp_dir=temp_dir,
        extensions=resolved_extensions,
    )


def message_roles(messages: list[AgentMessage]) -> list[str]:
    return [getattr(message, "role", "") for message in messages]


def user_msg(text: str) -> UserMessage:
    """Port of `test/utilities.ts`'s `userMsg`."""
    return UserMessage(content=text, timestamp=now_ms())


def assistant_msg(text: str) -> AssistantMessage:
    """Port of `test/utilities.ts`'s `assistantMsg`."""
    return AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model="test",
        content=[TextContent(text=text)],
        usage=Usage(input=1, output=1, cache_read=0, cache_write=0, total_tokens=2, cost=Cost()),
        stop_reason="stop",
        timestamp=now_ms(),
    )
