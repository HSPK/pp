"""Programmatic entry point for constructing an `AgentSession`.

Python port of `packages/coding-agent/src/core/sdk.ts`. `create_agent_session`
wires together the model runtime, session manager, settings manager, resource
loader and builtin tools into a real `pi_agent.agent.Agent` and the
`AgentSession` that wraps it -- the same construction path
`pi_coding_agent.core.agent_session_runtime` uses to build sessions the RPC
server drives.

**Extensions: accepted, not discovered.** TS's `createAgentSession` loads
extensions off the `ResourceLoader` itself and threads an
`extensionRunnerRef` through the `Agent` constructor (`onPayload`,
`onResponse`, `transformContext` all call into the extension runner), then
returns an `extensionsResult` for the interactive/RPC modes to bind UI
context against. Here the caller loads extensions
(`discover_and_load_extensions()`) and passes them via
`CreateAgentSessionOptions.extensions`; `AgentSession` owns the runner and
every wired hook (see `agent_session.py`'s module docstring). The
`extensionRunnerRef` indirection is ported as `_ExtensionRunnerRef`, so
`on_payload`, `on_response` and `transform_context` reach the runner exactly
as they do in TypeScript. `CreateAgentSessionResult` still has no
`extensions_result` field.

**Provider attribution headers are applied; the extension hook is not.** TS's
`streamFn` wrapper does two things through `transformHeaders`: it merges
`mergeProviderAttributionHeaders` (telemetry-gated attribution plus opencode
session headers) and it runs the extension `before_provider_headers` hook.
The first is ported -- `stream_fn` below merges the attribution headers into
`options.headers` -- and only the second is dropped. The blocker is not the
extension system (`ExtensionRunner.emit_before_provider_headers` is fully
implemented) but `pi_ai`: its `SimpleStreamOptions` has no `transformHeaders`
callback, and the provider/model/request headers the hook is supposed to see
are only assembled inside `ModelRegistry._resolve_request`, downstream of
anything this module can wrap. Wiring the hook therefore needs a `pi_ai`
change, so `emit_before_provider_headers` currently has no caller. The
attribution merge is unaffected: it happens up front against
`StreamOptions.headers` rather than at request time, and the result is the
same because nothing between here and the request removes those headers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from typing import Any

from pi_agent.agent import Agent, MutableAgentState
from pi_agent.harness.messages import convert_to_llm as harness_convert_to_llm
from pi_agent.types import AgentMessage, AgentTool, ThinkingLevel
from pi_ai.models import clamp_thinking_level
from pi_ai.types import Message, Model, ProviderResponse, SimpleStreamOptions, TextContent

from pi_coding_agent.core.agent_session import AgentSession
from pi_coding_agent.core.auth_guidance import format_no_models_available_message
from pi_coding_agent.core.config import get_agent_dir
from pi_coding_agent.core.extensions.runner import ExtensionRunner
from pi_coding_agent.core.extensions.types import AfterProviderResponseEvent, Extension
from pi_coding_agent.core.model_resolver import DEFAULT_THINKING_LEVEL, ScopedModel, find_initial_model
from pi_coding_agent.core.model_runtime import ModelRuntime
from pi_coding_agent.core.provider_attribution import merge_provider_attribution_headers
from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions
from pi_coding_agent.core.session_manager import SessionManager, get_default_session_dir
from pi_coding_agent.core.settings_manager import SettingsManager
from pi_coding_agent.core.timings import time as record_timing
from pi_coding_agent.tools import create_all_tools
from pi_coding_agent.utils.paths import resolve_path

_DEFAULT_ACTIVE_TOOL_NAMES: list[str] = ["read", "bash", "edit", "write"]


@dataclass
class CreateAgentSessionOptions:
    """Port of TS `CreateAgentSessionOptions`."""

    cwd: str | None = None
    agent_dir: str | None = None
    model_runtime: ModelRuntime | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None
    scoped_models: list[ScopedModel] | None = None
    no_tools: str | None = None
    """`"all"` or `"builtin"`. See the module docstring on `TOOL_PROMPT_CONTRIBUTIONS` in `agent_session.py`."""
    tools: list[str] | None = None
    exclude_tools: list[str] | None = None
    custom_tools: dict[str, AgentTool] = field(default_factory=dict)
    resource_loader: ResourceLoader | None = None
    session_manager: SessionManager | None = None
    settings_manager: SettingsManager | None = None
    extensions: list[Extension] | None = None
    """Already-loaded extensions to bind to the session.

    TypeScript's `createAgentSession` loads these itself from the
    `ResourceLoader` and returns an `extensionsResult`; this port's
    `ResourceLoader` does not own extensions (they are loaded through
    `discover_and_load_extensions()`), so the caller passes the loaded list in.
    Without this, `AgentSession`'s extension support would be unreachable from
    the SDK entry point."""


@dataclass
class CreateAgentSessionResult:
    """Port of TS `CreateAgentSessionResult`, minus `extensionsResult`."""

    session: AgentSession
    model_fallback_message: str | None = None


def _get_default_agent_dir() -> str:
    return get_agent_dir()


def _block_images_convert_to_llm(settings_manager: SettingsManager) -> Any:
    """Wrap `convert_to_llm` to replace image content when `blockImages` is enabled.

    Checks the setting dynamically (not just at construction time) so a
    mid-session settings change takes effect on the next turn, matching TS's
    `convertToLlmWithBlockImages`.
    """
    placeholder_text = "Image reading is disabled."

    def convert(messages: list[AgentMessage]) -> list[Message]:
        converted = harness_convert_to_llm(messages)
        if not settings_manager.get_block_images():
            return converted

        result: list[Message] = []
        for message in converted:
            if message.role in ("user", "toolResult") and isinstance(message.content, list):
                has_images = any(part.type == "image" for part in message.content)
                if has_images:
                    filtered: list[Any] = []
                    for part in message.content:
                        if part.type == "image":
                            if (
                                filtered
                                and getattr(filtered[-1], "type", None) == "text"
                                and filtered[-1].text == placeholder_text
                            ):
                                continue
                            filtered.append(TextContent(text=placeholder_text))
                        else:
                            filtered.append(part)
                    message = replace_content(message, filtered)
            result.append(message)
        return result

    return convert


def replace_content(message: Message, content: list[Any]) -> Message:
    """Return a copy of `message` with its `content` replaced (TS object spread equivalent)."""
    return dataclass_replace(message, content=content)


class _ExtensionRunnerRef:
    """Mutable holder for the session's `ExtensionRunner`.

    Port of TypeScript's `extensionRunnerRef`: the `Agent` needs the provider
    hooks at construction time, but the runner only exists once `AgentSession`
    has been built, so the callbacks resolve it lazily through this box.
    """

    def __init__(self) -> None:
        self.current: ExtensionRunner | None = None

    async def on_payload(self, payload: Any, _model: Model) -> Any:
        runner = self.current
        if runner is None or not runner.has_handlers("before_provider_request"):
            return payload
        return await runner.emit_before_provider_request(payload)

    async def on_response(self, response: ProviderResponse, _model: Model) -> None:
        runner = self.current
        if runner is None or not runner.has_handlers("after_provider_response"):
            return
        await runner.emit(AfterProviderResponseEvent(status=response.status, headers=dict(response.headers)))

    async def transform_context(self, messages: list[AgentMessage], signal: object | None = None) -> list[AgentMessage]:
        """`agent_loop` calls this as `transform_context(messages, signal)`.

        TypeScript declares `transformContext: async (messages) => ...`
        (`sdk.ts:350`) and `agent-loop.ts:291` calls it with two arguments;
        JavaScript discards the extra one, Python raises `TypeError`. The
        signal is accepted and ignored to match the upstream behaviour --
        without it every SDK-driven agent run dies mid-turn.
        """
        del signal
        runner = self.current
        if runner is None:
            return messages
        return await runner.emit_context(messages)


async def create_agent_session(options: CreateAgentSessionOptions | None = None) -> CreateAgentSessionResult:
    """Create an `AgentSession` with the specified options.

    Model selection order: explicit `options.model`, else the session's saved
    model (if authenticated), else `find_initial_model`'s settings/provider
    defaults. Thinking level follows the same restore-then-default order and
    is always clamped to the resolved model's capabilities.
    """
    options = options or CreateAgentSessionOptions()
    cwd = resolve_path(
        options.cwd or (options.session_manager.get_cwd() if options.session_manager else None) or os.getcwd()
    )
    agent_dir = resolve_path(options.agent_dir) if options.agent_dir else _get_default_agent_dir()

    model_runtime = options.model_runtime or await ModelRuntime.create(agent_dir=agent_dir)

    settings_manager = options.settings_manager or SettingsManager.create(cwd, agent_dir)
    session_manager = options.session_manager or SessionManager.create(cwd, get_default_session_dir(cwd, agent_dir))

    resource_loader = options.resource_loader
    if resource_loader is None:
        resource_loader = ResourceLoader(ResourceLoaderOptions(cwd=cwd, agent_dir=agent_dir))
        resource_loader.reload()
        record_timing("resourceLoader.reload")

    existing_session = session_manager.build_session_context()
    has_existing_session = len(existing_session.messages) > 0
    has_thinking_entry = any(entry.type == "thinking_level_change" for entry in session_manager.get_branch())

    model = options.model
    model_fallback_message: str | None = None

    if model is None and has_existing_session and existing_session.model is not None:
        restored_model = model_runtime.get_model(existing_session.model.provider, existing_session.model.model_id)
        if restored_model is not None and model_runtime.has_configured_auth(restored_model.provider):
            model = restored_model
        if model is None:
            model_fallback_message = (
                f"Could not restore model {existing_session.model.provider}/{existing_session.model.model_id}"
            )

    if model is None:
        result = find_initial_model(
            model_runtime,
            scoped_models=[],
            is_continuing=has_existing_session,
            default_provider=settings_manager.get_default_provider(),
            default_model_id=settings_manager.get_default_model(),
            default_thinking_level=settings_manager.get_default_thinking_level(),
        )
        model = result.model
        if model is None:
            model_fallback_message = format_no_models_available_message()
        elif model_fallback_message:
            model_fallback_message += f". Using {model.provider}/{model.id}"

    thinking_level = options.thinking_level

    if thinking_level is None and has_existing_session:
        thinking_level = (
            existing_session.thinking_level
            if has_thinking_entry
            else (settings_manager.get_default_thinking_level() or DEFAULT_THINKING_LEVEL)
        )

    if thinking_level is None:
        thinking_level = settings_manager.get_default_thinking_level() or DEFAULT_THINKING_LEVEL

    if model is None:
        thinking_level = "off"
    else:
        thinking_level = clamp_thinking_level(model, thinking_level)

    allowed_tool_names = options.tools if options.tools is not None else ([] if options.no_tools == "all" else None)
    excluded_tool_names = options.exclude_tools
    excluded_tool_name_set = set(excluded_tool_names) if excluded_tool_names else None
    initial_active_tool_names = (
        list(options.tools)
        if options.tools is not None
        else ([] if options.no_tools else list(_DEFAULT_ACTIVE_TOOL_NAMES))
    )
    if excluded_tool_name_set:
        initial_active_tool_names = [name for name in initial_active_tool_names if name not in excluded_tool_name_set]

    extension_runner_ref = _ExtensionRunnerRef()

    agent = Agent(
        _make_stream_fn(model_runtime, settings_manager),
        initial_state=MutableAgentState(
            system_prompt="",
            model=model if model is not None else MutableAgentState().model,
            thinking_level=thinking_level,
        ),
        convert_to_llm=_block_images_convert_to_llm(settings_manager),
        session_id=session_manager.get_session_id(),
        steering_mode=settings_manager.get_steering_mode(),
        follow_up_mode=settings_manager.get_follow_up_mode(),
        transport=settings_manager.get_transport(),
        thinking_budgets=settings_manager.get_thinking_budgets(),
        max_retry_delay_ms=settings_manager.get_provider_retry_settings().get("maxRetryDelayMs"),
        on_payload=extension_runner_ref.on_payload,
        on_response=extension_runner_ref.on_response,
        transform_context=extension_runner_ref.transform_context,
    )

    if has_existing_session:
        agent.state.messages = existing_session.messages
        if not has_thinking_entry:
            session_manager.append_thinking_level_change(thinking_level)
    else:
        if model is not None:
            session_manager.append_model_change(model.provider, model.id)
        session_manager.append_thinking_level_change(thinking_level)

    session = AgentSession(
        agent=agent,
        session_manager=session_manager,
        settings_manager=settings_manager,
        cwd=cwd,
        resource_loader=resource_loader,
        model_runtime=model_runtime,
        scoped_models=options.scoped_models,
        custom_tools=options.custom_tools,
        initial_active_tool_names=initial_active_tool_names,
        allowed_tool_names=allowed_tool_names,
        excluded_tool_names=excluded_tool_names,
        extensions=options.extensions,
    )
    extension_runner_ref.current = session.extension_runner

    return CreateAgentSessionResult(session=session, model_fallback_message=model_fallback_message)


def _coalesce(*values: Any) -> Any:
    """Python equivalent of TS's `??` chain: first non-`None` value (unlike `or`, doesn't
    skip falsy-but-meaningful values like `0`, which callers use to mean "no timeout")."""
    for value in values:
        if value is not None:
            return value
    return None


def _make_stream_fn(model_runtime: ModelRuntime, settings_manager: SettingsManager) -> Any:
    async def stream_fn(model: Model, context: Any, options: Any = None) -> Any:
        provider_retry_settings = settings_manager.get_provider_retry_settings()
        http_idle_timeout_ms = settings_manager.get_http_idle_timeout_ms()
        # SDKs treat timeout=0 as 0ms (immediate timeout), not "no timeout".
        # Use a very large value to effectively disable the timeout.
        effective_timeout_ms = 2_147_483_647 if http_idle_timeout_ms == 0 else http_idle_timeout_ms
        timeout_ms = _coalesce(
            getattr(options, "timeout_ms", None) if options else None,
            provider_retry_settings.get("timeoutMs"),
            effective_timeout_ms,
        )
        websocket_connect_timeout_ms = _coalesce(
            getattr(options, "websocket_connect_timeout_ms", None) if options else None,
            settings_manager.get_websocket_connect_timeout_ms(),
        )
        max_retries = _coalesce(
            getattr(options, "max_retries", None) if options else None,
            provider_retry_settings.get("maxRetries"),
        )
        max_retry_delay_ms = _coalesce(
            getattr(options, "max_retry_delay_ms", None) if options else None,
            provider_retry_settings.get("maxRetryDelayMs"),
        )
        # TS spreads `options` into a fresh object literal (`{ ...options, timeoutMs, ... }`),
        # which is a no-op when `options` is `undefined` -- so the merged timeout/retry
        # defaults must always be forwarded, even when the caller passed no options at all.
        options = replace_stream_options(
            options if options is not None else SimpleStreamOptions(),
            timeout_ms=timeout_ms,
            websocket_connect_timeout_ms=websocket_connect_timeout_ms,
            max_retries=max_retries,
            max_retry_delay_ms=max_retry_delay_ms,
        )
        # Caller headers are passed last so an explicit header always wins.
        attributed = merge_provider_attribution_headers(model, settings_manager, options.session_id, options.headers)
        if attributed is not None:
            options = replace_stream_options(options, headers=attributed)
        return await model_runtime.stream_simple(model, context, options)

    return stream_fn


def replace_stream_options(options: Any, **updates: Any) -> Any:
    return dataclass_replace(options, **{k: v for k, v in updates.items() if v is not None})


def load_default_tools(cwd: str) -> dict[str, AgentTool]:
    """All builtin coding tools for `cwd`, keyed by name. Re-exported for callers that need a plain dict."""
    return create_all_tools(cwd)


__all__ = [
    "CreateAgentSessionOptions",
    "CreateAgentSessionResult",
    "create_agent_session",
    "load_default_tools",
]
