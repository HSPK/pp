"""Adapt a `ToolDefinition` (extension-facing) into an `AgentTool` (agent-loop-facing).

Ported from ``packages/coding-agent/src/core/tools/tool-definition-wrapper.ts``.

`ToolDefinition.execute` always takes a required `ExtensionContext` as its
fifth argument, while the agent loop only ever calls
`AgentTool.execute(tool_call_id, params, signal, on_update)` with four. This
module bridges that gap: `wrap_tool_definition` defaults the missing context
from an optional `ctx_factory`, and `create_tool_definition_from_agent_tool`
does the reverse -- synthesizing a minimal `ToolDefinition` from a plain
`AgentTool` so a caller-supplied tool can be stored in a definition-first
registry without prompt metadata or renderers.

Note: `pi_coding_agent.core.extensions.runner.wrap_registered_tool` already
performs this same context-defaulting inline (fused with the tool-activation
diffing from ``core/extensions/wrapper.ts``) rather than calling into this
module, since it always has a concrete `ExtensionRunner` to source the
context from. This module exists for callers that need the two steps
(context-defaulting, and the reverse `AgentTool` -> `ToolDefinition`
synthesis) independently.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult, AgentToolUpdateCallback
from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.core.extensions.types import ExtensionContext, ToolDefinition


def wrap_tool_definition(
    definition: ToolDefinition,
    ctx_factory: Callable[[], ExtensionContext] | None = None,
) -> AgentTool:
    """Wrap a `ToolDefinition` into an `AgentTool` for the core runtime."""

    async def execute(
        tool_call_id: str,
        params: Any,
        signal: AbortSignal | None = None,
        on_update: AgentToolUpdateCallback | None = None,
        ctx: ExtensionContext | None = None,
    ) -> AgentToolResult:
        effective_ctx = ctx if ctx is not None else (ctx_factory() if ctx_factory is not None else None)
        return await definition.execute(tool_call_id, params, signal, on_update, effective_ctx)

    return AgentTool(
        name=definition.name,
        label=definition.label,
        description=definition.description,
        parameters=definition.parameters,
        constrained_sampling=definition.constrained_sampling,
        execute=execute,
        prepare_arguments=definition.prepare_arguments,
        execution_mode=definition.execution_mode,
    )


def wrap_tool_definitions(
    definitions: list[ToolDefinition],
    ctx_factory: Callable[[], ExtensionContext] | None = None,
) -> list[AgentTool]:
    """Wrap multiple `ToolDefinition`s into `AgentTool`s for the core runtime."""
    return [wrap_tool_definition(definition, ctx_factory) for definition in definitions]


def create_tool_definition_from_agent_tool(tool: AgentTool) -> ToolDefinition:
    """Synthesize a minimal `ToolDefinition` from an `AgentTool`.

    This keeps a definition-first tool registry consistent even when a caller
    provides plain `AgentTool` overrides that do not include prompt metadata
    or renderers.
    """

    async def execute(
        tool_call_id: str,
        params: Any,
        signal: AbortSignal | None,
        on_update: AgentToolUpdateCallback | None,
        _ctx: ExtensionContext,
    ) -> AgentToolResult:
        return await tool.execute(tool_call_id, params, signal, on_update)

    return ToolDefinition(
        name=tool.name,
        label=tool.label,
        description=tool.description,
        parameters=tool.parameters,
        execute=execute,
        constrained_sampling=tool.constrained_sampling,
        prepare_arguments=tool.prepare_arguments,
        execution_mode=tool.execution_mode,
    )
