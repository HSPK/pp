"""Coding-agent server integration.

Python port of `packages/coding-agent/src/server/`. Only `create-harness.ts`
lives here; the RPC server itself is the separate `pi_server` package.
"""

from __future__ import annotations

from .create_harness import (
    DEFAULT_HARNESS_TOOL_NAMES,
    CodingAgentHarnessTool,
    CreateCodingAgentHarnessOptions,
    build_coding_agent_harness_system_prompt,
    create_coding_agent_harness,
    create_coding_agent_harness_tools,
)

__all__ = [
    "DEFAULT_HARNESS_TOOL_NAMES",
    "CodingAgentHarnessTool",
    "CreateCodingAgentHarnessOptions",
    "build_coding_agent_harness_system_prompt",
    "create_coding_agent_harness",
    "create_coding_agent_harness_tools",
]
