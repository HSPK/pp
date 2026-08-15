"""Experimental feature gate.

Ported from ``packages/coding-agent/src/core/experimental.ts``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pi_ai.types import JsonSchemaConstrainedSampling


def are_experimental_features_enabled(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return env.get("PI_EXPERIMENTAL") == "1"


def get_experimental_tool_sampling(
    env: Mapping[str, str] | None = None,
) -> JsonSchemaConstrainedSampling | None:
    """Strict-schema sampling for built-in tools, only in experimental mode.

    ``strict="prefer"`` asks providers that support constrained sampling to
    enforce the tool's JSON Schema, while letting providers that do not just
    ignore it. Returns a fresh object each call so a caller mutating one
    tool's config cannot affect another's.
    """
    if not are_experimental_features_enabled(env):
        return None
    return JsonSchemaConstrainedSampling(strict="prefer")
