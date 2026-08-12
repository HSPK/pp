"""Experimental feature gate.

Ported from ``packages/coding-agent/src/core/experimental.ts``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping


def are_experimental_features_enabled(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return env.get("PI_EXPERIMENTAL") == "1"
