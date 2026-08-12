"""Python port of `packages/coding-agent/test/suite/regressions/2753-reload-stale-resource-settings.test.ts`.

**Not portable.** The TypeScript test builds an `AgentSessionRuntime`, writes a
new `settings.json` after startup, calls `session.reload()`, and asserts the
rebuilt runtime picked up the new top-level `prompts` setting (here
`["-prompts/test.md"]`, a force-exclude override).

Two pieces that test needs do not exist in this port:

1. `AgentSession.reload()` / the extension-runtime rebuild it drives
   (`_buildRuntime` + `bindExtensions`) is deliberately not ported -- see
   `core/agent_session.py`'s and `core/agent_session_runtime.py`'s module
   docstrings ("Dropped: `AgentSessionServices` / cwd-bound service
   recreation").
2. This port's `ResourceLoader` never reads `settings.json`; the
   enable/disable override patterns (`-`/`+`/`!`) live in
   `core/package_manager.py` and are not wired into prompt-template
   discovery, so there is nothing for a reload to pick up staleley or
   freshly.

Re-testing only the parts that do exist (write settings, call
`ResourceLoader.reload()`) would assert nothing about the bug, so the file is
kept as this explanation instead of a weakened test.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="AgentSession.reload() and settings-driven prompt overrides are not ported; see module docstring"
)


def test_applies_updated_top_level_prompt_settings_on_reload() -> None:
    raise AssertionError("unreachable")
