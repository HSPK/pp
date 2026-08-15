"""Python port of `packages/coding-agent/test/restore-sandbox-env.test.ts`.

`src/bun/restore-sandbox-env.ts` has no Python counterpart, and cannot have
one. It works around https://github.com/oven-sh/bun/issues/27802: a Bun
*compiled binary* sees an empty `process.env` inside a Linux sandbox, so the
function re-reads `/proc/self/environ` and repopulates `process.env`. CPython
has no such bug -- `os.environ` is populated from the real `environ(7)` block
by the interpreter itself -- and there is no Bun runtime to detect
(`process.versions.bun`). The same workaround in `pi_ai` was dropped for the
same reason; see `pi_ai/utils/provider_env.py`'s module docstring.

All three TypeScript cases are about that Bun-only branch, so there is nothing
behavioral left to pin. They are recorded below.
"""

from __future__ import annotations

import os

import pytest

import pi_coding_agent

_REASON = (
    "`src/bun/restore-sandbox-env.ts` is not ported: it repopulates an empty "
    "`process.env` from /proc/self/environ under Bun compiled binaries "
    "(oven-sh/bun#27802). CPython has no equivalent failure mode and no Bun "
    "runtime to detect (see pi_ai/utils/provider_env.py)."
)


def test_environment_is_populated_without_any_restore_step() -> None:
    """The premise of the omission: Python needs no restore step.

    `os.environ` is non-empty for a normally launched interpreter, which is
    exactly the condition `restoreSandboxEnv` exists to repair, and there is no
    `bun` submodule to call into.
    """
    assert len(os.environ) > 0
    assert not hasattr(pi_coding_agent, "bun")


@pytest.mark.skip(reason=_REASON)
def test_does_nothing_when_not_running_under_bun() -> None:
    """`it("does nothing when not running under bun")`.

    With `process.versions` stubbed to `{ node: "20.0.0" }`, `restoreSandboxEnv()`
    leaves `process.env` deep-equal to its prior snapshot.
    """


@pytest.mark.skip(reason=_REASON)
def test_does_nothing_when_process_env_already_has_entries() -> None:
    """`it("does nothing when process.env already has entries")`.

    With `process.versions` stubbed to `{ bun: "1.2.0", node: "20.0.0" }` and
    at least one env entry present, `process.env` is unchanged.
    """


@pytest.mark.skip(reason=_REASON)
def test_restores_environment_from_proc_self_environ_when_bun_env_is_empty() -> None:
    """`it("restores environment from /proc/self/environ when bun env is empty")`.

    With `process.versions.bun` set and `process.env` fully cleared, and a
    mocked `readFileSync` returning `"FOO=bar\\0BAZ=qux\\0"`, asserts
    `readFileSync` was called with exactly `("/proc/self/environ", "utf-8")`,
    and that `process.env.FOO == "bar"` and `process.env.BAZ == "qux"`.
    Note the NUL-separated parsing and the `idx > 0` guard, which drops both
    empty trailing entries and entries whose name is empty.
    """
