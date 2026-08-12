"""Python port of `packages/coding-agent/test/git-merge-and-resolve-extension.test.ts`.

Exercises `examples/extensions/git_merge_and_resolve.py` with a fully scripted
`pi.exec()` so no real git command ever runs.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pi_coding_agent.core.exec import ExecResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "extensions"))

from git_merge_and_resolve import pi_extension

OK = ExecResult(stdout="", stderr="", code=0, killed=False)
FAIL = ExecResult(stdout="", stderr="error", code=1, killed=False)


def _result(base: ExecResult, **overrides: Any) -> ExecResult:
    return ExecResult(
        stdout=overrides.get("stdout", base.stdout),
        stderr=overrides.get("stderr", base.stderr),
        code=overrides.get("code", base.code),
        killed=overrides.get("killed", base.killed),
    )


def with_upstream(results: dict[str, ExecResult]) -> dict[str, ExecResult]:
    """Standard exec results for a clean repo tracking origin/main, not in a merge."""
    results["git rev-parse --git-dir"] = OK
    results["git rev-parse MERGE_HEAD"] = FAIL
    results["git status --porcelain"] = OK
    results["git rev-parse --abbrev-ref --symbolic-full-name @{u}"] = _result(OK, stdout="origin/main\n")
    results["git fetch origin"] = OK
    return results


class _Harness:
    def __init__(self, cwd: str, exec_results: dict[str, ExecResult]) -> None:
        self.exec_calls: list[tuple[str, list[str]]] = []
        self.send_user_message_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.notify_calls: list[tuple[Any, ...]] = []
        self._exec_results = exec_results
        self._handler: Callable[..., Awaitable[None]] | None = None

        harness = self

        class _Api:
            def on(self, event: str, handler: Callable[..., Awaitable[None]]) -> None:
                if event == "agent_end":
                    harness._handler = handler

            async def exec(self, command: str, args: list[str], options: Any = None) -> ExecResult:
                harness.exec_calls.append((command, list(args)))
                key = " ".join([command, *args])
                return harness._exec_results.get(key, FAIL)

            def send_user_message(self, *args: Any, **kwargs: Any) -> None:
                harness.send_user_message_calls.append((args, kwargs))

        pi_extension(_Api())  # type: ignore[arg-type]

        self.ctx = SimpleNamespace(
            cwd=cwd,
            ui=SimpleNamespace(notify=lambda *a, **k: self.notify_calls.append(a)),
        )

    async def trigger(self) -> None:
        assert self._handler is not None
        await self._handler(SimpleNamespace(type="agent_end"), self.ctx)


def setup(cwd: str, exec_results: dict[str, ExecResult]) -> _Harness:
    return _Harness(cwd, exec_results)


class TestGitMergeAndResolveExample:
    @pytest.fixture
    def temp_dir(self):
        path = tempfile.mkdtemp(prefix="pi-merge-test-")
        try:
            yield path
        finally:
            shutil.rmtree(path, ignore_errors=True)

    async def test_skips_when_not_a_git_repository(self, temp_dir: str) -> None:
        h = setup(temp_dir, {"git rev-parse --git-dir": FAIL})
        await h.trigger()

        assert len(h.exec_calls) == 1
        assert h.send_user_message_calls == []

    async def test_skips_when_no_upstream_is_configured(self, temp_dir: str) -> None:
        h = setup(
            temp_dir,
            {
                "git rev-parse --git-dir": OK,
                "git rev-parse --abbrev-ref --symbolic-full-name @{u}": FAIL,
            },
        )
        await h.trigger()

        assert h.send_user_message_calls == []

    async def test_re_sends_conflicts_when_in_an_unfinished_merge(self, temp_dir: str) -> None:
        conflict_content = "\n".join(["<<<<<<< HEAD", "ours", "=======", "theirs", ">>>>>>> origin/main"])
        Path(temp_dir, "file.ts").write_text(conflict_content, encoding="utf-8")

        h = setup(
            temp_dir,
            {
                "git rev-parse --git-dir": OK,
                "git rev-parse MERGE_HEAD": OK,
                "git diff --name-only --diff-filter=U": _result(OK, stdout="file.ts\n"),
            },
        )
        await h.trigger()

        # Should not attempt a new fetch/merge
        assert ("git", ["fetch", "origin"]) not in h.exec_calls
        assert len(h.send_user_message_calls) == 1
        message = h.send_user_message_calls[0][0][0]
        assert "file.ts:1-5" in message

    async def test_skips_when_working_tree_is_dirty_and_not_in_a_merge(self, temp_dir: str) -> None:
        h = setup(
            temp_dir,
            {
                "git rev-parse --git-dir": OK,
                "git rev-parse MERGE_HEAD": FAIL,
                "git status --porcelain": _result(OK, stdout=" M src/index.ts\n"),
            },
        )
        await h.trigger()

        assert ("git", ["fetch", "origin"]) not in h.exec_calls
        assert h.send_user_message_calls == []

    async def test_skips_when_fetch_fails(self, temp_dir: str) -> None:
        results = with_upstream({})
        results["git fetch origin"] = FAIL

        h = setup(temp_dir, results)
        await h.trigger()

        assert h.send_user_message_calls == []

    async def test_skips_when_merge_is_clean(self, temp_dir: str) -> None:
        results = with_upstream({})
        results["git merge --no-ff origin/main"] = OK

        h = setup(temp_dir, results)
        await h.trigger()

        assert h.send_user_message_calls == []

    async def test_sends_conflict_report_as_a_follow_up(self, temp_dir: str) -> None:
        conflict_content = "\n".join(
            [
                "line 1",
                "<<<<<<< HEAD",
                "our change",
                "=======",
                "their change",
                ">>>>>>> origin/main",
                "line 7",
                "<<<<<<< HEAD",
                "second conflict",
                "=======",
                "their second",
                ">>>>>>> origin/main",
            ]
        )
        Path(temp_dir, "src").mkdir(parents=True, exist_ok=True)
        Path(temp_dir, "src/index.ts").write_text(conflict_content, encoding="utf-8")

        results = with_upstream({})
        results["git merge --no-ff origin/main"] = _result(FAIL, code=1)
        results["git diff --name-only --diff-filter=U"] = _result(OK, stdout="src/index.ts\n")

        h = setup(temp_dir, results)
        await h.trigger()

        assert len(h.send_user_message_calls) == 1
        args, kwargs = h.send_user_message_calls[0]
        message = args[0]
        assert "src/index.ts:2-6 (ours 3, theirs 5)" in message
        assert "src/index.ts:8-12 (ours 9, theirs 11)" in message
        # TS passes `{ deliverAs: "followUp" }` as a second positional options object;
        # this port's `send_user_message` takes `deliver_as` as a keyword.
        assert kwargs == {"deliver_as": "followUp"}

    async def test_handles_empty_ours_or_theirs_sections(self, temp_dir: str) -> None:
        conflict_content = "\n".join(["<<<<<<< HEAD", "=======", "only theirs", ">>>>>>> origin/main"])
        Path(temp_dir, "empty-ours.ts").write_text(conflict_content, encoding="utf-8")

        results = with_upstream({})
        results["git merge --no-ff origin/main"] = _result(FAIL, code=1)
        results["git diff --name-only --diff-filter=U"] = _result(OK, stdout="empty-ours.ts\n")

        h = setup(temp_dir, results)
        await h.trigger()

        assert len(h.send_user_message_calls) == 1
        message = h.send_user_message_calls[0][0][0]
        assert "empty-ours.ts:1-4 (ours empty, theirs 3)" in message

    async def test_skips_message_when_merge_fails_but_no_conflict_markers_found(self, temp_dir: str) -> None:
        results = with_upstream({})
        results["git merge --no-ff origin/main"] = _result(FAIL, code=1)
        results["git diff --name-only --diff-filter=U"] = _result(OK, stdout="")

        h = setup(temp_dir, results)
        await h.trigger()

        assert h.send_user_message_calls == []
