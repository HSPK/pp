"""Python port of `packages/coding-agent/test/input-transform-streaming-example.test.ts`.

Exercises `examples/extensions/input_transform_streaming.py`, the Python port of
`examples/extensions/input-transform-streaming.ts`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from pi_coding_agent.core.exec import ExecResult
from pi_coding_agent.core.extensions.types import InputEvent, InputEventResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "extensions"))

from input_transform_streaming import pi_extension


class _FakeExtensionAPI:
    def __init__(self, exec_result: ExecResult) -> None:
        self.handler: Any = None
        self.exec_calls: list[tuple[str, list[str]]] = []
        self._exec_result = exec_result

    def on(self, event: str, handler: Any) -> None:
        if event == "input":
            self.handler = handler

    async def exec(self, command: str, args: list[str], options: Any = None) -> ExecResult:
        self.exec_calls.append((command, args))
        return self._exec_result


class _Harness:
    def __init__(self, exec_result: ExecResult) -> None:
        self.api = _FakeExtensionAPI(exec_result)
        pi_extension(self.api)  # type: ignore[arg-type]

    async def emit(
        self, text: str, streaming_behavior: Literal["steer", "followUp"] | None = None
    ) -> InputEventResult | None:
        event = InputEvent(text=text, source="interactive", streaming_behavior=streaming_behavior)
        # The TypeScript test passes `{} as ExtensionContext`; the handler
        # under test never reads it.
        return await self.api.handler(event, SimpleNamespace())


DIFF_OUTPUT = " src/index.ts | 5 ++---\n 1 file changed, 2 insertions(+), 3 deletions(-)"
GIT_SUCCESS = ExecResult(stdout=DIFF_OUTPUT, stderr="", code=0, killed=False)
GIT_EMPTY = ExecResult(stdout="", stderr="", code=0, killed=False)
GIT_FAIL = ExecResult(stdout="", stderr="not a git repo", code=128, killed=False)


async def test_skips_exec_during_steering() -> None:
    harness = _Harness(GIT_SUCCESS)
    result = await harness.emit("what changes did I make?", "steer")
    assert result == InputEventResult(action="continue")
    assert harness.api.exec_calls == []


async def test_transforms_when_idle_and_text_matches_trigger() -> None:
    harness = _Harness(GIT_SUCCESS)
    result = await harness.emit("review my changes")
    assert harness.api.exec_calls == [("git", ["diff", "--stat"])]
    assert result is not None
    assert result.action == "transform"
    assert result.text is not None
    assert "review my changes" in result.text
    assert "src/index.ts" in result.text


async def test_transforms_when_queued_as_follow_up() -> None:
    harness = _Harness(GIT_SUCCESS)
    result = await harness.emit("show me the diff", "followUp")
    assert harness.api.exec_calls
    assert result is not None
    assert result.action == "transform"


async def test_continues_when_text_does_not_match_trigger() -> None:
    harness = _Harness(GIT_SUCCESS)
    result = await harness.emit("explain this function")
    assert result == InputEventResult(action="continue")
    assert harness.api.exec_calls == []


async def test_continues_when_git_diff_is_empty() -> None:
    harness = _Harness(GIT_EMPTY)
    result = await harness.emit("any changes?")
    assert result == InputEventResult(action="continue")


async def test_continues_when_git_fails() -> None:
    harness = _Harness(GIT_FAIL)
    result = await harness.emit("show modified files")
    assert result == InputEventResult(action="continue")
