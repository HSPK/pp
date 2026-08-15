"""Python port of `packages/coding-agent/test/extensions-input-event.test.ts`.

The TypeScript test writes throwaway `.ts` extension files into a temp
`extensions/` directory and loads them with `discoverAndLoadExtensions`. The
Python loader takes `.py` files exporting `pi_extension`, so the extension
bodies are translated but the mechanism is the same. Where the TypeScript
extensions record state on `globalThis`, the Python ones record it on
`builtins`.
"""

from __future__ import annotations

import builtins
import shutil
import textwrap
from pathlib import Path

from pi_ai.types import ImageContent

from pi_coding_agent.core.extensions.loader import discover_and_load_extensions
from pi_coding_agent.core.extensions.runner import ExtensionRunner
from pi_coding_agent.core.session_manager import SessionManager

_PRELUDE = "import builtins\nfrom pi_coding_agent.core.extensions.types import InputEventResult\n"


def _ext(body: str) -> str:
    return _PRELUDE + textwrap.dedent(body)


def _test_var() -> object:
    return getattr(builtins, "pi_test_var", None)


async def _create_runner(tmp_path: Path, *extensions: str) -> ExtensionRunner:
    extensions_dir = tmp_path / "extensions"
    shutil.rmtree(extensions_dir, ignore_errors=True)
    extensions_dir.mkdir()
    for index, source in enumerate(extensions):
        (extensions_dir / f"e{index}.py").write_text(source, encoding="utf-8")
    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))
    return ExtensionRunner(result.extensions, cwd=str(tmp_path), session_manager=SessionManager.in_memory())


_NO_RESULT = _ext(
    """
    async def _on_input(event, ctx):
        return None

    def pi_extension(pi):
        pi.on("input", _on_input)
    """
)

_CONTINUE = _ext(
    """
    async def _on_input(event, ctx):
        return InputEventResult(action="continue")

    def pi_extension(pi):
        pi.on("input", _on_input)
    """
)


async def test_continue_when_no_handlers_no_result_or_explicit_continue(tmp_path: Path) -> None:
    runner = await _create_runner(tmp_path)
    assert (await runner.emit_input("x", None, "interactive")).action == "continue"

    runner = await _create_runner(tmp_path, _NO_RESULT)
    assert (await runner.emit_input("x", None, "interactive")).action == "continue"

    runner = await _create_runner(tmp_path, _CONTINUE)
    assert (await runner.emit_input("x", None, "interactive")).action == "continue"


async def test_transforms_text_and_preserves_images_when_omitted(tmp_path: Path) -> None:
    runner = await _create_runner(
        tmp_path,
        _ext(
            """
            async def _on_input(event, ctx):
                return InputEventResult(action="transform", text="T:" + event.text)

            def pi_extension(pi):
                pi.on("input", _on_input)
            """
        ),
    )
    images = [ImageContent(data="orig", mime_type="image/png")]
    result = await runner.emit_input("hi", images, "interactive")
    assert result.action == "transform"
    assert result.text == "T:hi"
    assert result.images == images


async def test_transforms_and_replaces_images_when_provided(tmp_path: Path) -> None:
    runner = await _create_runner(
        tmp_path,
        _ext(
            """
            from pi_ai.types import ImageContent

            async def _on_input(event, ctx):
                return InputEventResult(
                    action="transform",
                    text="X",
                    images=[ImageContent(data="new", mime_type="image/jpeg")],
                )

            def pi_extension(pi):
                pi.on("input", _on_input)
            """
        ),
    )
    result = await runner.emit_input("hi", [ImageContent(data="orig", mime_type="image/png")], "interactive")
    assert result.action == "transform"
    assert result.text == "X"
    assert result.images == [ImageContent(data="new", mime_type="image/jpeg")]


async def test_chains_transforms_across_multiple_handlers(tmp_path: Path) -> None:
    runner = await _create_runner(
        tmp_path,
        _ext(
            """
            async def _on_input(event, ctx):
                return InputEventResult(action="transform", text=event.text + "[1]")

            def pi_extension(pi):
                pi.on("input", _on_input)
            """
        ),
        _ext(
            """
            async def _on_input(event, ctx):
                return InputEventResult(action="transform", text=event.text + "[2]")

            def pi_extension(pi):
                pi.on("input", _on_input)
            """
        ),
    )
    result = await runner.emit_input("X", None, "interactive")
    assert result.action == "transform"
    assert result.text == "X[1][2]"
    assert result.images is None


async def test_short_circuits_on_handled_and_skips_subsequent_handlers(tmp_path: Path) -> None:
    builtins.pi_test_var = False  # type: ignore[attr-defined]
    runner = await _create_runner(
        tmp_path,
        _ext(
            """
            async def _on_input(event, ctx):
                return InputEventResult(action="handled")

            def pi_extension(pi):
                pi.on("input", _on_input)
            """
        ),
        _ext(
            """
            async def _on_input(event, ctx):
                builtins.pi_test_var = True

            def pi_extension(pi):
                pi.on("input", _on_input)
            """
        ),
    )
    result = await runner.emit_input("X", None, "interactive")
    assert result.action == "handled"
    assert result.text is None
    assert result.images is None
    assert _test_var() is False


async def test_passes_source_correctly_for_all_source_types(tmp_path: Path) -> None:
    runner = await _create_runner(
        tmp_path,
        _ext(
            """
            async def _on_input(event, ctx):
                builtins.pi_test_var = event.source
                return InputEventResult(action="continue")

            def pi_extension(pi):
                pi.on("input", _on_input)
            """
        ),
    )
    for source in ("interactive", "rpc", "extension"):
        await runner.emit_input("x", None, source)
        assert _test_var() == source


async def test_passes_streaming_behavior_correctly(tmp_path: Path) -> None:
    runner = await _create_runner(
        tmp_path,
        _ext(
            """
            async def _on_input(event, ctx):
                builtins.pi_test_var = event.streaming_behavior
                return InputEventResult(action="continue")

            def pi_extension(pi):
                pi.on("input", _on_input)
            """
        ),
    )
    await runner.emit_input("x", None, "interactive", "steer")
    assert _test_var() == "steer"
    await runner.emit_input("x", None, "interactive", "followUp")
    assert _test_var() == "followUp"
    await runner.emit_input("x", None, "interactive")
    assert _test_var() is None


async def test_catches_handler_errors_and_continues(tmp_path: Path) -> None:
    runner = await _create_runner(
        tmp_path,
        _ext(
            """
            async def _on_input(event, ctx):
                raise RuntimeError("boom")

            def pi_extension(pi):
                pi.on("input", _on_input)
            """
        ),
    )
    errors: list[str] = []
    runner.on_error(lambda error: errors.append(error.error))
    result = await runner.emit_input("x", None, "interactive")
    assert result.action == "continue"
    assert "boom" in errors


async def test_has_handlers_returns_correct_value(tmp_path: Path) -> None:
    runner = await _create_runner(tmp_path)
    assert runner.has_handlers("input") is False

    runner = await _create_runner(tmp_path, _NO_RESULT)
    assert runner.has_handlers("input") is True
