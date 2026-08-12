"""Python port of `packages/coding-agent/test/file-mutation-queue.test.ts`."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from pi_ai.utils.abort import AbortController
from pi_coding_agent.tools.edit import EditOperations, create_edit_tool
from pi_coding_agent.tools.file_mutation_queue import with_file_mutation_queue
from pi_coding_agent.tools.write import WriteOperations, create_write_tool


async def _delay(ms: int) -> None:
    await asyncio.sleep(ms / 1000)


async def _resolves_within(future: asyncio.Future[None] | asyncio.Task[None], ms: int) -> bool:
    try:
        await asyncio.wait_for(asyncio.shield(future), ms / 1000)
    except TimeoutError:
        return False
    return True


async def test_serializes_operations_for_the_same_file(tmp_path: Path) -> None:
    order: list[str] = []
    path = str(tmp_path / "file-mutation-queue-same")

    async def first() -> None:
        order.append("first:start")
        await _delay(30)
        order.append("first:end")

    async def second() -> None:
        order.append("second:start")
        order.append("second:end")

    await asyncio.gather(
        with_file_mutation_queue(path, first),
        with_file_mutation_queue(path, second),
    )

    assert order == ["first:start", "first:end", "second:start", "second:end"]


async def test_allows_different_files_to_proceed_in_parallel(tmp_path: Path) -> None:
    order: list[str] = []

    async def a() -> None:
        order.append("a:start")
        await _delay(30)
        order.append("a:end")

    async def b() -> None:
        order.append("b:start")
        await _delay(30)
        order.append("b:end")

    await asyncio.gather(
        with_file_mutation_queue(str(tmp_path / "file-mutation-queue-a"), a),
        with_file_mutation_queue(str(tmp_path / "file-mutation-queue-b"), b),
    )

    assert order.index("a:start") < order.index("a:end")
    assert order.index("b:start") < order.index("b:end")
    assert order.index("b:start") < order.index("a:end")


async def test_uses_the_same_queue_for_symlink_aliases(tmp_path: Path) -> None:
    target_path = tmp_path / "target.txt"
    symlink_path = tmp_path / "alias.txt"
    target_path.write_text("hello\n", encoding="utf-8")
    symlink_path.symlink_to(target_path)

    order: list[str] = []

    async def target() -> None:
        order.append("target:start")
        await _delay(30)
        order.append("target:end")

    async def alias() -> None:
        order.append("alias:start")
        order.append("alias:end")

    await asyncio.gather(
        with_file_mutation_queue(str(target_path), target),
        with_file_mutation_queue(str(symlink_path), alias),
    )

    assert order == ["target:start", "target:end", "alias:start", "alias:end"]


def _slow_edit_operations() -> EditOperations:
    async def access(path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    async def read_file(path: str) -> bytes:
        with open(path, "rb") as fh:
            data = fh.read()
        await _delay(30)
        return data

    async def write_file(path: str, content: str) -> None:
        await _delay(30)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    return EditOperations(read_file=read_file, write_file=write_file, access=access)


async def test_preserves_both_parallel_edits_on_the_same_file(tmp_path: Path) -> None:
    file_path = tmp_path / "parallel-edit.txt"
    file_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    edit_tool = create_edit_tool(str(tmp_path), _slow_edit_operations())

    await asyncio.gather(
        edit_tool.execute("call-1", {"path": str(file_path), "edits": [{"oldText": "alpha", "newText": "ALPHA"}]}),
        edit_tool.execute("call-2", {"path": str(file_path), "edits": [{"oldText": "beta", "newText": "BETA"}]}),
    )

    assert file_path.read_text(encoding="utf-8") == "ALPHA\nBETA\ngamma\n"


async def test_shares_the_queue_between_edit_and_write(tmp_path: Path) -> None:
    file_path = tmp_path / "mixed.txt"
    file_path.write_text("original\n", encoding="utf-8")

    edit_tool = create_edit_tool(str(tmp_path), _slow_edit_operations())

    async def mkdir(_directory: str) -> None:
        return None

    async def write_file(path: str, content: str) -> None:
        await _delay(10)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    write_tool = create_write_tool(str(tmp_path), WriteOperations(write_file=write_file, mkdir=mkdir))

    edit_task = asyncio.ensure_future(
        edit_tool.execute("call-1", {"path": str(file_path), "edits": [{"oldText": "original", "newText": "edited"}]})
    )
    await _delay(5)
    write_task = asyncio.ensure_future(
        write_tool.execute("call-2", {"path": str(file_path), "content": "replacement\n"})
    )

    await asyncio.gather(edit_task, write_task)

    assert file_path.read_text(encoding="utf-8") == "replacement\n"


async def test_keeps_write_queue_locked_while_an_aborted_write_is_still_in_flight(tmp_path: Path) -> None:
    file_path = tmp_path / "abort-write.txt"
    loop = asyncio.get_running_loop()
    first_write_started: asyncio.Future[None] = loop.create_future()
    finish_first_write: asyncio.Future[None] = loop.create_future()
    second_write_started: asyncio.Future[None] = loop.create_future()
    first_write_settled = False

    async def mkdir(_directory: str) -> None:
        return None

    async def write_file(path: str, content: str) -> None:
        nonlocal first_write_settled
        if content == "first\n":
            first_write_started.set_result(None)
            await finish_first_write
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            first_write_settled = True
            return

        if content == "second\n":
            assert first_write_settled is True
            second_write_started.set_result(None)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    write_tool = create_write_tool(str(tmp_path), WriteOperations(write_file=write_file, mkdir=mkdir))

    controller = AbortController()
    first_write = asyncio.ensure_future(
        write_tool.execute("call-1", {"path": str(file_path), "content": "first\n"}, controller.signal)
    )
    await first_write_started
    controller.abort()

    second_write = asyncio.ensure_future(write_tool.execute("call-2", {"path": str(file_path), "content": "second\n"}))
    assert await _resolves_within(second_write_started, 20) is False

    finish_first_write.set_result(None)
    with pytest.raises(RuntimeError, match="Operation aborted"):
        await first_write
    await second_write

    assert file_path.read_text(encoding="utf-8") == "second\n"


async def test_keeps_edit_queue_locked_while_an_aborted_edit_write_is_still_in_flight(tmp_path: Path) -> None:
    file_path = tmp_path / "abort-edit.txt"
    file_path.write_text("alpha\nbeta\n", encoding="utf-8")
    loop = asyncio.get_running_loop()
    first_write_started: asyncio.Future[None] = loop.create_future()
    finish_first_write: asyncio.Future[None] = loop.create_future()
    second_write_started: asyncio.Future[None] = loop.create_future()
    first_write_settled = False

    async def access(path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    async def read_file(path: str) -> bytes:
        with open(path, "rb") as fh:
            return fh.read()

    async def write_file(path: str, content: str) -> None:
        nonlocal first_write_settled
        if content == "ALPHA\nbeta\n":
            first_write_started.set_result(None)
            await finish_first_write
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(content)
            first_write_settled = True
            return

        if content in ("ALPHA\nBETA\n", "alpha\nBETA\n"):
            assert first_write_settled is True
            second_write_started.set_result(None)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    edit_tool = create_edit_tool(
        str(tmp_path), EditOperations(read_file=read_file, write_file=write_file, access=access)
    )

    controller = AbortController()
    first_edit = asyncio.ensure_future(
        edit_tool.execute(
            "call-1",
            {"path": str(file_path), "edits": [{"oldText": "alpha", "newText": "ALPHA"}]},
            controller.signal,
        )
    )
    await first_write_started
    controller.abort()

    second_edit = asyncio.ensure_future(
        edit_tool.execute("call-2", {"path": str(file_path), "edits": [{"oldText": "beta", "newText": "BETA"}]})
    )
    assert await _resolves_within(second_write_started, 20) is False

    finish_first_write.set_result(None)
    with pytest.raises(RuntimeError, match="Operation aborted"):
        await first_edit
    await second_edit

    assert file_path.read_text(encoding="utf-8") == "ALPHA\nBETA\n"
