"""Python port of `packages/coding-agent/test/suite/agent-session-tool-result-images.test.ts`."""

from __future__ import annotations

import asyncio
import base64
import struct
import zlib
from pathlib import Path

import pytest
from harness import Harness, create_harness
from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.providers.faux import faux_assistant_message, faux_tool_call
from pi_ai.types import ImageContent, TextContent


def _png_chunk(chunk_type: str, body: bytes) -> bytes:
    header = struct.pack(">I", len(body)) + chunk_type.encode("ascii")
    checksum = struct.pack(">I", zlib.crc32(header[4:] + body) & 0xFFFFFFFF)
    return header + body + checksum


def create_png(width: int, height: int) -> bytes:
    """Port of `createPng`: an 8-bit grayscale PNG without pulling in an encoder."""
    ihdr = struct.pack(">II", width, height) + bytes([8, 0, 0, 0, 0])
    raw = bytearray((width + 1) * height)
    for row in range(height):
        start = row * (width + 1) + 1
        end = (row + 1) * (width + 1)
        for i in range(start, end):
            raw[i] = row % 256
    return b"".join(
        [
            bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
            _png_chunk("IHDR", ihdr),
            _png_chunk("IDAT", zlib.compress(bytes(raw))),
            _png_chunk("IEND", b""),
        ]
    )


def read_png_dimensions(base64_data: str) -> tuple[int, int]:
    buffer = base64.b64decode(base64_data)
    return struct.unpack(">I", buffer[16:20])[0], struct.unpack(">I", buffer[20:24])[0]


OVERSIZED_PNG_BASE64 = base64.b64encode(create_png(2400, 4800)).decode("ascii")


def _screenshot_tool() -> AgentTool:
    """Stands in for extension, MCP bridge, or screenshot tools returning images."""

    async def execute(tool_call_id: str, params, signal=None, on_update=None) -> AgentToolResult:
        return AgentToolResult(
            content=[
                TextContent(text="captured"),
                ImageContent(data=OVERSIZED_PNG_BASE64, mime_type="image/png"),
            ],
            details={},
        )

    return AgentTool(
        name="screenshot",
        label="Screenshot",
        description="Return an oversized screenshot",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


def get_tool_result_images(harness: Harness) -> list[ImageContent]:
    return [
        block
        for message in harness.session.messages
        if getattr(message, "role", "") == "toolResult"
        for block in message.content
        if isinstance(block, ImageContent)
    ]


async def test_resizes_oversized_tool_result_images_before_they_enter_history(tmp_path: Path) -> None:
    pytest.importorskip("PIL", reason="image resizing needs Pillow, this port's stand-in for the TS Photon/WASM codec")
    harness = await create_harness(tmp_path, tools=[_screenshot_tool()])
    try:
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("screenshot", {})], stop_reason="toolUse"),
                faux_assistant_message("done"),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("take a screenshot"), timeout=30)

        images = get_tool_result_images(harness)
        assert len(images) == 1
        width, height = read_png_dimensions(images[0].data)
        assert width <= 2000
        assert height <= 2000
    finally:
        harness.cleanup()


async def test_honors_images_auto_resize_being_disabled(tmp_path: Path) -> None:
    harness = await create_harness(tmp_path, tools=[_screenshot_tool()], settings={"images": {"autoResize": False}})
    try:
        harness.set_responses(
            [
                faux_assistant_message([faux_tool_call("screenshot", {})], stop_reason="toolUse"),
                faux_assistant_message("done"),
            ]
        )

        await asyncio.wait_for(harness.session.prompt("take a screenshot"), timeout=20)

        images = get_tool_result_images(harness)
        assert len(images) == 1
        assert images[0].data == OVERSIZED_PNG_BASE64
    finally:
        harness.cleanup()
