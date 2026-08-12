"""Edit a file with one or more exact-text replacements.

Python port of `packages/coding-agent/src/core/tools/edit.ts`. The TUI preview
(`renderCall`/`renderResult`, `computeEditsDiff` used for live diff rendering)
is TS-only presentation logic with no equivalent in the headless `AgentTool`
interface and is not ported; `compute_edits_diff` itself lives in
`edit_diff.py` and remains available for callers that want a preview.
"""

from __future__ import annotations

import errno
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent
from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.tools.edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    format_os_error,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from pi_coding_agent.tools.file_mutation_queue import with_file_mutation_queue
from pi_coding_agent.tools.path_utils import resolve_to_cwd


@dataclass
class EditToolDetails:
    diff: str
    patch: str
    first_changed_line: int | None = None


@dataclass
class EditOperations:
    """Pluggable filesystem operations for the edit tool.

    Port of `EditOperations` in `edit.ts`: override these to delegate file
    editing to a remote system (for example SSH).
    """

    read_file: Callable[[str], Awaitable[bytes]]
    """Read file contents as bytes."""
    write_file: Callable[[str, str], Awaitable[None]]
    """Write content to a file."""
    access: Callable[[str], Awaitable[None]]
    """Check the file is readable and writable; raise if not."""


async def _default_read_file(absolute_path: str) -> bytes:
    with open(absolute_path, "rb") as fh:
        return fh.read()


async def _default_write_file(absolute_path: str, content: str) -> None:
    with open(absolute_path, "w", encoding="utf-8") as fh:
        fh.write(content)


async def _default_access(absolute_path: str) -> None:
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), absolute_path)
    if not os.access(absolute_path, os.R_OK | os.W_OK):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), absolute_path)


def default_edit_operations() -> EditOperations:
    return EditOperations(read_file=_default_read_file, write_file=_default_write_file, access=_default_access)


def _prepare_edit_arguments(params: Any) -> Any:
    """Fold legacy top-level `oldText`/`newText` into `edits`, and parse stringified `edits`.

    Mirrors `prepareEditArguments` in `edit.ts`: some models send `edits` as a
    JSON string instead of an array, or use the pre-multi-edit `oldText`/
    `newText` top-level shape.
    """
    if not isinstance(params, dict):
        return params

    args = params

    if isinstance(args.get("edits"), str):
        try:
            parsed = json.loads(args["edits"])
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            args["edits"] = parsed

    old_text = args.get("oldText")
    new_text = args.get("newText")
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        return args

    edits = list(args["edits"]) if isinstance(args.get("edits"), list) else []
    edits.append({"oldText": old_text, "newText": new_text})
    rest = {k: v for k, v in args.items() if k not in ("oldText", "newText")}
    rest["edits"] = edits
    return rest


def _validate_edit_input(params: dict[str, Any]) -> tuple[str, list[Edit]]:
    edits_raw = params.get("edits")
    if not isinstance(edits_raw, list) or len(edits_raw) == 0:
        raise ValueError("Edit tool input is invalid. edits must contain at least one replacement.")
    edits = [Edit(old_text=e["oldText"], new_text=e["newText"]) for e in edits_raw]
    return params["path"], edits


def create_edit_tool(cwd: str, operations: EditOperations | None = None) -> AgentTool:
    ops = operations if operations is not None else default_edit_operations()

    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
    ) -> AgentToolResult:
        path, edits = _validate_edit_input(params)
        absolute_path = resolve_to_cwd(path, cwd)

        async def mutate() -> AgentToolResult:
            def throw_if_aborted() -> None:
                if signal is not None and signal.aborted:
                    raise RuntimeError("Operation aborted")

            throw_if_aborted()

            try:
                await ops.access(absolute_path)
            except Exception as error:
                throw_if_aborted()
                message = format_os_error(error) if isinstance(error, OSError) else str(error)
                raise RuntimeError(f"Could not edit file: {path}. {message}.") from error
            throw_if_aborted()

            raw_content = (await ops.read_file(absolute_path)).decode("utf-8", errors="replace")
            throw_if_aborted()

            bom, content = strip_bom(raw_content)
            original_ending = detect_line_ending(content)
            normalized_content = normalize_to_lf(content)
            applied = apply_edits_to_normalized_content(normalized_content, edits, path)
            throw_if_aborted()

            final_content = bom + restore_line_endings(applied.new_content, original_ending)
            await ops.write_file(absolute_path, final_content)
            throw_if_aborted()

            diff, first_changed_line = generate_diff_string(applied.base_content, applied.new_content)
            patch = generate_unified_patch(path, applied.base_content, applied.new_content)

            return AgentToolResult(
                content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
                details=EditToolDetails(diff=diff, patch=patch, first_changed_line=first_changed_line),
            )

        try:
            return await with_file_mutation_queue(absolute_path, mutate)
        except OSError as err:
            raise RuntimeError(f"Could not edit file: {path}. {format_os_error(err)}.") from err

    tool = AgentTool(
        name="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
            "non-overlapping region of the original file. If two changes affect the same block or nearby lines, "
            "merge them into one edit instead of emitting overlapping edits. Do not include large unchanged "
            "regions just to connect distant changes."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
                "edits": {
                    "type": "array",
                    "description": (
                        "One or more targeted replacements. Each edit is matched against the original file, "
                        "not incrementally. Do not include overlapping or nested edits. If two changes touch the "
                        "same block or nearby lines, merge them into one edit instead."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {
                                "type": "string",
                                "description": (
                                    "Exact text for one targeted replacement. It must be unique in the original "
                                    "file and must not overlap with any other edits[].oldText in the same call."
                                ),
                            },
                            "newText": {"type": "string", "description": "Replacement text for this targeted edit."},
                        },
                        "required": ["oldText", "newText"],
                    },
                },
            },
            "required": ["path", "edits"],
        },
        execute=execute,
    )
    tool.prepare_arguments = _prepare_edit_arguments
    return tool


# --------------------------------------------------------------------------
# Rendering
#
# Port of `edit.ts`'s `formatEditCall` / `formatEditResult`. The live preview
# component upstream wraps these in a coloured header box (success/error/
# pending backgrounds); that box belongs to the unported extension UI host, so
# what is ported here is the text both paths render.
# --------------------------------------------------------------------------


def format_edit_call(args: Any, theme: Any, cwd: str) -> str:
    """Port of `formatEditCall`."""
    from pi_coding_agent.tools.render_utils import render_tool_path, str_arg

    a = args if isinstance(args, dict) else {}
    raw = str_arg(a.get("file_path") if a.get("file_path") is not None else a.get("path"))
    return f"{theme.fg('toolTitle', theme.bold('edit'))} {render_tool_path(raw, theme, cwd)}"


def format_edit_result(args: Any, result: Any, theme: Any, is_error: bool) -> str | None:
    """Port of `formatEditResult`. `None` means "render nothing".

    An error whose text merely repeats the preview's own error is suppressed,
    matching upstream: the message is already on screen above.
    """
    from pi_coding_agent.modes.interactive.components.diff import RenderDiffOptions, render_diff
    from pi_coding_agent.tools.render_utils import str_arg

    a = args if isinstance(args, dict) else {}
    raw_path = str_arg(a.get("file_path") if a.get("file_path") is not None else a.get("path"))

    if is_error:
        error_text = "\n".join(
            getattr(c, "text", "") or "" for c in getattr(result, "content", []) if getattr(c, "type", None) == "text"
        )
        if not error_text:
            return None
        return theme.fg("error", error_text)

    result_diff = getattr(getattr(result, "details", None), "diff", None)
    if result_diff:
        return render_diff(result_diff, RenderDiffOptions(file_path=raw_path or None))
    return None
