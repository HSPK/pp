"""Search file contents for a pattern, respecting `.gitignore`.

Python port of `packages/coding-agent/src/core/tools/grep.ts`. Prefers
shelling out to `rg` (ripgrep) with `--json` output, matching the TypeScript
behavior exactly when available. `rg` is resolved through
`pi_coding_agent.utils.tools_manager.ensure_tool`, which checks the agent's
managed `bin/` directory and `PATH` before downloading the current release --
the same path TypeScript takes. When `rg` is neither installed nor
downloadable (offline mode, Android, a failed download), this falls back to a
pure-Python regex scan honoring the local gitignore matcher, which upstream
has no equivalent of: TypeScript errors out instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pi_agent.types import AgentTool, AgentToolResult
from pi_ai.types import TextContent
from pi_ai.utils.abort import AbortSignal

from pi_coding_agent.tools.gitignore import build_matcher_for_tree
from pi_coding_agent.tools.path_utils import resolve_to_cwd
from pi_coding_agent.tools.truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
)
from pi_coding_agent.utils import tools_manager

_DEFAULT_LIMIT = 100


@dataclass
class GrepToolDetails:
    truncation: TruncationResult | None = None
    match_limit_reached: int | None = None
    lines_truncated: bool = False


@dataclass
class _Match:
    file_path: str
    line_number: int
    line_text: str | None = None


def _format_path(file_path: str, search_path: str, is_directory: bool) -> str:
    if is_directory:
        relative = os.path.relpath(file_path, search_path)
        if relative and not relative.startswith(".."):
            return relative.replace(os.sep, "/")
    return os.path.basename(file_path)


async def _run_rg(rg_path: str, args: list[str], signal: AbortSignal | None) -> tuple[list[str], str, int]:
    """Run rg, streaming stdout lines, honoring abort. Returns (lines, stderr, returncode)."""
    proc = await asyncio.create_subprocess_exec(
        rg_path, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    async def _read_stdout() -> list[str]:
        assert proc.stdout is not None
        raw = await proc.stdout.read()
        text = raw.decode("utf-8", errors="replace")
        return text.splitlines()

    async def _read_stderr() -> str:
        assert proc.stderr is not None
        raw = await proc.stderr.read()
        return raw.decode("utf-8", errors="replace")

    stdout_task = asyncio.ensure_future(_read_stdout())
    stderr_task = asyncio.ensure_future(_read_stderr())
    wait_task = asyncio.ensure_future(asyncio.gather(stdout_task, stderr_task, proc.wait()))

    if signal is not None:
        abort_task = asyncio.ensure_future(signal.wait())
        done, _pending = await asyncio.wait({wait_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
        if abort_task in done and not wait_task.done():
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            wait_task.cancel()
            stdout_task.cancel()
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await wait_task
            raise RuntimeError("Operation aborted")
        abort_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await abort_task
    else:
        await wait_task

    lines = stdout_task.result()
    stderr = stderr_task.result()
    returncode = proc.returncode if proc.returncode is not None else -1
    return lines, stderr, returncode


async def _grep_via_rg(
    rg_path: str,
    pattern: str,
    search_path: str,
    is_directory: bool,
    glob: str | None,
    ignore_case: bool,
    literal: bool,
    context_value: int,
    effective_limit: int,
    signal: AbortSignal | None,
) -> tuple[list[str], GrepToolDetails]:
    args = ["--json", "--line-number", "--color=never", "--hidden"]
    if ignore_case:
        args.append("--ignore-case")
    if literal:
        args.append("--fixed-strings")
    if glob:
        args.extend(["--glob", glob])
    args.extend(["--", pattern, search_path])

    lines, stderr, returncode = await _run_rg(rg_path, args, signal)

    matches: list[_Match] = []
    match_count = 0
    match_limit_reached = False
    for line in lines:
        if not line.strip() or match_count >= effective_limit:
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if event.get("type") != "match":
            continue
        match_count += 1
        data = event.get("data") or {}
        file_path = (data.get("path") or {}).get("text")
        line_number = data.get("line_number")
        line_text = (data.get("lines") or {}).get("text")
        if file_path and isinstance(line_number, int):
            matches.append(_Match(file_path=file_path, line_number=line_number, line_text=line_text))
        if match_count >= effective_limit:
            match_limit_reached = True
            break

    if returncode not in (0, 1) and not match_limit_reached:
        error_msg = stderr.strip() or f"ripgrep exited with code {returncode}"
        raise RuntimeError(error_msg)

    if match_count == 0:
        return [], GrepToolDetails()

    file_cache: dict[str, list[str]] = {}

    def get_file_lines(file_path: str) -> list[str]:
        if file_path not in file_cache:
            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                file_cache[file_path] = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            except OSError:
                file_cache[file_path] = []
        return file_cache[file_path]

    details = GrepToolDetails()
    output_lines: list[str] = []
    for match in matches:
        if context_value == 0 and match.line_text is not None:
            relative_path = _format_path(match.file_path, search_path, is_directory)
            sanitized = match.line_text.replace("\r\n", "\n").replace("\r", "")
            if sanitized.endswith("\n"):
                sanitized = sanitized[:-1]
            truncated_text, was_truncated = truncate_line(sanitized)
            if was_truncated:
                details.lines_truncated = True
            output_lines.append(f"{relative_path}:{match.line_number}: {truncated_text}")
        else:
            output_lines.extend(
                _format_block(
                    match.file_path,
                    match.line_number,
                    search_path,
                    is_directory,
                    get_file_lines,
                    context_value,
                    details,
                )
            )

    if match_limit_reached:
        details.match_limit_reached = effective_limit
    return output_lines, details


def _format_block(
    file_path: str,
    line_number: int,
    search_path: str,
    is_directory: bool,
    get_file_lines: Callable[[str], list[str]],
    context_value: int,
    details: GrepToolDetails,
) -> list[str]:
    relative_path = _format_path(file_path, search_path, is_directory)
    lines = get_file_lines(file_path)
    if not lines:
        return [f"{relative_path}:{line_number}: (unable to read file)"]
    block: list[str] = []
    start = max(1, line_number - context_value) if context_value > 0 else line_number
    end = min(len(lines), line_number + context_value) if context_value > 0 else line_number
    for current in range(start, end + 1):
        line_text = lines[current - 1] if current - 1 < len(lines) else ""
        sanitized = line_text.replace("\r", "")
        is_match_line = current == line_number
        truncated_text, was_truncated = truncate_line(sanitized)
        if was_truncated:
            details.lines_truncated = True
        if is_match_line:
            block.append(f"{relative_path}:{current}: {truncated_text}")
        else:
            block.append(f"{relative_path}-{current}- {truncated_text}")
    return block


def _fallback_regex(pattern: str, ignore_case: bool, literal: bool) -> re.Pattern[str]:
    flags = re.IGNORECASE if ignore_case else 0
    if literal:
        return re.compile(re.escape(pattern), flags)
    return re.compile(pattern, flags)


def _iter_search_files(search_path: str, is_directory: bool, glob: str | None) -> list[str]:
    if not is_directory:
        return [search_path]

    matcher = build_matcher_for_tree(search_path)
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(search_path):
        rel_dir = os.path.relpath(dirpath, search_path)
        rel_dir_posix = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")

        kept_dirnames = []
        for name in sorted(dirnames):
            if name == ".git":
                continue
            rel_child = name if not rel_dir_posix else f"{rel_dir_posix}/{name}"
            if matcher.is_ignored(rel_child, is_dir=True):
                continue
            kept_dirnames.append(name)
        dirnames[:] = kept_dirnames

        for name in sorted(filenames):
            rel_file = name if not rel_dir_posix else f"{rel_dir_posix}/{name}"
            if matcher.is_ignored(rel_file, is_dir=False):
                continue
            if glob and not fnmatch.fnmatch(rel_file, glob) and not fnmatch.fnmatch(name, glob):
                continue
            files.append(os.path.join(dirpath, name))
    return files


def _grep_via_fallback(
    pattern: str,
    search_path: str,
    is_directory: bool,
    glob: str | None,
    ignore_case: bool,
    literal: bool,
    context_value: int,
    effective_limit: int,
) -> tuple[list[str], GrepToolDetails]:
    regex = _fallback_regex(pattern, ignore_case, literal)
    files = _iter_search_files(search_path, is_directory, glob)

    details = GrepToolDetails()
    output_lines: list[str] = []
    match_count = 0
    match_limit_reached = False

    for file_path in files:
        if match_count >= effective_limit:
            break
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            continue
        lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        relative_path = _format_path(file_path, search_path, is_directory)

        for idx, line in enumerate(lines):
            if not regex.search(line):
                continue
            match_count += 1
            line_number = idx + 1
            if context_value == 0:
                truncated_text, was_truncated = truncate_line(line)
                if was_truncated:
                    details.lines_truncated = True
                output_lines.append(f"{relative_path}:{line_number}: {truncated_text}")
            else:
                start = max(1, line_number - context_value)
                end = min(len(lines), line_number + context_value)
                for current in range(start, end + 1):
                    ctx_line = lines[current - 1]
                    truncated_text, was_truncated = truncate_line(ctx_line)
                    if was_truncated:
                        details.lines_truncated = True
                    if current == line_number:
                        output_lines.append(f"{relative_path}:{current}: {truncated_text}")
                    else:
                        output_lines.append(f"{relative_path}-{current}- {truncated_text}")
            if match_count >= effective_limit:
                match_limit_reached = True
                break

    if match_limit_reached:
        details.match_limit_reached = effective_limit
    return output_lines, details


def create_grep_tool(cwd: str) -> AgentTool:
    async def execute(
        tool_call_id: str,
        params: dict[str, Any],
        signal: AbortSignal | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
    ) -> AgentToolResult:
        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        pattern = params["pattern"]
        search_dir = params.get("path")
        glob = params.get("glob")
        ignore_case = bool(params.get("ignoreCase"))
        literal = bool(params.get("literal"))
        context = params.get("context")
        limit = params.get("limit")

        search_path = resolve_to_cwd(search_dir or ".", cwd)
        if not os.path.exists(search_path):
            raise RuntimeError(f"Path not found: {search_path}")
        is_directory = os.path.isdir(search_path)

        context_value = context if context and context > 0 else 0
        effective_limit = max(1, limit if limit is not None else _DEFAULT_LIMIT)

        rg_path = await tools_manager.ensure_tool("rg", silent=True)
        if rg_path:
            output_lines, details = await _grep_via_rg(
                rg_path,
                pattern,
                search_path,
                is_directory,
                glob,
                ignore_case,
                literal,
                context_value,
                effective_limit,
                signal,
            )
        else:
            output_lines, details = _grep_via_fallback(
                pattern, search_path, is_directory, glob, ignore_case, literal, context_value, effective_limit
            )

        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        if not output_lines:
            return AgentToolResult(content=[TextContent(text="No matches found")])

        raw_output = "\n".join(output_lines)
        truncation = truncate_head(raw_output, max_lines=2**31)
        output = truncation.content
        notices: list[str] = []
        if details.match_limit_reached:
            notices.append(
                f"{effective_limit} matches limit reached. Use limit={effective_limit * 2} for more, or refine pattern"
            )
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
        if details.lines_truncated:
            notices.append(f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines")
        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        has_details = (
            details.truncation is not None or details.match_limit_reached is not None or details.lines_truncated
        )
        return AgentToolResult(content=[TextContent(text=output)], details=details if has_details else None)

    return AgentTool(
        name="grep",
        description=(
            "Search file contents for a pattern. Returns matching lines with file paths and line numbers. "
            f"Respects .gitignore. Output is truncated to {_DEFAULT_LIMIT} matches or "
            f"{DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            f"Long lines are truncated to {GREP_MAX_LINE_LENGTH} chars."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (default: current directory)",
                },
                "glob": {
                    "type": "string",
                    "description": "Filter files by glob pattern, e.g. '*.ts' or '**/*.spec.ts'",
                },
                "ignoreCase": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
                "literal": {
                    "type": "boolean",
                    "description": "Treat pattern as literal string instead of regex (default: false)",
                },
                "context": {
                    "type": "number",
                    "description": "Number of lines to show before and after each match (default: 0)",
                },
                "limit": {"type": "number", "description": "Maximum number of matches to return (default: 100)"},
            },
            "required": ["pattern"],
        },
        execute=execute,
    )
