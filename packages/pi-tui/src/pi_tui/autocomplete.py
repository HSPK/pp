"""Autocomplete providers for `packages/tui/src/autocomplete.ts`.

Provides slash-command completion, `@`-prefixed fuzzy file attachment
completion (backed by the `fd` binary when available), and plain path
completion for file references typed directly into the input.

`AbortSignal` is modelled as a plain `asyncio.Event`: callers set the event to
request cancellation of an in-flight `fd` subprocess, mirroring how the
TypeScript code listens for the DOM `abort` event. `pi-tui` does not depend on
`pi-ai`, so this module does not reuse `pi_ai.utils.abort.AbortSignal`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from pi_tui.fuzzy import fuzzy_filter

_PATH_DELIMITERS = frozenset({" ", "\t", '"', "'", "="})


def _to_display_path(value: str) -> str:
    return value.replace("\\", "/")


def _node_basename(path: str) -> str:
    """`os.path.basename` equivalent to Node's `path.basename`.

    Node's `path.basename` strips trailing separators before extracting the
    last path segment (`path.basename("src/") === "src"`), while Python's
    `os.path.basename` treats a trailing separator as an empty final segment
    (`os.path.basename("src/") == ""`).
    """
    return os.path.basename(path.rstrip("/")) if path != "/" else ""


def _escape_regex(value: str) -> str:
    return re.sub(r"[.*+?^${}()|\[\]\\]", lambda m: "\\" + m.group(0), value)


def _build_fd_path_query(query: str) -> str:
    normalized = _to_display_path(query)
    if "/" not in normalized:
        return normalized

    has_trailing_separator = normalized.endswith("/")
    trimmed = re.sub(r"^/+|/+$", "", normalized)
    if not trimmed:
        return normalized

    separator_pattern = "[\\\\/]"
    segments = [_escape_regex(segment) for segment in trimmed.split("/") if segment]
    if not segments:
        return normalized

    pattern = separator_pattern.join(segments)
    if has_trailing_separator:
        pattern += separator_pattern
    return pattern


def _find_last_delimiter(text: str) -> int:
    for i in range(len(text) - 1, -1, -1):
        if text[i] in _PATH_DELIMITERS:
            return i
    return -1


def _find_unclosed_quote_start(text: str) -> int | None:
    in_quotes = False
    quote_start = -1

    for i, ch in enumerate(text):
        if ch == '"':
            in_quotes = not in_quotes
            if in_quotes:
                quote_start = i

    return quote_start if in_quotes else None


def _is_token_start(text: str, index: int) -> bool:
    return index == 0 or text[index - 1] in _PATH_DELIMITERS


def _extract_quoted_prefix(text: str) -> str | None:
    quote_start = _find_unclosed_quote_start(text)
    if quote_start is None:
        return None

    if quote_start > 0 and text[quote_start - 1] == "@":
        if not _is_token_start(text, quote_start - 1):
            return None
        return text[quote_start - 1 :]

    if not _is_token_start(text, quote_start):
        return None

    return text[quote_start:]


@dataclass
class _ParsedPathPrefix:
    raw_prefix: str
    is_at_prefix: bool
    is_quoted_prefix: bool


def _parse_path_prefix(prefix: str) -> _ParsedPathPrefix:
    if prefix.startswith('@"'):
        return _ParsedPathPrefix(prefix[2:], True, True)
    if prefix.startswith('"'):
        return _ParsedPathPrefix(prefix[1:], False, True)
    if prefix.startswith("@"):
        return _ParsedPathPrefix(prefix[1:], True, False)
    return _ParsedPathPrefix(prefix, False, False)


def _build_completion_value(path: str, *, is_directory: bool, is_at_prefix: bool, is_quoted_prefix: bool) -> str:
    del is_directory  # kept for parity with the TS signature; unused there too
    needs_quotes = is_quoted_prefix or " " in path
    prefix = "@" if is_at_prefix else ""

    if not needs_quotes:
        return f"{prefix}{path}"

    return f'{prefix}"{path}"'


@dataclass
class _FdEntry:
    path: str
    is_directory: bool


async def _walk_directory_with_fd(
    base_dir: str, fd_path: str, query: str, max_results: int, signal: asyncio.Event
) -> list[_FdEntry]:
    """Use `fd` to walk a directory tree (fast, respects `.gitignore`)."""
    args = [
        "--base-directory",
        base_dir,
        "--max-results",
        str(max_results),
        "--type",
        "f",
        "--type",
        "d",
        "--follow",
        "--hidden",
        "--exclude",
        ".git",
        "--exclude",
        ".git/*",
        "--exclude",
        ".git/**",
    ]

    if "/" in _to_display_path(query):
        args.append("--full-path")

    if query:
        args.append(_build_fd_path_query(query))

    if signal.is_set():
        return []

    try:
        proc = await asyncio.create_subprocess_exec(
            fd_path,
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError:
        return []

    communicate_task = asyncio.ensure_future(proc.communicate())
    abort_task = asyncio.ensure_future(signal.wait())
    try:
        done, _pending = await asyncio.wait({communicate_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)

        if abort_task in done and communicate_task not in done:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            communicate_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await communicate_task
            return []

        stdout_bytes, _stderr_bytes = communicate_task.result()
    finally:
        if not abort_task.done():
            abort_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await abort_task

    if signal.is_set() or proc.returncode != 0 or not stdout_bytes:
        return []

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    lines = [line for line in stdout.strip().split("\n") if line]
    results: list[_FdEntry] = []

    for line in lines:
        display_line = _to_display_path(line)
        has_trailing_separator = display_line.endswith("/")
        normalized_path = display_line[:-1] if has_trailing_separator else display_line
        if normalized_path == ".git" or normalized_path.startswith(".git/") or "/.git/" in normalized_path:
            continue

        results.append(_FdEntry(path=display_line, is_directory=has_trailing_separator))

    return results


@dataclass
class AutocompleteItem:
    value: str
    label: str
    description: str | None = None


GetArgumentCompletions = Callable[[str], Awaitable["list[AutocompleteItem] | None"]]


@dataclass
class SlashCommand:
    name: str
    description: str | None = None
    argument_hint: str | None = None
    # Callable returning argument completions for this command, or None if no
    # argument completion is available. Always awaited, even if the
    # underlying implementation is effectively synchronous (the TS type is
    # `Awaitable<T>`, i.e. `T | Promise<T>`; the Python port always uses an
    # `async def`/coroutine).
    get_argument_completions: GetArgumentCompletions | None = None


@dataclass
class AutocompleteSuggestions:
    items: list[AutocompleteItem]
    prefix: str  # What we're matching against (e.g., "/" or "src/")


@dataclass
class AppliedCompletion:
    lines: list[str]
    cursor_line: int
    cursor_col: int


@runtime_checkable
class AutocompleteProvider(Protocol):
    """Structural interface mirrored from the TS `AutocompleteProvider`."""

    # Characters that should naturally trigger this provider at token boundaries.
    trigger_characters: list[str] | None

    async def get_suggestions(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        *,
        signal: asyncio.Event,
        force: bool = False,
    ) -> AutocompleteSuggestions | None: ...

    def apply_completion(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        item: AutocompleteItem,
        prefix: str,
    ) -> AppliedCompletion: ...

    def should_trigger_file_completion(self, lines: list[str], cursor_line: int, cursor_col: int) -> bool: ...


@dataclass
class _ScopedFuzzyQuery:
    base_dir: str
    query: str
    display_base: str


class CombinedAutocompleteProvider:
    """Combined provider that handles both slash commands and file paths."""

    trigger_characters: list[str] | None = None

    def __init__(
        self,
        commands: list[SlashCommand | AutocompleteItem] | None = None,
        base_path: str = "",
        fd_path: str | None = None,
    ) -> None:
        self._commands: list[SlashCommand | AutocompleteItem] = commands if commands is not None else []
        self._base_path = base_path
        self._fd_path = fd_path

    async def get_suggestions(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        *,
        signal: asyncio.Event,
        force: bool = False,
    ) -> AutocompleteSuggestions | None:
        current_line = lines[cursor_line] if 0 <= cursor_line < len(lines) else ""
        text_before_cursor = current_line[:cursor_col]

        at_prefix = self._extract_at_prefix(text_before_cursor)
        if at_prefix:
            parsed = _parse_path_prefix(at_prefix)
            suggestions = await self._get_fuzzy_file_suggestions(
                parsed.raw_prefix, is_quoted_prefix=parsed.is_quoted_prefix, signal=signal
            )
            if not suggestions:
                return None

            return AutocompleteSuggestions(items=suggestions, prefix=at_prefix)

        if not force and text_before_cursor.startswith("/"):
            space_index = text_before_cursor.find(" ")

            if space_index == -1:
                prefix = text_before_cursor[1:]
                command_items = [
                    _CommandItem(
                        name=_command_name(cmd),
                        label=_command_name(cmd),
                        description=_command_full_description(cmd),
                    )
                    for cmd in self._commands
                ]

                filtered = fuzzy_filter(command_items, prefix, lambda item: item.name)
                items = [
                    AutocompleteItem(value=item.name, label=item.label, description=item.description)
                    for item in filtered
                ]

                if not items:
                    return None

                return AutocompleteSuggestions(items=items, prefix=text_before_cursor)

            command_name = text_before_cursor[1:space_index]
            argument_text = text_before_cursor[space_index + 1 :]

            command = next((cmd for cmd in self._commands if _command_name(cmd) == command_name), None)
            if command is None or not isinstance(command, SlashCommand) or command.get_argument_completions is None:
                return None

            argument_suggestions = await command.get_argument_completions(argument_text)
            if not argument_suggestions:
                return None

            return AutocompleteSuggestions(items=argument_suggestions, prefix=argument_text)

        path_match = self._extract_path_prefix(text_before_cursor, force)
        if path_match is None:
            return None

        suggestions = self._get_file_suggestions(path_match)
        if not suggestions:
            return None

        return AutocompleteSuggestions(items=suggestions, prefix=path_match)

    def apply_completion(
        self,
        lines: list[str],
        cursor_line: int,
        cursor_col: int,
        item: AutocompleteItem,
        prefix: str,
    ) -> AppliedCompletion:
        current_line = lines[cursor_line] if 0 <= cursor_line < len(lines) else ""
        before_prefix = current_line[: cursor_col - len(prefix)]
        after_cursor = current_line[cursor_col:]
        is_quoted_prefix = prefix.startswith('"') or prefix.startswith('@"')
        has_leading_quote_after_cursor = after_cursor.startswith('"')
        has_trailing_quote_in_item = item.value.endswith('"')
        adjusted_after_cursor = (
            after_cursor[1:]
            if is_quoted_prefix and has_trailing_quote_in_item and has_leading_quote_after_cursor
            else after_cursor
        )

        # Check if we're completing a slash command (prefix starts with "/" but
        # NOT a file path). Slash commands are at the start of the line and
        # don't contain path separators after the first /.
        is_slash_command = prefix.startswith("/") and before_prefix.strip() == "" and "/" not in prefix[1:]
        if is_slash_command:
            new_line = f"{before_prefix}/{item.value} {adjusted_after_cursor}"
            new_lines = list(lines)
            new_lines[cursor_line] = new_line

            return AppliedCompletion(
                lines=new_lines,
                cursor_line=cursor_line,
                cursor_col=len(before_prefix) + len(item.value) + 2,  # +2 for "/" and space
            )

        # Check if we're completing a file attachment (prefix starts with "@").
        if prefix.startswith("@"):
            # Don't add space after directories so user can continue autocompleting.
            is_directory = item.label.endswith("/")
            suffix = "" if is_directory else " "
            new_line = f"{before_prefix + item.value}{suffix}{adjusted_after_cursor}"
            new_lines = list(lines)
            new_lines[cursor_line] = new_line

            has_trailing_quote = item.value.endswith('"')
            cursor_offset = len(item.value) - 1 if is_directory and has_trailing_quote else len(item.value)

            return AppliedCompletion(
                lines=new_lines,
                cursor_line=cursor_line,
                cursor_col=len(before_prefix) + cursor_offset + len(suffix),
            )

        # Check if we're in a slash command context (beforePrefix contains "/command ").
        text_before_cursor = current_line[:cursor_col]
        if "/" in text_before_cursor and " " in text_before_cursor:
            new_line = before_prefix + item.value + adjusted_after_cursor
            new_lines = list(lines)
            new_lines[cursor_line] = new_line

            is_directory = item.label.endswith("/")
            has_trailing_quote = item.value.endswith('"')
            cursor_offset = len(item.value) - 1 if is_directory and has_trailing_quote else len(item.value)

            return AppliedCompletion(
                lines=new_lines,
                cursor_line=cursor_line,
                cursor_col=len(before_prefix) + cursor_offset,
            )

        # For file paths, complete the path.
        new_line = before_prefix + item.value + adjusted_after_cursor
        new_lines = list(lines)
        new_lines[cursor_line] = new_line

        is_directory = item.label.endswith("/")
        has_trailing_quote = item.value.endswith('"')
        cursor_offset = len(item.value) - 1 if is_directory and has_trailing_quote else len(item.value)

        return AppliedCompletion(
            lines=new_lines,
            cursor_line=cursor_line,
            cursor_col=len(before_prefix) + cursor_offset,
        )

    def should_trigger_file_completion(self, lines: list[str], cursor_line: int, cursor_col: int) -> bool:
        """Check if we should trigger file completion (called on Tab key)."""
        current_line = lines[cursor_line] if 0 <= cursor_line < len(lines) else ""
        text_before_cursor = current_line[:cursor_col]

        # Don't trigger if we're typing a slash command at the start of the line.
        stripped = text_before_cursor.strip()
        return not (stripped.startswith("/") and " " not in stripped)

    # -- @ prefix extraction ------------------------------------------------

    def _extract_at_prefix(self, text: str) -> str | None:
        quoted_prefix = _extract_quoted_prefix(text)
        if quoted_prefix is not None and quoted_prefix.startswith('@"'):
            return quoted_prefix

        last_delimiter_index = _find_last_delimiter(text)
        token_start = 0 if last_delimiter_index == -1 else last_delimiter_index + 1

        if token_start < len(text) and text[token_start] == "@":
            return text[token_start:]

        return None

    def _extract_path_prefix(self, text: str, force_extract: bool = False) -> str | None:
        """Extract a path-like prefix from the text before cursor."""
        quoted_prefix = _extract_quoted_prefix(text)
        if quoted_prefix:
            return quoted_prefix

        last_delimiter_index = _find_last_delimiter(text)
        path_prefix = text if last_delimiter_index == -1 else text[last_delimiter_index + 1 :]

        # For forced extraction (Tab key), always return something.
        if force_extract:
            return path_prefix

        # For natural triggers, return if it looks like a path: ends with /,
        # starts with ~/, or starts with ".".
        if "/" in path_prefix or path_prefix.startswith(".") or path_prefix.startswith("~/"):
            return path_prefix

        # Return empty string only after a space (not for completely empty
        # text). Empty text should not trigger file suggestions - that's for
        # forced Tab completion.
        if path_prefix == "" and text.endswith(" "):
            return path_prefix

        return None

    def _expand_home_path(self, path: str) -> str:
        """Expand home directory (~/) to actual home path."""
        if path.startswith("~/"):
            expanded_path = os.path.join(str(Path.home()), path[2:])
            # Preserve trailing slash if original path had one.
            return f"{expanded_path}/" if path.endswith("/") and not expanded_path.endswith("/") else expanded_path
        if path == "~":
            return str(Path.home())
        return path

    def _resolve_scoped_fuzzy_query(self, raw_query: str) -> _ScopedFuzzyQuery | None:
        normalized_query = _to_display_path(raw_query)
        slash_index = normalized_query.rfind("/")
        if slash_index == -1:
            return None

        display_base = normalized_query[: slash_index + 1]
        query = normalized_query[slash_index + 1 :]

        if display_base.startswith("~/"):
            base_dir = self._expand_home_path(display_base)
        elif display_base.startswith("/"):
            base_dir = display_base
        else:
            base_dir = os.path.join(self._base_path, display_base)

        if not os.path.isdir(base_dir):
            return None

        return _ScopedFuzzyQuery(base_dir=base_dir, query=query, display_base=display_base)

    def _scoped_path_for_display(self, display_base: str, relative_path: str) -> str:
        normalized_relative_path = _to_display_path(relative_path)
        if display_base == "/":
            return f"/{normalized_relative_path}"
        return f"{_to_display_path(display_base)}{normalized_relative_path}"

    # -- Plain filesystem suggestions ---------------------------------------

    def _get_file_suggestions(self, prefix: str) -> list[AutocompleteItem]:
        """Get file/directory suggestions for a given path prefix."""
        try:
            parsed = _parse_path_prefix(prefix)
            raw_prefix = parsed.raw_prefix
            is_at_prefix = parsed.is_at_prefix
            is_quoted_prefix = parsed.is_quoted_prefix
            expanded_prefix = raw_prefix

            if expanded_prefix.startswith("~"):
                expanded_prefix = self._expand_home_path(expanded_prefix)

            is_root_prefix = raw_prefix in ("", "./", "../", "~", "~/", "/") or (is_at_prefix and raw_prefix == "")

            if is_root_prefix or raw_prefix.endswith("/"):
                if raw_prefix.startswith("~") or expanded_prefix.startswith("/"):
                    search_dir = expanded_prefix
                else:
                    search_dir = os.path.join(self._base_path, expanded_prefix)
                search_prefix = ""
            else:
                directory = os.path.dirname(expanded_prefix)
                file_part = os.path.basename(expanded_prefix)
                if raw_prefix.startswith("~") or expanded_prefix.startswith("/"):
                    search_dir = directory
                else:
                    search_dir = os.path.join(self._base_path, directory)
                search_prefix = file_part

            entries = list(os.scandir(search_dir))
            suggestions: list[AutocompleteItem] = []

            for entry in entries:
                if not entry.name.lower().startswith(search_prefix.lower()):
                    continue

                is_directory = entry.is_dir()
                if not is_directory and entry.is_symlink():
                    with contextlib.suppress(OSError):
                        is_directory = os.path.isdir(os.path.join(search_dir, entry.name))

                name = entry.name
                display_prefix = raw_prefix

                if display_prefix.endswith("/"):
                    relative_path = display_prefix + name
                elif "/" in display_prefix or "\\" in display_prefix:
                    if display_prefix.startswith("~/"):
                        home_relative_dir = display_prefix[2:]  # remove ~/
                        directory = os.path.dirname(home_relative_dir)
                        relative_path = f"~/{name}" if directory == "." else f"~/{os.path.join(directory, name)}"
                    elif display_prefix.startswith("/"):
                        directory = os.path.dirname(display_prefix)
                        relative_path = f"/{name}" if directory == "/" else f"{directory}/{name}"
                    else:
                        relative_path = os.path.join(os.path.dirname(display_prefix), name)
                        # os.path.join normalizes away ./ prefix; preserve it.
                        if display_prefix.startswith("./") and not relative_path.startswith("./"):
                            relative_path = f"./{relative_path}"
                else:
                    relative_path = f"~/{name}" if display_prefix.startswith("~") else name

                relative_path = _to_display_path(relative_path)
                path_value = f"{relative_path}/" if is_directory else relative_path
                value = _build_completion_value(
                    path_value,
                    is_directory=is_directory,
                    is_at_prefix=is_at_prefix,
                    is_quoted_prefix=is_quoted_prefix,
                )

                suggestions.append(AutocompleteItem(value=value, label=name + ("/" if is_directory else "")))

            # Sort directories first, then alphabetically.
            suggestions.sort(key=lambda item: (not item.value.endswith("/"), item.label))

            return suggestions
        except OSError:
            # Directory doesn't exist or not accessible.
            return []

    def _score_entry(self, file_path: str, query: str, is_directory: bool) -> int:
        """Score an entry against the query (higher = better match).

        `is_directory` adds a bonus to prioritize folders.
        """
        file_name = _node_basename(file_path)
        lower_file_name = file_name.lower()
        lower_query = query.lower()

        score = 0

        if lower_file_name == lower_query:
            score = 100
        elif lower_file_name.startswith(lower_query):
            score = 80
        elif lower_query in lower_file_name:
            score = 50
        elif lower_query in file_path.lower():
            score = 30

        if is_directory and score > 0:
            score += 10

        return score

    async def _get_fuzzy_file_suggestions(
        self, query: str, *, is_quoted_prefix: bool, signal: asyncio.Event
    ) -> list[AutocompleteItem]:
        """Fuzzy file search using `fd` (fast, respects `.gitignore`)."""
        if not self._fd_path or signal.is_set():
            return []

        try:
            scoped_query = self._resolve_scoped_fuzzy_query(query)
            fd_base_dir = scoped_query.base_dir if scoped_query else self._base_path
            fd_query = scoped_query.query if scoped_query else query
            entries = await _walk_directory_with_fd(fd_base_dir, self._fd_path, fd_query, 100, signal)
            if signal.is_set():
                return []

            scored_entries = [
                (entry, self._score_entry(entry.path, fd_query, entry.is_directory) if fd_query else 1)
                for entry in entries
            ]
            scored_entries = [(entry, score) for entry, score in scored_entries if score > 0]
            scored_entries.sort(key=lambda pair: pair[1], reverse=True)
            top_entries = scored_entries[:20]

            suggestions: list[AutocompleteItem] = []
            for entry, _score in top_entries:
                path_without_slash = entry.path[:-1] if entry.is_directory else entry.path
                display_path = (
                    self._scoped_path_for_display(scoped_query.display_base, path_without_slash)
                    if scoped_query
                    else path_without_slash
                )
                entry_name = os.path.basename(path_without_slash)
                completion_path = f"{display_path}/" if entry.is_directory else display_path
                value = _build_completion_value(
                    completion_path,
                    is_directory=entry.is_directory,
                    is_at_prefix=True,
                    is_quoted_prefix=is_quoted_prefix,
                )

                suggestions.append(
                    AutocompleteItem(
                        value=value,
                        label=entry_name + ("/" if entry.is_directory else ""),
                        description=display_path,
                    )
                )

            return suggestions
        except OSError:
            return []


@dataclass
class _CommandItem:
    name: str
    label: str
    description: str | None


def _command_name(command: SlashCommand | AutocompleteItem) -> str:
    return command.name if isinstance(command, SlashCommand) else command.value


def _command_full_description(command: SlashCommand | AutocompleteItem) -> str | None:
    hint = command.argument_hint if isinstance(command, SlashCommand) else None
    desc = command.description or ""
    full_desc = (f"{hint} — {desc}" if desc else hint) if hint else desc
    return full_desc or None
