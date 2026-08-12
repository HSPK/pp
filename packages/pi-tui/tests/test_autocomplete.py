"""Tests ported from packages/tui/test/autocomplete.test.ts."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

import pytest
from pi_tui.autocomplete import (
    AppliedCompletion,
    AutocompleteItem,
    AutocompleteSuggestions,
    CombinedAutocompleteProvider,
    SlashCommand,
)


def _resolve_fd_path() -> str | None:
    """Locate an `fd`-compatible binary.

    Debian/Ubuntu package the `fd` binary as `fdfind` (there's a naming
    clash with an existing package). It behaves identically to upstream
    `fd`, so we fall back to it here to maximize test coverage of the
    `fd`-backed code path when plain `fd` isn't on `PATH`.
    """
    return shutil.which("fd") or shutil.which("fdfind")


_FD_PATH = _resolve_fd_path()
_IS_FD_INSTALLED = _FD_PATH is not None
requires_fd = pytest.mark.skipif(not _IS_FD_INSTALLED, reason="fd (or fdfind) is not installed")


def _require_fd_path() -> str:
    assert _FD_PATH is not None
    return _FD_PATH


def _setup_folder(base_dir: str, dirs: list[str] | None = None, files: dict[str, str] | None = None) -> None:
    for directory in dirs or []:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)
    for file_path, contents in (files or {}).items():
        full_path = os.path.join(base_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(contents)


def _get_suggestions(
    provider: CombinedAutocompleteProvider,
    lines: list[str],
    cursor_line: int,
    cursor_col: int,
    force: bool = False,
) -> AutocompleteSuggestions | None:
    return asyncio.run(provider.get_suggestions(lines, cursor_line, cursor_col, signal=asyncio.Event(), force=force))


class TestExtractPathPrefix:
    def test_extracts_slash_from_hey_slash_when_forced(self):
        provider = CombinedAutocompleteProvider([], "/tmp")
        lines = ["hey /"]
        result = _get_suggestions(provider, lines, 0, 5, True)

        assert result is not None, "Should return suggestions for root directory"
        assert result.prefix == "/"

    def test_extracts_slash_a_from_slash_a_when_forced(self):
        provider = CombinedAutocompleteProvider([], "/tmp")
        lines = ["/A"]
        result = _get_suggestions(provider, lines, 0, 2, True)

        # This might return None if /A doesn't match anything, which is fine.
        if result is not None:
            assert result.prefix == "/A"

    def test_does_not_trigger_for_slash_commands(self):
        provider = CombinedAutocompleteProvider([], "/tmp")
        lines = ["/model"]
        result = _get_suggestions(provider, lines, 0, 6, True)

        assert result is None, "Should not trigger for slash commands"

    def test_triggers_for_absolute_paths_after_slash_command_argument(self):
        provider = CombinedAutocompleteProvider([], "/tmp")
        lines = ["/command /"]
        result = _get_suggestions(provider, lines, 0, 10, True)

        assert result is not None, "Should trigger for absolute paths in command arguments"
        assert result.prefix == "/"


@requires_fd
class TestFdAtFileSuggestions:
    root_dir: str
    base_dir: str
    outside_dir: str

    def setup_method(self):
        self.root_dir = tempfile.mkdtemp(prefix="pi-autocomplete-root-")
        self.base_dir = os.path.join(self.root_dir, "cwd")
        self.outside_dir = os.path.join(self.root_dir, "outside")
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.outside_dir, exist_ok=True)

    def teardown_method(self):
        shutil.rmtree(self.root_dir, ignore_errors=True)

    def test_returns_all_files_and_folders_for_empty_at_query(self):
        _setup_folder(self.base_dir, dirs=["src"], files={"README.md": "readme"})

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = "@"
        result = _get_suggestions(provider, [line], 0, len(line))

        values = sorted(item.value for item in result.items)
        assert values == sorted(["@README.md", "@src/"])

    def test_matches_file_with_extension_in_query(self):
        _setup_folder(self.base_dir, files={"file.txt": "content"})

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = "@file.txt"
        result = _get_suggestions(provider, [line], 0, len(line))

        values = [item.value for item in result.items]
        assert "@file.txt" in values

    def test_filters_are_case_insensitive(self):
        _setup_folder(self.base_dir, dirs=["src"], files={"README.md": "readme"})

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = "@re"
        result = _get_suggestions(provider, [line], 0, len(line))

        values = sorted(item.value for item in result.items)
        assert values == ["@README.md"]

    def test_ranks_directories_before_files(self):
        _setup_folder(self.base_dir, dirs=["src"], files={"src.txt": "text"})

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = "@src"
        result = _get_suggestions(provider, [line], 0, len(line))

        first_value = result.items[0].value
        has_src_file = any(item.value == "@src.txt" for item in result.items)
        assert first_value == "@src/"
        assert has_src_file

    def test_returns_nested_file_paths(self):
        _setup_folder(self.base_dir, files={"src/index.ts": "export {};\n"})

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = "@index"
        result = _get_suggestions(provider, [line], 0, len(line))

        values = [item.value for item in result.items]
        assert "@src/index.ts" in values

    def test_matches_deeply_nested_paths(self):
        _setup_folder(
            self.base_dir,
            files={
                "packages/tui/src/autocomplete.ts": "export {};",
                "packages/ai/src/autocomplete.ts": "export {};",
            },
        )

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = "@tui/src/auto"
        result = _get_suggestions(provider, [line], 0, len(line))

        values = [item.value for item in result.items]
        assert "@packages/tui/src/autocomplete.ts" in values
        assert "@packages/ai/src/autocomplete.ts" not in values

    def test_matches_directory_in_middle_of_path_with_full_path(self):
        _setup_folder(
            self.base_dir,
            files={
                "src/components/Button.tsx": "export {};",
                "src/utils/helpers.ts": "export {};",
            },
        )

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = "@components/"
        result = _get_suggestions(provider, [line], 0, len(line))

        values = [item.value for item in result.items]
        assert "@src/components/Button.tsx" in values
        assert "@src/utils/helpers.ts" not in values

    def test_scopes_fuzzy_search_to_relative_directories_and_searches_recursively(self):
        _setup_folder(
            self.outside_dir,
            files={
                "nested/alpha.ts": "export {};",
                "nested/deeper/also-alpha.ts": "export {};",
                "nested/deeper/zzz.ts": "export {};",
            },
        )

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = "@../outside/a"
        result = _get_suggestions(provider, [line], 0, len(line))

        values = [item.value for item in result.items]
        assert "@../outside/nested/alpha.ts" in values
        assert "@../outside/nested/deeper/also-alpha.ts" in values
        assert "@../outside/nested/deeper/zzz.ts" not in values

    def test_quotes_paths_with_spaces_for_at_suggestions(self):
        _setup_folder(self.base_dir, dirs=["my folder"], files={"my folder/test.txt": "content"})

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = "@my"
        result = _get_suggestions(provider, [line], 0, len(line))

        values = [item.value for item in result.items]
        assert '@"my folder/"' in values

    def test_includes_hidden_paths_but_excludes_git(self):
        _setup_folder(
            self.base_dir,
            dirs=[".pi", ".github", ".git"],
            files={
                ".pi/config.json": "{}",
                ".github/workflows/ci.yml": "name: ci",
                ".git/config": "[core]",
            },
        )

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = "@"
        result = _get_suggestions(provider, [line], 0, len(line))

        values = [item.value for item in result.items] if result else []
        assert "@.pi/" in values
        assert "@.github/" in values
        assert not any(value == "@.git" or value.startswith("@.git/") for value in values)

    def test_follows_symlinked_directories_for_fuzzy_at_search(self):
        _setup_folder(self.base_dir, files={"dir/some_file.txt": "real"})
        _setup_folder(self.outside_dir, files={"some_file.txt": "symlinked"})
        os.symlink("../outside", os.path.join(self.base_dir, "symlinked_dir"))

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = "@some"
        result = _get_suggestions(provider, [line], 0, len(line))

        values = [item.value for item in result.items] if result else []
        assert "@dir/some_file.txt" in values
        assert "@symlinked_dir/some_file.txt" in values

    def test_returns_symlinked_directories_when_matching_their_name(self):
        _setup_folder(self.outside_dir, files={"nested/file.txt": "symlinked"})
        os.symlink("../outside", os.path.join(self.base_dir, "symlinked_dir"))

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = "@symlinked"
        result = _get_suggestions(provider, [line], 0, len(line))

        values = [item.value for item in result.items] if result else []
        assert "@symlinked_dir/" in values

    def test_returns_symlinked_files_without_requiring_type_l(self):
        _setup_folder(self.base_dir, files={"original.txt": "content"})
        link_path = os.path.join(self.base_dir, "link.txt")
        os.symlink("original.txt", link_path)

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = "@link"
        result = _get_suggestions(provider, [line], 0, len(line))

        values = [item.value for item in result.items] if result else []
        assert "@link.txt" in values

    def test_returns_the_same_at_suggestions_when_the_cwd_path_contains_the_query(self):
        normal_base_dir = os.path.join(self.root_dir, "cwd-normal")
        query_in_path_base_dir = os.path.join(self.root_dir, "cwd-plan-repro")
        os.makedirs(normal_base_dir, exist_ok=True)
        os.makedirs(query_in_path_base_dir, exist_ok=True)

        dirs = ["packages/coding-agent/examples/extensions/plan-mode"]
        files = {
            "packages/coding-agent/examples/extensions/plan-mode/README.md": "readme",
            "packages/tui/docs/plan.md": "plan",
        }
        _setup_folder(normal_base_dir, dirs=dirs, files=files)
        _setup_folder(query_in_path_base_dir, dirs=dirs, files=files)

        query = "@plan"
        normal_provider = CombinedAutocompleteProvider([], normal_base_dir, _require_fd_path())
        query_in_path_provider = CombinedAutocompleteProvider([], query_in_path_base_dir, _require_fd_path())

        normal_result = _get_suggestions(normal_provider, [query], 0, len(query))
        query_in_path_result = _get_suggestions(query_in_path_provider, [query], 0, len(query))

        def normalize(result: AutocompleteSuggestions | None) -> list[str]:
            items = result.items if result else []
            return sorted(f"{item.label} :: {item.description or ''}" for item in items)

        assert normalize(query_in_path_result) == normalize(normal_result)
        assert "plan-mode/ :: packages/coding-agent/examples/extensions/plan-mode" in normalize(normal_result)
        assert "plan.md :: packages/tui/docs/plan.md" in normalize(normal_result)

    def test_continues_autocomplete_inside_quoted_at_paths(self):
        _setup_folder(
            self.base_dir,
            files={"my folder/test.txt": "content", "my folder/other.txt": "content"},
        )

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = '@"my folder/"'
        result = _get_suggestions(provider, [line], 0, len(line) - 1)

        assert result is not None, "Should return suggestions for quoted folder path"
        values = [item.value for item in result.items]
        assert '@"my folder/test.txt"' in values
        assert '@"my folder/other.txt"' in values

    def test_applies_quoted_at_completion_without_duplicating_closing_quote(self):
        _setup_folder(self.base_dir, files={"my folder/test.txt": "content"})

        provider = CombinedAutocompleteProvider([], self.base_dir, _require_fd_path())
        line = '@"my folder/te"'
        cursor_col = len(line) - 1
        result = _get_suggestions(provider, [line], 0, cursor_col)

        assert result is not None, "Should return suggestions for quoted @ path"
        item = next((entry for entry in result.items if entry.value == '@"my folder/test.txt"'), None)
        assert item is not None, "Should find test.txt suggestion"

        applied: AppliedCompletion = provider.apply_completion([line], 0, cursor_col, item, result.prefix)
        assert applied.lines[0] == '@"my folder/test.txt" '


class TestDotSlashPathCompletion:
    base_dir: str

    def setup_method(self):
        self.base_dir = tempfile.mkdtemp(prefix="pi-autocomplete-")

    def teardown_method(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_preserves_dot_slash_prefix_when_completing_paths(self):
        _setup_folder(self.base_dir, files={"update.sh": "#!/bin/bash", "utils.ts": "export {};"})

        provider = CombinedAutocompleteProvider([], self.base_dir)
        line = "./up"
        result = _get_suggestions(provider, [line], 0, len(line), True)

        assert result is not None, "Should return suggestions for ./ path"
        values = [item.value for item in result.items]
        assert "./update.sh" in values, f"Expected ./update.sh in {values}"

    def test_preserves_dot_slash_prefix_for_directory_completions(self):
        _setup_folder(self.base_dir, dirs=["src"], files={"src/index.ts": "export {};"})

        provider = CombinedAutocompleteProvider([], self.base_dir)
        line = "./sr"
        result = _get_suggestions(provider, [line], 0, len(line), True)

        assert result is not None, "Should return suggestions for ./ directory path"
        values = [item.value for item in result.items]
        assert "./src/" in values, f"Expected ./src/ in {values}"


class TestQuotedPathCompletion:
    base_dir: str

    def setup_method(self):
        self.base_dir = tempfile.mkdtemp(prefix="pi-autocomplete-")

    def teardown_method(self):
        shutil.rmtree(self.base_dir, ignore_errors=True)

    def test_quotes_paths_with_spaces_for_direct_completion(self):
        _setup_folder(self.base_dir, dirs=["my folder"], files={"my folder/test.txt": "content"})

        provider = CombinedAutocompleteProvider([], self.base_dir)
        line = "my"
        result = _get_suggestions(provider, [line], 0, len(line), True)

        assert result is not None, "Should return suggestions for path completion"
        values = [item.value for item in result.items]
        assert '"my folder/"' in values

    def test_continues_completion_inside_quoted_paths(self):
        _setup_folder(
            self.base_dir,
            files={"my folder/test.txt": "content", "my folder/other.txt": "content"},
        )

        provider = CombinedAutocompleteProvider([], self.base_dir)
        line = '"my folder/"'
        result = _get_suggestions(provider, [line], 0, len(line) - 1, True)

        assert result is not None, "Should return suggestions for quoted folder path"
        values = [item.value for item in result.items]
        assert '"my folder/test.txt"' in values
        assert '"my folder/other.txt"' in values

    def test_applies_quoted_completion_without_duplicating_closing_quote(self):
        _setup_folder(self.base_dir, files={"my folder/test.txt": "content"})

        provider = CombinedAutocompleteProvider([], self.base_dir)
        line = '"my folder/te"'
        cursor_col = len(line) - 1
        result = _get_suggestions(provider, [line], 0, cursor_col, True)

        assert result is not None, "Should return suggestions for quoted path"
        item = next((entry for entry in result.items if entry.value == '"my folder/test.txt"'), None)
        assert item is not None, "Should find test.txt suggestion"

        applied = provider.apply_completion([line], 0, cursor_col, item, result.prefix)
        assert applied.lines[0] == '"my folder/test.txt"'


class TestSlashCommandCompletion:
    """Original coverage: slash-command matching/argument completion isn't in autocomplete.test.ts."""

    def test_lists_all_commands_for_bare_slash(self):
        commands = [
            SlashCommand(name="model", description="Switch model"),
            SlashCommand(name="mcp", description="Manage MCP servers"),
        ]
        provider = CombinedAutocompleteProvider(commands, "/tmp")
        line = "/"
        result = _get_suggestions(provider, [line], 0, len(line))

        assert result is not None
        assert result.prefix == "/"
        assert {item.value for item in result.items} == {"model", "mcp"}

    def test_fuzzy_filters_commands_by_prefix(self):
        commands = [
            SlashCommand(name="model", description="Switch model"),
            SlashCommand(name="mcp", description="Manage MCP servers"),
            SlashCommand(name="help", description="Show help"),
        ]
        provider = CombinedAutocompleteProvider(commands, "/tmp")
        line = "/mo"
        result = _get_suggestions(provider, [line], 0, len(line))

        assert result is not None
        assert [item.value for item in result.items] == ["model"]

    def test_combines_argument_hint_and_description(self):
        commands = [SlashCommand(name="model", description="Switch model", argument_hint="<name>")]
        provider = CombinedAutocompleteProvider(commands, "/tmp")
        line = "/model"
        result = _get_suggestions(provider, [line], 0, len(line))

        assert result is not None
        assert result.items[0].description == "<name> — Switch model"

    def test_returns_none_for_unknown_command_with_no_matches(self):
        provider = CombinedAutocompleteProvider([SlashCommand(name="model")], "/tmp")
        line = "/zzz"
        result = _get_suggestions(provider, [line], 0, len(line))

        assert result is None

    def test_includes_plain_autocomplete_items_as_commands(self):
        commands = [AutocompleteItem(value="model", label="model", description="Switch model")]
        provider = CombinedAutocompleteProvider(commands, "/tmp")
        line = "/mod"
        result = _get_suggestions(provider, [line], 0, len(line))

        assert result is not None
        assert result.items[0].value == "model"

    def test_delegates_to_argument_completions_when_command_matches(self):
        async def get_argument_completions(prefix: str) -> list[AutocompleteItem] | None:
            return [AutocompleteItem(value=f"{prefix}-gpt", label=f"{prefix}-gpt")]

        commands = [SlashCommand(name="model", get_argument_completions=get_argument_completions)]
        provider = CombinedAutocompleteProvider(commands, "/tmp")
        line = "/model gp"
        result = _get_suggestions(provider, [line], 0, len(line))

        assert result is not None
        assert result.prefix == "gp"
        assert result.items[0].value == "gp-gpt"

    def test_returns_none_when_command_has_no_argument_completions(self):
        commands = [SlashCommand(name="help")]
        provider = CombinedAutocompleteProvider(commands, "/tmp")
        line = "/help arg"
        result = _get_suggestions(provider, [line], 0, len(line))

        assert result is None

    def test_returns_none_when_argument_completions_resolve_empty(self):
        async def get_argument_completions(prefix: str) -> list[AutocompleteItem] | None:
            return []

        commands = [SlashCommand(name="model", get_argument_completions=get_argument_completions)]
        provider = CombinedAutocompleteProvider(commands, "/tmp")
        line = "/model gp"
        result = _get_suggestions(provider, [line], 0, len(line))

        assert result is None

    def test_applies_slash_command_completion(self):
        provider = CombinedAutocompleteProvider([], "/tmp")
        item = AutocompleteItem(value="model", label="model")
        applied = provider.apply_completion(["/mo"], 0, 3, item, "/mo")

        assert applied.lines[0] == "/model "
        assert applied.cursor_col == len("/model ")


class TestShouldTriggerFileCompletion:
    def test_false_for_bare_slash_command(self):
        provider = CombinedAutocompleteProvider([], "/tmp")
        assert provider.should_trigger_file_completion(["/model"], 0, 6) is False

    def test_true_once_command_has_an_argument(self):
        provider = CombinedAutocompleteProvider([], "/tmp")
        assert provider.should_trigger_file_completion(["/command arg"], 0, 12) is True

    def test_true_for_plain_text(self):
        provider = CombinedAutocompleteProvider([], "/tmp")
        assert provider.should_trigger_file_completion(["some text"], 0, 9) is True


class TestApplyCompletionFilePaths:
    def test_appends_value_and_keeps_trailing_text(self):
        provider = CombinedAutocompleteProvider([], "/tmp")
        item = AutocompleteItem(value="src/index.ts", label="index.ts")
        applied: AppliedCompletion = provider.apply_completion(["src/ind rest"], 0, 7, item, "src/ind")

        assert applied.lines[0] == "src/index.ts rest"
        assert applied.cursor_col == len("src/index.ts")

    def test_at_completion_skips_space_after_directory(self):
        provider = CombinedAutocompleteProvider([], "/tmp")
        item = AutocompleteItem(value="@src/", label="src/")
        applied = provider.apply_completion(["@sr"], 0, 3, item, "@sr")

        assert applied.lines[0] == "@src/"
        assert applied.cursor_col == len("@src/")
