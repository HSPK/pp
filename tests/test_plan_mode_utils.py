"""Python port of `packages/coding-agent/test/plan-mode-utils.test.ts`.

Exercises `examples/extensions/plan_mode/utils.py`, the Python port of
`examples/extensions/plan-mode/utils.ts`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "extensions"))

from plan_mode.utils import (
    TodoItem,
    clean_step_text,
    extract_done_steps,
    extract_todo_items,
    is_safe_command,
    mark_completed_steps,
)


class TestIsSafeCommandSafeCommands:
    def test_allows_basic_read_commands(self):
        assert is_safe_command("ls -la") is True
        assert is_safe_command("cat file.txt") is True
        assert is_safe_command("head -n 10 file.txt") is True
        assert is_safe_command("tail -f log.txt") is True
        assert is_safe_command("grep pattern file") is True
        assert is_safe_command("find . -name '*.ts'") is True

    def test_allows_git_read_commands(self):
        assert is_safe_command("git status") is True
        assert is_safe_command("git log --oneline") is True
        assert is_safe_command("git diff") is True
        assert is_safe_command("git branch") is True

    def test_allows_npm_yarn_read_commands(self):
        assert is_safe_command("npm list") is True
        assert is_safe_command("npm outdated") is True
        assert is_safe_command("yarn info react") is True

    def test_allows_other_safe_commands(self):
        assert is_safe_command("pwd") is True
        assert is_safe_command("echo hello") is True
        assert is_safe_command("wc -l file.txt") is True
        assert is_safe_command("du -sh .") is True
        assert is_safe_command("df -h") is True


class TestIsSafeCommandDestructiveCommands:
    def test_blocks_file_modification_commands(self):
        assert is_safe_command("rm file.txt") is False
        assert is_safe_command("rm -rf dir") is False
        assert is_safe_command("mv old new") is False
        assert is_safe_command("cp src dst") is False
        assert is_safe_command("mkdir newdir") is False
        assert is_safe_command("touch newfile") is False

    def test_blocks_git_write_commands(self):
        assert is_safe_command("git add .") is False
        assert is_safe_command("git commit -m 'msg'") is False
        assert is_safe_command("git push") is False
        assert is_safe_command("git checkout main") is False
        assert is_safe_command("git reset --hard") is False

    def test_blocks_package_manager_installs(self):
        assert is_safe_command("npm install lodash") is False
        assert is_safe_command("yarn add react") is False
        assert is_safe_command("pip install requests") is False
        assert is_safe_command("brew install node") is False

    def test_blocks_redirects(self):
        assert is_safe_command("echo hello > file.txt") is False
        assert is_safe_command("cat foo >> bar") is False
        assert is_safe_command(">file.txt") is False

    def test_blocks_dangerous_commands(self):
        assert is_safe_command("sudo rm -rf /") is False
        assert is_safe_command("kill -9 1234") is False
        assert is_safe_command("reboot") is False

    def test_blocks_editors(self):
        assert is_safe_command("vim file.txt") is False
        assert is_safe_command("nano file.txt") is False
        assert is_safe_command("code .") is False


class TestIsSafeCommandEdgeCases:
    def test_requires_command_to_be_in_safe_list(self):
        assert is_safe_command("unknown-command") is False
        assert is_safe_command("my-script.sh") is False

    def test_handles_commands_with_leading_whitespace(self):
        assert is_safe_command("  ls -la") is True
        assert is_safe_command("  rm file") is False


class TestCleanStepText:
    def test_removes_markdown_bold_italic(self):
        assert clean_step_text("**bold text**") == "Bold text"
        assert clean_step_text("*italic text*") == "Italic text"

    def test_removes_markdown_code(self):
        # "run" is stripped as an action word.
        assert clean_step_text("run `npm install`") == "Npm install"
        assert clean_step_text("check the `config.json` file") == "Config.json file"

    def test_removes_leading_action_words(self):
        assert clean_step_text("Create the new file") == "New file"
        assert clean_step_text("Run the tests") == "Tests"
        assert clean_step_text("Check the status") == "Status"

    def test_capitalizes_first_letter(self):
        assert clean_step_text("update config") == "Config"

    def test_truncates_long_text(self):
        long_text = "This is a very long step description that exceeds the maximum allowed length for display"
        result = clean_step_text(long_text)
        assert len(result) == 50
        assert result.endswith("...")

    def test_normalizes_whitespace(self):
        assert clean_step_text("multiple   spaces   here") == "Multiple spaces here"


class TestExtractTodoItems:
    def test_extracts_numbered_items_after_plan_header(self):
        message = """Here's what we'll do:

Plan:
1. First step here
2. Second step here
3. Third step here"""

        items = extract_todo_items(message)
        assert len(items) == 3
        assert items[0].step == 1
        assert items[0].text == "First step here"
        assert items[0].completed is False

    def test_handles_bold_plan_header(self):
        message = "**Plan:**\n1. Do something"

        items = extract_todo_items(message)
        assert len(items) == 1

    def test_handles_parenthesis_style_numbering(self):
        message = "Plan:\n1) First item\n2) Second item"

        items = extract_todo_items(message)
        assert len(items) == 2

    def test_returns_empty_list_without_plan_header(self):
        message = "Here are some steps:\n1. First step\n2. Second step"

        items = extract_todo_items(message)
        assert len(items) == 0

    def test_filters_out_short_items(self):
        message = "Plan:\n1. OK\n2. This is a proper step"

        items = extract_todo_items(message)
        assert len(items) == 1
        assert "proper" in items[0].text

    def test_filters_out_code_like_items(self):
        message = "Plan:\n1. `npm install`\n2. Run the build process"

        items = extract_todo_items(message)
        assert len(items) == 1


class TestExtractDoneSteps:
    def test_extracts_single_done_marker(self):
        assert extract_done_steps("I've completed the first step [DONE:1]") == [1]

    def test_extracts_multiple_done_markers(self):
        assert extract_done_steps("Did steps [DONE:1] and [DONE:2] and [DONE:3]") == [1, 2, 3]

    def test_handles_case_insensitivity(self):
        assert extract_done_steps("[done:1] [DONE:2] [Done:3]") == [1, 2, 3]

    def test_returns_empty_list_with_no_markers(self):
        assert extract_done_steps("No markers here") == []

    def test_ignores_malformed_markers(self):
        assert extract_done_steps("[DONE:abc] [DONE:] [DONE:1]") == [1]


class TestMarkCompletedSteps:
    def test_marks_matching_items_as_completed(self):
        items = [
            TodoItem(step=1, text="First", completed=False),
            TodoItem(step=2, text="Second", completed=False),
            TodoItem(step=3, text="Third", completed=False),
        ]

        count = mark_completed_steps("[DONE:1] [DONE:3]", items)

        assert count == 2
        assert items[0].completed is True
        assert items[1].completed is False
        assert items[2].completed is True

    def test_returns_count_of_completed_items(self):
        items = [TodoItem(step=1, text="First", completed=False)]

        assert mark_completed_steps("[DONE:1]", items) == 1
        assert mark_completed_steps("no markers", items) == 0

    def test_ignores_markers_for_non_existent_steps(self):
        items = [TodoItem(step=1, text="First", completed=False)]

        count = mark_completed_steps("[DONE:99]", items)

        assert count == 1  # Still counts the marker found.
        assert items[0].completed is False  # But doesn't mark anything.

    def test_does_not_double_complete_already_completed_items(self):
        items = [TodoItem(step=1, text="First", completed=True)]

        mark_completed_steps("[DONE:1]", items)
        assert items[0].completed is True
