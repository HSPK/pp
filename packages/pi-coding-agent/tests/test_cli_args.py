"""Tests for CLI argument parsing.

The parser is a hand-rolled scanner, so the cases that matter are the ambiguous
ones: which flags consume the next argument, when `--print` swallows a prompt,
and how unrecognised flags become extension flags.
"""

from __future__ import annotations

import pytest
from pi_coding_agent.cli.args import (
    VALID_THINKING_LEVELS,
    is_valid_thinking_level,
    parse_args,
)


def test_empty_command_line():
    args = parse_args([])
    assert args.messages == []
    assert args.file_args == []
    assert args.unknown_flags == {}
    assert args.diagnostics == []
    assert args.help is None


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_flag(flag):
    assert parse_args([flag]).help is True


@pytest.mark.parametrize("flag", ["--version", "-v"])
def test_version_flag(flag):
    assert parse_args([flag]).version is True


def test_positional_messages_are_collected_in_order():
    args = parse_args(["first message", "second message"])
    assert args.messages == ["first message", "second message"]


def test_file_args_strip_the_at_prefix():
    args = parse_args(["@prompt.md", "@image.png", "what is this?"])
    assert args.file_args == ["prompt.md", "image.png"]
    assert args.messages == ["what is this?"]


# --------------------------------------------------------------------------
# value-taking flags
# --------------------------------------------------------------------------


def test_simple_value_flags():
    args = parse_args(
        [
            "--provider",
            "openai",
            "--model",
            "gpt-4o",
            "--api-key",
            "sk-1",
            "--system-prompt",
            "be brief",
            "--session",
            "abc",
            "--session-id",
            "id-1",
            "--fork",
            "def",
            "--session-dir",
            "/tmp/s",
            "--export",
            "out.html",
        ]
    )
    assert args.provider == "openai"
    assert args.model == "gpt-4o"
    assert args.api_key == "sk-1"
    assert args.system_prompt == "be brief"
    assert args.session == "abc"
    assert args.session_id == "id-1"
    assert args.fork == "def"
    assert args.session_dir == "/tmp/s"
    assert args.export == "out.html"


def test_repeatable_flags_accumulate():
    args = parse_args(
        [
            "--append-system-prompt",
            "one",
            "--append-system-prompt",
            "two",
            "--extension",
            "a.py",
            "-e",
            "b.py",
            "--skill",
            "s1",
            "--prompt-template",
            "t1",
            "--theme",
            "th1",
        ]
    )
    assert args.append_system_prompt == ["one", "two"]
    assert args.extensions == ["a.py", "b.py"]
    assert args.skills == ["s1"]
    assert args.prompt_templates == ["t1"]
    assert args.themes == ["th1"]


def test_comma_separated_lists():
    args = parse_args(["--models", "a , b,c", "--tools", "read, bash", "--exclude-tools", "write"])
    assert args.models == ["a", "b", "c"]
    assert args.tools == ["read", "bash"]
    assert args.exclude_tools == ["write"]


def test_tool_lists_drop_empty_entries():
    args = parse_args(["--tools", "read,,bash,", "--exclude-tools", ",,"])
    assert args.tools == ["read", "bash"]
    assert args.exclude_tools == []


def test_models_list_keeps_empty_entries():
    # --models does not filter, matching the TypeScript.
    assert parse_args(["--models", "a,,b"]).models == ["a", "", "b"]


def test_name_requires_a_value():
    args = parse_args(["--name"])
    assert args.name is None
    assert args.diagnostics == [args.diagnostics[0]]
    assert args.diagnostics[0].type == "error"
    assert "--name requires a value" in args.diagnostics[0].message


def test_name_accepts_a_value():
    assert parse_args(["-n", "my session"]).name == "my session"


# --------------------------------------------------------------------------
# boolean flags
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "attribute"),
    [
        ("--no-session", "no_session"),
        ("--no-tools", "no_tools"),
        ("-nt", "no_tools"),
        ("--no-builtin-tools", "no_builtin_tools"),
        ("-nbt", "no_builtin_tools"),
        ("--no-extensions", "no_extensions"),
        ("-ne", "no_extensions"),
        ("--no-skills", "no_skills"),
        ("-ns", "no_skills"),
        ("--no-prompt-templates", "no_prompt_templates"),
        ("-np", "no_prompt_templates"),
        ("--no-themes", "no_themes"),
        ("--no-context-files", "no_context_files"),
        ("-nc", "no_context_files"),
        ("--verbose", "verbose"),
        ("--offline", "offline"),
        ("--continue", "continue_"),
        ("-c", "continue_"),
        ("--resume", "resume"),
        ("-r", "resume"),
    ],
)
def test_boolean_flags(flag, attribute):
    assert getattr(parse_args([flag]), attribute) is True


def test_project_trust_override():
    assert parse_args(["--approve"]).project_trust_override is True
    assert parse_args(["-a"]).project_trust_override is True
    assert parse_args(["--no-approve"]).project_trust_override is False
    assert parse_args(["-na"]).project_trust_override is False
    assert parse_args([]).project_trust_override is None


# --------------------------------------------------------------------------
# --print, which optionally swallows the next argument
# --------------------------------------------------------------------------


def test_print_takes_a_following_prompt():
    args = parse_args(["-p", "do the thing"])
    assert args.print is True
    assert args.messages == ["do the thing"]


def test_print_does_not_swallow_a_following_flag():
    args = parse_args(["-p", "--verbose"])
    assert args.print is True
    assert args.messages == []
    assert args.verbose is True


def test_print_does_not_swallow_a_file_arg():
    args = parse_args(["-p", "@file.md"])
    assert args.messages == []
    assert args.file_args == ["file.md"]


def test_print_swallows_a_triple_dash_value_as_text():
    args = parse_args(["-p", "---weird"])
    assert args.messages == ["---weird"]


def test_print_at_end_of_line():
    args = parse_args(["--print"])
    assert args.print is True
    assert args.messages == []


# --------------------------------------------------------------------------
# validated enumerations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("level", VALID_THINKING_LEVELS)
def test_valid_thinking_levels(level):
    assert is_valid_thinking_level(level) is True
    assert parse_args(["--thinking", level]).thinking == level


def test_invalid_thinking_level_warns_and_is_ignored():
    args = parse_args(["--thinking", "ultra"])
    assert args.thinking is None
    assert args.diagnostics[0].type == "warning"
    assert "ultra" in args.diagnostics[0].message


@pytest.mark.parametrize("mode", ["text", "json", "rpc"])
def test_valid_modes(mode):
    assert parse_args(["--mode", mode]).mode == mode


def test_invalid_mode_is_silently_ignored():
    # The TypeScript records no diagnostic for an invalid --mode.
    args = parse_args(["--mode", "nope"])
    assert args.mode is None
    assert args.diagnostics == []


@pytest.mark.parametrize("mode", ["regular", "fullscreen"])
def test_valid_tui_modes(mode):
    assert parse_args(["--tui-mode", mode]).tui_mode == mode


def test_tui_mode_without_a_value_errors():
    args = parse_args(["--tui-mode"])
    assert args.tui_mode is None
    assert "requires regular or fullscreen" in args.diagnostics[0].message


def test_tui_mode_followed_by_a_flag_errors():
    args = parse_args(["--tui-mode", "--verbose"])
    assert args.tui_mode is None
    assert args.verbose is True
    assert "requires regular or fullscreen" in args.diagnostics[0].message


def test_invalid_tui_mode_errors_and_consumes_the_value():
    args = parse_args(["--tui-mode", "weird", "message"])
    assert args.tui_mode is None
    assert args.messages == ["message"]
    assert 'Invalid TUI mode "weird"' in args.diagnostics[0].message


# --------------------------------------------------------------------------
# --list-models with an optional search term
# --------------------------------------------------------------------------


def test_list_models_without_a_search_term():
    assert parse_args(["--list-models"]).list_models is True


def test_list_models_with_a_search_term():
    assert parse_args(["--list-models", "sonnet"]).list_models == "sonnet"


def test_list_models_does_not_swallow_a_flag_or_file_arg():
    assert parse_args(["--list-models", "--verbose"]).list_models is True
    assert parse_args(["--list-models", "@f.md"]).list_models is True


# --------------------------------------------------------------------------
# unknown flags become extension flags
# --------------------------------------------------------------------------


def test_unknown_long_flag_without_a_value_is_boolean():
    args = parse_args(["--plan"])
    assert args.unknown_flags == {"plan": True}
    assert args.diagnostics == []


def test_unknown_long_flag_takes_a_following_value():
    assert parse_args(["--plan", "aggressive"]).unknown_flags == {"plan": "aggressive"}


def test_unknown_long_flag_supports_equals_syntax():
    assert parse_args(["--plan=aggressive"]).unknown_flags == {"plan": "aggressive"}


def test_unknown_long_flag_with_empty_equals_value():
    assert parse_args(["--plan="]).unknown_flags == {"plan": ""}


def test_unknown_long_flag_does_not_swallow_a_flag_or_file_arg():
    assert parse_args(["--plan", "--verbose"]).unknown_flags == {"plan": True}
    assert parse_args(["--plan", "@f.md"]).unknown_flags == {"plan": True}


def test_unknown_short_flag_is_an_error():
    args = parse_args(["-z"])
    assert args.diagnostics[0].type == "error"
    assert "Unknown option: -z" in args.diagnostics[0].message
    assert args.unknown_flags == {}


# --------------------------------------------------------------------------
# combinations
# --------------------------------------------------------------------------


def test_realistic_command_line():
    args = parse_args(
        [
            "--model",
            "openai/gpt-4o",
            "--thinking",
            "high",
            "-t",
            "read,bash",
            "@context.md",
            "-p",
            "refactor this",
            "--plan",
        ]
    )
    assert args.model == "openai/gpt-4o"
    assert args.thinking == "high"
    assert args.tools == ["read", "bash"]
    assert args.file_args == ["context.md"]
    assert args.print is True
    assert args.messages == ["refactor this"]
    assert args.unknown_flags == {"plan": True}
    assert args.diagnostics == []


def test_value_flag_at_end_without_a_value_is_ignored():
    # `--model` with nothing after it does not consume anything and is dropped,
    # matching the TypeScript's `i + 1 < args.length` guard.
    args = parse_args(["--model"])
    assert args.model is None
    assert args.messages == []


def test_flags_after_positional_messages_still_parse():
    args = parse_args(["hello", "--verbose", "world"])
    assert args.messages == ["hello", "world"]
    assert args.verbose is True
