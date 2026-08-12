"""Python port of `packages/coding-agent/test/args.test.ts`.

`unknownFlags` is a `Map` upstream and a `dict` here, so `.get(name)` becomes
`[name]`; `result.continue` becomes `continue_` (reserved word).
"""

from __future__ import annotations

import pytest
from pi_coding_agent.cli.args import Diagnostic, parse_args

# ---------------------------------------------------------------------------
# --version / --help
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--version", "-v"])
def test_parses_version_flag(flag: str) -> None:
    assert parse_args([flag]).version is True


def test_version_takes_precedence_over_other_args() -> None:
    result = parse_args(["--version", "--help", "some message"])

    assert result.version is True
    assert result.help is True
    assert "some message" in result.messages


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_parses_help_flag(flag: str) -> None:
    assert parse_args([flag]).help is True


# ---------------------------------------------------------------------------
# --print
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--print", "-p"])
def test_parses_print_flag(flag: str) -> None:
    assert parse_args([flag]).print is True


def test_parses_prompt_after_p_even_with_yaml_frontmatter() -> None:
    prompt = "---\ntitle: hello\n---\nSay hi."

    result = parse_args(["-p", prompt])

    assert result.print is True
    assert result.messages == [prompt]
    assert len(result.unknown_flags) == 0


def test_does_not_consume_options_after_p_as_prompts() -> None:
    result = parse_args(["-p", "--provider", "openai", "Say hi."])

    assert result.print is True
    assert result.provider == "openai"
    assert result.messages == ["Say hi."]


# ---------------------------------------------------------------------------
# --continue / --resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--continue", "-c"])
def test_parses_continue_flag(flag: str) -> None:
    assert parse_args([flag]).continue_ is True


@pytest.mark.parametrize("flag", ["--resume", "-r"])
def test_parses_resume_flag(flag: str) -> None:
    assert parse_args([flag]).resume is True


# ---------------------------------------------------------------------------
# flags with values
# ---------------------------------------------------------------------------


def test_parses_provider() -> None:
    assert parse_args(["--provider", "openai"]).provider == "openai"


def test_parses_model() -> None:
    assert parse_args(["--model", "gpt-4o"]).model == "gpt-4o"


def test_parses_api_key() -> None:
    assert parse_args(["--api-key", "sk-test-key"]).api_key == "sk-test-key"


def test_parses_system_prompt() -> None:
    assert parse_args(["--system-prompt", "You are a helpful assistant"]).system_prompt == "You are a helpful assistant"


def test_parses_append_system_prompt() -> None:
    assert parse_args(["--append-system-prompt", "Additional context"]).append_system_prompt == ["Additional context"]


def test_parses_multiple_append_system_prompt_flags() -> None:
    result = parse_args(["--append-system-prompt", "Context A", "--append-system-prompt", "Context B"])

    assert result.append_system_prompt == ["Context A", "Context B"]


def test_parses_mode() -> None:
    assert parse_args(["--mode", "json"]).mode == "json"


def test_parses_mode_rpc() -> None:
    assert parse_args(["--mode", "rpc"]).mode == "rpc"


def test_parses_session() -> None:
    assert parse_args(["--session", "/path/to/session.jsonl"]).session == "/path/to/session.jsonl"


def test_parses_session_id() -> None:
    assert parse_args(["--session-id", "orchestrated-session"]).session_id == "orchestrated-session"


def test_parses_fork() -> None:
    result = parse_args(["--fork", "1234abcd"])

    assert result.fork == "1234abcd"
    assert result.messages == []


def test_parses_export() -> None:
    assert parse_args(["--export", "session.jsonl"]).export == "session.jsonl"


def test_parses_thinking() -> None:
    assert parse_args(["--thinking", "high"]).thinking == "high"


def test_parses_models_as_comma_separated_list() -> None:
    result = parse_args(["--models", "gpt-4o,claude-sonnet,gemini-pro"])

    assert result.models == ["gpt-4o", "claude-sonnet", "gemini-pro"]


# ---------------------------------------------------------------------------
# --name
# ---------------------------------------------------------------------------


def test_parses_name_flag_with_value() -> None:
    assert parse_args(["--name", "my-session"]).name == "my-session"


def test_parses_n_shorthand() -> None:
    assert parse_args(["-n", "quick-session"]).name == "quick-session"


def test_preserves_empty_name_values_for_main_validation() -> None:
    assert parse_args(["--name", ""]).name == ""


def test_reports_missing_name_value() -> None:
    assert parse_args(["--name"]).diagnostics == [Diagnostic(type="error", message="--name requires a value")]


def test_name_works_alongside_other_flags() -> None:
    result = parse_args(["--name", "named-run", "--print", "--model", "gpt-4o", "hello"])

    assert result.name == "named-run"
    assert result.print is True
    assert result.model == "gpt-4o"
    assert result.messages == ["hello"]


# ---------------------------------------------------------------------------
# extensions, skills, prompt templates, themes
# ---------------------------------------------------------------------------


def test_parses_no_session_flag() -> None:
    assert parse_args(["--no-session"]).no_session is True


def test_parses_single_extension() -> None:
    assert parse_args(["--extension", "./my-extension.ts"]).extensions == ["./my-extension.ts"]


def test_parses_e_shorthand() -> None:
    assert parse_args(["-e", "./my-extension.ts"]).extensions == ["./my-extension.ts"]


def test_parses_multiple_extension_flags() -> None:
    assert parse_args(["--extension", "./ext1.ts", "-e", "./ext2.ts"]).extensions == ["./ext1.ts", "./ext2.ts"]


def test_parses_no_extensions_flag() -> None:
    assert parse_args(["--no-extensions"]).no_extensions is True


def test_parses_no_extensions_with_explicit_e_flags() -> None:
    result = parse_args(["--no-extensions", "-e", "foo.ts", "-e", "bar.ts"])

    assert result.no_extensions is True
    assert result.extensions == ["foo.ts", "bar.ts"]


def test_parses_single_skill() -> None:
    assert parse_args(["--skill", "./skill-dir"]).skills == ["./skill-dir"]


def test_parses_multiple_skill_flags() -> None:
    assert parse_args(["--skill", "./skill-a", "--skill", "./skill-b"]).skills == ["./skill-a", "./skill-b"]


def test_parses_single_prompt_template() -> None:
    assert parse_args(["--prompt-template", "./prompts"]).prompt_templates == ["./prompts"]


def test_parses_multiple_prompt_template_flags() -> None:
    result = parse_args(["--prompt-template", "./one", "--prompt-template", "./two"])

    assert result.prompt_templates == ["./one", "./two"]


def test_parses_single_theme() -> None:
    assert parse_args(["--theme", "./theme.json"]).themes == ["./theme.json"]


def test_parses_multiple_theme_flags() -> None:
    result = parse_args(["--theme", "./dark.json", "--theme", "./light.json"])

    assert result.themes == ["./dark.json", "./light.json"]


def test_parses_no_skills_flag() -> None:
    assert parse_args(["--no-skills"]).no_skills is True


def test_parses_no_prompt_templates_flag() -> None:
    assert parse_args(["--no-prompt-templates"]).no_prompt_templates is True


def test_parses_no_themes_flag() -> None:
    assert parse_args(["--no-themes"]).no_themes is True


@pytest.mark.parametrize("flag", ["--no-context-files", "-nc"])
def test_parses_no_context_files_flag(flag: str) -> None:
    assert parse_args([flag]).no_context_files is True


# ---------------------------------------------------------------------------
# project approval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--approve", "-a"])
def test_parses_approve(flag: str) -> None:
    assert parse_args([flag]).project_trust_override is True


@pytest.mark.parametrize("flag", ["--no-approve", "-na"])
def test_parses_no_approve(flag: str) -> None:
    assert parse_args([flag]).project_trust_override is False


def test_parses_verbose_flag() -> None:
    assert parse_args(["--verbose"]).verbose is True


def test_parses_offline_flag() -> None:
    assert parse_args(["--offline"]).offline is True


# ---------------------------------------------------------------------------
# --tui-mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["regular", "fullscreen"])
def test_parses_tui_mode(mode: str) -> None:
    assert parse_args(["--tui-mode", mode]).tui_mode == mode


def test_rejects_invalid_tui_modes() -> None:
    result = parse_args(["--tui-mode", "other"])

    assert result.diagnostics == [
        Diagnostic(type="error", message='Invalid TUI mode "other". Valid values: regular, fullscreen')
    ]


def test_tui_mode_requires_a_mode() -> None:
    result = parse_args(["--tui-mode"])

    assert result.diagnostics == [Diagnostic(type="error", message="--tui-mode requires regular or fullscreen")]


def test_does_not_recognize_the_old_ui_mode_flag() -> None:
    result = parse_args(["--ui-mode", "fullscreen"])

    assert result.tui_mode is None
    assert result.unknown_flags["ui-mode"] == "fullscreen"


# ---------------------------------------------------------------------------
# tool flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--no-tools", "-nt"])
def test_parses_no_tools_flag(flag: str) -> None:
    assert parse_args([flag]).no_tools is True


@pytest.mark.parametrize("flag", ["--no-builtin-tools", "-nbt"])
def test_parses_no_builtin_tools_flag(flag: str) -> None:
    assert parse_args([flag]).no_builtin_tools is True


@pytest.mark.parametrize("flag", ["--tools", "-t"])
def test_parses_tools_flag(flag: str) -> None:
    assert parse_args([flag, "read,bash"]).tools == ["read", "bash"]


@pytest.mark.parametrize("flag", ["--exclude-tools", "-xt"])
def test_parses_exclude_tools_flag(flag: str) -> None:
    assert parse_args([flag, "read,bash"]).exclude_tools == ["read", "bash"]


def test_parses_no_tools_with_explicit_tools_flags() -> None:
    result = parse_args(["--no-tools", "--tools", "read,bash"])

    assert result.no_tools is True
    assert result.tools == ["read", "bash"]


def test_parses_no_builtin_tools_with_explicit_tools_flags() -> None:
    result = parse_args(["--no-builtin-tools", "--tools", "read,bash"])

    assert result.no_builtin_tools is True
    assert result.tools == ["read", "bash"]


# ---------------------------------------------------------------------------
# messages and file args
# ---------------------------------------------------------------------------


def test_parses_plain_text_messages() -> None:
    assert parse_args(["hello", "world"]).messages == ["hello", "world"]


def test_parses_at_file_arguments() -> None:
    assert parse_args(["@README.md", "@src/main.ts"]).file_args == ["README.md", "src/main.ts"]


def test_parses_mixed_messages_and_file_args() -> None:
    result = parse_args(["@file.txt", "explain this", "@image.png"])

    assert result.file_args == ["file.txt", "image.png"]
    assert result.messages == ["explain this"]


def test_captures_unknown_long_flags_with_string_values() -> None:
    result = parse_args(["--unknown-flag", "message"])

    assert result.messages == []
    assert result.unknown_flags["unknown-flag"] == "message"


def test_captures_unknown_boolean_long_flags() -> None:
    assert parse_args(["--unknown-flag"]).unknown_flags["unknown-flag"] is True


def test_captures_unknown_long_flags_with_equals_syntax() -> None:
    assert parse_args(["--unknown-flag=value"]).unknown_flags["unknown-flag"] == "value"


# ---------------------------------------------------------------------------
# complex combinations
# ---------------------------------------------------------------------------


def test_parses_multiple_flags_together() -> None:
    result = parse_args(
        [
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet",
            "--print",
            "--thinking",
            "high",
            "@prompt.md",
            "Do the task",
        ]
    )

    assert result.provider == "anthropic"
    assert result.model == "claude-sonnet"
    assert result.print is True
    assert result.thinking == "high"
    assert result.file_args == ["prompt.md"]
    assert result.messages == ["Do the task"]
