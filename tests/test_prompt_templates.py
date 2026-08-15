"""Python port of `packages/coding-agent/test/prompt-templates.test.ts`.

Prompt template argument parsing and substitution: quote-aware argument
splitting, `$1` / `$@` / `$ARGUMENTS` placeholders, `${N:-default}` defaults,
bash-style `${@:N:L}` slicing, the rule that argument *values* are never
recursively substituted, and `argument-hint` frontmatter.

The functions live in `core/resource_loader.py` in this port rather than a
separate `prompt-templates` module.
"""

from __future__ import annotations

import os
from pathlib import Path

from pi_coding_agent.core.resource_loader import (
    PromptTemplate,
    expand_prompt_template,
    load_prompt_templates,
    parse_command_args,
    substitute_args,
)
from pi_coding_agent.core.source_info import create_synthetic_source_info

# ============================================================================
# substitute_args
# ============================================================================


def test_replaces_arguments_with_all_args_joined() -> None:
    assert substitute_args("Test: $ARGUMENTS", ["a", "b", "c"]) == "Test: a b c"


def test_replaces_at_with_all_args_joined() -> None:
    assert substitute_args("Test: $@", ["a", "b", "c"]) == "Test: a b c"


def test_replaces_at_and_arguments_identically() -> None:
    args = ["foo", "bar", "baz"]
    assert substitute_args("Test: $@", args) == substitute_args("Test: $ARGUMENTS", args)


def test_does_not_recursively_substitute_patterns_in_argument_values() -> None:
    assert substitute_args("$ARGUMENTS", ["$1", "$ARGUMENTS"]) == "$1 $ARGUMENTS"
    assert substitute_args("$@", ["$100", "$1"]) == "$100 $1"
    assert substitute_args("$ARGUMENTS", ["$100", "$1"]) == "$100 $1"


def test_supports_mixed_numbered_and_arguments() -> None:
    assert substitute_args("$1: $ARGUMENTS", ["prefix", "a", "b"]) == "prefix: prefix a b"


def test_supports_mixed_numbered_and_at() -> None:
    assert substitute_args("$1: $@", ["prefix", "a", "b"]) == "prefix: prefix a b"


def test_handles_empty_arguments_array_with_arguments() -> None:
    assert substitute_args("Test: $ARGUMENTS", []) == "Test: "


def test_handles_empty_arguments_array_with_at() -> None:
    assert substitute_args("Test: $@", []) == "Test: "


def test_handles_empty_arguments_array_with_positional() -> None:
    assert substitute_args("Test: $1", []) == "Test: "


def test_handles_multiple_occurrences_of_arguments() -> None:
    assert substitute_args("$ARGUMENTS and $ARGUMENTS", ["a", "b"]) == "a b and a b"


def test_handles_multiple_occurrences_of_at() -> None:
    assert substitute_args("$@ and $@", ["a", "b"]) == "a b and a b"


def test_handles_mixed_occurrences_of_at_and_arguments() -> None:
    assert substitute_args("$@ and $ARGUMENTS", ["a", "b"]) == "a b and a b"


def test_handles_special_characters_in_arguments() -> None:
    # `$100` inside an argument is never partially matched: whole values go in.
    assert substitute_args("$1 $2: $ARGUMENTS", ["arg100", "@user"]) == "arg100 @user: arg100 @user"


def test_handles_out_of_range_numbered_placeholders() -> None:
    # Out-of-range placeholders become empty strings, preserving template spaces.
    assert substitute_args("$1 $2 $3 $4 $5", ["a", "b"]) == "a b   "


def test_handles_unicode_characters() -> None:
    assert substitute_args("$ARGUMENTS", ["日本語", "🎉", "café"]) == "日本語 🎉 café"


def test_preserves_newlines_and_tabs_in_argument_values() -> None:
    assert substitute_args("$1 $2", ["line1\nline2", "tab\tthere"]) == "line1\nline2 tab\tthere"


def test_handles_consecutive_dollar_patterns() -> None:
    assert substitute_args("$1$2", ["a", "b"]) == "ab"


def test_handles_quoted_arguments_with_spaces() -> None:
    assert substitute_args("$ARGUMENTS", ["first arg", "second arg"]) == "first arg second arg"


def test_handles_single_argument_with_arguments() -> None:
    assert substitute_args("Test: $ARGUMENTS", ["only"]) == "Test: only"


def test_handles_single_argument_with_at() -> None:
    assert substitute_args("Test: $@", ["only"]) == "Test: only"


def test_handles_dollar_zero() -> None:
    assert substitute_args("$0", ["a", "b"]) == ""


def test_handles_decimal_number_in_pattern() -> None:
    assert substitute_args("$1.5", ["a"]) == "a.5"


def test_handles_arguments_as_part_of_word() -> None:
    assert substitute_args("pre$ARGUMENTS", ["a", "b"]) == "prea b"


def test_handles_at_as_part_of_word() -> None:
    assert substitute_args("pre$@", ["a", "b"]) == "prea b"


def test_handles_empty_arguments_in_middle_of_list() -> None:
    assert substitute_args("$ARGUMENTS", ["a", "", "c"]) == "a  c"


def test_handles_trailing_and_leading_spaces_in_arguments() -> None:
    assert substitute_args("$ARGUMENTS", ["  leading  ", "trailing  "]) == "  leading   trailing  "


def test_handles_argument_containing_pattern_partially() -> None:
    assert substitute_args("Prefix $ARGUMENTS suffix", ["ARGUMENTS"]) == "Prefix ARGUMENTS suffix"


def test_handles_non_matching_patterns() -> None:
    assert substitute_args("$A $$ $ $ARGS", ["a"]) == "$A $$ $ $ARGS"


def test_handles_case_variations() -> None:
    assert substitute_args("$arguments $Arguments $ARGUMENTS", ["a", "b"]) == "$arguments $Arguments a b"


def test_handles_both_syntaxes_in_same_command_with_same_result() -> None:
    args = ["x", "y", "z"]
    result1 = substitute_args("$@ and $ARGUMENTS", args)
    result2 = substitute_args("$ARGUMENTS and $@", args)
    assert result1 == result2
    assert result1 == "x y z and x y z"


def test_handles_very_long_argument_lists() -> None:
    args = [f"arg{i}" for i in range(100)]
    assert substitute_args("$ARGUMENTS", args) == " ".join(args)


def test_handles_numbered_placeholders_with_single_digit() -> None:
    assert substitute_args("$1 $2 $3", ["a", "b", "c"]) == "a b c"


def test_handles_numbered_placeholders_with_multiple_digits() -> None:
    args = [f"val{i}" for i in range(15)]
    assert substitute_args("$10 $12 $15", args) == "val9 val11 val14"


def test_handles_escaped_dollar_signs_literally() -> None:
    # There is no escape mechanism: the backslash stays and `$100` substitutes
    # to the (missing) 100th argument.
    assert substitute_args("Price: \\$100", []) == "Price: \\"


def test_handles_mixed_numbered_and_wildcard_placeholders() -> None:
    assert (
        substitute_args("$1: $@ ($ARGUMENTS)", ["first", "second", "third"])
        == "first: first second third (first second third)"
    )


def test_handles_command_with_no_placeholders() -> None:
    assert substitute_args("Just plain text", ["a", "b"]) == "Just plain text"


def test_handles_command_with_only_placeholders() -> None:
    assert substitute_args("$1 $2 $@", ["a", "b", "c"]) == "a b a b c"


# ============================================================================
# substitute_args - positional defaults
# ============================================================================


def test_uses_default_when_positional_arg_is_missing() -> None:
    assert substitute_args("List exactly ${1:-7} next steps", []) == "List exactly 7 next steps"


def test_supports_defaults_for_all_arguments() -> None:
    template = "${@:-default}\n${ARGUMENTS:-default}"
    assert substitute_args(template, []) == "default\ndefault"
    assert (
        substitute_args(template, ["This", "would", "be", "the", "arguments"])
        == "This would be the arguments\nThis would be the arguments"
    )


def test_uses_positional_arg_when_present() -> None:
    assert substitute_args("List exactly ${1:-7} next steps", ["3"]) == "List exactly 3 next steps"


def test_uses_default_when_positional_arg_is_empty() -> None:
    assert substitute_args("Mode: ${1:-brief}", [""]) == "Mode: brief"


def test_supports_multiple_positional_defaults() -> None:
    assert substitute_args("${1:-7} ${2:-brief}", []) == "7 brief"
    assert substitute_args("${1:-7} ${2:-brief}", ["3"]) == "3 brief"
    assert substitute_args("${1:-7} ${2:-brief}", ["3", "verbose"]) == "3 verbose"


def test_defaults_do_not_recursively_substitute_patterns_in_arg_values() -> None:
    assert substitute_args("${1:-7}", ["$ARGUMENTS"]) == "$ARGUMENTS"
    assert substitute_args("${1:-7}", ["$1"]) == "$1"


def test_defaults_do_not_recursively_substitute_patterns_in_default_values() -> None:
    assert substitute_args("${1:-$ARGUMENTS}", ["a", "b"]) == "a"
    assert substitute_args("${3:-$ARGUMENTS}", ["a", "b"]) == "$ARGUMENTS"


def test_supports_defaults_with_spaces() -> None:
    assert substitute_args("${1:-seven steps}", []) == "seven steps"


def test_supports_out_of_range_positional_defaults() -> None:
    assert substitute_args("${3:-fallback}", ["a", "b"]) == "fallback"


def test_mixes_positional_defaults_with_existing_placeholders() -> None:
    assert substitute_args("$1 ${2:-x} $ARGUMENTS", ["a"]) == "a x a"


# ============================================================================
# substitute_args - array slicing
# ============================================================================


def test_slices_from_index() -> None:
    assert substitute_args("${@:2}", ["a", "b", "c", "d"]) == "b c d"
    assert substitute_args("${@:1}", ["a", "b", "c"]) == "a b c"
    assert substitute_args("${@:3}", ["a", "b", "c", "d"]) == "c d"


def test_slices_with_length() -> None:
    assert substitute_args("${@:2:2}", ["a", "b", "c", "d"]) == "b c"
    assert substitute_args("${@:1:1}", ["a", "b", "c"]) == "a"
    assert substitute_args("${@:3:1}", ["a", "b", "c", "d"]) == "c"
    assert substitute_args("${@:2:3}", ["a", "b", "c", "d", "e"]) == "b c d"


def test_handles_out_of_range_slices() -> None:
    assert substitute_args("${@:99}", ["a", "b"]) == ""
    assert substitute_args("${@:5}", ["a", "b"]) == ""
    assert substitute_args("${@:10:5}", ["a", "b"]) == ""


def test_handles_zero_length_slices() -> None:
    assert substitute_args("${@:2:0}", ["a", "b", "c"]) == ""
    assert substitute_args("${@:1:0}", ["a", "b"]) == ""


def test_handles_length_exceeding_array() -> None:
    assert substitute_args("${@:2:99}", ["a", "b", "c"]) == "b c"
    assert substitute_args("${@:1:10}", ["a", "b"]) == "a b"


def test_processes_slice_before_simple_at() -> None:
    assert substitute_args("${@:2} vs $@", ["a", "b", "c"]) == "b c vs a b c"
    assert substitute_args("First: ${@:1:1}, All: $@", ["x", "y", "z"]) == "First: x, All: x y z"


def test_does_not_recursively_substitute_slice_patterns_in_args() -> None:
    assert substitute_args("${@:1}", ["${@:2}", "test"]) == "${@:2} test"
    assert substitute_args("${@:2}", ["a", "${@:3}", "c"]) == "${@:3} c"


def test_handles_slices_mixed_with_positional_args() -> None:
    assert substitute_args("$1: ${@:2}", ["cmd", "arg1", "arg2"]) == "cmd: arg1 arg2"
    assert substitute_args("$1 $2 ${@:3}", ["a", "b", "c", "d"]) == "a b c d"


def test_treats_slice_from_zero_as_all_args() -> None:
    assert substitute_args("${@:0}", ["a", "b", "c"]) == "a b c"


def test_slicing_handles_empty_args_array() -> None:
    assert substitute_args("${@:2}", []) == ""
    assert substitute_args("${@:1}", []) == ""


def test_slicing_handles_single_arg_array() -> None:
    assert substitute_args("${@:1}", ["only"]) == "only"
    assert substitute_args("${@:2}", ["only"]) == ""


def test_handles_slice_in_middle_of_text() -> None:
    assert substitute_args("Process ${@:2} with $1", ["tool", "file1", "file2"]) == "Process file1 file2 with tool"


def test_handles_multiple_slices_in_one_template() -> None:
    assert substitute_args("${@:1:1} and ${@:2}", ["a", "b", "c"]) == "a and b c"
    assert substitute_args("${@:1:2} vs ${@:3:2}", ["a", "b", "c", "d", "e"]) == "a b vs c d"


def test_handles_quoted_arguments_in_slices() -> None:
    assert substitute_args("${@:2}", ["cmd", "first arg", "second arg"]) == "first arg second arg"


def test_handles_special_characters_in_sliced_args() -> None:
    assert substitute_args("${@:2}", ["cmd", "$100", "@user", "#tag"]) == "$100 @user #tag"


def test_handles_unicode_in_sliced_args() -> None:
    assert substitute_args("${@:1}", ["日本語", "🎉", "café"]) == "日本語 🎉 café"


def test_combines_positional_slice_and_wildcard_placeholders() -> None:
    template = "Run $1 on ${@:2:2}, then process $@"
    args = ["eslint", "file1.ts", "file2.ts", "file3.ts"]
    assert (
        substitute_args(template, args)
        == "Run eslint on file1.ts file2.ts, then process eslint file1.ts file2.ts file3.ts"
    )


def test_handles_slice_with_no_spacing() -> None:
    assert substitute_args("prefix${@:2}suffix", ["a", "b", "c"]) == "prefixb csuffix"


def test_handles_large_slice_lengths_gracefully() -> None:
    args = [f"arg{i + 1}" for i in range(10)]
    assert substitute_args("${@:5:100}", args) == "arg5 arg6 arg7 arg8 arg9 arg10"


# ============================================================================
# parse_command_args
# ============================================================================


def test_parses_simple_space_separated_arguments() -> None:
    assert parse_command_args("a b c") == ["a", "b", "c"]


def test_parses_quoted_arguments_with_spaces() -> None:
    assert parse_command_args('"first arg" second') == ["first arg", "second"]


def test_parses_single_quoted_arguments() -> None:
    assert parse_command_args("'first arg' second") == ["first arg", "second"]


def test_parses_mixed_quote_styles() -> None:
    assert parse_command_args('"double" \'single\' "double again"') == ["double", "single", "double again"]


def test_parse_handles_empty_string() -> None:
    assert parse_command_args("") == []


def test_parse_handles_extra_spaces() -> None:
    assert parse_command_args("a  b   c") == ["a", "b", "c"]


def test_parse_handles_tabs_as_separators() -> None:
    assert parse_command_args("a\tb\tc") == ["a", "b", "c"]


def test_parse_handles_quoted_empty_string() -> None:
    # Empty quotes are skipped by the current implementation.
    assert parse_command_args('"" " "') == [" "]


def test_parse_handles_arguments_with_special_characters() -> None:
    assert parse_command_args("$100 @user #tag") == ["$100", "@user", "#tag"]


def test_parse_handles_unicode_characters() -> None:
    assert parse_command_args("日本語 🎉 café") == ["日本語", "🎉", "café"]


def test_parse_handles_newlines_in_quoted_arguments() -> None:
    assert parse_command_args('"line1\nline2" second') == ["line1\nline2", "second"]


def test_parse_treats_unquoted_newlines_as_separators() -> None:
    assert parse_command_args("label-2\n\nHere is some description #2.") == [
        "label-2",
        "Here",
        "is",
        "some",
        "description",
        "#2.",
    ]


def test_parse_collapses_mixed_unquoted_whitespace() -> None:
    assert parse_command_args("a\n\n\tb  c") == ["a", "b", "c"]


def test_parse_handles_escaped_quotes_inside_quoted_strings() -> None:
    # No escape handling: the backslash is literal and closes/opens nothing.
    assert parse_command_args('"quoted \\"text\\""') == ["quoted \\text\\"]


def test_parse_handles_trailing_spaces() -> None:
    assert parse_command_args("a b c   ") == ["a", "b", "c"]


def test_parse_handles_leading_spaces() -> None:
    assert parse_command_args("   a b c") == ["a", "b", "c"]


# ============================================================================
# expand_prompt_template
# ============================================================================


def _template(name: str, content: str) -> PromptTemplate:
    path = "/tmp/arg-test.md"
    return PromptTemplate(
        name=name,
        description="test",
        content=content,
        source_info=create_synthetic_source_info(path, "local"),
        file_path=path,
    )


def test_splits_template_arguments_on_unquoted_newlines() -> None:
    result = expand_prompt_template(
        "/arg-test label-2\n\nHere is some description #2.",
        [_template("arg-test", "- arg1: $1\n- rest: ${@:2}")],
    )
    assert result == "- arg1: label-2\n- rest: Here is some description #2."


def test_supports_template_command_separated_from_args_by_newline() -> None:
    result = expand_prompt_template("/arg-test\nlabel-2", [_template("arg-test", "arg1: $1")])
    assert result == "arg1: label-2"


# ============================================================================
# parse_command_args + substitute_args integration
# ============================================================================


def test_parses_and_substitutes_together_correctly() -> None:
    args = parse_command_args('Button "onClick handler" "disabled support"')
    result = substitute_args("Create component $1 with features: $ARGUMENTS", args)
    assert result == "Create component Button with features: Button onClick handler disabled support"


def test_handles_the_example_from_the_readme() -> None:
    args = parse_command_args('Button "onClick handler" "disabled support"')
    result = substitute_args("Create a React component named $1 with features: $ARGUMENTS", args)
    assert result == "Create a React component named Button with features: Button onClick handler disabled support"


def test_produces_same_result_with_at_and_arguments() -> None:
    args = parse_command_args("feature1 feature2 feature3")
    assert substitute_args("Implement: $@", args) == substitute_args("Implement: $ARGUMENTS", args)


# ============================================================================
# load_prompt_templates - argument-hint frontmatter
# ============================================================================


def _load(tmp_path: Path, name: str, content: str) -> list[PromptTemplate]:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / f"{name}.md").write_text(content, encoding="utf-8")
    return load_prompt_templates(
        cwd=os.getcwd(),
        agent_dir=str(tmp_path / "agent"),
        prompt_paths=[str(prompts_dir)],
        include_defaults=False,
    )


def _find(templates: list[PromptTemplate], name: str) -> PromptTemplate | None:
    return next((t for t in templates if t.name == name), None)


def test_parses_required_argument_hint_from_frontmatter(tmp_path: Path) -> None:
    templates = _load(
        tmp_path,
        "pr",
        "---\ndescription: Review PRs from URLs with structured issue and code analysis\n"
        'argument-hint: "<PR-URL>"\n---\nYou are given one or more GitHub PR URLs: $@',
    )
    pr = _find(templates, "pr")
    assert pr is not None
    assert pr.argument_hint == "<PR-URL>"
    assert pr.description == "Review PRs from URLs with structured issue and code analysis"


def test_parses_optional_argument_hint_from_frontmatter(tmp_path: Path) -> None:
    templates = _load(
        tmp_path,
        "wr",
        "---\ndescription: Finish the current task end-to-end with changelog, commit, and push\n"
        'argument-hint: "[instructions]"\n---\nWrap it. Additional instructions: $ARGUMENTS',
    )
    wr = _find(templates, "wr")
    assert wr is not None
    assert wr.argument_hint == "[instructions]"
    assert wr.description == "Finish the current task end-to-end with changelog, commit, and push"


def test_leaves_argument_hint_unset_when_not_specified(tmp_path: Path) -> None:
    templates = _load(
        tmp_path,
        "cl",
        "---\ndescription: Audit changelog entries before release\n---\n"
        "Audit changelog entries for all commits since the last release.",
    )
    cl = _find(templates, "cl")
    assert cl is not None
    assert cl.argument_hint is None


def test_ignores_empty_argument_hint(tmp_path: Path) -> None:
    templates = _load(
        tmp_path,
        "empty-hint",
        '---\ndescription: A command with empty hint\nargument-hint: ""\n---\nDo something',
    )
    template = _find(templates, "empty-hint")
    assert template is not None
    assert template.argument_hint is None


def test_preserves_argument_hint_with_special_characters(tmp_path: Path) -> None:
    templates = _load(
        tmp_path,
        "is",
        "---\ndescription: Analyze GitHub issues (bugs or feature requests)\n"
        'argument-hint: "<issue>"\n---\nAnalyze GitHub issue(s): $ARGUMENTS',
    )
    template = _find(templates, "is")
    assert template is not None
    assert template.argument_hint == "<issue>"
