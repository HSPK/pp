"""Tests for the `edit` tool. Ported from `tools.test.ts` (describe("edit tool"),
describe("edit tool fuzzy matching"), describe("edit tool CRLF handling")) and
`edit-tool-legacy-input.test.ts`.

The TS tests round-trip `applyPatch(originalContent, result.details.patch)` using
the `diff` npm package's patch-applier. There is no Python equivalent of that
library to import, so `apply_unified_patch` below shells out to the system
`patch(1)` binary instead (present on every POSIX dev/CI box, not a new Python
dependency) to actually apply the generated unified patch and check the result,
which is what the TS assertion is really pinning: that `result.details.patch`
is a *valid, correctly targeted* unified diff, not just a string containing the
right substrings.
"""

from __future__ import annotations

import os
import subprocess
import tempfile

import pytest

from pi_coding_agent.tools.edit import EditOperations, create_edit_tool
from pi_coding_agent.tools.edit_diff import compute_edits_diff


def get_text(result) -> str:
    return "\n".join(c.text for c in result.content if c.type == "text")


def apply_unified_patch(original: str, patch_text: str) -> str:
    """Apply `patch_text` (as produced by `generate_unified_patch`) to `original`.

    Mirrors the TS suite's `applyPatch(originalContent, result.details.patch)`
    calls (from the `diff` npm package) using the real `patch(1)` utility so a
    malformed or mistargeted hunk actually fails the test instead of only being
    checked by substring.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = os.path.join(tmp_dir, "target")
        with open(target, "w", encoding="utf-8", newline="") as fh:
            fh.write(original)
        proc = subprocess.run(
            ["patch", "--quiet", target],
            input=patch_text,
            text=True,
            capture_output=True,
        )
        assert proc.returncode == 0, f"patch failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
        with open(target, encoding="utf-8", newline="") as fh:
            return fh.read()


async def test_replace_text_in_file(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "edit-test.txt"
    original_content = "Hello, world!"
    test_file.write_text(original_content)

    result = await tool.execute(
        "call-1", {"path": str(test_file), "edits": [{"oldText": "world", "newText": "testing"}]}
    )

    assert "Successfully replaced" in get_text(result)
    assert result.details is not None
    assert isinstance(result.details.diff, str)
    assert "testing" in result.details.diff
    assert "--- " in result.details.patch
    assert "+++ " in result.details.patch
    assert "@@" in result.details.patch
    assert "-Hello, world!" in result.details.patch
    assert "+Hello, testing!" in result.details.patch
    assert test_file.read_text() == "Hello, testing!"
    assert apply_unified_patch(original_content, result.details.patch) == "Hello, testing!"


async def test_fails_if_text_not_found(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "edit-test.txt"
    test_file.write_text("Hello, world!")

    with pytest.raises(ValueError, match="Could not find the exact text"):
        await tool.execute(
            "call-2", {"path": str(test_file), "edits": [{"oldText": "nonexistent", "newText": "testing"}]}
        )


async def test_enoent_when_target_missing(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(RuntimeError, match=rf"Could not edit file: {missing_file}\. Error code: ENOENT\."):
        await tool.execute("call-3", {"path": str(missing_file), "edits": [{"oldText": "hello", "newText": "world"}]})


async def test_fails_if_text_appears_multiple_times(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "edit-test.txt"
    test_file.write_text("foo foo foo")

    with pytest.raises(ValueError, match="Found 3 occurrences"):
        await tool.execute("call-4", {"path": str(test_file), "edits": [{"oldText": "foo", "newText": "bar"}]})


async def test_multiple_disjoint_regions_in_one_call(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "edit-multi.txt"
    test_file.write_text("alpha\nbeta\ngamma\ndelta\n")

    result = await tool.execute(
        "call-5",
        {
            "path": str(test_file),
            "edits": [
                {"oldText": "alpha\n", "newText": "ALPHA\n"},
                {"oldText": "gamma\n", "newText": "GAMMA\n"},
            ],
        },
    )

    assert "Successfully replaced 2 block(s)" in get_text(result)
    assert test_file.read_text() == "ALPHA\nbeta\nGAMMA\ndelta\n"
    assert "ALPHA" in result.details.diff
    assert "GAMMA" in result.details.diff


async def test_collapses_large_unchanged_gaps_in_multi_edit_diffs(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "edit-multi-large-gap.txt"
    lines = [f"line {i + 1:03d}" for i in range(600)]
    test_file.write_text("\n".join(lines) + "\n")

    result = await tool.execute(
        "call-6",
        {
            "path": str(test_file),
            "edits": [
                {"oldText": "line 100\n", "newText": "LINE 100\n"},
                {"oldText": "line 300\n", "newText": "LINE 300\n"},
                {"oldText": "line 500\n", "newText": "LINE 500\n"},
            ],
        },
    )

    diff = result.details.diff
    assert "LINE 100" in diff
    assert "LINE 300" in diff
    assert "LINE 500" in diff
    assert "..." in diff
    assert "line 250" not in diff
    assert len(diff.split("\n")) < 50


async def test_edits_match_original_file_not_incrementally(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "edit-multi-original.txt"
    test_file.write_text("foo\nbar\nbaz\n")

    await tool.execute(
        "call-7",
        {
            "path": str(test_file),
            "edits": [
                {"oldText": "foo\n", "newText": "foo bar\n"},
                {"oldText": "bar\n", "newText": "BAR\n"},
            ],
        },
    )

    assert test_file.read_text() == "foo bar\nBAR\nbaz\n"


async def test_fails_when_edits_empty(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "edit-empty-edits.txt"
    test_file.write_text("hello\nworld\n")

    with pytest.raises(ValueError, match="edits must contain at least one replacement"):
        await tool.execute("call-8", {"path": str(test_file), "edits": []})


async def test_fails_when_multi_edit_regions_overlap(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "edit-overlap.txt"
    test_file.write_text("one\ntwo\nthree\n")

    with pytest.raises(ValueError, match="overlap"):
        await tool.execute(
            "call-9",
            {
                "path": str(test_file),
                "edits": [
                    {"oldText": "one\ntwo\n", "newText": "ONE\nTWO\n"},
                    {"oldText": "two\nthree\n", "newText": "TWO\nTHREE\n"},
                ],
            },
        )


async def test_no_partial_apply_when_one_edit_fails(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "edit-no-partial.txt"
    original_content = "alpha\nbeta\ngamma\n"
    test_file.write_text(original_content)

    with pytest.raises(ValueError, match="Could not find"):
        await tool.execute(
            "call-10",
            {
                "path": str(test_file),
                "edits": [
                    {"oldText": "alpha\n", "newText": "ALPHA\n"},
                    {"oldText": "missing\n", "newText": "MISSING\n"},
                ],
            },
        )

    assert test_file.read_text() == original_content


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permission checks")
async def test_eacces_for_readonly_files(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "edit-readonly.txt"
    test_file.write_text("hello\n")
    os.chmod(test_file, 0o444)
    try:
        with pytest.raises(RuntimeError, match=rf"Could not edit file: {test_file}\. Error code: EACCES\."):
            await tool.execute("call-11", {"path": str(test_file), "edits": [{"oldText": "hello", "newText": "world"}]})
    finally:
        os.chmod(test_file, 0o644)


async def test_includes_original_error_message_for_unknown_edit_access_errors(tmp_path):
    async def failing_access(_path: str) -> None:
        raise RuntimeError("disk offline")

    async def read_file(_path: str) -> bytes:
        return b"hello\n"

    async def write_file(_path: str, _content: str) -> None:
        return None

    tool = create_edit_tool(
        str(tmp_path),
        EditOperations(read_file=read_file, write_file=write_file, access=failing_access),
    )

    # TS asserts `Could not edit file: broken.txt. Error: disk offline.` -- the
    # `Error: ` prefix comes from JavaScript's `String(new Error(...))`, which
    # stringifies the class name. Python's `str(exc)` is just the message, so the
    # equivalent claim is that the original message is threaded through verbatim.
    with pytest.raises(RuntimeError, match=r"^Could not edit file: broken\.txt\. disk offline\.$"):
        await tool.execute("call-16", {"path": "broken.txt", "edits": [{"oldText": "hello", "newText": "world"}]})


def test_compute_edits_diff_enoent_preview(tmp_path):
    missing_file = tmp_path / "missing-preview.txt"

    result = compute_edits_diff(str(missing_file), [{"oldText": "hello", "newText": "world"}], str(tmp_path))

    assert result == {"error": f"Could not edit file: {missing_file}. Error code: ENOENT."}


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permission checks")
def test_compute_edits_diff_eacces_preview(tmp_path):
    unreadable_file = tmp_path / "unreadable-preview.txt"
    unreadable_file.write_text("hello\n")
    os.chmod(unreadable_file, 0o222)
    try:
        result = compute_edits_diff(str(unreadable_file), [{"oldText": "hello", "newText": "world"}], str(tmp_path))
        assert result == {"error": f"Could not edit file: {unreadable_file}. Error code: EACCES."}
    finally:
        os.chmod(unreadable_file, 0o644)


# --- edit tool fuzzy matching ---


async def test_fuzzy_trailing_whitespace_stripped(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "trailing-ws.txt"
    test_file.write_text("line one   \nline two  \nline three\n")

    result = await tool.execute(
        "fuzzy-1",
        {"path": str(test_file), "edits": [{"oldText": "line one\nline two\n", "newText": "replaced\n"}]},
    )

    assert "Successfully replaced" in get_text(result)
    assert test_file.read_text() == "replaced\nline three\n"


async def test_fuzzy_fullwidth_chinese_punctuation(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "chinese-punctuation.txt"
    test_file.write_text("你好，世界\n你好（世界）\n")

    result = await tool.execute(
        "fuzzy-chinese",
        {
            "path": str(test_file),
            "edits": [{"oldText": "你好,世界\n你好(世界)\n", "newText": "你好，pi\n你好(pi)\n"}],
        },
    )

    assert "Successfully replaced" in get_text(result)
    assert test_file.read_text() == "你好，pi\n你好(pi)\n"


async def test_fuzzy_unicode_compatibility_forms(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "unicode-compatibility.txt"
    test_file.write_text("\uff21\uff22\uff23\uff11\uff12\uff13\ncafe\u0301\n")

    result = await tool.execute(
        "fuzzy-unicode",
        {"path": str(test_file), "edits": [{"oldText": "ABC123\ncaf\u00e9\n", "newText": "XYZ789\ncoffee\n"}]},
    )

    assert "Successfully replaced" in get_text(result)
    assert test_file.read_text() == "XYZ789\ncoffee\n"


async def test_fuzzy_smart_single_quotes(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "smart-quotes.txt"
    test_file.write_text("console.log(\u2018hello\u2019);\n")

    result = await tool.execute(
        "fuzzy-2",
        {
            "path": str(test_file),
            "edits": [{"oldText": "console.log('hello');", "newText": "console.log('world');"}],
        },
    )

    assert "Successfully replaced" in get_text(result)
    assert "world" in test_file.read_text()


async def test_fuzzy_smart_double_quotes(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "smart-double-quotes.txt"
    test_file.write_text("const msg = \u201cHello World\u201d;\n")

    result = await tool.execute(
        "fuzzy-3",
        {
            "path": str(test_file),
            "edits": [{"oldText": 'const msg = "Hello World";', "newText": 'const msg = "Goodbye";'}],
        },
    )

    assert "Successfully replaced" in get_text(result)
    assert "Goodbye" in test_file.read_text()


async def test_fuzzy_unicode_dashes(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "unicode-dashes.txt"
    test_file.write_text("range: 1\u20135\nbreak\u2014here\n")

    result = await tool.execute(
        "fuzzy-4",
        {
            "path": str(test_file),
            "edits": [{"oldText": "range: 1-5\nbreak-here", "newText": "range: 10-50\nbreak--here"}],
        },
    )

    assert "Successfully replaced" in get_text(result)
    assert "10-50" in test_file.read_text()


async def test_fuzzy_non_breaking_space(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "nbsp.txt"
    test_file.write_text("hello\u00a0world\n")

    result = await tool.execute(
        "fuzzy-5", {"path": str(test_file), "edits": [{"oldText": "hello world", "newText": "hello universe"}]}
    )

    assert "Successfully replaced" in get_text(result)
    assert "universe" in test_file.read_text()


async def test_fuzzy_prefers_exact_match(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "exact-preferred.txt"
    test_file.write_text("const x = 'exact';\nconst y = 'other';\n")

    result = await tool.execute(
        "fuzzy-6",
        {
            "path": str(test_file),
            "edits": [{"oldText": "const x = 'exact';", "newText": "const x = 'changed';"}],
        },
    )

    assert "Successfully replaced" in get_text(result)
    assert test_file.read_text() == "const x = 'changed';\nconst y = 'other';\n"


async def test_fuzzy_still_fails_when_not_found(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "no-match.txt"
    test_file.write_text("completely different content\n")

    with pytest.raises(ValueError, match="Could not find the exact text"):
        await tool.execute(
            "fuzzy-7",
            {"path": str(test_file), "edits": [{"oldText": "this does not exist", "newText": "replacement"}]},
        )


async def test_fuzzy_detects_duplicates_after_normalization(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "fuzzy-dups.txt"
    test_file.write_text("hello world   \nhello world\n")

    with pytest.raises(ValueError, match="Found 2 occurrences"):
        await tool.execute(
            "fuzzy-8", {"path": str(test_file), "edits": [{"oldText": "hello world", "newText": "replaced"}]}
        )


async def test_fuzzy_multi_edit_mode(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "fuzzy-multi.txt"
    test_file.write_text("console.log(\u2018hello\u2019);\nhello\u00a0world\n")

    await tool.execute(
        "fuzzy-9",
        {
            "path": str(test_file),
            "edits": [
                {"oldText": "console.log('hello');\n", "newText": "console.log('world');\n"},
                {"oldText": "hello world\n", "newText": "hello universe\n"},
            ],
        },
    )

    assert test_file.read_text() == "console.log('world');\nhello universe\n"


async def test_fuzzy_preserves_correct_occurrence_near_duplicate_line(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "fuzzy-preserve-duplicate-line.txt"
    original_content = "\n".join(["replace me   ", "after   ", ""])
    test_file.write_text(original_content)

    result = await tool.execute(
        "fuzzy-preserve-duplicate-line",
        {"path": str(test_file), "edits": [{"oldText": "replace me\n", "newText": "after\n"}]},
    )

    expected_content = "\n".join(["after", "after   ", ""])
    assert test_file.read_text() == expected_content
    assert "--- " in result.details.patch
    assert "+++ " in result.details.patch
    assert apply_unified_patch(original_content, result.details.patch) == expected_content


async def test_fuzzy_preserve_untouched_lines_multi_edit(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "fuzzy-preserve-multi.txt"
    original_content = "\n".join(
        [
            "keep before  ",
            "first target  ",
            "first after",
            "keep middle   ",
            "second target  ",
            "second after",
            "keep after  ",
            "",
        ]
    )
    test_file.write_text(original_content)

    result = await tool.execute(
        "fuzzy-preserve-multi",
        {
            "path": str(test_file),
            "edits": [
                {"oldText": "first target\nfirst after", "newText": "FIRST\nFIRST2"},
                {"oldText": "second target\nsecond after", "newText": "SECOND\nSECOND2"},
            ],
        },
    )

    expected_content = "\n".join(
        [
            "keep before  ",
            "FIRST",
            "FIRST2",
            "keep middle   ",
            "SECOND",
            "SECOND2",
            "keep after  ",
            "",
        ]
    )
    assert test_file.read_text() == expected_content
    assert "--- " in result.details.patch
    assert apply_unified_patch(original_content, result.details.patch) == expected_content


# --- edit tool CRLF handling ---


async def test_crlf_lf_oldtext_matches_crlf_file(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "crlf-test.txt"
    test_file.write_bytes(b"line one\r\nline two\r\nline three\r\n")

    result = await tool.execute(
        "crlf-1", {"path": str(test_file), "edits": [{"oldText": "line two\n", "newText": "replaced line\n"}]}
    )

    assert "Successfully replaced" in get_text(result)


async def test_crlf_preserved_after_edit(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "crlf-preserve.txt"
    test_file.write_bytes(b"first\r\nsecond\r\nthird\r\n")

    await tool.execute("crlf-2", {"path": str(test_file), "edits": [{"oldText": "second\n", "newText": "REPLACED\n"}]})

    assert test_file.read_bytes() == b"first\r\nREPLACED\r\nthird\r\n"


async def test_lf_preserved_for_lf_files(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "lf-preserve.txt"
    test_file.write_text("first\nsecond\nthird\n")

    await tool.execute("lf-1", {"path": str(test_file), "edits": [{"oldText": "second\n", "newText": "REPLACED\n"}]})

    assert test_file.read_text() == "first\nREPLACED\nthird\n"


async def test_detects_duplicates_across_crlf_lf_variants(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "mixed-endings.txt"
    test_file.write_bytes(b"hello\r\nworld\r\n---\r\nhello\nworld\n")

    with pytest.raises(ValueError, match="Found 2 occurrences"):
        await tool.execute(
            "crlf-dup", {"path": str(test_file), "edits": [{"oldText": "hello\nworld\n", "newText": "replaced\n"}]}
        )


async def test_preserves_utf8_bom_after_edit(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "bom-test.txt"
    test_file.write_bytes("\ufefffirst\r\nsecond\r\nthird\r\n".encode())

    await tool.execute(
        "bom-test", {"path": str(test_file), "edits": [{"oldText": "second\n", "newText": "REPLACED\n"}]}
    )

    assert test_file.read_bytes() == "\ufefffirst\r\nREPLACED\r\nthird\r\n".encode()


async def test_preserves_crlf_and_bom_in_multi_edit(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    test_file = tmp_path / "bom-crlf-multi.txt"
    test_file.write_bytes("\ufefffirst\r\nsecond\r\nthird\r\nfourth\r\n".encode())

    await tool.execute(
        "crlf-multi",
        {
            "path": str(test_file),
            "edits": [
                {"oldText": "second\n", "newText": "SECOND\n"},
                {"oldText": "fourth\n", "newText": "FOURTH\n"},
            ],
        },
    )

    assert test_file.read_bytes() == "\ufefffirst\r\nSECOND\r\nthird\r\nFOURTH\r\n".encode()


# --- edit tool prepareArguments (ported from edit-tool-legacy-input.test.ts) ---


def test_legacy_schema_excludes_oldtext_newtext(tmp_path):
    tool = create_edit_tool(str(tmp_path))

    assert "oldText" not in tool.parameters["properties"]
    assert "newText" not in tool.parameters["properties"]


def test_folds_top_level_oldtext_newtext_into_edits(tmp_path):
    tool = create_edit_tool(str(tmp_path))

    prepared = tool.prepare_arguments({"path": "file.txt", "oldText": "before", "newText": "after"})

    assert prepared == {"path": "file.txt", "edits": [{"oldText": "before", "newText": "after"}]}


def test_appends_legacy_replacement_to_existing_edits(tmp_path):
    tool = create_edit_tool(str(tmp_path))

    prepared = tool.prepare_arguments(
        {"path": "file.txt", "edits": [{"oldText": "a", "newText": "b"}], "oldText": "c", "newText": "d"}
    )

    assert prepared == {
        "path": "file.txt",
        "edits": [{"oldText": "a", "newText": "b"}, {"oldText": "c", "newText": "d"}],
    }


def test_passes_through_valid_input_unchanged(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    input_ = {"path": "file.txt", "edits": [{"oldText": "a", "newText": "b"}]}

    prepared = tool.prepare_arguments(input_)

    # TS asserts `toBe(input)` -- the same object, not a copy.
    assert prepared is input_


def test_passes_through_non_object_input_unchanged(tmp_path):
    tool = create_edit_tool(str(tmp_path))

    assert tool.prepare_arguments(None) is None
    # TS also checks `undefined`; Python has no separate undefined value, so `None`
    # above covers both of TS's null/undefined assertions.
    assert tool.prepare_arguments("garbage") == "garbage"


async def test_prepared_args_execute_correctly(tmp_path):
    (tmp_path / "legacy.txt").write_text("before\n")
    tool = create_edit_tool(str(tmp_path))

    prepared = tool.prepare_arguments({"path": "legacy.txt", "oldText": "before", "newText": "after"})
    result = await tool.execute("tool-1", prepared)

    assert get_text(result) == "Successfully replaced 1 block(s) in legacy.txt."
    assert (tmp_path / "legacy.txt").read_text() == "after\n"


def test_parses_edits_from_json_string(tmp_path):
    tool = create_edit_tool(str(tmp_path))
    input_ = {"path": "file.txt", "edits": '[{"oldText": "a", "newText": "b"}]'}

    prepared = tool.prepare_arguments(input_)

    assert prepared == {"path": "file.txt", "edits": [{"oldText": "a", "newText": "b"}]}
    # TS writes the parsed array back onto the caller's object (`args.edits = parsed`)
    # and returns that same object.
    assert prepared is input_


def test_leaves_edits_alone_when_string_is_not_valid_json(tmp_path):
    tool = create_edit_tool(str(tmp_path))

    prepared = tool.prepare_arguments({"path": "file.txt", "edits": "not json"})

    assert prepared == {"path": "file.txt", "edits": "not json"}
