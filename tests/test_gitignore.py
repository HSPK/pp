"""Tests for the gitignore pattern matcher (`pi_coding_agent.tools.gitignore`).

This module has no direct TypeScript source (the TS tools rely on the
`ignore` npm package via `fd`/`rg`); these tests exercise gitignore-semantics
pattern matching thoroughly per the porting task's explicit requirement.
"""

from __future__ import annotations

from pi_coding_agent.tools.gitignore import GitignoreMatcher, build_matcher_for_tree, compile_glob


def test_simple_filename_pattern_matches_any_depth():
    m = GitignoreMatcher()
    m.add(["*.log"])

    assert m.is_ignored("debug.log", is_dir=False)
    assert m.is_ignored("nested/dir/debug.log", is_dir=False)
    assert not m.is_ignored("debug.txt", is_dir=False)


def test_anchored_pattern_with_leading_slash():
    m = GitignoreMatcher()
    m.add(["/build"])

    assert m.is_ignored("build", is_dir=True)
    assert not m.is_ignored("nested/build", is_dir=True)


def test_directory_only_pattern_trailing_slash():
    m = GitignoreMatcher()
    m.add(["logs/"])

    assert m.is_ignored("logs", is_dir=True)
    assert not m.is_ignored("logs", is_dir=False)
    assert m.is_ignored("nested/logs", is_dir=True)


def test_double_star_matches_any_number_of_directories():
    m = GitignoreMatcher()
    m.add(["**/foo"])

    assert m.is_ignored("foo", is_dir=False)
    assert m.is_ignored("a/foo", is_dir=False)
    assert m.is_ignored("a/b/c/foo", is_dir=False)


def test_double_star_in_middle():
    m = GitignoreMatcher()
    m.add(["a/**/b"])

    assert m.is_ignored("a/b", is_dir=False)
    assert m.is_ignored("a/x/b", is_dir=False)
    assert m.is_ignored("a/x/y/b", is_dir=False)
    assert not m.is_ignored("a/c", is_dir=False)


def test_single_star_does_not_cross_directory_boundary():
    m = GitignoreMatcher()
    m.add(["*.txt"])

    assert m.is_ignored("a.txt", is_dir=False)
    assert m.is_ignored("dir/a.txt", is_dir=False)


def test_question_mark_matches_single_char():
    m = GitignoreMatcher()
    m.add(["file?.txt"])

    assert m.is_ignored("file1.txt", is_dir=False)
    assert not m.is_ignored("file12.txt", is_dir=False)


def test_character_class():
    m = GitignoreMatcher()
    m.add(["file[0-2].txt"])

    assert m.is_ignored("file0.txt", is_dir=False)
    assert m.is_ignored("file2.txt", is_dir=False)
    assert not m.is_ignored("file3.txt", is_dir=False)


def test_negation_reincludes_file():
    m = GitignoreMatcher()
    m.add(["*.log", "!important.log"])

    assert m.is_ignored("debug.log", is_dir=False)
    assert not m.is_ignored("important.log", is_dir=False)


def test_later_pattern_overrides_earlier():
    m = GitignoreMatcher()
    m.add(["!keep.txt", "*.txt"])

    # *.txt comes after the negation, so it wins (last match wins).
    assert m.is_ignored("keep.txt", is_dir=False)


def test_comments_and_blank_lines_ignored():
    m = GitignoreMatcher()
    m.add(["# a comment", "", "*.log"])

    assert m.is_ignored("a.log", is_dir=False)


def test_add_text_splits_lines():
    m = GitignoreMatcher()
    m.add_text("*.log\n!keep.log\n")

    assert m.is_ignored("a.log", is_dir=False)
    assert not m.is_ignored("keep.log", is_dir=False)


def test_anchored_pattern_with_internal_slash():
    m = GitignoreMatcher()
    m.add(["src/build"])

    assert m.is_ignored("src/build", is_dir=True)
    assert not m.is_ignored("other/src/build", is_dir=True)


def test_empty_path_never_ignored():
    m = GitignoreMatcher()
    m.add(["*"])

    assert not m.is_ignored("", is_dir=False)


def test_compile_glob_full_match_semantics():
    regex = compile_glob("*.ts")
    assert regex.match("foo.ts")
    assert not regex.match("dir/foo.ts")
    assert not regex.match("foo.tsx")


def test_compile_glob_double_star_matches_nested_paths():
    regex = compile_glob("src/**/*.ts")
    assert regex.match("src/a.ts")
    assert regex.match("src/a/b/c.ts")
    assert not regex.match("other/a.ts")


def test_build_matcher_for_tree_uses_nested_gitignore_scoping(tmp_path):
    (tmp_path / ".gitignore").write_text("*.log\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / ".gitignore").write_text("*.tmp\n!keep.tmp\n")

    matcher = build_matcher_for_tree(str(tmp_path))

    assert matcher.is_ignored("a.log", is_dir=False)
    assert matcher.is_ignored("sub/a.log", is_dir=False)
    assert matcher.is_ignored("sub/b.tmp", is_dir=False)
    assert not matcher.is_ignored("sub/keep.tmp", is_dir=False)
    # A .tmp file outside `sub` is not covered by the nested .gitignore.
    assert not matcher.is_ignored("top.tmp", is_dir=False)


def test_build_matcher_for_tree_missing_root_returns_empty_matcher(tmp_path):
    matcher = build_matcher_for_tree(str(tmp_path / "does-not-exist"))
    assert not matcher.is_ignored("anything.txt", is_dir=False)
