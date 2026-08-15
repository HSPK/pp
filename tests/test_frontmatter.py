"""Python port of `packages/coding-agent/test/frontmatter.test.ts`.

`utils/frontmatter.ts` is ported into `core/resource_loader.py`, which returns
a `(frontmatter, body)` tuple where TypeScript returns an object.
"""

from __future__ import annotations

import pytest
import yaml

from pi_coding_agent.core.resource_loader import parse_frontmatter, strip_frontmatter


def test_parses_keys_strips_quotes_and_returns_body() -> None:
    text = "---\nname: \"skill-name\"\ndescription: 'A desc'\nfoo-bar: value\n---\n\nBody text"

    frontmatter, body = parse_frontmatter(text)

    assert frontmatter["name"] == "skill-name"
    assert frontmatter["description"] == "A desc"
    assert frontmatter["foo-bar"] == "value"
    assert body == "Body text"


def test_normalizes_newlines_and_handles_crlf() -> None:
    text = "---\r\nname: test\r\n---\r\nLine one\r\nLine two"

    _frontmatter, body = parse_frontmatter(text)

    assert body == "Line one\nLine two"


def test_raises_on_invalid_yaml_frontmatter() -> None:
    text = "---\nfoo: [bar\n---\nBody"

    # TypeScript's `yaml` package reports `at line 1, column 10` (the point the
    # unterminated flow sequence runs out); PyYAML raises with the same kind of
    # positional mark but anchors it at the opening bracket, so the port pins
    # "an error carrying a line/column mark" rather than the exact offsets.
    with pytest.raises(yaml.YAMLError, match=r"line \d+, column \d+") as error:
        parse_frontmatter(text)
    assert "expected ',' or ']'" in str(error.value)


def test_parses_pipe_multiline_yaml_syntax() -> None:
    text = "---\ndescription: |\n  Line one\n  Line two\n---\n\nBody"

    frontmatter, body = parse_frontmatter(text)

    assert frontmatter["description"] == "Line one\nLine two\n"
    assert body == "Body"


def test_returns_original_content_when_frontmatter_is_missing_or_unterminated() -> None:
    no_frontmatter = "Just text\nsecond line"
    missing_end = "---\nname: test\nBody without terminator"

    assert parse_frontmatter(no_frontmatter)[1] == "Just text\nsecond line"
    assert parse_frontmatter(missing_end)[1] == missing_end


def test_returns_empty_mapping_for_comment_only_frontmatter() -> None:
    frontmatter, _body = parse_frontmatter("---\n# just a comment\n---\nBody")

    assert frontmatter == {}


def test_strip_frontmatter_removes_frontmatter_and_trims_body() -> None:
    assert strip_frontmatter("---\nkey: value\n---\n\nBody\n") == "Body"


def test_strip_frontmatter_returns_body_when_no_frontmatter_present() -> None:
    assert strip_frontmatter("\n  No frontmatter body  \n") == "\n  No frontmatter body  \n"
