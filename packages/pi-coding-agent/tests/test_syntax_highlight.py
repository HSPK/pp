"""Python port of `packages/coding-agent/test/syntax-highlight.test.ts`.

Only the `renderHighlightedHtml` cases are ported. The `highlight` /
`supportsLanguage` case and the whole `theme syntax highlighting` describe block
(`highlightCode`) drive highlight.js, which this port deliberately omits (see
the module docstring of `utils/syntax_highlight.py` and
`modes/interactive/theme/theme.py`).
"""

from __future__ import annotations

import pytest
from pi_coding_agent.utils.syntax_highlight import render_highlighted_html


def test_renders_highlighted_spans_with_the_provided_theme() -> None:
    rendered = render_highlighted_html(
        '<span class="hljs-keyword">const</span> value',
        {"keyword": lambda text: f"[keyword:{text}]"},
    )
    assert rendered == "[keyword:const] value"


def test_decodes_html_entities_emitted_by_highlight_js() -> None:
    rendered = render_highlighted_html("&lt;tag attr=&quot;value&quot;&gt;&amp;#x41;&#65;&lt;/tag&gt;")
    assert rendered == '<tag attr="value">&#x41;A</tag>'


def test_inherits_parent_formatting_for_unmapped_nested_scopes() -> None:
    interpolation = "${x}"
    rendered = render_highlighted_html(
        f'<span class="hljs-string">a<span class="hljs-subst">{interpolation}</span>b</span>',
        {"string": lambda text: f"[string:{text}]"},
    )
    assert rendered == f"[string:a][string:{interpolation}][string:b]"


def test_keeps_parent_formatting_across_unscoped_nested_spans() -> None:
    rendered = render_highlighted_html(
        '<span class="hljs-string">a<span class="language-xml">b</span>c</span>',
        {"string": lambda text: f"[string:{text}]"},
    )
    assert rendered == "[string:a][string:b][string:c]"


# The remaining three TypeScript cases are pinned individually below so each has
# an identifiable counterpart, rather than one lumped placeholder.


@pytest.mark.skip(
    reason=(
        "TS 'highlights code through highlight.js' asserts supportsLanguage('typescript') "
        "and tokenises `const value = 1` into keyword/number scopes. Neither `highlight` "
        "nor `supports_language` is ported: they wrap highlight.js, which has no Python "
        "equivalent here (see `utils/syntax_highlight.py`, which ports only the pure "
        "`render_highlighted_html`; the omission is listed in the top-level README "
        "under 'Not ported, by decision')."
    )
)
def test_highlights_code_through_highlight_js() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "TS 'colors diff additions and deletions in fenced diff blocks' asserts exact "
        "truecolor escapes from `highlightCode('-old\\n+new\\n', 'diff')`. `highlight_code` "
        "does not exist in `modes/interactive/theme/theme.py` (get_markdown_theme leaves "
        "MarkdownTheme.highlight_code at None), because it is highlight.js-backed."
    )
)
def test_colors_diff_additions_and_deletions_in_fenced_diff_blocks() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "TS 'keeps cli-highlight default styled scopes mapped to theme styles' asserts "
        "highlightCode() output for javascript regex, python decorator and html tag "
        "scopes. Same missing surface as above: no `highlight_code`, and no cli-highlight "
        "default scope table to map."
    )
)
def test_keeps_cli_highlight_default_styled_scopes_mapped_to_theme_styles() -> None:
    raise AssertionError("unreachable")
