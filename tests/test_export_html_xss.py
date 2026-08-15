"""Python port of `packages/coding-agent/test/export-html-xss.test.ts`.

The TypeScript test does not exercise any TypeScript code: it reads
`packages/coding-agent/src/core/export-html/template.js` -- a vendored browser
bundle that ships inside the exported HTML document and re-renders the session
client-side with `marked` -- and greps its source for the sanitisation calls
(`sanitizeMarkdownUrl`, `escapeHtml(...)`) that keep session-controlled strings
out of attribute and `innerHTML` positions.

That bundle is not part of this port. `pi_coding_agent.core.export_html` ports
only `ansi_to_html.py` and `colors.py`; the exporter's document assembly
(`index.ts`) and the vendored `marked`/`highlight.js`/`template.js` browser
assets are excluded (see that package's module docstring and the README's
"Not ported, by decision" list). There is no `.js` file anywhere under
`packages/pi-coding-agent`, so each source-grep assertion below is skipped
individually with that reason rather than the file being dropped.

The one piece of the sanitisation chain that *is* ported -- `escape_html`, the
`ansi-to-html.ts` helper that the exporter uses for server-side escaping -- is
pinned for real at the bottom.
"""

from __future__ import annotations

import pytest

from pi_coding_agent.core.export_html import ansi_to_html, escape_html

TEMPLATE_JS_NOT_PORTED = (
    "core/export-html/template.js is a vendored browser bundle; this port "
    "ships only ansi_to_html.py and colors.py from core/export-html/ (see "
    "pi_coding_agent.core.export_html.__doc__), so there is no file to grep."
)


@pytest.mark.skip(reason=TEMPLATE_JS_NOT_PORTED)
def test_overrides_the_marked_link_renderer_to_use_scheme_allow_list_sanitization() -> None:
    r"""`/link\s*\(\s*token\s*\)/`, `/sanitizeMarkdownUrl\(token\.href\)/`,
    `/\^\(https\?\|mailto\|tel\|ftp\)/`."""
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=TEMPLATE_JS_NOT_PORTED)
def test_overrides_the_marked_image_renderer_to_use_scheme_allow_list_sanitization() -> None:
    r"""`/image\s*\(\s*token\s*\)/`, `/sanitizeMarkdownUrl\(token\.href\)/`."""
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=TEMPLATE_JS_NOT_PORTED)
def test_strips_c0_controls_before_checking_and_emitting_markdown_urls() -> None:
    r"""`toContain("replace(/[\x00-\x1f\x7f]/g, '')")` and no
    `/\^\\s\*\(javascript\|vbscript\|data\):/i` denylist."""
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=TEMPLATE_JS_NOT_PORTED)
def test_escapes_href_attributes_in_the_custom_link_renderer() -> None:
    r"""`/escapeHtml\(href\)/`."""
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=TEMPLATE_JS_NOT_PORTED)
def test_escapes_image_mime_type_attributes() -> None:
    r"""No `${img.mimeType}`; `/escapeHtml\(img\.mimeType/`."""
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=TEMPLATE_JS_NOT_PORTED)
def test_escapes_image_data_attributes() -> None:
    r"""No `;base64,${img.data}"`; `;base64,${escapeHtml(img.data || '')}"`."""
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=TEMPLATE_JS_NOT_PORTED)
def test_escapes_entry_ids_before_inserting_them_into_attributes() -> None:
    r"""No `id="${entryId}"` / `data-entry-id="${entryId}"`; instead
    `entry-${escapeHtml(entry.id)}` and `data-entry-id="${escapeHtml(entryId)}"`."""
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=TEMPLATE_JS_NOT_PORTED)
def test_escapes_tree_metadata_rendered_from_session_fields() -> None:
    r"""`msg.toolName`, `msg.role`, `entry.modelId`, `entry.thinkingLevel` and
    `entry.type` must all go through `escapeHtml(...)`."""
    raise AssertionError("unreachable")


@pytest.mark.skip(reason=TEMPLATE_JS_NOT_PORTED)
def test_escapes_model_names_in_the_exported_header() -> None:
    r"""`${escapeHtml(globalStats.models.join(', ') || 'unknown')}`."""
    raise AssertionError("unreachable")


# -- The ported half of the sanitisation chain ------------------------------


def test_escape_html_neutralises_every_attribute_breakout_character() -> None:
    """`escapeHtml` from `core/export-html/ansi-to-html.ts` is ported verbatim.

    Note the apostrophe entity differs between the two TypeScript copies:
    `ansi-to-html.ts` emits `&#039;` while the vendored `template.js` emits
    `&#39;`. The ported helper follows `ansi-to-html.ts`, the file it came from.
    """
    assert escape_html("&") == "&amp;"
    assert escape_html("<script>") == "&lt;script&gt;"
    assert escape_html('"') == "&quot;"
    assert escape_html("'") == "&#039;"
    assert escape_html("<img src=x onerror=alert('1')>") == "&lt;img src=x onerror=alert(&#039;1&#039;)&gt;"
    # `&` is replaced first, so already-escaped input double-escapes rather
    # than leaving a live entity behind.
    assert escape_html("&lt;") == "&amp;lt;"


def test_ansi_to_html_escapes_markup_in_session_text() -> None:
    """Session-controlled text reaching the exporter cannot open a tag."""
    html = ansi_to_html("<script>alert('xss')</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&#039;xss&#039;" in html
