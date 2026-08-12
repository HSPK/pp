"""Python port of `packages/coding-agent/test/export-html-skill-block.test.ts`.

The TypeScript test reads `src/core/export-html/template.js` -- the browser
bundle the HTML exporter inlines -- and greps it for the identifiers that
implement skill-block rendering (`parseSkillBlock`, `skillBlock.userMessage`,
`skill-invocation`, `hasUserContent`, `safeMarkedParse(skillBlock.content)`,
`tree-role-skill`). That file is a vendored `marked`/`highlight.js`-driven
document assembly with no Python counterpart (see the repository README:
"The HTML exporter's document assembly (`export-html/index.ts`) ... Its
ANSI-to-HTML converter and colour maths are ported"). Each grep assertion is
therefore skipped individually below, at the spot where it would have gone.

What the greps stand for *is* ported: the wrapper-detection function the
template duplicates is `core/agent-session.ts`'s `parseSkillBlock`, ported as
`pi_coding_agent.core.agent_session.parse_skill_block`. The tests below pin
that function against the same wrapper shape the template test describes, so
the behaviour the exporter must not regress (render only the user-visible
prompt, never the Pi-generated `<skill>` XML) is asserted against real code.
"""

from __future__ import annotations

import pytest
from pi_coding_agent.core.agent_session import parse_skill_block

SKILL_TEXT = (
    '<skill name="review" location="/skills/review/SKILL.md">\n'
    "# Review\n\nSteps to review a change.\n"
    "</skill>\n\n"
    "review the parser changes"
)


def test_strips_skill_wrapper_xml_from_user_message_rendering() -> None:
    # Skipped: `expect(templateJs).toMatch(/parseSkillBlock/)` and
    # `/skillBlock\.userMessage/` grep the unported `export-html/template.js`.
    # The behaviour they guard is asserted directly instead.
    parsed = parse_skill_block(SKILL_TEXT)

    assert parsed is not None
    assert parsed.name == "review"
    assert parsed.location == "/skills/review/SKILL.md"
    assert parsed.user_message == "review the parser changes"
    assert "<skill" not in parsed.user_message
    assert "</skill>" not in parsed.user_message


def test_renders_skill_invocation_and_user_message_as_separate_sibling_blocks() -> None:
    # Skipped: `expect(templateJs).toMatch(/skill-invocation/)` and
    # `/hasUserContent/` name CSS classes and a local variable inside the
    # unported template. The port-visible half of the contract is that the
    # skill body and the user prompt come back as two separate fields, and
    # that `user_message` is `None` exactly when there is no user prompt --
    # which is what the template's `hasUserContent` check keys off.
    parsed = parse_skill_block(SKILL_TEXT)
    assert parsed is not None
    assert parsed.content == "# Review\n\nSteps to review a change."
    assert parsed.user_message == "review the parser changes"

    without_prompt = parse_skill_block('<skill name="review" location="/skills/review/SKILL.md">\n# Review\n</skill>')
    assert without_prompt is not None
    assert without_prompt.content == "# Review"
    assert without_prompt.user_message is None

    # A whitespace-only prompt is also "no user content" upstream
    # (`match[4]?.trim() || undefined`).
    whitespace_prompt = parse_skill_block(
        '<skill name="review" location="/skills/review/SKILL.md">\n# Review\n</skill>\n\n   \n'
    )
    assert whitespace_prompt is not None
    assert whitespace_prompt.user_message is None


def test_renders_skill_content_as_markdown_not_raw_text() -> None:
    # Skipped: `expect(templateJs).toMatch(/safeMarkedParse\(skillBlock\.content\)/)`
    # asserts the template pipes the body through the vendored `marked`
    # bundle. There is no Markdown-to-HTML renderer in this port's exporter.
    # What is assertable is that `content` is handed over as the raw
    # Markdown source of the SKILL.md file, unescaped and unmodified.
    parsed = parse_skill_block(
        '<skill name="review" location="/s/SKILL.md">\n'
        "# Heading\n\n- bullet `code`\n\n<em>raw html</em>\n"
        "</skill>\n\ngo"
    )
    assert parsed is not None
    assert parsed.content == "# Heading\n\n- bullet `code`\n\n<em>raw html</em>"


def test_shows_skill_name_and_user_message_in_the_sidebar_tree() -> None:
    # Skipped: `expect(templateJs).toMatch(/tree-role-skill/)` is a CSS class
    # in the unported template. Both values the sidebar shows are available.
    parsed = parse_skill_block(SKILL_TEXT)
    assert parsed is not None
    assert parsed.name == "review"
    assert parsed.user_message == "review the parser changes"


@pytest.mark.parametrize(
    "text",
    [
        "just a prompt",
        '<skill name="a">\nbody\n</skill>',
        '<skill name="a" location="b">\nbody\n</skill>trailing',
        # A trailing newline after the closing tag is *not* a skill block
        # upstream: JavaScript's `$` anchors at the very end of the string.
        '<skill name="a" location="b">\nbody\n</skill>\n',
    ],
)
def test_returns_none_for_text_that_is_not_a_skill_block(text: str) -> None:
    assert parse_skill_block(text) is None
