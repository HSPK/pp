"""Python port of `packages/coding-agent/test/mermaid.test.ts`.

Not portable: every case calls `createMermaidMarkdownTransformer` from
`src/modes/interactive/components/mermaid.ts`, which renders diagrams through
the `grok-mermaid` npm package. The top-level README lists this under "Not
ported, by decision": "**Mermaid diagram rendering**, which renders through the
`grok-mermaid` npm package; there is no Python equivalent." There is no
`components/mermaid.py` and no transformer to call, so there is no code under
test.

The seven TypeScript cases and what each needs:

1. "replaces Mermaid code blocks with Unicode diagrams" -- the box-drawing
   layout engine inside `grok-mermaid`.
2. "leaves unsupported and oversized diagrams unchanged" -- the same engine's
   support matrix (`pie` is unsupported) and its width budget.
3. "maps semantic spans through the Pi theme" -- `grok-mermaid`'s semantic span
   output plus `modes/interactive/theme/theme.ts`, and the interactive theme
   layer is not ported either.
4. "renders incomplete Mermaid blocks during streaming" -- partial-parse
   rendering.
5. "falls back to the code block with a warning after streaming" -- the
   renderer's diagnostic strings ("dropped, expected a link: ...").
6. "summarizes additional partial-render warnings" -- the "(+1 more)" rollup
   over those same diagnostics.
7. "respects rendering modes and skips thinking blocks" -- the only case with a
   ported dependency (`SettingsManager.get_mermaid_rendering_mode`, which does
   exist here), but it still asserts on the transformer's output.

The setting itself is covered by `tests/test_settings_manager.py`; rewriting
any of the above against it would assert nothing about rendering, so nothing is
substituted here.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason=(
        "modes/interactive/components/mermaid.ts is not ported: rendering needs the "
        "grok-mermaid npm package (README, 'Not ported, by decision'). Covers 7 "
        "TypeScript cases."
    )
)
def test_mermaid_rendering() -> None:
    raise AssertionError("unreachable")
