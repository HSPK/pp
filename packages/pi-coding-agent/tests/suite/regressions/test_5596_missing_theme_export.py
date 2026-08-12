"""Python port of `packages/coding-agent/test/suite/regressions/5596-missing-theme-export.test.ts`.

Not portable: the whole test is `session.exportToHtml(...)` resolving while the
configured theme is missing. The theme half *is* ported (`get_theme_by_name`
returns `None` for an unknown name exactly like `getThemeByName`), but this port
omits the HTML document assembly -- `AgentSession.export_to_html()` raises
`NotImplementedError` because `exportSessionToHtml` stitches the transcript into
`template.html`/`template.css`/`template.js` around vendored `marked` and
`highlight.js` browser bundles (see the "Not ported, by decision" list in the
repo README) -- so the export this test awaits never happens and there is no
behavior here to pin.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="AgentSession.export_to_html is deliberately not ported (no HTML document assembly)"
)


def test_exports_with_the_active_fallback_theme_when_the_configured_theme_is_missing() -> None:
    raise AssertionError("unreachable")
