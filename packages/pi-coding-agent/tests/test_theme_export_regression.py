"""Regression tests for defects found by auditing the theme and editor ports."""

from __future__ import annotations

import json

# --------------------------------------------------------------------------
# theme: an unresolvable export var-ref must not escape
# --------------------------------------------------------------------------


def write_theme(directory, name: str, body: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(body), encoding="utf-8")


def test_unresolvable_export_var_ref_returns_empty_colors(tmp_path):
    """Regression: `resolve_var_refs` raised out of `get_theme_export_colors`.

    The schema accepts any string for the export colours, so a theme that
    validates cleanly can still carry an unresolvable reference; the TypeScript
    catches that and returns no colours rather than failing the HTML export.
    """
    from pi_coding_agent.modes.interactive.theme.theme import get_theme_export_colors

    themes_dir = tmp_path / "themes"
    write_theme(
        themes_dir,
        "broken",
        {
            "name": "broken",
            "dark": True,
            "colors": {},
            "export": {"pageBg": "doesnotexist"},
        },
    )

    colors = get_theme_export_colors("broken", custom_themes_dir=str(themes_dir))

    assert colors.page_bg is None
    assert colors.card_bg is None
    assert colors.info_bg is None


def test_missing_theme_returns_empty_export_colors(tmp_path):
    from pi_coding_agent.modes.interactive.theme.theme import get_theme_export_colors

    colors = get_theme_export_colors("nope", custom_themes_dir=str(tmp_path))
    assert colors.page_bg is None


def test_builtin_theme_export_colors_still_resolve():
    from pi_coding_agent.modes.interactive.theme.theme import get_theme_export_colors

    # The guard must not swallow the working path.
    for name in ("dark", "light"):
        colors = get_theme_export_colors(name)
        assert colors is not None
