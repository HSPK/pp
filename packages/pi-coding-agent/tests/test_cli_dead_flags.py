"""Flags that parse must also do something.

Both flags here were accepted by the parser and consumed nowhere in `src/`,
which is the worst shape a flag can have: `--help` advertises it, the user
passes it, and nothing happens. `--extension`/`--no-extensions` had the same
defect and is covered by `test_cli_extension_wiring.py`.

The two are fixed differently on purpose. `--no-themes` has a real Python
behaviour to wire, so it is wired. `--export` needs the HTML document assembly
this port does not carry, so it fails loudly instead of pretending -- the point
is that the user learns their intent was dropped, not that the feature appears.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pi_coding_agent.cli import entry as cli
from pi_coding_agent.modes.interactive.theme.theme import (
    _THEME_PACKAGE,
    get_available_themes,
    init_theme,
    load_theme,
    set_custom_theme_discovery_enabled,
)


@pytest.fixture(autouse=True)
def _restore_theme_discovery():
    """The gate is module-level state; leaving it off would leak into other tests."""
    init_theme("dark")
    yield
    set_custom_theme_discovery_enabled(True)


def _write_custom_theme(themes_dir: Path, name: str) -> None:
    """Copy a real built-in theme under a new name.

    Hand-rolling the JSON risks writing something the loader silently rejects
    (`_get_custom_theme_infos` swallows parse errors), which would make the
    "hidden" assertion below pass for the wrong reason.
    """
    themes_dir.mkdir(parents=True, exist_ok=True)
    builtin = json.loads((_THEME_PACKAGE / "dark.json").read_text())
    builtin["name"] = name
    (themes_dir / f"{name}.json").write_text(json.dumps(builtin))


def test_custom_themes_are_discovered_by_default(tmp_path):
    _write_custom_theme(tmp_path, "midnight")

    names = get_available_themes(custom_themes_dir=str(tmp_path))

    assert "midnight" in names


def test_no_themes_hides_custom_themes_but_keeps_builtins(tmp_path):
    """Port of `resource-loader.ts:501`'s `noThemes` branch.

    This port has no package or extension theme discovery, so the discovered
    set is the user's themes directory; built-ins ship inside the package and
    are never "discovered", so they survive exactly as upstream's do.
    """
    _write_custom_theme(tmp_path, "midnight")
    set_custom_theme_discovery_enabled(False)

    names = get_available_themes(custom_themes_dir=str(tmp_path))

    assert "midnight" not in names
    assert "dark" in names


def test_no_themes_also_refuses_to_load_a_custom_theme_by_name(tmp_path):
    """Hiding it from the selector is not enough.

    A saved `theme` setting naming a custom theme would otherwise still be
    loaded at startup, so the flag would suppress the list and not the effect.
    """
    _write_custom_theme(tmp_path, "midnight")
    set_custom_theme_discovery_enabled(False)

    with pytest.raises(ValueError, match="Theme not found: midnight"):
        load_theme("midnight", custom_themes_dir=str(tmp_path))


def test_export_reports_that_it_is_not_ported(capsys):
    """`--export` used to parse, do nothing, and let the run continue.

    The user then saw "No prompt given." -- an error about something they never
    asked for, with no hint that `--export` had been dropped.
    """
    code = cli.main(["--export", "/tmp/does-not-matter.jsonl"])

    assert code == 1
    stderr = capsys.readouterr().err
    assert "--export is not available in this Python port" in stderr
    assert "No prompt given" not in stderr
