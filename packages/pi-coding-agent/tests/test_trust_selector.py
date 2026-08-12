"""Python port of `packages/coding-agent/test/trust-selector.test.ts`."""

from __future__ import annotations

import pytest
from pi_coding_agent.core.app_keybindings import KeybindingsManager
from pi_coding_agent.core.trust_manager import ProjectTrustStoreEntry, ProjectTrustUpdate
from pi_coding_agent.modes.interactive.components.trust_selector import (
    TrustSelection,
    TrustSelectorComponent,
)
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi
from pi_tui.keybindings import set_keybindings


@pytest.fixture(autouse=True)
def _theme_and_keybindings() -> None:
    init_theme("dark")
    set_keybindings(KeybindingsManager.create())


def _render(selector: TrustSelectorComponent) -> str:
    return strip_ansi("\n".join(selector.render(120)))


class TestTrustSelectorComponent:
    def test_marks_the_saved_trusted_decision(self) -> None:
        selector = TrustSelectorComponent(
            cwd="/project",
            saved_decision=ProjectTrustStoreEntry(path="/project", decision=True),
            project_trusted=True,
            on_select=lambda _selection: None,
            on_cancel=lambda: None,
        )

        output = _render(selector)

        assert "Saved decision: trusted (/project)" in output
        assert "Current session: trusted" in output
        assert "Trust ✓" in output
        assert "Do not trust ✓" not in output

    def test_selects_a_trust_decision(self) -> None:
        selections: list[TrustSelection] = []
        selector = TrustSelectorComponent(
            cwd="/project",
            saved_decision=None,
            project_trusted=False,
            on_select=selections.append,
            on_cancel=lambda: None,
        )

        selector.handle_input("\n")

        assert selections == [
            TrustSelection(trusted=True, updates=[ProjectTrustUpdate(path="/project", decision=True)])
        ]

    def test_labels_saved_ancestor_decisions_as_inherited(self) -> None:
        selector = TrustSelectorComponent(
            cwd="/parent/project/nested",
            saved_decision=ProjectTrustStoreEntry(path="/parent", decision=True),
            project_trusted=True,
            on_select=lambda _selection: None,
            on_cancel=lambda: None,
        )

        assert "Saved decision: trusted (inherited from /parent)" in _render(selector)

    def test_adds_a_trust_parent_option(self) -> None:
        selections: list[TrustSelection] = []
        selector = TrustSelectorComponent(
            cwd="/parent/project",
            saved_decision=ProjectTrustStoreEntry(path="/parent", decision=True),
            project_trusted=True,
            on_select=selections.append,
            on_cancel=lambda: None,
        )

        output = _render(selector)
        assert "Saved decision: trusted (inherited from /parent)" in output
        assert "Trust parent folder (/parent) ✓" in output

        selector.handle_input("\n")

        assert selections == [
            TrustSelection(
                trusted=True,
                updates=[
                    ProjectTrustUpdate(path="/parent", decision=True),
                    ProjectTrustUpdate(path="/parent/project", decision=None),
                ],
            )
        ]
