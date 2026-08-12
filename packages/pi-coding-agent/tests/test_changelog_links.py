"""Python port of `packages/coding-agent/test/changelog.test.ts`."""

from __future__ import annotations

from pi_coding_agent.utils.changelog import ChangelogEntry, normalize_changelog_links

ENTRY = ChangelogEntry(major=0, minor=79, patch=0, content="")


def test_rewrites_package_relative_links_to_tag_pinned_source_links() -> None:
    markdown = "\n".join(
        [
            "[Project Trust](README.md#project-trust)",
            "[Extensions](docs/extensions.md#project_trust)",
            "[Examples](examples/extensions/)",
            "[Root README](../../README.md#supply-chain-hardening)",
        ]
    )

    assert normalize_changelog_links(markdown, ENTRY) == "\n".join(
        [
            "[Project Trust](https://github.com/earendil-works/pi/blob/v0.79.0/packages/coding-agent/README.md#project-trust)",
            "[Extensions](https://github.com/earendil-works/pi/blob/v0.79.0/packages/coding-agent/docs/extensions.md#project_trust)",
            "[Examples](https://github.com/earendil-works/pi/tree/v0.79.0/packages/coding-agent/examples/extensions/)",
            "[Root README](https://github.com/earendil-works/pi/blob/v0.79.0/README.md#supply-chain-hardening)",
        ]
    )


def test_canonicalizes_old_repository_urls_without_changing_external_links() -> None:
    markdown = "\n".join(
        [
            "[#5167](https://github.com/earendil-works/pi-mono/pull/5167)",
            "[#4163](https://github.com/badlogic/pi-mono/issues/4163)",
            "[Agent README](https://github.com/badlogic/pi-mono/blob/main/packages/agent/README.md)",
            "[External](https://example.com/docs)",
            "[Local anchor](#settings)",
        ]
    )

    assert normalize_changelog_links(markdown, "0.79.0") == "\n".join(
        [
            "[#5167](https://github.com/earendil-works/pi/pull/5167)",
            "[#4163](https://github.com/earendil-works/pi/issues/4163)",
            "[Agent README](https://github.com/earendil-works/pi/blob/v0.79.0/packages/agent/README.md)",
            "[External](https://example.com/docs)",
            "[Local anchor](#settings)",
        ]
    )
