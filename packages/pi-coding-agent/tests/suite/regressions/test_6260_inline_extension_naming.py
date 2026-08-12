"""Python port of `packages/coding-agent/test/suite/regressions/6260-inline-extension-naming.test.ts`.

The TypeScript test drives `DefaultResourceLoader({ extensionFactories })`.
This port's `ResourceLoader` deliberately does not load extensions (see its
module docstring); the equivalent code path lives in
`core/extensions/loader.py::load_extension_factories`, which is what
`discover_and_load_extensions` calls for inline factories. The assertions --
`<inline:N>` for bare factories, `<inline:name>` for named wrappers, `hidden`
preserved, and stable numbering when the two are mixed -- are unchanged.
"""

from __future__ import annotations

from pathlib import Path

from pi_coding_agent.core.extensions.loader import (
    ExtensionAPI,
    NamedInlineExtension,
    discover_and_load_extensions,
    load_extension_factories,
)


def _noop(pi: ExtensionAPI) -> None:
    return None


async def test_displays_bare_factories_as_inline_numbers(tmp_path: Path) -> None:
    result = await load_extension_factories([_noop, _noop], str(tmp_path))

    assert len(result.extensions) == 2
    assert result.extensions[0].path == "<inline:1>"
    assert result.extensions[1].path == "<inline:2>"


async def test_displays_named_wrappers_as_inline_name(tmp_path: Path) -> None:
    result = await load_extension_factories(
        [
            NamedInlineExtension(name="my-provider", factory=_noop),
            NamedInlineExtension(name="my-commands", factory=_noop),
        ],
        str(tmp_path),
    )

    assert len(result.extensions) == 2
    assert result.extensions[0].path == "<inline:my-provider>"
    assert result.extensions[1].path == "<inline:my-commands>"


async def test_preserves_hidden_state_for_named_factories(tmp_path: Path) -> None:
    result = await load_extension_factories(
        [NamedInlineExtension(name="built-in", factory=_noop, hidden=True)],
        str(tmp_path),
    )

    assert len(result.extensions) == 1
    assert result.extensions[0].path == "<inline:built-in>"
    assert result.extensions[0].hidden is True


async def test_supports_mixed_bare_and_named_factories(tmp_path: Path) -> None:
    result = await load_extension_factories(
        [_noop, NamedInlineExtension(name="named-ext", factory=_noop), _noop],
        str(tmp_path),
    )

    assert len(result.extensions) == 3
    assert result.extensions[0].path == "<inline:1>"
    assert result.extensions[1].path == "<inline:named-ext>"
    assert result.extensions[2].path == "<inline:3>"


async def test_discovery_appends_inline_factories(tmp_path: Path) -> None:
    """Not in TS: pins that the discovery entry point plumbs factories through."""
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    cwd.mkdir()
    agent_dir.mkdir()

    result = await discover_and_load_extensions(
        [],
        str(cwd),
        agent_dir=str(agent_dir),
        extension_factories=[NamedInlineExtension(name="built-in", factory=_noop, hidden=True)],
    )

    assert [extension.path for extension in result.extensions] == ["<inline:built-in>"]
    assert result.extensions[0].hidden is True
