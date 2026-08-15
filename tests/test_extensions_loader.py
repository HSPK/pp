"""Tests for pi_coding_agent.core.extensions.loader.

Ported from packages/coding-agent/test/extensions-discovery.test.ts. Cases
relying on TypeScript/`jiti`-specific behavior with no Python equivalent are
skipped: the two `jiti` alias cases (`loads the coding-agent entrypoint
without rewriting pi-ai provider subpaths`, `keeps the type-only pi-ai OAuth
compatibility barrel resolvable`), `resolves dependencies from extension's own
node_modules` (Python has no per-extension dependency isolation equivalent),
and `registers message and entry
renderers`/`loads extension with shortcuts`/`loads extension with flags`
(types.py's module docstring -- no `pi_tui` consumer).

TS's `.js`/`index.js` and `index.ts`-over-`index.js` variants have no direct
counterpart (Python has one extension suffix, `.py`); their load-bearing half
is ported as `test_ignores_direct_files_that_are_not_py`, and the
precedence half as `test_pi_json_takes_precedence_over_init_py`.

`package.json`'s nested `"pi"` field maps to a top-level `pi.json` here (see
`core/pi_manifest.py`), so the manifest-discovery cases are ported against
`pi.json`.

All extension fixtures are written into `tmp_path` as real `.py` files and
loaded through the real loader -- nothing is ever loaded from outside
`tmp_path`.
"""

import json
import os

import pytest

from pi_coding_agent.core.extensions.loader import (
    discover_and_load_extensions,
    discover_extensions_in_dir,
    load_extensions,
)

COMMAND_EXTENSION = """
def pi_extension(pi):
    async def _handler(args, ctx):
        return None

    pi.register_command("test", handler=_handler)
"""


def _tool_extension(tool_name: str) -> str:
    return f"""
from pi_ai.types import TextContent
from pi_coding_agent.core.extensions.types import ToolDefinition


async def _execute(tool_call_id, params, signal, on_update, ctx):
    from pi_agent.types import AgentToolResult

    return AgentToolResult(content=[TextContent(text="ok")])


def pi_extension(pi):
    pi.register_tool(
        ToolDefinition(
            name="{tool_name}",
            label="{tool_name}",
            description="Test tool",
            execute=_execute,
        )
    )
"""


def _write(path, content: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path


async def test_discovers_direct_py_files_in_extensions_dir(tmp_path):
    ext_dir = tmp_path / "extensions"
    _write(str(ext_dir / "foo.py"), COMMAND_EXTENSION)
    _write(str(ext_dir / "bar.py"), COMMAND_EXTENSION)

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 2
    assert sorted(os.path.basename(e.path) for e in result.extensions) == ["bar.py", "foo.py"]


async def test_ignores_direct_files_that_are_not_py(tmp_path):
    """Counterpart of TS's three `.js`/`index.js` discovery cases.

    TypeScript loads `.ts` *and* `.js` and pins `index.ts` winning over
    `index.js`; Python has one extension suffix, so the load-bearing half of
    those cases here is the negative: everything else in `extensions/` is left
    alone. Without this, loosening `_is_extension_file` would go unnoticed.
    """
    ext_dir = tmp_path / "extensions"
    _write(str(ext_dir / "real.py"), COMMAND_EXTENSION)
    _write(str(ext_dir / "notes.txt"), "not an extension")
    _write(str(ext_dir / "extension.js"), "export default () => {}")
    _write(str(ext_dir / "extension.ts"), "export default () => {}")
    _write(str(ext_dir / "cached.pyc"), "")

    assert [os.path.basename(path) for path in discover_extensions_in_dir(str(ext_dir))] == ["real.py"]

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert [os.path.basename(e.path) for e in result.extensions] == ["real.py"]


async def test_discovers_subdirectory_with_init_py(tmp_path):
    ext_dir = tmp_path / "extensions" / "sub"
    _write(str(ext_dir / "__init__.py"), COMMAND_EXTENSION)

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 1
    assert result.extensions[0].path.endswith("__init__.py")


async def test_ignores_subdirectory_without_init_py(tmp_path):
    ext_dir = tmp_path / "extensions"
    sub = ext_dir / "sub"
    sub.mkdir(parents=True)
    (sub / "helper.py").write_text(COMMAND_EXTENSION)
    (sub / "utils.py").write_text(COMMAND_EXTENSION)

    assert discover_extensions_in_dir(str(ext_dir)) == []

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert result.extensions == []


async def test_does_not_recurse_beyond_one_level(tmp_path):
    ext_dir = tmp_path / "extensions"
    nested = ext_dir / "container" / "nested"
    nested.mkdir(parents=True)
    (nested / "__init__.py").write_text(COMMAND_EXTENSION)
    # `container` itself has no `__init__.py`, only a nested subdirectory does.

    assert discover_extensions_in_dir(str(ext_dir)) == []

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert result.extensions == []


async def test_loads_extension_and_registers_command(tmp_path):
    ext_dir = tmp_path / "extensions"
    _write(str(ext_dir / "with_command.py"), COMMAND_EXTENSION)

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "test" in result.extensions[0].commands


async def test_loads_extension_and_registers_tool(tmp_path):
    ext_dir = tmp_path / "extensions"
    _write(str(ext_dir / "with_tool.py"), _tool_extension("my_tool"))

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "my_tool" in result.extensions[0].tools


async def test_reports_errors_for_invalid_extension_code(tmp_path):
    ext_dir = tmp_path / "extensions"
    _write(str(ext_dir / "invalid.py"), "this is not valid python !!! syntax {{{")

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert len(result.errors) == 1
    assert "invalid.py" in result.errors[0]["path"]
    assert result.extensions == []


async def test_handles_explicitly_configured_paths(tmp_path):
    custom_path = tmp_path / "custom-location" / "my_ext.py"
    _write(str(custom_path), COMMAND_EXTENSION)

    result = await discover_and_load_extensions(
        [str(custom_path)], str(tmp_path), agent_dir=str(tmp_path / "agent-dir-does-not-exist")
    )

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "my_ext.py" in result.extensions[0].path


async def test_reports_error_when_extension_raises_during_initialization(tmp_path):
    ext_dir = tmp_path / "extensions"
    _write(
        str(ext_dir / "throws.py"),
        """
def pi_extension(pi):
    raise RuntimeError("Initialization failed!")
""",
    )

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert len(result.errors) == 1
    assert "Initialization failed!" in result.errors[0]["error"]
    assert result.extensions == []


async def test_reports_error_when_extension_has_no_entry_point(tmp_path):
    ext_dir = tmp_path / "extensions"
    _write(
        str(ext_dir / "no_entry_point.py"),
        """
def not_the_entry_point(pi):
    pass
""",
    )

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert len(result.errors) == 1
    # TypeScript asserts the error contains "does not export a valid factory
    # function". The port keeps that wording but names its entry point instead
    # of JS's default export, so the whole message is pinned here with that one
    # substitution -- matching only "pi_extension" would let the rest of the
    # sentence drift.
    assert result.errors[0]["error"] == (
        f"Extension does not export a valid `pi_extension` callable: {ext_dir / 'no_entry_point.py'}"
    )
    assert result.errors[0]["path"] == str(ext_dir / "no_entry_point.py")
    assert result.extensions == []


async def test_allows_multiple_extensions_to_register_different_tools(tmp_path):
    ext_dir = tmp_path / "extensions"
    _write(str(ext_dir / "tool_a.py"), _tool_extension("tool_a"))
    _write(str(ext_dir / "tool_b.py"), _tool_extension("tool_b"))

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 2
    all_tools = {name for ext in result.extensions for name in ext.tools}
    assert all_tools == {"tool_a", "tool_b"}


async def test_loads_extension_with_event_handlers(tmp_path):
    ext_dir = tmp_path / "extensions"
    _write(
        str(ext_dir / "with_handlers.py"),
        """
def pi_extension(pi):
    async def on_agent_start(event, ctx):
        pass

    async def on_tool_call(event, ctx):
        return None

    async def on_agent_end(event, ctx):
        pass

    pi.on("agent_start", on_agent_start)
    pi.on("tool_call", on_tool_call)
    pi.on("agent_end", on_agent_end)
""",
    )

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 1
    handlers = result.extensions[0].handlers
    assert "agent_start" in handlers
    assert "tool_call" in handlers
    assert "agent_end" in handlers


async def test_load_extensions_only_loads_explicit_paths_without_discovery(tmp_path):
    ext_dir = tmp_path / "extensions"
    _write(str(ext_dir / "discovered.py"), _tool_extension("discovered"))

    explicit_path = tmp_path / "explicit.py"
    _write(str(explicit_path), _tool_extension("explicit"))

    result = await load_extensions([str(explicit_path)], str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "explicit" in result.extensions[0].tools
    assert "discovered" not in result.extensions[0].tools


async def test_load_extensions_with_no_paths_loads_nothing(tmp_path):
    ext_dir = tmp_path / "extensions"
    _write(str(ext_dir / "discovered.py"), COMMAND_EXTENSION)

    result = await load_extensions([], str(tmp_path))

    assert result.errors == []
    assert result.extensions == []


# ---------------------------------------------------------------------------
# Trust gating: project-local `.pi/extensions/` is only discovered when the
# project is trusted; the global/user `agent_dir/extensions/` directory and
# explicitly configured paths are always discovered regardless of trust.
# ---------------------------------------------------------------------------


async def test_untrusted_project_refuses_project_local_extensions(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    local_ext_dir = cwd / ".pi" / "extensions"
    global_ext_dir = agent_dir / "extensions"
    _write(str(local_ext_dir / "untrusted_local.py"), _tool_extension("untrusted_local_tool"))
    _write(str(global_ext_dir / "trusted_global.py"), _tool_extension("trusted_global_tool"))

    result = await discover_and_load_extensions([], str(cwd), project_trusted=False, agent_dir=str(agent_dir))

    assert result.errors == []
    all_tools = {name for ext in result.extensions for name in ext.tools}
    assert "untrusted_local_tool" not in all_tools
    assert "trusted_global_tool" in all_tools


async def test_trusted_project_loads_project_local_extensions(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    local_ext_dir = cwd / ".pi" / "extensions"
    global_ext_dir = agent_dir / "extensions"
    _write(str(local_ext_dir / "trusted_local.py"), _tool_extension("trusted_local_tool"))
    _write(str(global_ext_dir / "trusted_global.py"), _tool_extension("trusted_global_tool"))

    result = await discover_and_load_extensions([], str(cwd), project_trusted=True, agent_dir=str(agent_dir))

    assert result.errors == []
    all_tools = {name for ext in result.extensions for name in ext.tools}
    assert "trusted_local_tool" in all_tools
    assert "trusted_global_tool" in all_tools


async def test_untrusted_project_still_loads_explicitly_configured_paths(tmp_path):
    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent-empty"
    explicit_path = tmp_path / "explicit-dir" / "cli_provided.py"
    _write(str(explicit_path), _tool_extension("cli_provided_tool"))

    result = await discover_and_load_extensions(
        [str(explicit_path)], str(cwd), project_trusted=False, agent_dir=str(agent_dir)
    )

    assert result.errors == []
    all_tools = {name for ext in result.extensions for name in ext.tools}
    assert "cli_provided_tool" in all_tools


# ---------------------------------------------------------------------------
# pi.json directory manifests (TypeScript's package.json "pi" field)
# ---------------------------------------------------------------------------


async def test_discovers_subdirectory_with_pi_json_manifest(tmp_path):
    subdir = tmp_path / "extensions" / "my-package"
    _write(str(subdir / "src" / "main.py"), COMMAND_EXTENSION)
    _write(str(subdir / "pi.json"), json.dumps({"name": "my-package", "extensions": ["./src/main.py"]}))

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 1
    assert "src" in result.extensions[0].path
    assert result.extensions[0].path.endswith("main.py")


async def test_pi_json_entries_with_leading_tilde_stay_package_relative(tmp_path):
    subdir = tmp_path / "extensions" / "tilde-package"
    direct_path = _write(str(subdir / "~entry.py"), COMMAND_EXTENSION)
    slash_path = _write(str(subdir / "~" / "entry.py"), COMMAND_EXTENSION)
    _write(
        str(subdir / "pi.json"),
        json.dumps({"name": "tilde-package", "extensions": ["~entry.py", "~/entry.py"]}),
    )

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert sorted(e.path for e in result.extensions) == sorted([direct_path, slash_path])


async def test_pi_json_can_declare_multiple_extensions(tmp_path):
    subdir = tmp_path / "extensions" / "my-package"
    _write(str(subdir / "ext1.py"), COMMAND_EXTENSION)
    _write(str(subdir / "ext2.py"), COMMAND_EXTENSION)
    _write(str(subdir / "pi.json"), json.dumps({"name": "my-package", "extensions": ["./ext1.py", "./ext2.py"]}))

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 2


async def test_pi_json_takes_precedence_over_init_py(tmp_path):
    subdir = tmp_path / "extensions" / "my-package"
    _write(str(subdir / "__init__.py"), _tool_extension("from_index"))
    _write(str(subdir / "custom.py"), _tool_extension("from_custom"))
    _write(str(subdir / "pi.json"), json.dumps({"name": "my-package", "extensions": ["./custom.py"]}))

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 1
    assert result.extensions[0].path.endswith("custom.py")
    assert "from_custom" in result.extensions[0].tools
    assert "from_index" not in result.extensions[0].tools


async def test_pi_json_without_extensions_field_falls_back_to_init_py(tmp_path):
    subdir = tmp_path / "extensions" / "my-package"
    _write(str(subdir / "__init__.py"), COMMAND_EXTENSION)
    _write(str(subdir / "pi.json"), json.dumps({"name": "my-package", "version": "1.0.0"}))

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 1
    assert result.extensions[0].path.endswith("__init__.py")


async def test_skips_non_existent_paths_declared_in_pi_json(tmp_path):
    subdir = tmp_path / "extensions" / "my-package"
    _write(str(subdir / "exists.py"), COMMAND_EXTENSION)
    _write(str(subdir / "pi.json"), json.dumps({"extensions": ["./exists.py", "./missing.py"]}))

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 1
    assert result.extensions[0].path.endswith("exists.py")


async def test_handles_mixed_direct_files_and_subdirectories(tmp_path):
    ext_dir = tmp_path / "extensions"
    _write(str(ext_dir / "direct.py"), COMMAND_EXTENSION)
    _write(str(ext_dir / "with-index" / "__init__.py"), COMMAND_EXTENSION)
    _write(str(ext_dir / "with-manifest" / "entry.py"), COMMAND_EXTENSION)
    _write(str(ext_dir / "with-manifest" / "pi.json"), json.dumps({"extensions": ["./entry.py"]}))

    result = await discover_and_load_extensions([], str(tmp_path), agent_dir=str(tmp_path))

    assert result.errors == []
    assert len(result.extensions) == 3


# ---------------------------------------------------------------------------
# TypeScript cases with no Python counterpart. Recorded here rather than only
# in the module docstring so the omission is visible when the file is read or
# collected.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        'TS "loads the coding-agent entrypoint without rewriting pi-ai provider subpaths" and '
        '"keeps the type-only pi-ai OAuth compatibility barrel resolvable" both pin `jiti` alias '
        "configuration in loader.ts (import-specifier rewriting for bundled subpath exports). "
        "The Python loader uses `importlib` on a real file path and has no specifier-rewriting "
        "stage, so there is nothing to assert."
    )
)
def test_jiti_alias_configuration():
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "TS \"resolves dependencies from extension's own node_modules\" pins `jiti`'s per-extension "
        "module resolution root. Python extensions are imported into the host interpreter and "
        "resolve imports through the host `sys.path`; there is no per-extension dependency root."
    )
)
def test_resolves_dependencies_from_extensions_own_node_modules():
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        'TS "registers message and entry renderers" asserts `pi.registerMessageRenderer` / '
        "`pi.registerEntryRenderer` populate `extension.messageRenderers` / `entryRenderers`. "
        "Per-tool and per-message TUI renderers are omitted from this port (see the "
        "core/extensions/types.py module docstring), so `Extension` has no such fields."
    )
)
def test_registers_message_and_entry_renderers():
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        'TS "loads extension with shortcuts" asserts `extension.shortcuts.has("ctrl+t")` after '
        "`pi.registerShortcut`. Extension-registered keyboard shortcuts are omitted from this "
        "port (core/extensions/types.py); `ExtensionAPI` has no `register_shortcut`."
    )
)
def test_loads_extension_with_shortcuts():
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        'TS "loads extension with flags" asserts `extension.flags.has("my-flag")` after '
        "`pi.registerFlag`. Extension-registered CLI flags are omitted from this port "
        "(core/extensions/types.py: flag parsing belongs to the CLI layer here)."
    )
)
def test_loads_extension_with_flags():
    raise AssertionError("unreachable")


async def test_a_shared_extension_dir_symlinked_into_both_roots_loads_once(tmp_path):
    """Port of `DefaultResourceLoader.mergePaths`, which dedups on the canonical path.

    A user who keeps one extension directory and symlinks it into both
    `agent_dir/extensions` and `cwd/.pi/extensions` is the realistic setup this
    guards. Deduping on `abspath` alone leaves the two aliases distinct, so the
    same file loads twice: duplicate commands, plus a tool-name conflict the
    user cannot resolve because both sides are the same file.
    """
    shared = tmp_path / "shared"
    _write(str(shared / "dup.py"), COMMAND_EXTENSION)

    cwd = tmp_path / "project"
    agent_dir = tmp_path / "agent"
    (cwd / ".pi").mkdir(parents=True)
    agent_dir.mkdir(parents=True)
    os.symlink(shared, cwd / ".pi" / "extensions", target_is_directory=True)
    os.symlink(shared, agent_dir / "extensions", target_is_directory=True)

    result = await discover_and_load_extensions([], str(cwd), agent_dir=str(agent_dir))

    assert result.errors == []
    assert [os.path.basename(e.path) for e in result.extensions] == ["dup.py"]
    assert result.conflicts == []
