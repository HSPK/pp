"""The system prompt must point the model at pi's own documentation.

This is the entire mechanism behind pi explaining its own features, and it is
easy to mistake for something cleverer. There is no retrieval index and no
embedded copy of the manual: `system-prompt.ts:131-138` writes three absolute
paths into the prompt -- README.md, `docs/`, `examples/` -- plus a map from
topic to filename, and the model opens those real files with the ordinary read
tool. If the paragraph is missing, or the paths do not resolve, the capability
silently disappears; nothing raises and every test that only checks tools and
guidelines still passes.

So these tests pin both halves: that the paths resolve to files that actually
exist, and that the paragraph naming them reaches the prompt.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pi_coding_agent.core.config import (
    get_docs_path,
    get_examples_path,
    get_package_dir,
    get_readme_path,
)
from pi_coding_agent.core.system_prompt import BuildSystemPromptOptions, build_system_prompt


def _prompt() -> str:
    return build_system_prompt(BuildSystemPromptOptions(cwd="/tmp"))


def test_docs_directory_ships_with_the_package():
    """A path in the prompt that resolves to nothing is worse than no path.

    The model would read the failure as "this install is broken" rather than
    "I should answer from what I know".
    """
    assert os.path.isdir(get_docs_path()), get_docs_path()


def test_documented_topic_files_all_exist():
    """The prompt names specific filenames; every one must be readable.

    The topic->file map is what stops the model guessing filenames, so a name
    in that list that isn't on disk actively misleads it.
    """
    docs = Path(get_docs_path())
    referenced = [
        "extensions.md",
        "themes.md",
        "skills.md",
        "prompt-templates.md",
        "tui.md",
        "keybindings.md",
        "sdk.md",
        "custom-provider.md",
        "models.md",
        "packages.md",
        "environment-variables.md",
    ]
    missing = [name for name in referenced if not (docs / name).is_file()]
    assert missing == [], f"prompt names docs that do not exist: {missing}"


def test_prompt_includes_the_documentation_paths():
    prompt = _prompt()
    assert "Pi documentation" in prompt
    assert get_docs_path() in prompt
    assert get_examples_path() in prompt


def test_prompt_maps_topics_to_doc_files():
    """Without the topic map the model has paths but no index."""
    prompt = _prompt()
    for reference in ("docs/extensions.md", "docs/tui.md", "docs/custom-provider.md"):
        assert reference in prompt, reference


def test_prompt_omits_the_section_when_docs_are_absent(monkeypatch, tmp_path):
    """An installation without docs must say nothing rather than point at air.

    `PI_PACKAGE_DIR` is the supported override (port of the same variable in
    `config.ts`), so pointing it at an empty directory is the honest way to
    simulate a stripped install.
    """
    monkeypatch.setenv("PI_PACKAGE_DIR", str(tmp_path))
    prompt = _prompt()
    assert "Pi documentation" not in prompt
    assert str(tmp_path) not in prompt


def test_package_dir_honours_the_environment_override(tmp_path):
    override = str(tmp_path)
    assert get_package_dir(env={"PI_PACKAGE_DIR": override}) == override
    assert get_readme_path(env={"PI_PACKAGE_DIR": override}) == str(tmp_path / "README.md")
    assert get_docs_path(env={"PI_PACKAGE_DIR": override}) == str(tmp_path / "docs")


@pytest.mark.parametrize("helper", [get_readme_path, get_docs_path, get_examples_path])
def test_paths_are_absolute(helper):
    """The prompt tells the model to resolve these independently of the cwd."""
    assert os.path.isabs(helper())
