"""Tests for pi_coding_agent.core.skills.

Ported from packages/coding-agent/test/skills.test.ts. `skills.py` is a thin
facade re-exporting `resource_loader.py`'s already-complete skills logic (see
its module docstring), so these tests exercise the facade's public surface
directly (rather than through `resource_loader.py`'s own `ResourceLoader`
tests in `test_resource_loader.py`) to prove the re-export path itself works
end to end, plus the `formatSkillsForPrompt`/collision-handling cases that
`test_resource_loader.py` does not already cover.
"""

from pi_coding_agent.core import resource_loader as _resource_loader_module
from pi_coding_agent.core.resource_loader import create_synthetic_source_info
from pi_coding_agent.core.skills import (
    LoadSkillsResult,
    Skill,
    format_skills_for_prompt,
    load_skills,
    load_skills_from_dir,
)

SKILL_MD = """---
name: {name}
description: {description}
---
{content}"""


def _write_skill(directory, name: str | None, description: str | None, content: str = "Skill body.") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frontmatter_lines = ["---"]
    if name is not None:
        frontmatter_lines.append(f"name: {name}")
    if description is not None:
        frontmatter_lines.append(f"description: {description}")
    frontmatter_lines.append("---")
    text = "\n".join(frontmatter_lines) + f"\n{content}"
    (directory / "SKILL.md").write_text(text)


def _make_test_skill(
    name: str, description: str, file_path: str, base_dir: str, disable_model_invocation=False
) -> Skill:
    return Skill(
        name=name,
        description=description,
        file_path=file_path,
        base_dir=base_dir,
        source_info=create_synthetic_source_info(file_path, "test"),
        disable_model_invocation=disable_model_invocation,
    )


# ---------------------------------------------------------------------------
# Facade re-export identity: `skills.py` must not fork the implementation.
# ---------------------------------------------------------------------------


def test_facade_reexports_the_same_objects_as_resource_loader():
    assert Skill is _resource_loader_module.Skill
    assert LoadSkillsResult is _resource_loader_module.LoadSkillsResult
    assert load_skills_from_dir is _resource_loader_module.load_skills_from_dir
    assert load_skills is _resource_loader_module.load_skills
    assert format_skills_for_prompt is _resource_loader_module.format_skills_for_prompt


# ---------------------------------------------------------------------------
# load_skills_from_dir()
# ---------------------------------------------------------------------------


def test_loads_a_valid_skill(tmp_path):
    skill_dir = tmp_path / "valid-skill"
    _write_skill(skill_dir, "valid-skill", "A valid skill for testing purposes.")

    result = load_skills_from_dir(str(skill_dir), "test")

    assert len(result.skills) == 1
    assert result.skills[0].name == "valid-skill"
    assert result.skills[0].description == "A valid skill for testing purposes."
    assert result.skills[0].source_info.source == "test"
    assert result.diagnostics == []


def test_warns_when_name_contains_invalid_characters(tmp_path):
    skill_dir = tmp_path / "invalid-name-chars"
    _write_skill(skill_dir, "Invalid_Name!", "A skill with an invalid name.")

    result = load_skills_from_dir(str(skill_dir), "test")

    assert len(result.skills) == 1
    assert any("invalid characters" in d.message for d in result.diagnostics)


def test_warns_and_skips_skill_when_description_is_missing(tmp_path):
    skill_dir = tmp_path / "missing-description"
    _write_skill(skill_dir, "missing-description", None)

    result = load_skills_from_dir(str(skill_dir), "test")

    assert result.skills == []
    assert any("description is required" in d.message for d in result.diagnostics)


def test_parses_disable_model_invocation_frontmatter_field(tmp_path):
    skill_dir = tmp_path / "disable-model-invocation"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: disable-model-invocation\ndescription: A hidden skill.\ndisable-model-invocation: true\n---\nBody."
    )

    result = load_skills_from_dir(str(skill_dir), "test")

    assert len(result.skills) == 1
    assert result.skills[0].name == "disable-model-invocation"
    assert result.skills[0].disable_model_invocation is True
    assert not any("unknown frontmatter field" in d.message for d in result.diagnostics)


def test_defaults_disable_model_invocation_to_false(tmp_path):
    skill_dir = tmp_path / "valid-skill"
    _write_skill(skill_dir, "valid-skill", "A valid skill.")

    result = load_skills_from_dir(str(skill_dir), "test")

    assert len(result.skills) == 1
    assert result.skills[0].disable_model_invocation is False


def test_returns_empty_for_non_existent_directory(tmp_path):
    result = load_skills_from_dir(str(tmp_path / "does-not-exist"), "test")

    assert result.skills == []
    assert result.diagnostics == []


def test_allows_names_that_do_not_match_the_parent_directory(tmp_path):
    skill_dir = tmp_path / "name-mismatch"
    _write_skill(skill_dir, "different-name", "A skill with a name that doesn't match the directory.")

    result = load_skills_from_dir(str(skill_dir), "test")

    assert len(result.skills) == 1
    assert result.skills[0].name == "different-name"
    assert not any("does not match parent directory" in d.message for d in result.diagnostics)


def test_warns_when_name_exceeds_64_characters(tmp_path):
    skill_dir = tmp_path / "long-name"
    long_name = "this-is-a-very-long-skill-name-that-exceeds-the-sixty-four-character-limit-set-by-the-standard"
    _write_skill(skill_dir, long_name, "A skill with a name that exceeds 64 characters.")

    result = load_skills_from_dir(str(skill_dir), "test")

    assert len(result.skills) == 1
    assert any("exceeds 64 characters" in d.message for d in result.diagnostics)


def test_warns_when_name_contains_consecutive_hyphens(tmp_path):
    skill_dir = tmp_path / "consecutive-hyphens"
    _write_skill(skill_dir, "bad--name", "A skill with consecutive hyphens in the name.")

    result = load_skills_from_dir(str(skill_dir), "test")

    assert len(result.skills) == 1
    assert any("consecutive hyphens" in d.message for d in result.diagnostics)


def test_ignores_unknown_frontmatter_fields(tmp_path):
    skill_dir = tmp_path / "unknown-field"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: unknown-field\ndescription: A skill with an unknown frontmatter field.\n"
        "author: someone\nversion: 1.0\n---\n\n# Unknown Field\n"
    )

    result = load_skills_from_dir(str(skill_dir), "test")

    assert len(result.skills) == 1
    assert result.diagnostics == []


def test_loads_nested_skills_recursively(tmp_path):
    _write_skill(tmp_path / "nested" / "child-skill", "child-skill", "A nested skill in a subdirectory.")

    result = load_skills_from_dir(str(tmp_path / "nested"), "test")

    assert len(result.skills) == 1
    assert result.skills[0].name == "child-skill"
    assert result.diagnostics == []


def test_prefers_a_directorys_root_skill_md_over_nested_skill_md_files(tmp_path):
    root = tmp_path / "root-skill-preferred"
    (root / "nested-child").mkdir(parents=True)
    (root / "SKILL.md").write_text("---\ndescription: Root skill should win.\n---\n")
    (root / "nested-child" / "SKILL.md").write_text("---\ndescription: Nested skill should be ignored.\n---\n")

    result = load_skills_from_dir(str(root), "test")

    assert len(result.skills) == 1
    assert result.skills[0].name == "root-skill-preferred"
    assert result.skills[0].description == "Root skill should win."
    assert result.diagnostics == []


def test_skips_files_without_frontmatter(tmp_path):
    skill_dir = tmp_path / "no-frontmatter"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# No Frontmatter\n\nThis skill has no YAML frontmatter at all.\n")

    result = load_skills_from_dir(str(skill_dir), "test")

    assert result.skills == []
    assert any("description is required" in d.message for d in result.diagnostics)


def test_warns_and_skips_skill_when_yaml_frontmatter_is_invalid(tmp_path):
    skill_dir = tmp_path / "invalid-yaml"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: invalid-yaml\ndescription: [unclosed bracket\n---\n\n# Invalid YAML Skill\n"
    )

    result = load_skills_from_dir(str(skill_dir), "test")

    assert result.skills == []
    # TypeScript asserts the message contains "at line" -- the `yaml` package
    # phrases it "... at line 2, column 31". PyYAML localizes the same parse
    # error as 'in "<unicode string>", line 2, column 14'. The pinned behavior
    # is that the diagnostic points at the offending line, not the wording of
    # whichever YAML library produced it.
    assert any("line 2" in d.message for d in result.diagnostics)


def test_preserves_multiline_descriptions_from_yaml(tmp_path):
    skill_dir = tmp_path / "multiline-description"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: multiline-description\ndescription: |\n  This is a multiline description.\n"
        "  It spans multiple lines.\n  And should be normalized.\n---\n\n# Multiline Description Skill\n"
    )

    result = load_skills_from_dir(str(skill_dir), "test")

    assert len(result.skills) == 1
    assert "\n" in result.skills[0].description
    assert "This is a multiline description." in result.skills[0].description
    assert result.diagnostics == []


def test_uses_parent_directory_name_when_name_not_in_frontmatter(tmp_path):
    skill_dir = tmp_path / "valid-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\ndescription: A valid skill for testing purposes.\n---\nBody.\n")

    result = load_skills_from_dir(str(skill_dir), "test")

    assert len(result.skills) == 1
    assert result.skills[0].name == "valid-skill"


def test_loads_all_skills_from_a_directory_of_skill_directories(tmp_path):
    _write_skill(tmp_path / "valid-skill", "valid-skill", "A valid skill.")
    _write_skill(tmp_path / "name-mismatch", "different-name", "A mismatched name.")
    _write_skill(tmp_path / "invalid-name-chars", "Invalid_Name!", "An invalid name.")
    _write_skill(tmp_path / "long-name", "x" * 70, "A long name.")
    _write_skill(tmp_path / "unknown-field", "unknown-field", "An unknown field.")
    _write_skill(tmp_path / "nested" / "child-skill", "child-skill", "A nested skill.")
    _write_skill(tmp_path / "consecutive-hyphens", "bad--name", "Consecutive hyphens.")
    _write_skill(tmp_path / "missing-description", "missing-description", None)

    result = load_skills_from_dir(str(tmp_path), "test")

    assert len(result.skills) >= 6
    assert "missing-description" not in {skill.name for skill in result.skills}


# ---------------------------------------------------------------------------
# format_skills_for_prompt()
# ---------------------------------------------------------------------------


def test_format_skills_for_prompt_returns_empty_string_for_no_skills():
    assert format_skills_for_prompt([]) == ""


def test_format_skills_for_prompt_formats_skills_as_xml():
    skills = [_make_test_skill("test-skill", "A test skill.", "/path/to/skill/SKILL.md", "/path/to/skill")]

    result = format_skills_for_prompt(skills)

    assert "<available_skills>" in result
    assert "</available_skills>" in result
    assert "<skill>" in result
    assert "<name>test-skill</name>" in result
    assert "<description>A test skill.</description>" in result
    assert "<location>/path/to/skill/SKILL.md</location>" in result


def test_format_skills_for_prompt_includes_intro_text_before_xml():
    skills = [_make_test_skill("test-skill", "A test skill.", "/path/to/skill/SKILL.md", "/path/to/skill")]

    result = format_skills_for_prompt(skills)
    intro_text = result[: result.index("<available_skills>")]

    assert "The following skills provide specialized instructions" in intro_text
    assert "Use the read tool to load a skill's file" in intro_text


def test_format_skills_for_prompt_escapes_xml_special_characters():
    skills = [
        _make_test_skill(
            "test-skill",
            'A skill with <special> & "characters".',
            "/path/to/skill/SKILL.md",
            "/path/to/skill",
        )
    ]

    result = format_skills_for_prompt(skills)

    assert "&lt;special&gt;" in result
    assert "&amp;" in result
    assert "&quot;characters&quot;" in result


def test_format_skills_for_prompt_formats_multiple_skills():
    skills = [
        _make_test_skill("skill-one", "First skill.", "/path/one/SKILL.md", "/path/one"),
        _make_test_skill("skill-two", "Second skill.", "/path/two/SKILL.md", "/path/two"),
    ]

    result = format_skills_for_prompt(skills)

    assert "<name>skill-one</name>" in result
    assert "<name>skill-two</name>" in result
    assert result.count("<skill>") == 2


def test_format_skills_for_prompt_excludes_disabled_skills():
    skills = [
        _make_test_skill("visible-skill", "A visible skill.", "/path/visible/SKILL.md", "/path/visible"),
        _make_test_skill(
            "hidden-skill",
            "A hidden skill.",
            "/path/hidden/SKILL.md",
            "/path/hidden",
            disable_model_invocation=True,
        ),
    ]

    result = format_skills_for_prompt(skills)

    assert "<name>visible-skill</name>" in result
    assert "<name>hidden-skill</name>" not in result
    assert result.count("<skill>") == 1


def test_format_skills_for_prompt_returns_empty_when_all_disabled():
    skills = [
        _make_test_skill(
            "hidden-skill",
            "A hidden skill.",
            "/path/hidden/SKILL.md",
            "/path/hidden",
            disable_model_invocation=True,
        )
    ]

    assert format_skills_for_prompt(skills) == ""


# ---------------------------------------------------------------------------
# load_skills() with explicit skill_paths
# ---------------------------------------------------------------------------


def test_load_skills_loads_from_explicit_skill_paths(tmp_path):
    explicit_dir = tmp_path / "explicit-skills"
    _write_skill(explicit_dir / "my-skill", "my-skill", "An explicitly-pathed skill.")

    result = load_skills(
        cwd=str(tmp_path / "project"),
        agent_dir=str(tmp_path / "agent"),
        skill_paths=[str(explicit_dir)],
        # TS passes `includeDefaults: true` against empty agent/cwd fixture
        # dirs, so this also pins that the default user/project scans
        # contribute nothing (and no diagnostics) when those dirs are absent.
        include_defaults=True,
    )

    assert len(result.skills) == 1
    assert result.skills[0].name == "my-skill"
    assert result.skills[0].source_info.scope == "temporary"
    assert result.diagnostics == []


def test_load_skills_warns_when_skill_path_does_not_exist(tmp_path):
    result = load_skills(
        cwd=str(tmp_path / "project"),
        agent_dir=str(tmp_path / "agent"),
        skill_paths=[str(tmp_path / "does-not-exist")],
        include_defaults=True,
    )

    assert result.skills == []
    assert any("does not exist" in d.message for d in result.diagnostics)


def test_load_skills_expands_tilde_in_skill_paths(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home_skills_dir = home / ".pi" / "agent" / "skills" / "home-skill"
    _write_skill(home_skills_dir, "home-skill", "A skill under the home directory.")
    monkeypatch.setenv("HOME", str(home))

    with_tilde = load_skills(
        cwd=str(tmp_path / "project"),
        agent_dir=str(tmp_path / "agent"),
        skill_paths=["~/.pi/agent/skills"],
        include_defaults=True,
    )
    without_tilde = load_skills(
        cwd=str(tmp_path / "project"),
        agent_dir=str(tmp_path / "agent"),
        skill_paths=[str(home / ".pi" / "agent" / "skills")],
        include_defaults=True,
    )

    assert [skill.name for skill in with_tilde.skills] == ["home-skill"]
    assert [skill.name for skill in with_tilde.skills] == [skill.name for skill in without_tilde.skills]


# ---------------------------------------------------------------------------
# Collision handling: user (agent_dir) skills win over project (cwd) skills.
# ---------------------------------------------------------------------------


def test_load_skills_detects_name_collisions_and_keeps_first_loaded(tmp_path):
    agent_dir = tmp_path / "agent"
    cwd = tmp_path / "project"
    _write_skill(agent_dir / "skills" / "shared", "shared", "The user-level version.")
    _write_skill(cwd / ".pi" / "skills" / "shared", "shared", "The project-level version.")

    result = load_skills(cwd=str(cwd), agent_dir=str(agent_dir), skill_paths=[], include_defaults=True)

    shared_skills = [s for s in result.skills if s.name == "shared"]
    assert len(shared_skills) == 1
    assert shared_skills[0].description == "The user-level version."
    # TS builds the diagnostic by hand; the port's real one must carry the same
    # winner/loser pair, so both paths stay recoverable from the diagnostic.
    collisions = [d for d in result.diagnostics if d.type == "collision"]
    assert len(collisions) == 1
    assert collisions[0].message == 'name "shared" collision'
    assert collisions[0].path == str(cwd / ".pi" / "skills" / "shared" / "SKILL.md")
    assert collisions[0].collision is not None
    assert collisions[0].collision.resource_type == "skill"
    assert collisions[0].collision.name == "shared"
    assert collisions[0].collision.winner_path == str(agent_dir / "skills" / "shared" / "SKILL.md")
    assert collisions[0].collision.loser_path == str(cwd / ".pi" / "skills" / "shared" / "SKILL.md")
