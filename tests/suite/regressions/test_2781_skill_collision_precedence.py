"""Python port of `packages/coding-agent/test/suite/regressions/2781-skill-collision-precedence.test.ts`.

The TypeScript test installs a fake npm-style package (a `package.json` with a
`pi.skills` manifest listed in `settings.json`'s `packages`) and asserts through
`DefaultResourceLoader`. This port's `ResourceLoader` deliberately does not
consume `settings.json` `packages` (package resolution lives in
`core/package_manager.py`, and the npm source kind is not ported at all -- see
its module docstring), so the package skill directory is handed to the loader
the way a resolved package would reach it here: as an
`additional_skill_paths` entry. The behaviour under test -- user and project
auto-discovered skills win a name collision against a package skill, and the
collision diagnostic names the package copy as the loser -- is unchanged.
"""

from __future__ import annotations

from pathlib import Path

from pi_coding_agent.core.resource_loader import ResourceLoader, ResourceLoaderOptions


def _create_package_with_skill(temp_dir: Path, name: str, description: str) -> Path:
    pkg_dir = temp_dir / f"fake-package-{name}"
    skill_dir = pkg_dir / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nPackage skill content",
        encoding="utf-8",
    )
    return pkg_dir / "skills" / name


def _create_user_skill(agent_dir: Path, name: str, description: str) -> Path:
    skill_dir = agent_dir / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nUser skill content",
        encoding="utf-8",
    )
    return skill_path


def _create_project_skill(cwd: Path, name: str, description: str) -> Path:
    skill_dir = cwd / ".pi" / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nProject skill content",
        encoding="utf-8",
    )
    return skill_path


def _make_loader(cwd: Path, agent_dir: Path, package_skill_dir: Path) -> ResourceLoader:
    loader = ResourceLoader(
        ResourceLoaderOptions(
            cwd=str(cwd),
            agent_dir=str(agent_dir),
            additional_skill_paths=[str(package_skill_dir)],
        )
    )
    loader.reload()
    return loader


def _dirs(tmp_path: Path) -> tuple[Path, Path]:
    agent_dir = tmp_path / "agent"
    cwd = tmp_path / "project"
    agent_dir.mkdir(parents=True, exist_ok=True)
    cwd.mkdir(parents=True, exist_ok=True)
    return agent_dir, cwd


def test_user_skill_overrides_package_skill(tmp_path: Path) -> None:
    agent_dir, cwd = _dirs(tmp_path)
    package_skill_dir = _create_package_with_skill(tmp_path, "web-fetch", "Package web-fetch skill")
    user_skill_path = _create_user_skill(agent_dir, "web-fetch", "User web-fetch override")

    loader = _make_loader(cwd, agent_dir, package_skill_dir)

    skills = loader.get_skills().skills
    web_fetch = next((s for s in skills if s.name == "web-fetch"), None)
    assert web_fetch is not None
    assert web_fetch.file_path == str(user_skill_path)
    assert web_fetch.description == "User web-fetch override"


def test_project_skill_overrides_package_skill(tmp_path: Path) -> None:
    agent_dir, cwd = _dirs(tmp_path)
    package_skill_dir = _create_package_with_skill(tmp_path, "web-fetch", "Package web-fetch skill")
    project_skill_path = _create_project_skill(cwd, "web-fetch", "Project web-fetch override")

    loader = _make_loader(cwd, agent_dir, package_skill_dir)

    skills = loader.get_skills().skills
    web_fetch = next((s for s in skills if s.name == "web-fetch"), None)
    assert web_fetch is not None
    assert web_fetch.file_path == str(project_skill_path)
    assert web_fetch.description == "Project web-fetch override"


def test_project_overrides_user_which_overrides_package(tmp_path: Path) -> None:
    agent_dir, cwd = _dirs(tmp_path)
    package_skill_dir = _create_package_with_skill(tmp_path, "web-fetch", "Package web-fetch skill")
    _create_user_skill(agent_dir, "web-fetch", "User web-fetch override")
    project_skill_path = _create_project_skill(cwd, "web-fetch", "Project web-fetch override")

    loader = _make_loader(cwd, agent_dir, package_skill_dir)

    skills = loader.get_skills().skills
    web_fetch = next((s for s in skills if s.name == "web-fetch"), None)
    assert web_fetch is not None
    assert web_fetch.file_path == str(project_skill_path)
    assert web_fetch.description == "Project web-fetch override"


def test_collision_diagnostics_report_package_skill_as_loser(tmp_path: Path) -> None:
    agent_dir, cwd = _dirs(tmp_path)
    package_skill_dir = _create_package_with_skill(tmp_path, "web-fetch", "Package web-fetch skill")
    _create_user_skill(agent_dir, "web-fetch", "User web-fetch override")

    loader = _make_loader(cwd, agent_dir, package_skill_dir)

    diagnostics = loader.get_skills().diagnostics
    collision = next(
        (
            d
            for d in diagnostics
            if d.type == "collision" and d.collision is not None and d.collision.name == "web-fetch"
        ),
        None,
    )
    assert collision is not None
    assert collision.collision is not None
    assert "fake-package" in collision.collision.loser_path
