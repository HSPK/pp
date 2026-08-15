"""Python port of `packages/coding-agent/test/suite/regressions/7187-malformed-package-manifest.test.ts`.

Two substitutions, both documented scope deviations of this port (see
`core/package_manager.py` and `core/pi_manifest.py`):

- TypeScript installs the package through an `npm:bad-package` source into
  `agentDir/npm/node_modules/`; there is no npm equivalent here, so the same
  package directory is referenced as a **local path** source. The manifest
  reading being tested is identical for both source kinds.
- The manifest is a top-level `pi.json` instead of the `"pi"` key of a
  `package.json`.

The behavior under test is unchanged: a manifest whose `skills` field has the
wrong shape (a string instead of an array of strings) must be ignored --
without falling back to directory-convention discovery for skills, and
without dropping the well-formed `prompts` field.
"""

from __future__ import annotations

import json
from pathlib import Path

from pi_coding_agent.core.package_manager import PackageManager
from pi_coding_agent.core.settings_manager import SettingsManager


async def test_ignores_invalid_resource_fields_without_dropping_valid_fields(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    package_dir = agent_dir / "packages" / "bad-package"
    skill_path = package_dir / "skills" / "bad" / "SKILL.md"
    prompt_path = package_dir / "prompts" / "valid.md"
    skill_path.parent.mkdir(parents=True)
    prompt_path.parent.mkdir(parents=True)
    skill_path.write_text("---\nname: bad\ndescription: Must not load\n---\n", encoding="utf-8")
    prompt_path.write_text("Valid prompt\n", encoding="utf-8")
    (package_dir / "pi.json").write_text(
        json.dumps({"skills": "./skills", "prompts": ["./prompts"]}),
        encoding="utf-8",
    )

    package_manager = PackageManager(
        str(tmp_path),
        str(agent_dir),
        SettingsManager.in_memory({"packages": [str(package_dir)]}),
    )

    resources = await package_manager.resolve()

    assert str(skill_path) not in [str(skill.path) for skill in resources.skills]
    assert str(prompt_path) in [str(prompt.path) for prompt in resources.prompts]
