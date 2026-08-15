"""Agent Skills discovery and formatting.

Port of `packages/coding-agent/src/core/skills.ts`.

This module is a thin, file-layout-parity facade: `resource_loader.py`
already contains a complete port of the skills logic (`Skill`,
`LoadSkillsResult`, `load_skills_from_dir`, `load_skills`,
`format_skills_for_prompt`, ignore-file handling via `GitignoreMatcher`,
name/description validation), reused directly by `ResourceLoader`. This
module re-exports that public surface under the `core/skills.py` path so the
Python tree mirrors the TypeScript source tree
(`core/skills.ts` <-> `core/skills.py`) without duplicating logic or risking
the two implementations drifting apart.
"""

from __future__ import annotations

from pi_coding_agent.core.resource_loader import (
    LoadSkillsResult,
    Skill,
    format_skills_for_prompt,
    load_skills,
    load_skills_from_dir,
)

__all__ = [
    "LoadSkillsResult",
    "Skill",
    "format_skills_for_prompt",
    "load_skills",
    "load_skills_from_dir",
]
