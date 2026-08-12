"""Per-package resource manifest.

Python port of `packages/coding-agent/src/core/pi-manifest.ts`, which reads a
``"pi"`` field (``{ extensions, skills, prompts, themes }``, each a string
array of relative paths/globs) out of a JavaScript package's ``package.json``.

**``package.json`` substitution.** Python extension packages have no
``package.json``/npm manifest convention. This port reads the same four
fields directly from a top-level ``pi.json`` file at the package root
instead of a nested ``"pi"`` key inside ``package.json`` -- there is no
surrounding npm manifest to nest under. A package that wants to declare non-
convention resource paths (e.g. an ``extensions/`` entry point other than a
bare ``*.py``/``__init__.py``) ships a ``pi.json`` like::

    {"extensions": ["./extensions/main.py"], "skills": [], "prompts": [], "themes": []}

Any error (missing file, invalid JSON, wrong shape) is swallowed and treated
as "no manifest", exactly like the TypeScript `readPiManifest`'s try/catch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

RESOURCE_FIELDS = ("extensions", "skills", "prompts", "themes")
MANIFEST_FILE_NAME = "pi.json"


@dataclass
class PiManifest:
    extensions: list[str] | None = None
    skills: list[str] | None = None
    prompts: list[str] | None = None
    themes: list[str] | None = None

    def get(self, resource_type: str) -> list[str] | None:
        return getattr(self, resource_type, None)


def read_pi_manifest(manifest_path: str) -> PiManifest | None:
    """Read a package's ``pi.json`` manifest, or ``None`` if absent/invalid.

    ``manifest_path`` is the full path to the manifest file (mirrors the
    TypeScript function taking a full ``package.json`` path rather than a
    package root directory).
    """
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return None

    if not isinstance(raw, dict):
        return None

    manifest = PiManifest()
    for field_name in RESOURCE_FIELDS:
        entries = raw.get(field_name)
        if isinstance(entries, list) and all(isinstance(entry, str) for entry in entries):
            setattr(manifest, field_name, entries)
    return manifest


def manifest_path_for_package_root(package_root: str) -> str:
    return os.path.join(package_root, MANIFEST_FILE_NAME)


__all__ = ["MANIFEST_FILE_NAME", "PiManifest", "manifest_path_for_package_root", "read_pi_manifest"]
