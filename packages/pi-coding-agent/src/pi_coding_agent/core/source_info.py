"""Where a discovered resource came from.

Python port of `packages/coding-agent/src/core/source-info.ts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SourceScope = Literal["user", "project", "temporary"]
SourceOrigin = Literal["package", "top-level"]


@dataclass
class SourceInfo:
    """Provenance of a skill, prompt template, theme or extension."""

    path: str
    source: str
    scope: SourceScope = "temporary"
    origin: SourceOrigin = "top-level"
    base_dir: str | None = None


def create_source_info(path: str, metadata: Any) -> SourceInfo:
    """Build provenance from a package manager `PathMetadata`."""
    return SourceInfo(
        path=path,
        source=metadata.source,
        scope=metadata.scope,
        origin=metadata.origin,
        base_dir=getattr(metadata, "base_dir", None),
    )


def create_synthetic_source_info(
    path: str,
    source: str,
    scope: SourceScope = "temporary",
    origin: SourceOrigin = "top-level",
    base_dir: str | None = None,
) -> SourceInfo:
    """Build provenance for a resource with no package behind it."""
    return SourceInfo(path=path, source=source, scope=scope, origin=origin, base_dir=base_dir)
