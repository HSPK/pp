"""Decide whether a project folder may load and execute its own config.

Ported from ``packages/coding-agent/src/core/project-trust.ts`` and
``cli/project-trust.ts``.

A project can ship a ``.pi`` directory containing settings, extensions and
packages. Loading those means running code the user may not have read, so
opening an unfamiliar repository must not silently execute it. `resolve_project_trusted`
answers "may we load this project's resources?" from, in order: an explicit
``--trust-project``/``--no-trust-project`` override, whether the project has
anything trust-requiring at all, a remembered decision, the
``defaultProjectTrust`` setting, and finally an interactive prompt. With no UI
available the answer is "no" — the safe default.

Extension participation runs through ``trust_decider`` rather than inline.
TypeScript's `resolveProjectTrusted` takes a ``LoadExtensionsResult`` and calls
``emitProjectTrustEvent`` itself; here the extensions are not loaded yet when
`cli/entry.py` resolves trust (this port loads them inside `AgentSession`,
after trust has already decided whether the project's own extensions may load
at all), so the caller passes a ``trust_decider`` that wraps
`core.extensions.runner.emit_project_trust_event` instead. The decision
semantics are identical: a handler answering yes/no wins and is remembered
when it asks to be, and "undecided" falls through to the remembered decision.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pi_coding_agent.core.config import CONFIG_DIR_NAME
from pi_coding_agent.core.trust_manager import (
    ProjectTrustOption,
    ProjectTrustStore,
    get_project_trust_options,
    has_trust_requiring_project_resources,
)

AppMode = Literal["interactive", "print", "json", "rpc"]
DefaultProjectTrust = Literal["ask", "always", "never"]
NotifyType = Literal["info", "warning", "error"]


class ProjectTrustUI(Protocol):
    """The subset of the extension-host UI that trust resolution needs."""

    async def select(self, title: str, options: Sequence[str]) -> str | None: ...


@dataclass
class ProjectTrustContext:
    cwd: str
    mode: str
    has_ui: bool
    ui: ProjectTrustUI | None = None


@dataclass
class ExtensionTrustDecision:
    """What an extension answered when asked whether to trust the project."""

    trusted: bool
    remember: bool = False


TrustDecider = Callable[[str], Awaitable[ExtensionTrustDecision | None]]


def format_project_trust_prompt(cwd: str) -> str:
    return (
        f"Trust project folder?\n{cwd}\n\n"
        f"This allows pi to load {CONFIG_DIR_NAME} settings and resources, "
        "install missing project packages, and execute project extensions."
    )


async def _select_project_trust_option(cwd: str, ctx: ProjectTrustContext) -> ProjectTrustOption | None:
    if ctx.ui is None:
        return None
    options = get_project_trust_options(cwd, include_session_only=True)
    selected = await ctx.ui.select(format_project_trust_prompt(cwd), [option.label for option in options])
    for option in options:
        if option.label == selected:
            return option
    return None


async def resolve_project_trusted(
    *,
    cwd: str,
    trust_store: ProjectTrustStore,
    project_trust_context: ProjectTrustContext,
    trust_override: bool | None = None,
    default_project_trust: DefaultProjectTrust | None = None,
    trust_decider: TrustDecider | None = None,
) -> bool:
    if trust_override is not None:
        return trust_override
    if not has_trust_requiring_project_resources(cwd):
        return True

    if trust_decider is not None:
        decision = await trust_decider(cwd)
        if decision is not None:
            if decision.remember:
                trust_store.set(cwd, decision.trusted)
            return decision.trusted

    remembered = trust_store.get(cwd)
    if remembered is not None:
        return remembered

    mode = default_project_trust or "ask"
    if mode == "always":
        return True
    if mode == "never":
        return False

    if not project_trust_context.has_ui:
        return False

    selected = await _select_project_trust_option(cwd, project_trust_context)
    if selected is None:
        return False
    if selected.updates:
        trust_store.set_many(selected.updates)
    return selected.trusted


def create_project_trust_context(
    *, cwd: str, mode: AppMode, has_ui: bool, ui: ProjectTrustUI | None = None
) -> ProjectTrustContext:
    """Build the context handed to trust resolution.

    Only interactive mode can prompt, so every other mode reports ``has_ui``
    false and resolution falls back to the remembered/settings answer.
    """
    interactive = mode == "interactive"
    return ProjectTrustContext(
        cwd=cwd,
        mode="tui" if interactive else mode,
        has_ui=has_ui and interactive,
        ui=ui if interactive else None,
    )


__all__ = [
    "AppMode",
    "DefaultProjectTrust",
    "ExtensionTrustDecision",
    "ProjectTrustContext",
    "ProjectTrustUI",
    "create_project_trust_context",
    "format_project_trust_prompt",
    "resolve_project_trusted",
]
