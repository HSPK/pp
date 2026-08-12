"""Python port of `packages/coding-agent/test/plan-mode-extension.test.ts`.

Not portable: the TypeScript test imports the plan-mode extension's *entry
point*, `examples/extensions/plan-mode/index.ts` (390 lines), and drives it
through a hand-rolled `ExtensionAPI`/`ExtensionContext` double. This port only
carries the pure helpers of that extension
(`examples/extensions/plan_mode/utils.py`, whose own docstring says so and which
`tests/test_plan_mode_utils.py` covers); the entry point itself is unported.

Checked against what does exist in this port (`core/extensions/types.py`,
`core/extensions/loader.py`) rather than taking the prior skip at face value:
`index.ts` needs `pi.registerFlag`, `pi.registerShortcut`, `ctx.ui.editor`,
`ctx.ui.setWidget`, and `ctx.ui.theme`. None of the four exist on this port's
`ExtensionAPI`/`ExtensionUIContext` -- `core/extensions/types.py`'s own module
docstring documents dropping exactly this set (`registerFlag`, `registerShortcut`,
`setWidget`/`setFooter`/`setHeader`, `editor`, `theme` and its accessors) because
they are shaped around `pi_tui`/the interactive theme system, which the extension
API surface does not expose. This matches the top-level README's "Not ported, by
decision" entry for "the extension UI host (widgets, custom header/footer,
extension-driven dialogs, terminal input listeners)". `pi.registerCommand`,
`pi.on(event, ...)`, `pi.getActiveTools`/`setActiveTools`, and `pi.appendEntry` -
the primitives all four cases also rely on - do exist in this port, but every one
of the four TypeScript cases below drives its scenario through `index.ts`'s
`togglePlanMode`/`updateStatus`/`agent_end` handler, which unconditionally calls
into the dropped surface (`updateStatus` always calls `ctx.ui.setWidget`, even
when idle), so there is no slice of `index.ts` that runs without it. Each case
is listed individually below with the exact TypeScript assertions it would need,
rather than one blanket skip.
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(
    reason=(
        "TS 'preserves custom active tools while toggling plan mode' calls the "
        "registered `/plan` command handler (`togglePlanMode`) and asserts "
        "`setActiveTools` was called with the recomputed tool list on each toggle. "
        "`togglePlanMode` unconditionally calls `ctx.ui.setStatus`/`updateStatus`, "
        "which calls the dropped `ctx.ui.setWidget` and `ctx.ui.theme.fg`, so even "
        "this UI-select-free case cannot run against index.ts without them."
    )
)
def test_preserves_custom_active_tools_while_toggling_plan_mode() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "TS 'does not prompt when the assistant response contains no plan' triggers "
        "the registered `agent_end` handler and asserts `ctx.ui.select` was never "
        "called. The handler is only reachable through index.ts's default export, "
        "which needs the dropped `pi.registerFlag`/`pi.registerShortcut` at module "
        "setup time before any handler can even be registered."
    )
)
def test_does_not_prompt_when_the_assistant_response_contains_no_plan() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "TS 'queues plan refinement as a follow-up user message' asserts "
        "`sendUserMessage` is called with the dropped `ctx.ui.editor`'s return value "
        "and `{deliverAs: 'followUp'}`. `ctx.ui.editor` does not exist on this port's "
        "`ExtensionUIContext` (see `core/extensions/types.py`'s module docstring)."
    )
)
def test_queues_plan_refinement_as_a_follow_up_user_message() -> None:
    raise AssertionError("unreachable")


@pytest.mark.skip(
    reason=(
        "TS 'queues plan execution as a follow-up custom message' asserts "
        "`sendMessage` is called with `{customType: 'plan-mode-execute'}` after "
        "`ctx.ui.select` resolves to the execute choice, and that active tools are "
        "restored via the dropped `updateStatus`/`ctx.ui.setWidget` path along the "
        "way. The `agent_end` handler that drives this is only reachable through "
        "index.ts's default export (needs `pi.registerFlag`/`pi.registerShortcut` at "
        "setup, per the module docstring)."
    )
)
def test_queues_plan_execution_as_a_follow_up_custom_message() -> None:
    raise AssertionError("unreachable")
