"""Pi evals: behavioral, model-backed checks for Pi workflows.

Python port of the `@earendil-works/pi-evals` package
(`packages/evals/README.md`, `packages/evals/src/pi-harness.ts`,
`packages/evals/src/smoke.eval.ts`, `packages/evals/src/extensions.eval.ts`,
`packages/evals/scripts/run-evals.mjs`, and everything under
`packages/evals/src/vitest-evals/`).

The TypeScript package is built on the npm library `vitest-evals` plus a
Vitest reporter. There is no Python equivalent of `vitest-evals`, so
`pi_evals.harness` ports the slice of its interface this package uses on top
of pytest, and `pi_evals.vitest_evals.reporter` replaces the Vitest reporter
with a pytest plugin that writes the same `.eval/` artifact layout. See the
package README for the full substitution table.
"""

from __future__ import annotations

from pi_evals.harness import (
    EvalCase,
    EvalOptions,
    Harness,
    HarnessContext,
    HarnessRun,
    SimpleHarnessResult,
    create_harness,
    create_judge,
    describe_eval,
)
from pi_evals.pi_harness import (
    PiCodingAgentHarnessOptions,
    PiCodingAgentInput,
    PiCodingAgentModelSelection,
    create_pi_coding_agent_harness,
    resolve_model_selection,
)
from pi_evals.vitest_evals.harness_table import eval_harness_table

__all__ = [
    "EvalCase",
    "EvalOptions",
    "Harness",
    "HarnessContext",
    "HarnessRun",
    "PiCodingAgentHarnessOptions",
    "PiCodingAgentInput",
    "PiCodingAgentModelSelection",
    "SimpleHarnessResult",
    "create_harness",
    "create_judge",
    "create_pi_coding_agent_harness",
    "describe_eval",
    "eval_harness_table",
    "resolve_model_selection",
]
