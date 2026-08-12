"""Eval infrastructure ported from `packages/evals/src/vitest-evals/`.

Python ports of `packages/evals/src/vitest-evals/artifacts.ts`,
`packages/evals/src/vitest-evals/harness-table.ts`,
`packages/evals/src/vitest-evals/reporter.ts`,
`packages/evals/src/vitest-evals/setup.ts` and
`packages/evals/src/vitest-evals/summary.ts`.

The directory name is kept so the file layout mirrors the TypeScript one; the
npm library `vitest-evals` itself has no Python equivalent and is ported as
`pi_evals.harness`.
"""

from __future__ import annotations
