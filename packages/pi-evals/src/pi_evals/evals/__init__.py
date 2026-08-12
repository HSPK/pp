"""The eval suites themselves.

Python ports of `packages/evals/src/smoke.eval.ts` and
`packages/evals/src/extensions.eval.ts`. Both need a real provider and model
(`PI_PROVIDER`/`PI_MODEL`, or the `pp-evals` runner's `--provider`/`--model`);
they are not part of this package's offline test suite.
"""

from __future__ import annotations
