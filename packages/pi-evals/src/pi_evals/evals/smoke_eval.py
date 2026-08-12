"""The Pi coding-agent smoke eval.

Python port of `packages/evals/src/smoke.eval.ts`.

Runs a single factual prompt end to end through a real session with all tools
disabled, and asserts the answer, the absence of harness errors, and that the
usage telemetry names the runner's default model.
"""

from __future__ import annotations

import os

from pi_evals.harness import EvalCase, EvalOptions, describe_eval
from pi_evals.pi_harness import PiCodingAgentHarnessOptions, create_pi_coding_agent_harness

pi_coding_agent_harness = create_pi_coding_agent_harness(PiCodingAgentHarnessOptions(no_tools="all"))


def _define(it) -> None:
    async def runs_a_basic_prompt_end_to_end(case: EvalCase) -> None:
        result = await case.run("What's the capital of France? Respond with only the city name.")

        assert result.output.strip() == "Paris"
        assert result.errors == []
        assert result.usage.provider == os.environ.get("PI_PROVIDER")
        assert result.usage.model == os.environ.get("PI_MODEL")
        assert (result.usage.total_tokens or 0) > 0

    it("runs a basic prompt end to end", runs_a_basic_prompt_end_to_end)


describe_eval("Pi Coding Agent smoke", EvalOptions(harness=pi_coding_agent_harness), _define)
