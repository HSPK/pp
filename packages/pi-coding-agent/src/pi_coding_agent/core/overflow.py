"""Context-overflow and recoverable-length detection.

Python port of `packages/ai/src/utils/overflow.ts`. This belongs conceptually
to `pi_ai` (it is a pure function of `AssistantMessage`/`Usage`), but `pi_ai`
has not ported it yet and is owned by a different concurrent session. It is
duplicated here, narrowly, because `AgentSession`'s auto-compaction-on-overflow
and length-retry logic depend on it directly, mirroring
`packages/coding-agent/src/core/agent-session.ts`'s imports of
`isContextOverflow`/`isRecoverableLength` from `@earendil-works/pi-ai/compat`.
"""

from __future__ import annotations

import re

from pi_ai.types import AssistantMessage

_OVERFLOW_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"prompt is too long",
        r"request_too_large",
        r"input is too long for requested model",
        r"exceeds the context window",
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
        r"input token count.*exceeds the maximum",
        r"maximum prompt length is \d+",
        r"reduce the length of the messages",
        r"maximum context length is \d+ tokens",
        r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        r"exceeds the limit of \d+",
        r"exceeds the available context size",
        r"greater than the context length",
        r"context window exceeds limit",
        r"exceeded model token limit",
        r"too large for model with \d+ maximum context length",
        r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
        r"model_context_window_exceeded",
        r"prompt too long; exceeded (?:max )?context length",
        r"range of input length should be",
        r"context[_ ]length[_ ]exceeded",
        r"too many tokens",
        r"token limit exceeded",
        r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)",
    )
]

_NON_OVERFLOW_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^(Throttling error|Service unavailable):",
        r"rate limit",
        r"too many requests",
    )
]


def is_context_overflow(message: AssistantMessage, context_window: int | None = None) -> bool:
    """Detect a context-overflow error, silent overflow, or length-stop overflow.

    See `packages/ai/src/utils/overflow.ts`'s `isContextOverflow` for the full
    per-provider rationale in the port's docstring; the three cases mirrored
    here are: (1) error message pattern match, (2) successful response whose
    usage already exceeds `context_window` (z.ai style silent overflow), and
    (3) a `"length"` stop with zero output that filled the context window
    (Xiaomi MiMo style).
    """
    if message.stop_reason == "error" and message.error_message:
        is_non_overflow = any(p.search(message.error_message) for p in _NON_OVERFLOW_PATTERNS)
        if not is_non_overflow and any(p.search(message.error_message) for p in _OVERFLOW_PATTERNS):
            return True

    if context_window and message.stop_reason == "stop":
        input_tokens = message.usage.input + message.usage.cache_read
        if input_tokens > context_window:
            return True

    if context_window and message.stop_reason == "length" and message.usage.output == 0:
        input_tokens = message.usage.input + message.usage.cache_read
        if input_tokens >= context_window * 0.99:
            return True

    return False


def is_recoverable_length(message: AssistantMessage, desired_max_output: int) -> bool:
    """Whether a `"length"` stop ended below the caller's intended output limit."""
    return message.stop_reason == "length" and desired_max_output > 0 and message.usage.output < desired_max_output


__all__ = ["is_context_overflow", "is_recoverable_length"]
