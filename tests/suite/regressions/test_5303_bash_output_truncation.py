"""Python port of `packages/coding-agent/test/suite/regressions/5303-bash-output-truncation.test.ts`.

The TypeScript test drives `waitForChildProcess` with a fake `ChildProcess` and
vitest fake timers. This port drives `wait_for_child_streams` (see
`utils/child_process.py` for why the shape differs -- Node exposes stream
`end`/`close` events, asyncio does not) with real `asyncio.StreamReader` pipes
that are never given EOF, which is what a detached descendant holding the pipe
open looks like here.

asyncio has no fake-timer facility, and the TypeScript file says explicitly why
it uses virtual time: "so host load cannot reorder a real subprocess's writes
and the grace timer". Sleeping real time between writes and then asserting the
wait has not resolved would reintroduce exactly that race under a parallel test
run. Instead both cases are stated in the direction load cannot break:

  * "an actively writing pipe keeps us reading" is driven through the
    `last_data_at` callback, which the production code re-reads on every
    wakeup. Returning `loop.time() - _virtual_age` models "the last chunk
    arrived `_virtual_age` ago", so while `_virtual_age` is 0 the deadline is
    recomputed as a full grace no matter how long the host actually stalls. A
    fixed timer armed once at exit -- the #5303 bug -- still fires and is still
    caught.
  * "a quiet pipe releases after the grace" is asserted as `elapsed >= GRACE`
    at the moment it resolves. That is the same claim as the TypeScript's
    `expect(resolved).toBe(false)` one millisecond before the grace, and host
    load can only make `elapsed` larger.
"""

from __future__ import annotations

import asyncio

from pi_coding_agent.tools.bash import _pump_stream
from pi_coding_agent.utils.child_process import wait_for_child_streams

GRACE = 0.05
WRITE_WINDOW = 3 * GRACE


async def test_captures_output_emitted_after_exit_while_a_descendant_holds_stdout_open() -> None:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    output: list[str] = []
    virtual_age = 0.0

    def on_data(chunk: bytes) -> None:
        output.append(chunk.decode())

    pump = asyncio.ensure_future(_pump_stream(reader, on_data))
    reader.feed_data(b"HEAD\n")

    waiting = asyncio.ensure_future(wait_for_child_streams([pump], lambda: loop.time() - virtual_age, grace=GRACE))

    for index in range(1, 7):
        reader.feed_data(f"TICK{index}\n".encode())
        await asyncio.sleep(0)
    # Spans several graces of real time, so the wait loop wakes repeatedly and
    # must re-arm from `last_data_at` each time rather than from a single
    # deadline armed at exit.
    await asyncio.sleep(WRITE_WINDOW)

    assert not waiting.done()

    virtual_age = GRACE
    await waiting

    # TypeScript accumulates into one string and uses `toContain`; asyncio
    # readers coalesce adjacent writes into a single chunk, so join first.
    joined = "".join(output)
    assert "HEAD\n" in joined
    assert "TICK6\n" in joined


async def test_resolves_after_the_grace_when_a_descendant_holds_stdout_open_but_stays_quiet() -> None:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    last_data_at = loop.time()

    def on_data(_chunk: bytes) -> None:
        nonlocal last_data_at
        last_data_at = loop.time()

    pump = asyncio.ensure_future(_pump_stream(reader, on_data))
    reader.feed_data(b"DONE\n")

    started = loop.time()
    waiting = asyncio.ensure_future(wait_for_child_streams([pump], lambda: last_data_at, grace=GRACE))

    await waiting

    # Subsumes the TypeScript's `expect(resolved).toBe(false)` at grace-minus-1ms:
    # resolving early would put `elapsed` below the grace.
    assert loop.time() - started >= GRACE
    assert pump.cancelled()
