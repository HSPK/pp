"""Python port of `packages/coding-agent/test/custom-message.test.ts`."""

from __future__ import annotations

import time

from pi_agent.harness.messages import CustomMessage
from pi_coding_agent.modes.interactive.components.custom_message import (
    CustomMessageComponent,
    MessageRenderContext,
)
from pi_coding_agent.modes.interactive.theme.theme import init_theme
from pi_coding_agent.utils.ansi import strip_ansi
from pi_tui.components.text import Text


def test_provides_output_padding_to_custom_renderers_and_updates_it() -> None:
    init_theme("dark")
    options_seen: list[MessageRenderContext] = []

    def renderer(_message: CustomMessage, options: MessageRenderContext, _theme: object) -> Text:
        options_seen.append(options)
        return Text("custom", options.output_pad, 0)

    message = CustomMessage(
        custom_type="test",
        content="custom",
        display=True,
        timestamp=int(time.time() * 1000),
    )
    component = CustomMessageComponent(message, renderer, None, 1)

    assert options_seen == [MessageRenderContext(expanded=False, output_pad=1)]
    assert any(strip_ansi(line).startswith(" custom") for line in component.render(40))

    component.set_output_pad(0)

    assert options_seen[-1] == MessageRenderContext(expanded=False, output_pad=0)
    assert any(strip_ansi(line).startswith("custom") for line in component.render(40))
