"""Tests for incremental stdin parsing.

Python port of `packages/tui/test/stdin-buffer.test.ts`.
"""

from __future__ import annotations

import pytest
from pi_tui.stdin_buffer import StdinBuffer
from pi_tui.testing import wait_until


def make_buffer(timeout: float = 0.01) -> tuple[StdinBuffer, list[str]]:
    buffer = StdinBuffer(timeout=timeout)
    emitted: list[str] = []
    buffer.on("data", emitted.append)
    return buffer, emitted


class TestOscDcsApcSequences:
    def test_osc_sequence_terminated_by_bel(self) -> None:
        buffer, emitted = make_buffer()
        osc = "\x1b]11;rgb:ffff/ffff/ffff\x07"
        buffer.process(osc)
        assert emitted == [osc]

    def test_osc_sequence_terminated_by_st(self) -> None:
        buffer, emitted = make_buffer()
        osc = "\x1b]11;rgb:0000/0000/0000\x1b\\"
        buffer.process(osc)
        assert emitted == [osc]

    async def test_osc_sequence_split_across_chunks(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b]11;rgb:")
        assert emitted == []

        buffer.process("ffff/ffff/ffff\x07")
        assert emitted == ["\x1b]11;rgb:ffff/ffff/ffff\x07"]

    def test_dcs_sequence_terminated_by_st(self) -> None:
        buffer, emitted = make_buffer()
        dcs = "\x1bP>|pi terminal\x1b\\"
        buffer.process(dcs)
        assert emitted == [dcs]

    async def test_dcs_sequence_split_across_chunks(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1bP>|pi")
        assert emitted == []

        buffer.process(" terminal\x1b\\")
        assert emitted == ["\x1bP>|pi terminal\x1b\\"]

    def test_apc_sequence_terminated_by_st(self) -> None:
        buffer, emitted = make_buffer()
        apc = "\x1b_Gi=1;OK\x1b\\"
        buffer.process(apc)
        assert emitted == [apc]

    async def test_apc_sequence_split_across_chunks(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b_Gi=1")
        assert emitted == []

        buffer.process(";OK\x1b\\")
        assert emitted == ["\x1b_Gi=1;OK\x1b\\"]


class TestOnEventValidation:
    def test_raises_for_unknown_event_name(self) -> None:
        buffer = StdinBuffer()
        with pytest.raises(ValueError, match="Unknown StdinBuffer event"):
            buffer.on("bogus", lambda _value: None)  # type: ignore[arg-type]


class TestHighByteConversion:
    def test_single_high_byte_converts_to_escape_plus_low_byte_char(self) -> None:
        buffer, emitted = make_buffer()
        # 0xe1 (225) is not valid standalone UTF-8; it must take the high-byte
        # conversion path (byte - 128 = 97 = 'a') rather than UTF-8 decoding.
        buffer.process(bytes([0xE1]))
        assert emitted == ["\x1ba"]


class TestBracketedPasteWithSurroundingContentInSameChunk:
    def make(self) -> tuple[StdinBuffer, list[str], list[str]]:
        buffer = StdinBuffer(timeout=0.01)
        emitted: list[str] = []
        pastes: list[str] = []
        buffer.on("data", emitted.append)
        buffer.on("paste", pastes.append)
        return buffer, emitted, pastes

    def test_characters_preceding_paste_start_in_same_chunk_are_emitted(self) -> None:
        buffer, emitted, pastes = self.make()
        buffer.process("a\x1b[200~pasted\x1b[201~")

        assert emitted == ["a"]
        assert pastes == ["pasted"]

    def test_trailing_content_after_paste_end_in_same_initial_chunk_is_processed(self) -> None:
        buffer, emitted, pastes = self.make()
        buffer.process("\x1b[200~pasted\x1b[201~b")

        assert pastes == ["pasted"]
        assert emitted == ["b"]

    def test_trailing_content_after_paste_end_while_already_in_paste_mode(self) -> None:
        buffer, emitted, pastes = self.make()
        buffer.process("\x1b[200~partial ")
        assert pastes == []

        buffer.process("content\x1b[201~trailing")

        assert pastes == ["partial content"]
        assert emitted == ["t", "r", "a", "i", "l", "i", "n", "g"]


class TestRegularCharacters:
    def test_single_character_passes_through_immediately(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("a")
        assert emitted == ["a"]

    def test_multiple_characters_pass_through(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("abc")
        assert emitted == ["a", "b", "c"]

    def test_unicode_characters(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("hello 世界")
        assert emitted == ["h", "e", "l", "l", "o", " ", "世", "界"]


class TestCompleteEscapeSequences:
    def test_mouse_sgr_sequence(self) -> None:
        buffer, emitted = make_buffer()
        mouse_seq = "\x1b[<35;20;5m"
        buffer.process(mouse_seq)
        assert emitted == [mouse_seq]

    def test_arrow_key_sequence(self) -> None:
        buffer, emitted = make_buffer()
        up_arrow = "\x1b[A"
        buffer.process(up_arrow)
        assert emitted == [up_arrow]

    def test_function_key_sequence(self) -> None:
        buffer, emitted = make_buffer()
        f1 = "\x1b[11~"
        buffer.process(f1)
        assert emitted == [f1]

    def test_meta_key_sequence(self) -> None:
        buffer, emitted = make_buffer()
        meta_a = "\x1ba"
        buffer.process(meta_a)
        assert emitted == [meta_a]

    def test_ss3_sequence(self) -> None:
        buffer, emitted = make_buffer()
        ss3 = "\x1bOA"
        buffer.process(ss3)
        assert emitted == [ss3]


class TestPartialEscapeSequences:
    @pytest.mark.asyncio
    async def test_buffer_incomplete_mouse_sgr_sequence(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b")
        assert emitted == []
        assert buffer.get_buffer() == "\x1b"

        buffer.process("[<35")
        assert emitted == []
        assert buffer.get_buffer() == "\x1b[<35"

        buffer.process(";20;5m")
        assert emitted == ["\x1b[<35;20;5m"]
        assert buffer.get_buffer() == ""

    @pytest.mark.asyncio
    async def test_buffer_incomplete_csi_sequence(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[")
        assert emitted == []

        buffer.process("1;")
        assert emitted == []

        buffer.process("5H")
        assert emitted == ["\x1b[1;5H"]

    @pytest.mark.asyncio
    async def test_buffer_split_across_many_chunks(self) -> None:
        buffer, emitted = make_buffer()
        for chunk in ["\x1b", "[", "<", "3", "5", ";", "2", "0", ";", "5", "m"]:
            buffer.process(chunk)

        assert emitted == ["\x1b[<35;20;5m"]

    @pytest.mark.asyncio
    async def test_flush_incomplete_sequence_after_timeout(self) -> None:
        buffer, emitted = make_buffer(timeout=0.01)
        buffer.process("\x1b[<35")
        assert emitted == []

        await wait_until(
            lambda: emitted == ["\x1b[<35"],
            message="the 10ms flush timeout never fired",
        )

        assert emitted == ["\x1b[<35"]


class TestMixedContent:
    @pytest.mark.asyncio
    async def test_characters_followed_by_escape_sequence(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("abc\x1b[A")
        assert emitted == ["a", "b", "c", "\x1b[A"]

    @pytest.mark.asyncio
    async def test_escape_sequence_followed_by_characters(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[Aabc")
        assert emitted == ["\x1b[A", "a", "b", "c"]

    @pytest.mark.asyncio
    async def test_multiple_complete_sequences(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[A\x1b[B\x1b[C")
        assert emitted == ["\x1b[A", "\x1b[B", "\x1b[C"]

    @pytest.mark.asyncio
    async def test_partial_sequence_with_preceding_characters(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("abc\x1b[<35")
        assert emitted == ["a", "b", "c"]
        assert buffer.get_buffer() == "\x1b[<35"

        buffer.process(";20;5m")
        assert emitted == ["a", "b", "c", "\x1b[<35;20;5m"]


class TestKittyKeyboardProtocol:
    @pytest.mark.asyncio
    async def test_kitty_csi_u_press_event(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[97u")
        assert emitted == ["\x1b[97u"]

    @pytest.mark.asyncio
    async def test_kitty_csi_u_release_event(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[97;1:3u")
        assert emitted == ["\x1b[97;1:3u"]

    @pytest.mark.asyncio
    async def test_batched_kitty_press_and_release(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[97u\x1b[97;1:3u")
        assert emitted == ["\x1b[97u", "\x1b[97;1:3u"]

    @pytest.mark.asyncio
    async def test_multiple_batched_kitty_events(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[97u\x1b[97;1:3u\x1b[98u\x1b[98;1:3u")
        assert emitted == ["\x1b[97u", "\x1b[97;1:3u", "\x1b[98u", "\x1b[98;1:3u"]

    @pytest.mark.asyncio
    async def test_kitty_arrow_keys_with_event_type(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[1;1:1A")
        assert emitted == ["\x1b[1;1:1A"]

    @pytest.mark.asyncio
    async def test_kitty_functional_keys_with_event_type(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[3;1:3~")
        assert emitted == ["\x1b[3;1:3~"]

    @pytest.mark.asyncio
    async def test_splits_esc_esc_csi_wezterm_escape_regression(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b\x1b[27;129:3u")
        assert emitted == ["\x1b", "\x1b[27;129:3u"]

    @pytest.mark.asyncio
    async def test_splits_esc_esc_csi_no_modifier(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b\x1b[27;1:3u")
        assert emitted == ["\x1b", "\x1b[27;1:3u"]

    @pytest.mark.asyncio
    async def test_esc_esc_stays_single_sequence_without_following_escape(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b\x1b")
        assert emitted == ["\x1b\x1b"]

    @pytest.mark.asyncio
    async def test_plain_characters_mixed_with_kitty_sequences(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("a\x1b[97;1:3u")
        assert emitted == ["a", "\x1b[97;1:3u"]

    @pytest.mark.asyncio
    async def test_drop_raw_duplicate_character_after_kitty_printable(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[224uà")
        assert emitted == ["\x1b[224u"]

    @pytest.mark.asyncio
    async def test_drop_raw_duplicate_character_across_chunks(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[64u")
        buffer.process("@")
        assert emitted == ["\x1b[64u"]

    @pytest.mark.asyncio
    async def test_keep_non_matching_plain_character_after_kitty_printable(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[97ub")
        assert emitted == ["\x1b[97u", "b"]

    @pytest.mark.asyncio
    async def test_keep_raw_character_after_modified_kitty_printable(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[64;3u@")
        assert emitted == ["\x1b[64;3u", "@"]

    @pytest.mark.asyncio
    async def test_rapid_typing_simulation(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[104u\x1b[104;1:3u\x1b[105u\x1b[105;1:3u")
        assert emitted == ["\x1b[104u", "\x1b[104;1:3u", "\x1b[105u", "\x1b[105;1:3u"]


class TestMouseEvents:
    @pytest.mark.asyncio
    async def test_mouse_press_event(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[<0;10;5M")
        assert emitted == ["\x1b[<0;10;5M"]

    @pytest.mark.asyncio
    async def test_mouse_release_event(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[<0;10;5m")
        assert emitted == ["\x1b[<0;10;5m"]

    @pytest.mark.asyncio
    async def test_mouse_move_event(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[<35;20;5m")
        assert emitted == ["\x1b[<35;20;5m"]

    @pytest.mark.asyncio
    async def test_split_mouse_events(self) -> None:
        buffer, emitted = make_buffer()
        for chunk in ["\x1b[<3", "5;1", "5;", "10m"]:
            buffer.process(chunk)
        assert emitted == ["\x1b[<35;15;10m"]

    @pytest.mark.asyncio
    async def test_multiple_mouse_events(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[<35;1;1m\x1b[<35;2;2m\x1b[<35;3;3m")
        assert emitted == ["\x1b[<35;1;1m", "\x1b[<35;2;2m", "\x1b[<35;3;3m"]

    @pytest.mark.asyncio
    async def test_old_style_mouse_sequence(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[M abc")
        assert emitted == ["\x1b[M ab", "c"]

    @pytest.mark.asyncio
    async def test_buffer_incomplete_old_style_mouse_sequence(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[M")
        assert buffer.get_buffer() == "\x1b[M"

        buffer.process(" a")
        assert buffer.get_buffer() == "\x1b[M a"

        buffer.process("b")
        assert emitted == ["\x1b[M ab"]


class TestEdgeCases:
    def test_empty_input_emits_empty_data_event(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("")
        assert emitted == [""]

    @pytest.mark.asyncio
    async def test_lone_escape_character_with_timeout(self) -> None:
        buffer, emitted = make_buffer(timeout=0.01)
        buffer.process("\x1b")
        assert emitted == []

        await wait_until(
            lambda: emitted == ["\x1b"],
            message="the 10ms flush timeout never fired",
        )
        assert emitted == ["\x1b"]

    @pytest.mark.asyncio
    async def test_lone_escape_character_with_explicit_flush(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b")
        assert emitted == []

        flushed = buffer.flush()
        assert flushed == ["\x1b"]

    @pytest.mark.asyncio
    async def test_bytes_input(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process(b"\x1b[A")
        assert emitted == ["\x1b[A"]

    @pytest.mark.asyncio
    async def test_very_long_sequences(self) -> None:
        buffer, emitted = make_buffer()
        long_seq = f"\x1b[{'1;' * 50}H"
        buffer.process(long_seq)
        assert emitted == [long_seq]


class TestFlush:
    @pytest.mark.asyncio
    async def test_flush_incomplete_sequences(self) -> None:
        buffer, _emitted = make_buffer()
        buffer.process("\x1b[<35")
        flushed = buffer.flush()
        assert flushed == ["\x1b[<35"]
        assert buffer.get_buffer() == ""

    def test_returns_empty_list_if_nothing_to_flush(self) -> None:
        buffer, _emitted = make_buffer()
        flushed = buffer.flush()
        assert flushed == []

    @pytest.mark.asyncio
    async def test_emit_flushed_data_via_timeout(self) -> None:
        buffer, emitted = make_buffer(timeout=0.01)
        buffer.process("\x1b[<35")
        assert emitted == []

        await wait_until(
            lambda: emitted == ["\x1b[<35"],
            message="the 10ms flush timeout never fired",
        )

        assert emitted == ["\x1b[<35"]


class TestClear:
    @pytest.mark.asyncio
    async def test_clear_buffered_content_without_emitting(self) -> None:
        buffer, emitted = make_buffer()
        buffer.process("\x1b[<35")
        assert buffer.get_buffer() == "\x1b[<35"

        buffer.clear()
        assert buffer.get_buffer() == ""
        assert emitted == []


class TestBracketedPaste:
    def make(self) -> tuple[StdinBuffer, list[str], list[str]]:
        buffer = StdinBuffer(timeout=0.01)
        emitted: list[str] = []
        pastes: list[str] = []
        buffer.on("data", emitted.append)
        buffer.on("paste", pastes.append)
        return buffer, emitted, pastes

    def test_complete_bracketed_paste(self) -> None:
        buffer, emitted, pastes = self.make()
        paste_start = "\x1b[200~"
        paste_end = "\x1b[201~"
        content = "hello world"

        buffer.process(paste_start + content + paste_end)

        assert pastes == ["hello world"]
        assert emitted == []

    def test_paste_arriving_in_chunks(self) -> None:
        buffer, emitted, pastes = self.make()
        buffer.process("\x1b[200~")
        assert pastes == []

        buffer.process("hello ")
        assert pastes == []

        buffer.process("world\x1b[201~")
        assert pastes == ["hello world"]
        assert emitted == []

    def test_paste_with_input_before_and_after(self) -> None:
        buffer, emitted, pastes = self.make()
        buffer.process("a")
        buffer.process("\x1b[200~pasted\x1b[201~")
        buffer.process("b")

        assert emitted == ["a", "b"]
        assert pastes == ["pasted"]

    def test_paste_with_newlines(self) -> None:
        buffer, emitted, pastes = self.make()
        buffer.process("\x1b[200~line1\nline2\nline3\x1b[201~")

        assert pastes == ["line1\nline2\nline3"]
        assert emitted == []

    def test_paste_with_unicode(self) -> None:
        buffer, emitted, pastes = self.make()
        buffer.process("\x1b[200~Hello 世界 🎉\x1b[201~")

        assert pastes == ["Hello 世界 🎉"]
        assert emitted == []


class TestDestroy:
    @pytest.mark.asyncio
    async def test_clear_buffer_on_destroy(self) -> None:
        buffer, _emitted = make_buffer()
        buffer.process("\x1b[<35")
        assert buffer.get_buffer() == "\x1b[<35"

        buffer.destroy()
        assert buffer.get_buffer() == ""

    @pytest.mark.asyncio
    async def test_clear_pending_timeouts_on_destroy(self) -> None:
        buffer, emitted = make_buffer(timeout=0.01)
        buffer.process("\x1b[<35")
        buffer.destroy()

        await wait_until(
            lambda: emitted == [],
            message="the 10ms flush timeout never fired",
        )

        assert emitted == []
