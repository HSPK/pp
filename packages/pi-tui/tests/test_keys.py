"""Python port of `packages/tui/test/keys.test.ts`.

One test function per TypeScript `it(...)`, grouped into classes that mirror the
`describe(...)` blocks. `parseKey` returns `undefined` in TypeScript where the
Python port returns `None`; the assertions are otherwise identical.

The final class holds Python-side additions that have no TypeScript
counterpart: they pin `is_key_release`/`is_key_repeat`, lock-bit masking, and
the shifted-codepoint path of `decode_kitty_printable`, all of which the port
implements but the TypeScript suite exercises only indirectly.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from pi_tui.keys import (
    Key,
    decode_kitty_printable,
    decode_printable_key,
    is_key_release,
    is_key_repeat,
    is_kitty_protocol_active,
    matches_key,
    parse_key,
    set_kitty_protocol_active,
)


@pytest.fixture(autouse=True)
def reset_kitty_protocol() -> Iterator[None]:
    set_kitty_protocol_active(False)
    yield
    set_kitty_protocol_active(False)


@contextmanager
def with_env_vars(vars_to_set: dict[str, str | None]) -> Iterator[None]:
    """Python stand-in for the TypeScript suite's `withEnv`/`withEnvVars`."""
    previous = {name: os.environ.get(name) for name in vars_to_set}
    for name, value in vars_to_set.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class TestMatchesKeyKittyAlternateKeys:
    """`matchesKey` > `Kitty protocol with alternate keys (non-Latin layouts)`.

    Kitty protocol flag 4 (report alternate keys) sends
    `CSI codepoint:shifted:base ; modifier:event u`, where `base` is the key in
    the standard PC-101 layout.
    """

    def test_should_match_ctrl_c_when_pressing_ctrl_cyrillic_es_with_base_layout_key(self) -> None:
        set_kitty_protocol_active(True)
        # Cyrillic 'с' = 1089, Latin 'c' = 99; ctrl=4, +1 = 5.
        cyrillic_ctrl_c = "\x1b[1089::99;5u"
        assert matches_key(cyrillic_ctrl_c, "ctrl+c") is True
        set_kitty_protocol_active(False)

    def test_should_match_ctrl_d_when_pressing_ctrl_cyrillic_ve_with_base_layout_key(self) -> None:
        set_kitty_protocol_active(True)
        # Cyrillic 'в' = 1074, Latin 'd' = 100.
        cyrillic_ctrl_d = "\x1b[1074::100;5u"
        assert matches_key(cyrillic_ctrl_d, "ctrl+d") is True
        set_kitty_protocol_active(False)

    def test_should_match_ctrl_z_when_pressing_ctrl_cyrillic_ya_with_base_layout_key(self) -> None:
        set_kitty_protocol_active(True)
        # Cyrillic 'я' = 1103, Latin 'z' = 122.
        cyrillic_ctrl_z = "\x1b[1103::122;5u"
        assert matches_key(cyrillic_ctrl_z, "ctrl+z") is True
        set_kitty_protocol_active(False)

    def test_should_match_ctrl_shift_p_with_base_layout_key(self) -> None:
        set_kitty_protocol_active(True)
        # Cyrillic 'з' = 1079, Latin 'p' = 112; ctrl=4, shift=1, +1 = 6.
        cyrillic_ctrl_shift_p = "\x1b[1079::112;6u"
        assert matches_key(cyrillic_ctrl_shift_p, "ctrl+shift+p") is True
        set_kitty_protocol_active(False)

    def test_should_still_match_direct_codepoint_when_no_base_layout_key(self) -> None:
        set_kitty_protocol_active(True)
        # Latin ctrl+c without a base layout key (terminal lacks flag 4).
        latin_ctrl_c = "\x1b[99;5u"
        assert matches_key(latin_ctrl_c, "ctrl+c") is True
        set_kitty_protocol_active(False)

    def test_should_match_super_modified_kitty_bindings_including_combined_modifiers(self) -> None:
        set_kitty_protocol_active(True)
        assert matches_key("\x1b[107;9u", "super+k") is True
        assert matches_key("\x1b[13;9u", "super+enter") is True
        assert matches_key("\x1b[107;13u", Key.ctrl_super("k")) is True
        assert matches_key("\x1b[107;13u", "ctrl+super+k") is True
        assert matches_key("\x1b[107;14u", "ctrl+shift+super+k") is True
        assert matches_key("\x1b[107;13u", "super+k") is False
        assert parse_key("\x1b[107;9u") == "super+k"
        assert parse_key("\x1b[13;9u") == "super+enter"
        assert parse_key("\x1b[107;13u") == "ctrl+super+k"
        assert parse_key("\x1b[107;14u") == "shift+ctrl+super+k"
        set_kitty_protocol_active(False)

    def test_should_match_digit_bindings_via_kitty_csi_u(self) -> None:
        set_kitty_protocol_active(True)
        assert matches_key("\x1b[49u", "1") is True
        assert matches_key("\x1b[49;5u", "ctrl+1") is True
        assert matches_key("\x1b[49;5u", "ctrl+2") is False
        assert parse_key("\x1b[49u") == "1"
        assert parse_key("\x1b[49;5u") == "ctrl+1"
        set_kitty_protocol_active(False)

    def test_should_normalize_kitty_keypad_functional_keys(self) -> None:
        set_kitty_protocol_active(True)
        assert matches_key("\x1b[57400u", "1") is True
        assert matches_key("\x1b[57410u", "/") is True
        assert matches_key("\x1b[57417u", "left") is True
        assert matches_key("\x1b[57426u", "delete") is True
        assert parse_key("\x1b[57399u") == "0"
        assert parse_key("\x1b[57409u") == "."
        assert parse_key("\x1b[57413u") == "+"
        assert parse_key("\x1b[57416u") == ","
        assert parse_key("\x1b[57417u") == "left"
        assert parse_key("\x1b[57418u") == "right"
        assert parse_key("\x1b[57419u") == "up"
        assert parse_key("\x1b[57420u") == "down"
        assert parse_key("\x1b[57421u") == "pageUp"
        assert parse_key("\x1b[57422u") == "pageDown"
        assert parse_key("\x1b[57423u") == "home"
        assert parse_key("\x1b[57424u") == "end"
        assert parse_key("\x1b[57425u") == "insert"
        assert parse_key("\x1b[57426u") == "delete"
        set_kitty_protocol_active(False)

    def test_should_handle_shifted_key_in_format(self) -> None:
        set_kitty_protocol_active(True)
        # Latin 'c' with shifted 'C' (67) and base 'c' (99); shift=1, +1 = 2.
        shifted_key = "\x1b[99:67:99;2u"
        assert matches_key(shifted_key, "shift+c") is True
        set_kitty_protocol_active(False)

    def test_should_handle_event_type_in_format(self) -> None:
        set_kitty_protocol_active(True)
        # Cyrillic ctrl+c release event (event type 3).
        release_event = "\x1b[1089::99;5:3u"
        assert matches_key(release_event, "ctrl+c") is True
        set_kitty_protocol_active(False)

    def test_should_handle_full_format_with_shifted_key_base_key_and_event_type(self) -> None:
        set_kitty_protocol_active(True)
        # Cyrillic 'с' = 1089, 'С' = 1057, Latin 'c' = 99; ctrl+shift = 6, repeat event = 2.
        full_format = "\x1b[1089:1057:99;6:2u"
        assert matches_key(full_format, "ctrl+shift+c") is True
        set_kitty_protocol_active(False)

    def test_should_prefer_codepoint_for_latin_letters_even_when_base_layout_differs(self) -> None:
        set_kitty_protocol_active(True)
        # Dvorak Ctrl+K reports codepoint 'k' (107) and base layout 'v' (118).
        dvorak_ctrl_k = "\x1b[107::118;5u"
        assert matches_key(dvorak_ctrl_k, "ctrl+k") is True
        assert matches_key(dvorak_ctrl_k, "ctrl+v") is False
        set_kitty_protocol_active(False)

    def test_should_prefer_codepoint_for_symbol_keys_even_when_base_layout_differs(self) -> None:
        set_kitty_protocol_active(True)
        # Dvorak Ctrl+/ reports codepoint '/' (47) and base layout '[' (91).
        dvorak_ctrl_slash = "\x1b[47::91;5u"
        assert matches_key(dvorak_ctrl_slash, "ctrl+/") is True
        assert matches_key(dvorak_ctrl_slash, "ctrl+[") is False
        set_kitty_protocol_active(False)

    def test_should_not_match_wrong_key_even_with_base_layout(self) -> None:
        set_kitty_protocol_active(True)
        cyrillic_ctrl_c = "\x1b[1089::99;5u"
        assert matches_key(cyrillic_ctrl_c, "ctrl+d") is False
        set_kitty_protocol_active(False)

    def test_should_not_match_wrong_modifiers_even_with_base_layout(self) -> None:
        set_kitty_protocol_active(True)
        cyrillic_ctrl_c = "\x1b[1089::99;5u"
        assert matches_key(cyrillic_ctrl_c, "ctrl+shift+c") is False
        set_kitty_protocol_active(False)


class TestModifyOtherKeysMatching:
    """`matchesKey` > `modifyOtherKeys matching`."""

    def test_should_match_xterm_modify_other_keys_ctrl_c(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b[27;5;99~", "ctrl+c") is True
        assert parse_key("\x1b[27;5;99~") == "ctrl+c"

    def test_should_match_xterm_modify_other_keys_ctrl_d(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b[27;5;100~", "ctrl+d") is True
        assert parse_key("\x1b[27;5;100~") == "ctrl+d"

    def test_should_match_xterm_modify_other_keys_ctrl_z(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b[27;5;122~", "ctrl+z") is True
        assert parse_key("\x1b[27;5;122~") == "ctrl+z"

    def test_should_match_xterm_modify_other_keys_enter_variants(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b[27;5;13~", "ctrl+enter") is True
        assert matches_key("\x1b[27;2;13~", "shift+enter") is True
        assert matches_key("\x1b[27;3;13~", "alt+enter") is True
        assert parse_key("\x1b[27;5;13~") == "ctrl+enter"
        assert parse_key("\x1b[27;2;13~") == "shift+enter"
        assert parse_key("\x1b[27;3;13~") == "alt+enter"

    def test_should_match_xterm_modify_other_keys_tab_variants(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b[27;2;9~", "shift+tab") is True
        assert matches_key("\x1b[27;5;9~", "ctrl+tab") is True
        assert matches_key("\x1b[27;3;9~", "alt+tab") is True
        assert parse_key("\x1b[27;2;9~") == "shift+tab"
        assert parse_key("\x1b[27;5;9~") == "ctrl+tab"
        assert parse_key("\x1b[27;3;9~") == "alt+tab"

    def test_should_match_xterm_modify_other_keys_backspace_variants(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b[27;1;127~", "backspace") is True
        assert matches_key("\x1b[27;5;127~", "ctrl+backspace") is True
        assert matches_key("\x1b[27;3;127~", "alt+backspace") is True
        assert parse_key("\x1b[27;1;127~") == "backspace"
        assert parse_key("\x1b[27;5;127~") == "ctrl+backspace"
        assert parse_key("\x1b[27;3;127~") == "alt+backspace"

    def test_should_match_xterm_modify_other_keys_escape(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b[27;1;27~", "escape") is True
        assert parse_key("\x1b[27;1;27~") == "escape"

    def test_should_match_xterm_modify_other_keys_space_variants(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b[27;1;32~", "space") is True
        assert matches_key("\x1b[27;5;32~", "ctrl+space") is True
        assert parse_key("\x1b[27;1;32~") == "space"
        assert parse_key("\x1b[27;5;32~") == "ctrl+space"

    def test_should_match_xterm_modify_other_keys_symbol_combos(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b[27;5;47~", "ctrl+/") is True
        assert parse_key("\x1b[27;5;47~") == "ctrl+/"

    def test_should_match_xterm_modify_other_keys_digit_combos(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b[27;5;49~", "ctrl+1") is True
        assert matches_key("\x1b[27;2;49~", "shift+1") is True
        assert parse_key("\x1b[27;5;49~") == "ctrl+1"
        assert parse_key("\x1b[27;2;49~") == "shift+1"

    def test_should_match_xterm_modify_other_keys_shifted_uppercase_letters(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b[27;2;69~", "shift+e") is True
        assert matches_key("\x1b[27;6;69~", "ctrl+shift+e") is True
        assert parse_key("\x1b[27;2;69~") == "shift+e"
        assert parse_key("\x1b[27;6;69~") == "shift+ctrl+e"

    def test_should_match_ctrl_alt_letter_via_csi_u_when_kitty_inactive(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b[104;7u", "ctrl+alt+h") is True
        assert parse_key("\x1b[104;7u") == "ctrl+alt+h"

    def test_should_match_ctrl_alt_letter_via_xterm_modify_other_keys(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b[27;7;104~", "ctrl+alt+h") is True
        assert parse_key("\x1b[27;7;104~") == "ctrl+alt+h"


class TestLegacyKeyMatching:
    """`matchesKey` > `Legacy key matching`."""

    def test_should_match_legacy_ctrl_c(self) -> None:
        set_kitty_protocol_active(False)
        # Ctrl+c sends ASCII 3 (ETX).
        assert matches_key("\x03", "ctrl+c") is True

    def test_should_match_legacy_ctrl_d(self) -> None:
        set_kitty_protocol_active(False)
        # Ctrl+d sends ASCII 4 (EOT).
        assert matches_key("\x04", "ctrl+d") is True

    def test_should_match_escape_key(self) -> None:
        assert matches_key("\x1b", "escape") is True

    def test_should_match_legacy_linefeed_as_enter(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\n", "enter") is True
        assert parse_key("\n") == "enter"

    def test_should_treat_linefeed_as_shift_enter_when_kitty_active(self) -> None:
        set_kitty_protocol_active(True)
        assert matches_key("\n", "shift+enter") is True
        assert matches_key("\n", "enter") is False
        assert parse_key("\n") == "shift+enter"
        set_kitty_protocol_active(False)

    def test_should_parse_ctrl_space(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x00", "ctrl+space") is True
        assert parse_key("\x00") == "ctrl+space"

    def test_should_match_legacy_ctrl_symbol(self) -> None:
        set_kitty_protocol_active(False)
        # Ctrl+\ sends ASCII 28 (File Separator) in legacy terminals.
        assert matches_key("\x1c", "ctrl+\\") is True
        assert parse_key("\x1c") == "ctrl+\\"
        # Ctrl+] sends ASCII 29 (Group Separator).
        assert matches_key("\x1d", "ctrl+]") is True
        assert parse_key("\x1d") == "ctrl+]"
        # Ctrl+_ sends ASCII 31 (Unit Separator); Ctrl+- is the same physical key on US keyboards.
        assert matches_key("\x1f", "ctrl+_") is True
        assert matches_key("\x1f", "ctrl+-") is True
        assert parse_key("\x1f") == "ctrl+-"

    def test_should_match_legacy_ctrl_alt_symbol(self) -> None:
        set_kitty_protocol_active(False)
        # Ctrl+Alt+[ sends ESC followed by ESC (Ctrl+[ = ESC).
        assert matches_key("\x1b\x1b", "ctrl+alt+[") is True
        assert parse_key("\x1b\x1b") == "ctrl+alt+["
        # Ctrl+Alt+\ sends ESC followed by ASCII 28.
        assert matches_key("\x1b\x1c", "ctrl+alt+\\") is True
        assert parse_key("\x1b\x1c") == "ctrl+alt+\\"
        # Ctrl+Alt+] sends ESC followed by ASCII 29.
        assert matches_key("\x1b\x1d", "ctrl+alt+]") is True
        assert parse_key("\x1b\x1d") == "ctrl+alt+]"
        assert matches_key("\x1b\x1f", "ctrl+alt+_") is True
        assert matches_key("\x1b\x1f", "ctrl+alt+-") is True
        assert parse_key("\x1b\x1f") == "ctrl+alt+-"

    def test_should_treat_raw_0x08_as_plain_backspace_outside_windows_terminal(self) -> None:
        set_kitty_protocol_active(False)
        with with_env_vars({"WT_SESSION": None}):
            assert matches_key("\x7f", "backspace") is True
            assert matches_key("\x7f", "ctrl+backspace") is False
            assert parse_key("\x7f") == "backspace"
            assert matches_key("\x08", "backspace") is True
            assert matches_key("\x08", "ctrl+backspace") is False
            assert parse_key("\x08") == "backspace"
            assert matches_key("\x08", "ctrl+h") is True

    def test_should_treat_raw_0x08_as_ctrl_backspace_in_local_windows_terminal(self) -> None:
        set_kitty_protocol_active(False)
        with with_env_vars(
            {
                "WT_SESSION": "test-session",
                "SSH_CONNECTION": None,
                "SSH_CLIENT": None,
                "SSH_TTY": None,
            }
        ):
            assert matches_key("\x08", "ctrl+backspace") is True
            assert matches_key("\x08", "backspace") is False
            assert parse_key("\x08") == "ctrl+backspace"
            assert matches_key("\x08", "ctrl+h") is True

    def test_should_treat_raw_0x08_as_plain_backspace_in_windows_terminal_over_ssh(self) -> None:
        set_kitty_protocol_active(False)
        with with_env_vars(
            {
                "WT_SESSION": "test-session",
                "SSH_CONNECTION": "1 2 3 4",
                "SSH_CLIENT": "1 2 3",
                "SSH_TTY": "/dev/pts/1",
            }
        ):
            assert matches_key("\x08", "ctrl+backspace") is False
            assert matches_key("\x08", "backspace") is True
            assert parse_key("\x08") == "backspace"
            assert matches_key("\x08", "ctrl+h") is True

    def test_should_parse_legacy_alt_prefixed_sequences_when_kitty_inactive(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b ", "alt+space") is True
        assert parse_key("\x1b ") == "alt+space"
        assert matches_key("\x1b\b", "alt+backspace") is True
        assert parse_key("\x1b\b") == "alt+backspace"
        assert matches_key("\x1b\x03", "ctrl+alt+c") is True
        assert parse_key("\x1b\x03") == "ctrl+alt+c"
        assert matches_key("\x1bB", "alt+left") is True
        assert parse_key("\x1bB") == "alt+left"
        assert matches_key("\x1bF", "alt+right") is True
        assert parse_key("\x1bF") == "alt+right"
        assert matches_key("\x1ba", "alt+a") is True
        assert parse_key("\x1ba") == "alt+a"
        assert matches_key("\x1b1", "alt+1") is True
        assert parse_key("\x1b1") == "alt+1"
        assert matches_key("\x1b,", "alt+,") is True
        assert parse_key("\x1b,") == "alt+,"
        assert matches_key("\x1b.", "alt+.") is True
        assert parse_key("\x1b.") == "alt+."
        assert matches_key("\x1by", "alt+y") is True
        assert parse_key("\x1by") == "alt+y"
        assert matches_key("\x1bz", "alt+z") is True
        assert parse_key("\x1bz") == "alt+z"

        set_kitty_protocol_active(True)
        assert matches_key("\x1b ", "alt+space") is False
        assert parse_key("\x1b ") is None
        # Alt+Backspace stays recognized under Kitty: it is not an alt-prefix guess.
        assert matches_key("\x1b\b", "alt+backspace") is True
        assert parse_key("\x1b\b") == "alt+backspace"
        assert matches_key("\x1b\x03", "ctrl+alt+c") is False
        assert parse_key("\x1b\x03") is None
        assert matches_key("\x1bB", "alt+left") is False
        assert parse_key("\x1bB") is None
        assert matches_key("\x1bF", "alt+right") is False
        assert parse_key("\x1bF") is None
        assert matches_key("\x1ba", "alt+a") is False
        assert parse_key("\x1ba") is None
        assert matches_key("\x1b1", "alt+1") is False
        assert parse_key("\x1b1") is None
        assert matches_key("\x1b,", "alt+,") is False
        assert parse_key("\x1b,") is None
        assert matches_key("\x1b.", "alt+.") is False
        assert parse_key("\x1b.") is None
        assert matches_key("\x1by", "alt+y") is False
        assert parse_key("\x1by") is None
        set_kitty_protocol_active(False)

    def test_should_match_arrow_keys(self) -> None:
        assert matches_key("\x1b[A", "up") is True
        assert matches_key("\x1b[B", "down") is True
        assert matches_key("\x1b[C", "right") is True
        assert matches_key("\x1b[D", "left") is True

    def test_should_match_ss3_arrows_and_home_end(self) -> None:
        assert matches_key("\x1bOA", "up") is True
        assert matches_key("\x1bOB", "down") is True
        assert matches_key("\x1bOC", "right") is True
        assert matches_key("\x1bOD", "left") is True
        assert matches_key("\x1bOH", "home") is True
        assert matches_key("\x1bOF", "end") is True

    def test_should_match_xterm_ctrl_modified_viewport_navigation(self) -> None:
        assert matches_key("\x1b[1;5H", "ctrl+home") is True
        assert matches_key("\x1b[1;5F", "ctrl+end") is True
        assert matches_key("\x1b[5;5~", "ctrl+pageUp") is True
        assert matches_key("\x1b[6;5~", "ctrl+pageDown") is True
        assert parse_key("\x1b[1;5H") == "ctrl+home"
        assert parse_key("\x1b[1;5F") == "ctrl+end"
        assert parse_key("\x1b[5;5~") == "ctrl+pageUp"
        assert parse_key("\x1b[6;5~") == "ctrl+pageDown"

    def test_should_match_legacy_function_keys_and_clear(self) -> None:
        assert matches_key("\x1bOP", "f1") is True
        assert matches_key("\x1b[24~", "f12") is True
        assert matches_key("\x1b[E", "clear") is True

    def test_should_match_alt_arrows(self) -> None:
        assert matches_key("\x1bp", "alt+up") is True
        assert matches_key("\x1bp", "up") is False

    def test_should_match_rxvt_modifier_sequences(self) -> None:
        assert matches_key("\x1b[a", "shift+up") is True
        assert matches_key("\x1bOa", "ctrl+up") is True
        assert matches_key("\x1b[2$", "shift+insert") is True
        assert matches_key("\x1b[2^", "ctrl+insert") is True
        assert matches_key("\x1b[7$", "shift+home") is True


class TestDecodeKittyPrintable:
    def test_should_decode_kitty_keypad_functional_keys_to_printable_characters(self) -> None:
        assert decode_kitty_printable("\x1b[57399u") == "0"
        assert decode_kitty_printable("\x1b[57400u") == "1"
        assert decode_kitty_printable("\x1b[57409u") == "."
        assert decode_kitty_printable("\x1b[57410u") == "/"
        assert decode_kitty_printable("\x1b[57411u") == "*"
        assert decode_kitty_printable("\x1b[57412u") == "-"
        assert decode_kitty_printable("\x1b[57413u") == "+"
        assert decode_kitty_printable("\x1b[57415u") == "="
        assert decode_kitty_printable("\x1b[57416u") == ","
        assert decode_kitty_printable("\x1b[57417u") is None


class TestDecodePrintableKey:
    def test_should_decode_printable_xterm_modify_other_keys_sequences(self) -> None:
        assert decode_printable_key("\x1b[27;2;69~") == "E"
        assert decode_printable_key("\x1b[27;2;196~") == "Ä"
        assert decode_printable_key("\x1b[27;2;32~") == " "
        assert decode_printable_key("\x1b[27;2;13~") is None
        assert decode_printable_key("\x1b[27;6;69~") is None


class TestParseKeyKittyAlternateKeys:
    """`parseKey` > `Kitty protocol with alternate keys`."""

    def test_should_return_latin_key_name_when_base_layout_key_is_present(self) -> None:
        set_kitty_protocol_active(True)
        cyrillic_ctrl_c = "\x1b[1089::99;5u"
        assert parse_key(cyrillic_ctrl_c) == "ctrl+c"
        set_kitty_protocol_active(False)

    def test_should_prefer_codepoint_for_latin_letters_when_base_layout_differs(self) -> None:
        set_kitty_protocol_active(True)
        dvorak_ctrl_k = "\x1b[107::118;5u"
        assert parse_key(dvorak_ctrl_k) == "ctrl+k"
        set_kitty_protocol_active(False)

    def test_should_prefer_codepoint_for_symbol_keys_when_base_layout_differs(self) -> None:
        set_kitty_protocol_active(True)
        dvorak_ctrl_slash = "\x1b[47::91;5u"
        assert parse_key(dvorak_ctrl_slash) == "ctrl+/"
        set_kitty_protocol_active(False)

    def test_should_return_key_name_from_codepoint_when_no_base_layout(self) -> None:
        set_kitty_protocol_active(True)
        latin_ctrl_c = "\x1b[99;5u"
        assert parse_key(latin_ctrl_c) == "ctrl+c"
        set_kitty_protocol_active(False)

    def test_should_parse_shifted_uppercase_csi_u_letters_as_shift_letter(self) -> None:
        set_kitty_protocol_active(True)
        assert matches_key("\x1b[69;2u", "shift+e") is True
        assert parse_key("\x1b[69;2u") == "shift+e"
        set_kitty_protocol_active(False)

    def test_should_ignore_kitty_csi_u_with_unsupported_modifiers(self) -> None:
        set_kitty_protocol_active(True)
        assert parse_key("\x1b[99;17u") is None
        set_kitty_protocol_active(False)


class TestLegacyKeyParsing:
    """`parseKey` > `Legacy key parsing`."""

    def test_should_parse_legacy_ctrl_letter(self) -> None:
        set_kitty_protocol_active(False)
        assert parse_key("\x03") == "ctrl+c"
        assert parse_key("\x04") == "ctrl+d"

    def test_should_parse_special_keys(self) -> None:
        assert parse_key("\x1b") == "escape"
        assert parse_key("\t") == "tab"
        assert parse_key("\r") == "enter"
        assert parse_key("\n") == "enter"
        assert parse_key("\x00") == "ctrl+space"
        assert parse_key(" ") == "space"
        assert parse_key("1") == "1"
        assert matches_key("1", "1") is True

    def test_should_parse_arrow_keys(self) -> None:
        assert parse_key("\x1b[A") == "up"
        assert parse_key("\x1b[B") == "down"
        assert parse_key("\x1b[C") == "right"
        assert parse_key("\x1b[D") == "left"

    def test_should_parse_ss3_arrows_and_home_end(self) -> None:
        assert parse_key("\x1bOA") == "up"
        assert parse_key("\x1bOB") == "down"
        assert parse_key("\x1bOC") == "right"
        assert parse_key("\x1bOD") == "left"
        assert parse_key("\x1bOH") == "home"
        assert parse_key("\x1bOF") == "end"

    def test_should_parse_legacy_function_and_modifier_sequences(self) -> None:
        assert parse_key("\x1bOP") == "f1"
        assert parse_key("\x1b[24~") == "f12"
        assert parse_key("\x1b[E") == "clear"
        assert parse_key("\x1b[2^") == "ctrl+insert"
        assert parse_key("\x1bp") == "alt+up"

    def test_should_parse_double_bracket_page_up(self) -> None:
        assert parse_key("\x1b[[5~") == "pageUp"


class TestPythonPortAdditions:
    """No TypeScript counterpart: behavior the port implements and must keep."""

    @pytest.mark.parametrize(
        ("builder", "expected"),
        [
            (lambda: Key.ctrl("c"), "ctrl+c"),
            (lambda: Key.alt_shift("x"), "alt+shift+x"),
            (lambda: Key.ctrl_super("k"), "ctrl+super+k"),
            (lambda: Key.page_up, "pageUp"),
            (lambda: Key.return_, "return"),
        ],
    )
    def test_key_helpers(self, builder, expected: str) -> None:
        assert builder() == expected

    def test_kitty_protocol_state_round_trip(self) -> None:
        assert is_kitty_protocol_active() is False
        set_kitty_protocol_active(True)
        assert is_kitty_protocol_active() is True
        set_kitty_protocol_active(False)
        assert is_kitty_protocol_active() is False

    def test_decode_kitty_printable_prefers_shifted_key_and_rejects_non_printable_modifiers(self) -> None:
        assert decode_kitty_printable("\x1b[49:33;2u") == "!"
        assert decode_kitty_printable("\x1b[97;5u") is None
        assert decode_kitty_printable("\x1b[97;3u") is None
        assert decode_kitty_printable("\x1b[9u") is None

    def test_release_and_repeat_detection_ignores_paste_content(self) -> None:
        assert is_key_release("\x1b[99;5:3u") is True
        assert is_key_repeat("\x1b[99;5:2u") is True
        pasted_release = "\x1b[200~90:62:3F:A5\x1b[201~"
        pasted_repeat = "\x1b[200~90:62:2F:A5\x1b[201~"
        assert is_key_release(pasted_release) is False
        assert is_key_repeat(pasted_repeat) is False

    def test_lock_bits_are_masked_for_matching_parsing_and_printable_decoding(self) -> None:
        set_kitty_protocol_active(True)
        assert matches_key("\x1b[99;193u", "c") is True
        assert parse_key("\x1b[99;193u") == "c"
        assert decode_kitty_printable("\x1b[97;193u") == "a"

    def test_alt_enter_is_recognized_in_legacy_mode(self) -> None:
        set_kitty_protocol_active(False)
        assert matches_key("\x1b\r", "alt+enter") is True
        assert parse_key("\x1b\r") == "alt+enter"
