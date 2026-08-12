"""Terminal string utilities: width, truncation, wrapping, ANSI handling.

Python port of `packages/tui/src/utils.ts`.

Two notable adaptations from the TypeScript source:

* **Code units vs. code points.** The TypeScript implementation indexes
  strings by UTF-16 code unit (`str.length`, `str[i]`), so a character outside
  the Basic Multilingual Plane (most emoji) occupies two indices. Python
  strings are sequences of Unicode code points, so this port operates on code
  points throughout. This is a behaviour improvement, not a bug: code-point
  indexing never splits a character in the middle, which the TypeScript
  version can do when slicing at an arbitrary UTF-16 offset.
* **Grapheme and word segmentation.** The TypeScript code uses the ICU-backed
  `Intl.Segmenter` (grapheme and word granularities), which implements the
  full Unicode text segmentation algorithm (UAX #29) from CLDR data. Node/V8
  ships this; Python's standard library does not, and this project avoids
  adding a heavy third-party dependency (e.g. `regex`, `PyICU`) for it. This
  module instead implements an approximation of the grapheme-cluster rules
  (`iter_graphemes`) directly from Unicode Character Database ranges
  (Extend/ZWJ, regional-indicator pairing, and the Unicode 15
  Indic_Conjunct_Break rule that keeps a consonant + virama + consonant
  sequence in Devanagari/Bengali/Gujarati/Oriya/Telugu/Malayalam/etc. as one
  cluster), and a simplified word segmenter (`iter_word_segments`) good enough
  for the editor word-navigation use case (see `word_navigation.py`). Exotic
  scripts or newly assigned emoji sequences not covered by the embedded
  Unicode ranges may segment slightly differently than `Intl.Segmenter`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

import wcwidth

# --------------------------------------------------------------------------
# Grapheme cluster segmentation (approximates Intl.Segmenter granularity:
# "grapheme", i.e. UAX #29 extended grapheme clusters).
# --------------------------------------------------------------------------

_MARK_CATEGORIES = frozenset({"Mn", "Mc", "Me"})
_ZWJ = "\u200d"
_ZWNJ = "\u200c"

# Indic_Conjunct_Break=Linker code points (Unicode 15+ DerivedCoreProperties.txt).
# These are the "virama"-like characters that keep a Consonant-Linker-Consonant
# sequence together as a single grapheme cluster (UAX #29 rule GB9c).
_INCB_LINKERS = frozenset(
    {
        0x094D,  # DEVANAGARI SIGN VIRAMA
        0x09CD,  # BENGALI SIGN VIRAMA
        0x0ACD,  # GUJARATI SIGN VIRAMA
        0x0B4D,  # ORIYA SIGN VIRAMA
        0x0C4D,  # TELUGU SIGN VIRAMA
        0x0D4D,  # MALAYALAM SIGN VIRAMA
        0x1039,  # MYANMAR SIGN VIRAMA
        0x17D2,  # KHMER SIGN COENG
        0x1A60,  # TAI THAM SIGN SAKOT
        0x1B44,  # BALINESE ADEG ADEG
        0x1BAB,  # SUNDANESE SIGN VIRAMA
        0xA9C0,  # JAVANESE PANGKON
        0xAAF6,  # MEETEI MAYEK VIRAMA
        0x10A3F,  # KHAROSHTHI VIRAMA
        0x11133,  # CHAKMA VIRAMA
        0x113D0,  # TULU-TIGALARI CONJOINER
        0x1193E,  # DIVES AKURU VIRAMA
        0x11A47,  # ZANABAZAR SQUARE SUBJOINER
        0x11A99,  # SOYOMBO SUBJOINER
        0x11F42,  # KAWI CONJOINER
    }
)

# Indic_Conjunct_Break=Consonant code point ranges (Unicode 15+
# DerivedCoreProperties.txt), inclusive [start, end] pairs.
_INCB_CONSONANT_RANGES: tuple[tuple[int, int], ...] = (
    (0x0915, 0x0939),
    (0x0958, 0x095F),
    (0x0978, 0x097F),
    (0x0995, 0x09A8),
    (0x09AA, 0x09B0),
    (0x09B2, 0x09B2),
    (0x09B6, 0x09B9),
    (0x09DC, 0x09DD),
    (0x09DF, 0x09DF),
    (0x09F0, 0x09F1),
    (0x0A95, 0x0AA8),
    (0x0AAA, 0x0AB0),
    (0x0AB2, 0x0AB3),
    (0x0AB5, 0x0AB9),
    (0x0AF9, 0x0AF9),
    (0x0B15, 0x0B28),
    (0x0B2A, 0x0B30),
    (0x0B32, 0x0B33),
    (0x0B35, 0x0B39),
    (0x0B5C, 0x0B5D),
    (0x0B5F, 0x0B5F),
    (0x0B71, 0x0B71),
    (0x0C15, 0x0C28),
    (0x0C2A, 0x0C39),
    (0x0C58, 0x0C5A),
    (0x0D15, 0x0D3A),
    (0x1000, 0x102A),
    (0x103F, 0x103F),
    (0x1050, 0x1055),
    (0x105A, 0x105D),
    (0x1061, 0x1061),
    (0x1065, 0x1066),
    (0x106E, 0x1070),
    (0x1075, 0x1081),
    (0x108E, 0x108E),
    (0x1780, 0x17B3),
    (0x1A20, 0x1A54),
    (0x1B0B, 0x1B0C),
    (0x1B13, 0x1B33),
    (0x1B45, 0x1B4C),
    (0x1B83, 0x1BA0),
    (0x1BAE, 0x1BAF),
    (0x1BBB, 0x1BBD),
    (0xA989, 0xA98B),
    (0xA98F, 0xA9B2),
    (0xA9E0, 0xA9E4),
    (0xA9E7, 0xA9EF),
    (0xA9FA, 0xA9FE),
    (0xAA60, 0xAA6F),
    (0xAA71, 0xAA73),
    (0xAA7A, 0xAA7A),
    (0xAA7E, 0xAA7F),
    (0xAAE0, 0xAAEA),
    (0xABC0, 0xABDA),
    (0x10A00, 0x10A00),
    (0x10A10, 0x10A13),
    (0x10A15, 0x10A17),
    (0x10A19, 0x10A35),
    (0x11103, 0x11126),
    (0x11144, 0x11144),
    (0x11147, 0x11147),
    (0x11380, 0x11389),
    (0x1138B, 0x1138B),
    (0x1138E, 0x1138E),
    (0x11390, 0x113B5),
    (0x11900, 0x11906),
    (0x11909, 0x11909),
    (0x1190C, 0x11913),
    (0x11915, 0x11916),
    (0x11918, 0x1192F),
    (0x11A00, 0x11A00),
    (0x11A0B, 0x11A32),
    (0x11A50, 0x11A50),
    (0x11A5C, 0x11A83),
    (0x11F04, 0x11F10),
    (0x11F12, 0x11F33),
)

_REGIONAL_INDICATOR_START = 0x1F1E6
_REGIONAL_INDICATOR_END = 0x1F1FF


def _is_mark(cp: int) -> bool:
    return unicodedata.category(chr(cp)) in _MARK_CATEGORIES


def _is_extend_or_joiner(cp: int) -> bool:
    """Approximates UAX #29 `Extend`/`ZWJ`: combining marks plus the joiners.

    Also includes the Fitzpatrick emoji skin-tone modifiers (U+1F3FB-U+1F3FF),
    which have `Grapheme_Cluster_Break=Extend` in the Unicode data despite
    their general category being `Sk` (Modifier_Symbol) rather than a mark.
    """
    if cp in (0x200C, 0x200D):
        return True
    if 0x1F3FB <= cp <= 0x1F3FF:
        return True
    return _is_mark(cp)


def _is_control_break(cp: int) -> bool:
    """Approximates UAX #29 `Control`: always breaks before/after (except CRLF)."""
    if cp in (0x0D, 0x0A, 0x200C, 0x200D):
        return cp not in (0x200C, 0x200D)
    category = unicodedata.category(chr(cp))
    return category in ("Cc", "Cf", "Zl", "Zp")


def _is_consonant(cp: int) -> bool:
    return any(start <= cp <= end for start, end in _INCB_CONSONANT_RANGES)


def _is_linker(cp: int) -> bool:
    return cp in _INCB_LINKERS


def _is_regional_indicator(cp: int) -> bool:
    return _REGIONAL_INDICATOR_START <= cp <= _REGIONAL_INDICATOR_END


def iter_graphemes(text: str) -> list[str]:
    """Split `text` into extended grapheme clusters (approximate UAX #29)."""
    if not text:
        return []

    clusters: list[str] = []
    current: list[str] = [text[0]]
    incb_state = "consonant" if _is_consonant(ord(text[0])) else "none"
    prev_ri_run = 1 if _is_regional_indicator(ord(text[0])) else 0

    for index in range(1, len(text)):
        prev_cp = ord(text[index - 1])
        cp = ord(text[index])
        merge: bool

        if prev_cp == 0x0D and cp == 0x0A:
            merge = True
        elif _is_control_break(prev_cp) or _is_control_break(cp):
            merge = False
        elif _is_extend_or_joiner(cp):
            merge = True
            if incb_state in ("consonant", "linker") and _is_linker(cp):
                incb_state = "linker"
        elif prev_cp in (0x200C, 0x200D):
            # Approximates GB11 (ZWJ + Extended_Pictographic): a joiner glues
            # to whatever base character follows, typically an emoji.
            merge = True
        elif incb_state == "linker" and _is_consonant(cp):
            merge = True
            incb_state = "consonant"
        elif _is_regional_indicator(prev_cp) and _is_regional_indicator(cp) and prev_ri_run % 2 == 1:
            merge = True
        else:
            merge = False

        if merge:
            current.append(text[index])
        else:
            clusters.append("".join(current))
            current = [text[index]]
            incb_state = "consonant" if _is_consonant(cp) else "none"

        prev_ri_run = (prev_ri_run + 1 if _is_regional_indicator(prev_cp) else 1) if _is_regional_indicator(cp) else 0

    clusters.append("".join(current))
    return clusters


# --------------------------------------------------------------------------
# Word segmentation (approximates Intl.Segmenter granularity: "word").
# --------------------------------------------------------------------------

# Unicode block ranges used to approximate the Script_Extensions test in
# `CJK_BREAK_PATTERN` below (Han, Hiragana, Katakana, Hangul, Bopomofo). CJK
# ideographs are treated as individual word-like segments rather than being
# grouped into runs, matching how ICU word-breaks CJK text (no dictionary).
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2EBEF),  # CJK Unified Ideographs Extension C-F
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0xFF66, 0xFF9D),  # Halfwidth Katakana
    (0xAC00, 0xD7A3),  # Hangul Syllables
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0x3100, 0x312F),  # Bopomofo
    (0x31A0, 0x31BF),  # Bopomofo Extended
)


def _is_cjk_break_char(char: str) -> bool:
    """Approximates `cjkBreakRegex` (Han/Hiragana/Katakana/Hangul/Bopomofo)."""
    cp = ord(char)
    return any(start <= cp <= end for start, end in _CJK_RANGES)


@dataclass
class WordSegment:
    """A single segment produced by `iter_word_segments`.

    Mirrors the subset of `Intl.SegmentData` (`segment`, `isWordLike`) that
    `word_navigation.py` actually consumes; `index`/`input` are not needed.
    """

    segment: str
    is_word_like: bool


def _is_word_char(char: str) -> bool:
    if char == "_":
        return True
    category = unicodedata.category(char)
    return category[0] in ("L", "N") or category in _MARK_CATEGORIES


def iter_word_segments(text: str) -> list[WordSegment]:
    """Split `text` into word-like/non-word-like segments.

    Approximates `Intl.Segmenter(locale, {granularity: "word"})`: runs of
    whitespace become one non-word-like segment, runs of ASCII/word
    characters become one word-like segment, each CJK character becomes its
    own word-like segment, and any other character (punctuation/symbols)
    becomes its own non-word-like segment.

    Approximation limit: real `Intl.Segmenter` uses ICU's dictionary-based
    word breaking for languages without space-delimited words (Chinese,
    Japanese, Thai, Khmer, Lao, Myanmar), which can group multiple CJK
    characters into one word-like segment when they form a known dictionary
    word (e.g. `"你好"` "hello" and `"世界"` "world" each segment as one
    two-character word in V8's ICU data). Replicating that would require
    embedding per-language dictionaries, which conflicts with this project's
    minimal-dependency policy, so this port always treats each CJK character
    as its own word-like segment instead. This only affects word-navigation
    step granularity within CJK runs; it does not affect non-CJK text.
    """
    if not text:
        return []

    segments: list[WordSegment] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if is_whitespace_char(char):
            end = index + 1
            while end < length and is_whitespace_char(text[end]):
                end += 1
            segments.append(WordSegment(text[index:end], False))
            index = end
        elif _is_cjk_break_char(char):
            segments.append(WordSegment(char, True))
            index += 1
        elif _is_word_char(char):
            end = index + 1
            while end < length and _is_word_char(text[end]) and not _is_cjk_break_char(text[end]):
                end += 1
            segments.append(WordSegment(text[index:end], True))
            index = end
        else:
            segments.append(WordSegment(char, False))
            index += 1
    return segments


_JS_WHITESPACE_PATTERN = re.compile("[\t\n\x0b\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]")

# Matches `PUNCTUATION_REGEX` in utils.ts; exported for `word_navigation.py`,
# which scans for the last/first ASCII-punctuation boundary inside a
# word-like segment.
PUNCTUATION_REGEX = re.compile(r"""[(){}\[\]<>.,;:'"!?+\-=*/\\|&%^$#@~`]""")


def is_whitespace_char(char: str) -> bool:
    """Check if a character (or any char within a short string) is whitespace."""
    return bool(_JS_WHITESPACE_PATTERN.search(char))


def is_punctuation_char(char: str) -> bool:
    """Check if a character is ASCII punctuation."""
    return bool(PUNCTUATION_REGEX.search(char))


# --------------------------------------------------------------------------
# Width calculation
# --------------------------------------------------------------------------

_SPACING_MARK_EXCLUDE = frozenset({0x1734, 0x302E, 0x302F})
_EXTRA_TERMINAL_SPACING_MARKS = frozenset(
    {0x065F, 0x0F7F, 0x102B, 0x102C, 0x1031, 0x1033, 0x1034, 0x1035, 0x1038, 0x103A, 0x103B, 0x103C, 0x103D, 0x103E}
)

WIDTH_CACHE_SIZE = 512
_width_cache: dict[str, int] = {}


def _is_terminal_spacing_mark(char: str) -> bool:
    """Marks that terminals allocate a cell for, matching `terminalSpacingMarkRegex`."""
    cp = ord(char)
    if cp in _SPACING_MARK_EXCLUDE:
        return False
    if unicodedata.category(char) == "Mc":
        return True
    return cp in _EXTRA_TERMINAL_SPACING_MARKS


def _is_nonprinting_char(char: str) -> bool:
    """Approximates `nonPrintingCharRegex`: ignorable/control/format/mark/surrogate."""
    cp = ord(char)
    if 0xD800 <= cp <= 0xDFFF:
        return True
    category = unicodedata.category(char)
    return category in ("Cc", "Cf", "Mn", "Mc", "Me")


def _is_probable_emoji_grapheme(segment: str) -> bool:
    """Approximates `couldBeEmoji(...) && rgiEmojiRegex.test(...)`.

    The TypeScript version validates against the full Unicode RGI_Emoji
    sequence data (thousands of emoji/ZWJ/flag/keycap/tag sequences), which
    is impractical to embed here. This checks the same broad code point
    blocks the TypeScript pre-filter (`couldBeEmoji`) uses, plus the presence
    of an emoji variation selector (U+FE0F) or zero-width joiner, which
    together account for the overwhelming majority of single- and
    multi-codepoint emoji grapheme clusters (flags, skin tones, ZWJ
    sequences). It deliberately does not treat "multiple non-mark code
    points" alone as emoji-like (unlike a naive reading of `couldBeEmoji`),
    since that would misclassify ordinary Indic consonant conjuncts.
    """
    cp = ord(segment[0])
    if 0x1F000 <= cp <= 0x1FBFF:
        return True
    if 0x2300 <= cp <= 0x23FF:
        return True
    if 0x2600 <= cp <= 0x27BF:
        return True
    if 0x2B50 <= cp <= 0x2B55:
        return True
    if "\ufe0f" in segment:
        return True
    return _ZWJ in segment


def _strip_leading_nonprinting(segment: str) -> str:
    index = 0
    while index < len(segment) and _is_nonprinting_char(segment[index]):
        index += 1
    return segment[index:]


def _east_asian_width(cp: int) -> int:
    """Port of `eastAsianWidth`: always 1 (narrow) or 2 (wide/fullwidth)."""
    return 2 if wcwidth.wcwidth(chr(cp)) == 2 else 1


def _grapheme_width(segment: str) -> int:
    """Terminal cell width of a single grapheme cluster."""
    if segment == "\t":
        return 3

    if segment and all(_is_terminal_spacing_mark(char) for char in segment):
        return len(segment)

    if segment and all(_is_nonprinting_char(char) for char in segment):
        return 0

    if _is_probable_emoji_grapheme(segment):
        return 2

    base = _strip_leading_nonprinting(segment)
    if not base:
        return 0
    cp = ord(base[0])

    if _is_regional_indicator(cp):
        return 2

    width = _east_asian_width(cp)

    follows_mark = False
    for char in base[1:]:
        if _is_terminal_spacing_mark(char):
            width += 1
            follows_mark = False
        elif _is_mark(ord(char)):
            follows_mark = True
        elif not _is_nonprinting_char(char):
            c = ord(char)
            if follows_mark or (0xFF00 <= c <= 0xFFEF):
                width += _east_asian_width(c)
            elif c in (0x0E33, 0x0EB3):
                width += 1
            follows_mark = False

    return width


def visible_width(text: str) -> int:
    """Calculate the visible width of a string in terminal columns."""
    if not text:
        return 0

    if text.isascii() and all(0x20 <= ord(c) <= 0x7E for c in text):
        return len(text)

    cached = _width_cache.get(text)
    if cached is not None:
        return cached

    clean = text
    if "\t" in clean:
        clean = clean.replace("\t", "   ")
    if "\x1b" in clean:
        stripped_chars: list[str] = []
        index = 0
        while index < len(clean):
            ansi = extract_ansi_code(clean, index)
            if ansi is not None:
                index += ansi[1]
                continue
            stripped_chars.append(clean[index])
            index += 1
        clean = "".join(stripped_chars)

    width = sum(_grapheme_width(segment) for segment in iter_graphemes(clean))

    if len(_width_cache) >= WIDTH_CACHE_SIZE:
        first_key = next(iter(_width_cache), None)
        if first_key is not None:
            del _width_cache[first_key]
    _width_cache[text] = width

    return width


def strip_terminal_sequences(text: str) -> str:
    """Remove ANSI, OSC, and APC control sequences while preserving visible text."""
    if "\x1b" not in text:
        return text
    result: list[str] = []
    index = 0
    while index < len(text):
        ansi = extract_ansi_code(text, index)
        if ansi is not None:
            index += ansi[1]
            continue
        result.append(text[index])
        index += 1
    return "".join(result)


def get_grapheme_cell_range(line: str, column: int) -> tuple[int, int] | None:
    """Return the terminal-cell range occupied by the grapheme at a visible column."""
    current_col = 0
    index = 0
    while index < len(line):
        ansi = extract_ansi_code(line, index)
        if ansi is not None:
            index += ansi[1]
            continue
        text_end = index
        while text_end < len(line) and extract_ansi_code(line, text_end) is None:
            text_end += 1
        for segment in iter_graphemes(line[index:text_end]):
            width = _grapheme_width(segment)
            if width > 0 and current_col <= column < current_col + width:
                return (current_col, current_col + width)
            current_col += width
        index = text_end
    return None


_OSC8_HYPERLINK_PATTERN = re.compile(r"^\x1b\]8;[^;]*;([^\x07\x1b]*)(?:\x07|\x1b\\)$")


def get_osc8_link_at_column(line: str, column: int) -> str | None:
    """Return the OSC 8 hyperlink covering a visible terminal column."""
    active_url: str | None = None
    current_col = 0
    index = 0
    while index < len(line):
        ansi = extract_ansi_code(line, index)
        if ansi is not None:
            match = _OSC8_HYPERLINK_PATTERN.match(ansi[0])
            if match:
                active_url = match.group(1) or None
            index += ansi[1]
            continue
        text_end = index
        while text_end < len(line) and extract_ansi_code(line, text_end) is None:
            text_end += 1
        for segment in iter_graphemes(line[index:text_end]):
            width = 3 if segment == "\t" else _grapheme_width(segment)
            if current_col <= column < current_col + width:
                return active_url
            current_col += width
        index = text_end
    return None


_THAI_LAO_AM_PATTERN = re.compile("[\u0e33\u0eb3]")
_THAI_LAO_AM_MAP = {"\u0e33": "\u0e4d\u0e32", "\u0eb3": "\u0ecd\u0eb2"}


def normalize_terminal_output(text: str) -> str:
    """Normalize text for terminal output without changing logical editor content.

    Some terminals render precomposed Thai/Lao AM vowels inconsistently
    during differential repaint. Their compatibility decompositions have the
    same cell width but avoid stale-cell artifacts in terminal renderers.
    Visible tabs are expanded to the fixed width used by layout so terminal
    tab stops cannot wrap a logical line, while tabs inside terminal string
    sequences stay untouched.
    """
    normalized = text
    if _THAI_LAO_AM_PATTERN.search(normalized):
        normalized = _THAI_LAO_AM_PATTERN.sub(lambda m: _THAI_LAO_AM_MAP[m.group(0)], normalized)
    if "\t" not in normalized:
        return normalized

    result: list[str] = []
    index = 0
    while index < len(normalized):
        ansi = extract_ansi_code(normalized, index)
        if ansi is not None:
            result.append(ansi[0])
            index += ansi[1]
            continue
        result.append("   " if normalized[index] == "\t" else normalized[index])
        index += 1
    return "".join(result)


_CSI_TERMINATOR_PATTERN = re.compile(r"[mGKHJ]")


def extract_ansi_code(text: str, pos: int) -> tuple[str, int] | None:
    """Extract an ANSI escape sequence from `text` at `pos`, if present."""
    if pos >= len(text) or text[pos] != "\x1b":
        return None

    next_char = text[pos + 1] if pos + 1 < len(text) else ""

    if next_char == "[":
        j = pos + 2
        while j < len(text) and not _CSI_TERMINATOR_PATTERN.match(text[j]):
            j += 1
        if j < len(text):
            return (text[pos : j + 1], j + 1 - pos)
        return None

    if next_char == "]":
        j = pos + 2
        while j < len(text):
            if text[j] == "\x07":
                return (text[pos : j + 1], j + 1 - pos)
            if text[j] == "\x1b" and j + 1 < len(text) and text[j + 1] == "\\":
                return (text[pos : j + 2], j + 2 - pos)
            j += 1
        return None

    if next_char == "_":
        j = pos + 2
        while j < len(text):
            if text[j] == "\x07":
                return (text[pos : j + 1], j + 1 - pos)
            if text[j] == "\x1b" and j + 1 < len(text) and text[j + 1] == "\\":
                return (text[pos : j + 2], j + 2 - pos)
            j += 1
        return None

    return None


_OSC8_TERMINATOR_BEL = "\x07"
_OSC8_TERMINATOR_ST = "\x1b\\"


@dataclass
class _ActiveHyperlink:
    params: str
    url: str
    terminator: str


def _parse_osc8_hyperlink(ansi_code: str) -> _ActiveHyperlink | Ellipsis | None:
    """Returns an `_ActiveHyperlink`, `None` for a close sequence, or `Ellipsis`
    (sentinel for TypeScript's `undefined`) if `ansi_code` is not OSC 8 at all."""
    if not ansi_code.startswith("\x1b]8;"):
        return Ellipsis

    terminator = _OSC8_TERMINATOR_BEL if ansi_code.endswith(_OSC8_TERMINATOR_BEL) else _OSC8_TERMINATOR_ST
    body = ansi_code[4 : -1 if terminator == _OSC8_TERMINATOR_BEL else -2]
    separator_index = body.find(";")
    if separator_index == -1:
        return Ellipsis

    params = body[:separator_index]
    url = body[separator_index + 1 :]
    if not url:
        return None
    return _ActiveHyperlink(params, url, terminator)


def _format_osc8_hyperlink(hyperlink: _ActiveHyperlink) -> str:
    return f"\x1b]8;{hyperlink.params};{hyperlink.url}{hyperlink.terminator}"


def _format_osc8_close(terminator: str) -> str:
    return f"\x1b]8;;{terminator}"


def _get_active_osc8_close(prefix: str) -> str:
    if "\x1b]8;" not in prefix:
        return ""

    active_hyperlink: _ActiveHyperlink | None = None
    index = 0
    while index < len(prefix):
        ansi = extract_ansi_code(prefix, index)
        if ansi is not None:
            hyperlink = _parse_osc8_hyperlink(ansi[0])
            if hyperlink is not Ellipsis:
                active_hyperlink = hyperlink
            index += ansi[1]
        else:
            index += 1
    return _format_osc8_close(active_hyperlink.terminator) if active_hyperlink else ""


_SGR_PARAMS_PATTERN = re.compile(r"\x1b\[([\d;]*)m")


class _AnsiCodeTracker:
    """Tracks active ANSI SGR codes to preserve styling across line breaks."""

    def __init__(self) -> None:
        self.bold = False
        self.dim = False
        self.italic = False
        self.underline = False
        self.blink = False
        self.inverse = False
        self.hidden = False
        self.strikethrough = False
        self.fg_color: str | None = None
        self.bg_color: str | None = None
        self.active_hyperlink: _ActiveHyperlink | None = None

    def process(self, ansi_code: str) -> None:
        hyperlink = _parse_osc8_hyperlink(ansi_code)
        if hyperlink is not Ellipsis:
            self.active_hyperlink = hyperlink
            return

        if not ansi_code.endswith("m"):
            return

        match = _SGR_PARAMS_PATTERN.match(ansi_code)
        if not match:
            return

        params = match.group(1)
        if params in ("", "0"):
            self._reset()
            return

        parts = params.split(";")
        i = 0
        while i < len(parts):
            code = int(parts[i]) if parts[i].isdigit() or (parts[i] and parts[i][0] == "-") else 0

            if code in (38, 48):
                if i + 1 < len(parts) and parts[i + 1] == "5" and i + 2 < len(parts) and parts[i + 2] != "":
                    color_code = f"{parts[i]};{parts[i + 1]};{parts[i + 2]}"
                    if code == 38:
                        self.fg_color = color_code
                    else:
                        self.bg_color = color_code
                    i += 3
                    continue
                if i + 1 < len(parts) and parts[i + 1] == "2" and i + 4 < len(parts) and parts[i + 4] != "":
                    color_code = f"{parts[i]};{parts[i + 1]};{parts[i + 2]};{parts[i + 3]};{parts[i + 4]}"
                    if code == 38:
                        self.fg_color = color_code
                    else:
                        self.bg_color = color_code
                    i += 5
                    continue

            if code == 0:
                self._reset()
            elif code == 1:
                self.bold = True
            elif code == 2:
                self.dim = True
            elif code == 3:
                self.italic = True
            elif code == 4:
                self.underline = True
            elif code == 5:
                self.blink = True
            elif code == 7:
                self.inverse = True
            elif code == 8:
                self.hidden = True
            elif code == 9:
                self.strikethrough = True
            elif code == 21:
                self.bold = False
            elif code == 22:
                self.bold = False
                self.dim = False
            elif code == 23:
                self.italic = False
            elif code == 24:
                self.underline = False
            elif code == 25:
                self.blink = False
            elif code == 27:
                self.inverse = False
            elif code == 28:
                self.hidden = False
            elif code == 29:
                self.strikethrough = False
            elif code == 39:
                self.fg_color = None
            elif code == 49:
                self.bg_color = None
            elif (30 <= code <= 37) or (90 <= code <= 97):
                self.fg_color = str(code)
            elif (40 <= code <= 47) or (100 <= code <= 107):
                self.bg_color = str(code)
            i += 1

    def _reset(self) -> None:
        self.bold = False
        self.dim = False
        self.italic = False
        self.underline = False
        self.blink = False
        self.inverse = False
        self.hidden = False
        self.strikethrough = False
        self.fg_color = None
        self.bg_color = None

    def clear(self) -> None:
        self._reset()
        self.active_hyperlink = None

    def get_active_codes(self) -> str:
        codes: list[str] = []
        if self.bold:
            codes.append("1")
        if self.dim:
            codes.append("2")
        if self.italic:
            codes.append("3")
        if self.underline:
            codes.append("4")
        if self.blink:
            codes.append("5")
        if self.inverse:
            codes.append("7")
        if self.hidden:
            codes.append("8")
        if self.strikethrough:
            codes.append("9")
        if self.fg_color:
            codes.append(self.fg_color)
        if self.bg_color:
            codes.append(self.bg_color)

        result = f"\x1b[{';'.join(codes)}m" if codes else ""
        if self.active_hyperlink:
            result += _format_osc8_hyperlink(self.active_hyperlink)
        return result

    def has_active_codes(self) -> bool:
        return (
            self.bold
            or self.dim
            or self.italic
            or self.underline
            or self.blink
            or self.inverse
            or self.hidden
            or self.strikethrough
            or self.fg_color is not None
            or self.bg_color is not None
            or self.active_hyperlink is not None
        )

    def get_line_end_reset(self) -> str:
        """Reset codes for attributes that need to be turned off at line end.

        Underline must be closed to prevent bleeding into padding. Active
        OSC 8 hyperlinks must be closed and re-opened on the next line.
        """
        result = ""
        if self.underline:
            result += "\x1b[24m"
        if self.active_hyperlink:
            result += _format_osc8_close(self.active_hyperlink.terminator)
        return result


def _update_tracker_from_text(text: str, tracker: _AnsiCodeTracker) -> None:
    index = 0
    while index < len(text):
        ansi = extract_ansi_code(text, index)
        if ansi is not None:
            tracker.process(ansi[0])
            index += ansi[1]
        else:
            index += 1


CJK_BREAK_REGEX = _is_cjk_break_char  # See module docstring re: Script_Extensions approximation.


def _split_into_tokens_with_ansi(text: str) -> list[str]:
    """Split text into words while keeping ANSI codes attached."""
    tokens: list[str] = []
    current = ""
    pending_ansi = ""
    current_kind: str | None = None
    index = 0

    def flush_current() -> None:
        nonlocal current, current_kind
        if current:
            tokens.append(current)
            current = ""
            current_kind = None

    while index < len(text):
        ansi = extract_ansi_code(text, index)
        if ansi is not None:
            pending_ansi += ansi[0]
            index += ansi[1]
            continue

        end = index
        while end < len(text) and extract_ansi_code(text, end) is None:
            end += 1

        for segment in iter_graphemes(text[index:end]):
            segment_is_space = segment == " "
            if not segment_is_space and _is_cjk_break_char(segment[0]):
                flush_current()
                token = pending_ansi + segment
                pending_ansi = ""
                tokens.append(token)
                continue

            segment_kind = "space" if segment_is_space else "word"
            if current and current_kind != segment_kind:
                flush_current()

            if pending_ansi:
                current += pending_ansi
                pending_ansi = ""

            current_kind = segment_kind
            current += segment

        index = end

    if pending_ansi:
        if current:
            current += pending_ansi
        elif tokens:
            tokens[-1] += pending_ansi
        else:
            current = pending_ansi

    if current:
        tokens.append(current)

    return tokens


def _break_long_word(word: str, width: int, tracker: _AnsiCodeTracker) -> list[str]:
    lines: list[str] = []
    current_line = tracker.get_active_codes()
    current_width = 0

    segments: list[tuple[str, str]] = []  # (kind, value) where kind is "ansi" or "grapheme"
    index = 0
    while index < len(word):
        ansi = extract_ansi_code(word, index)
        if ansi is not None:
            segments.append(("ansi", ansi[0]))
            index += ansi[1]
        else:
            end = index
            while end < len(word) and extract_ansi_code(word, end) is None:
                end += 1
            for segment in iter_graphemes(word[index:end]):
                segments.append(("grapheme", segment))
            index = end

    for kind, value in segments:
        if kind == "ansi":
            current_line += value
            tracker.process(value)
            continue

        if not value:
            continue

        grapheme_width = visible_width(value)
        if current_width + grapheme_width > width:
            line_end_reset = tracker.get_line_end_reset()
            if line_end_reset:
                current_line += line_end_reset
            lines.append(current_line)
            current_line = tracker.get_active_codes()
            current_width = 0

        current_line += value
        current_width += grapheme_width

    if current_line:
        lines.append(current_line)

    return lines if lines else [""]


def _wrap_single_line(line: str, width: int) -> list[str]:
    if not line:
        return [""]

    visible_length = visible_width(line)
    if visible_length <= width:
        return [line]

    wrapped: list[str] = []
    tracker = _AnsiCodeTracker()
    tokens = _split_into_tokens_with_ansi(line)

    current_line = ""
    current_visible_length = 0

    for token in tokens:
        token_visible_length = visible_width(token)
        is_whitespace = token.strip() == ""

        if token_visible_length > width and not is_whitespace:
            if current_line:
                line_end_reset = tracker.get_line_end_reset()
                if line_end_reset:
                    current_line += line_end_reset
                wrapped.append(current_line)
                current_line = ""
                current_visible_length = 0

            broken = _break_long_word(token, width, tracker)
            wrapped.extend(broken[:-1])
            current_line = broken[-1]
            current_visible_length = visible_width(current_line)
            continue

        total_needed = current_visible_length + token_visible_length

        if total_needed > width and current_visible_length > 0:
            line_to_wrap = current_line.rstrip()
            line_end_reset = tracker.get_line_end_reset()
            if line_end_reset:
                line_to_wrap += line_end_reset
            wrapped.append(line_to_wrap)
            if is_whitespace:
                current_line = tracker.get_active_codes()
                current_visible_length = 0
            else:
                current_line = tracker.get_active_codes() + token
                current_visible_length = token_visible_length
        else:
            current_line += token
            current_visible_length += token_visible_length

        _update_tracker_from_text(token, tracker)

    if current_line:
        wrapped.append(current_line)

    return [line.rstrip() for line in wrapped] if wrapped else [""]


def wrap_text_with_ansi(text: str, width: int) -> list[str]:
    """Wrap text with ANSI codes preserved.

    ONLY does word wrapping - NO padding, NO background colors. Returns lines
    where each line is <= width visible chars. Active ANSI codes are
    preserved across line breaks.
    """
    if not text:
        return [""]

    input_lines = re.split(r"\r\n|\r|\n", text)
    result: list[str] = []
    tracker = _AnsiCodeTracker()

    for input_line in input_lines:
        prefix = tracker.get_active_codes() if result else ""
        wrapped_lines = _wrap_single_line(prefix + input_line, width)
        result.extend(wrapped_lines)
        _update_tracker_from_text(input_line, tracker)

    return result if result else [""]


def apply_background_to_line(line: str, width: int, bg_fn: Callable[[str], str]) -> str:
    """Apply background color to a line, padding to full width."""
    visible_len = visible_width(line)
    padding_needed = max(0, width - visible_len)
    padding = " " * padding_needed
    return bg_fn(line + padding)


def _truncate_fragment_to_width(text: str, max_width: int) -> tuple[str, int]:
    if max_width <= 0 or not text:
        return ("", 0)

    if text.isascii() and all(0x20 <= ord(c) <= 0x7E for c in text):
        clipped = text[:max_width]
        return (clipped, len(clipped))

    has_ansi = "\x1b" in text
    has_tabs = "\t" in text
    if not has_ansi and not has_tabs:
        result = ""
        width = 0
        for segment in iter_graphemes(text):
            w = _grapheme_width(segment)
            if width + w > max_width:
                break
            result += segment
            width += w
        return (result, width)

    result = ""
    width = 0
    index = 0
    pending_ansi = ""

    while index < len(text):
        ansi = extract_ansi_code(text, index)
        if ansi is not None:
            pending_ansi += ansi[0]
            index += ansi[1]
            continue

        if text[index] == "\t":
            if width + 3 > max_width:
                break
            if pending_ansi:
                result += pending_ansi
                pending_ansi = ""
            result += "\t"
            width += 3
            index += 1
            continue

        end = index
        while end < len(text) and text[end] != "\t":
            if extract_ansi_code(text, end) is not None:
                break
            end += 1

        stopped = False
        for segment in iter_graphemes(text[index:end]):
            w = _grapheme_width(segment)
            if width + w > max_width:
                stopped = True
                break
            if pending_ansi:
                result += pending_ansi
                pending_ansi = ""
            result += segment
            width += w
        if stopped:
            break
        index = end

    return (result, width)


def _finalize_truncated_result(
    prefix: str, prefix_width: int, ellipsis: str, ellipsis_width: int, max_width: int, pad: bool
) -> str:
    reset = "\x1b[0m"
    hyperlink_close = _get_active_osc8_close(prefix)
    visible = prefix_width + ellipsis_width

    if ellipsis:
        result = f"{prefix}{hyperlink_close}{reset}{ellipsis}{reset}"
    else:
        result = f"{prefix}{hyperlink_close}{reset}"

    return result + " " * max(0, max_width - visible) if pad else result


def truncate_to_width(text: str, max_width: int, ellipsis: str = "...", pad: bool = False) -> str:
    """Truncate text to fit within a maximum visible width, adding ellipsis if needed.

    Optionally pads with spaces to reach exactly `max_width`. Properly
    handles ANSI escape codes (they don't count toward width).
    """
    if max_width <= 0:
        return ""

    if not text:
        return " " * max_width if pad else ""

    ellipsis_width = visible_width(ellipsis)
    if ellipsis_width >= max_width:
        text_width = visible_width(text)
        if text_width <= max_width:
            return text + " " * (max_width - text_width) if pad else text

        clipped_ellipsis_text, clipped_ellipsis_width = _truncate_fragment_to_width(ellipsis, max_width)
        if clipped_ellipsis_width == 0:
            return " " * max_width if pad else ""
        return _finalize_truncated_result("", 0, clipped_ellipsis_text, clipped_ellipsis_width, max_width, pad)

    if text.isascii() and all(0x20 <= ord(c) <= 0x7E for c in text):
        if len(text) <= max_width:
            return text + " " * (max_width - len(text)) if pad else text
        target_width = max_width - ellipsis_width
        return _finalize_truncated_result(text[:target_width], target_width, ellipsis, ellipsis_width, max_width, pad)

    target_width = max_width - ellipsis_width
    result = ""
    pending_ansi = ""
    visible_so_far = 0
    kept_width = 0
    keep_contiguous_prefix = True
    overflowed = False
    exhausted_input = False
    has_ansi = "\x1b" in text
    has_tabs = "\t" in text

    if not has_ansi and not has_tabs:
        for segment in iter_graphemes(text):
            width = _grapheme_width(segment)
            if keep_contiguous_prefix and kept_width + width <= target_width:
                result += segment
                kept_width += width
            else:
                keep_contiguous_prefix = False
            visible_so_far += width
            if visible_so_far > max_width:
                overflowed = True
                break
        exhausted_input = not overflowed
    else:
        index = 0
        while index < len(text):
            ansi = extract_ansi_code(text, index)
            if ansi is not None:
                pending_ansi += ansi[0]
                index += ansi[1]
                continue

            if text[index] == "\t":
                if keep_contiguous_prefix and kept_width + 3 <= target_width:
                    if pending_ansi:
                        result += pending_ansi
                        pending_ansi = ""
                    result += "\t"
                    kept_width += 3
                else:
                    keep_contiguous_prefix = False
                    pending_ansi = ""
                visible_so_far += 3
                if visible_so_far > max_width:
                    overflowed = True
                    break
                index += 1
                continue

            end = index
            while end < len(text) and text[end] != "\t":
                if extract_ansi_code(text, end) is not None:
                    break
                end += 1

            for segment in iter_graphemes(text[index:end]):
                width = _grapheme_width(segment)
                if keep_contiguous_prefix and kept_width + width <= target_width:
                    if pending_ansi:
                        result += pending_ansi
                        pending_ansi = ""
                    result += segment
                    kept_width += width
                else:
                    keep_contiguous_prefix = False
                    pending_ansi = ""

                visible_so_far += width
                if visible_so_far > max_width:
                    overflowed = True
                    break
            if overflowed:
                break
            index = end
        exhausted_input = index >= len(text)

    if not overflowed and exhausted_input:
        return text + " " * max(0, max_width - visible_so_far) if pad else text

    return _finalize_truncated_result(result, kept_width, ellipsis, ellipsis_width, max_width, pad)


def slice_by_column(line: str, start_col: int, length: int, strict: bool = False) -> str:
    """Extract a range of visible columns from a line.

    Handles ANSI codes and wide chars. If `strict`, exclude wide chars at the
    boundary that would extend past the range.
    """
    return slice_with_width(line, start_col, length, strict)[0]


def slice_with_width(line: str, start_col: int, length: int, strict: bool = False) -> tuple[str, int]:
    """Like `slice_by_column` but also returns the actual visible width of the result."""
    if length <= 0:
        return ("", 0)
    end_col = start_col + length
    result = ""
    result_width = 0
    current_col = 0
    index = 0
    pending_ansi = ""

    while index < len(line):
        ansi = extract_ansi_code(line, index)
        if ansi is not None:
            if start_col <= current_col < end_col:
                result += ansi[0]
            elif current_col < start_col:
                pending_ansi += ansi[0]
            index += ansi[1]
            continue

        text_end = index
        while text_end < len(line) and extract_ansi_code(line, text_end) is None:
            text_end += 1

        done = False
        for segment in iter_graphemes(line[index:text_end]):
            w = _grapheme_width(segment)
            in_range = start_col <= current_col < end_col
            fits = not strict or current_col + w <= end_col
            if in_range and fits:
                if pending_ansi:
                    result += pending_ansi
                    pending_ansi = ""
                result += segment
                result_width += w
            current_col += w
            if current_col >= end_col:
                done = True
                break
        index = text_end
        if done:
            break

    return (result, result_width)


def extract_segments(
    line: str, before_end: int, after_start: int, after_len: int, strict_after: bool = False
) -> tuple[str, int, str, int]:
    """Extract "before" and "after" segments from a line in a single pass.

    Used for overlay compositing where we need content before and after the
    overlay region. Preserves styling from before the overlay that should
    affect content after it. Returns `(before, before_width, after,
    after_width)`.
    """
    before = ""
    before_width = 0
    after = ""
    after_width = 0
    current_col = 0
    index = 0
    pending_ansi_before = ""
    after_started = False
    after_end = after_start + after_len

    tracker = _AnsiCodeTracker()

    while index < len(line):
        ansi = extract_ansi_code(line, index)
        if ansi is not None:
            tracker.process(ansi[0])
            if current_col < before_end:
                pending_ansi_before += ansi[0]
            elif after_start <= current_col < after_end and after_started:
                after += ansi[0]
            index += ansi[1]
            continue

        text_end = index
        while text_end < len(line) and extract_ansi_code(line, text_end) is None:
            text_end += 1

        done = False
        for segment in iter_graphemes(line[index:text_end]):
            w = _grapheme_width(segment)

            if current_col < before_end and current_col + w <= before_end:
                if pending_ansi_before:
                    before += pending_ansi_before
                    pending_ansi_before = ""
                before += segment
                before_width += w
            elif after_start <= current_col < after_end:
                fits = not strict_after or current_col + w <= after_end
                if fits:
                    if not after_started:
                        after += tracker.get_active_codes()
                        after_started = True
                    after += segment
                    after_width += w

            current_col += w
            if (current_col >= before_end) if after_len <= 0 else (current_col >= after_end):
                done = True
                break
        index = text_end
        if done:
            break

    return (before, before_width, after, after_width)
