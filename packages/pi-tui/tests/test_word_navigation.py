"""Tests ported from packages/tui/test/word-navigation.test.ts."""

from pi_tui.utils import WordSegment
from pi_tui.word_navigation import WordNavigationOptions, find_word_backward, find_word_forward


class TestFindWordBackward:
    def test_basic_words_hello_world(self):
        text = "hello world"
        assert find_word_backward(text, 11) == 6
        assert find_word_backward(text, 6) == 0

    def test_dotted_foo_bar(self):
        text = "foo.bar"
        assert find_word_backward(text, 7) == 4
        assert find_word_backward(text, 4) == 3
        assert find_word_backward(text, 3) == 0

    def test_colon_foo_bar(self):
        text = "foo:bar"
        assert find_word_backward(text, 7) == 4
        assert find_word_backward(text, 4) == 3
        assert find_word_backward(text, 3) == 0

    def test_path_path_to_file(self):
        text = "path/to/file"
        assert find_word_backward(text, 12) == 8
        assert find_word_backward(text, 8) == 7
        # "/to" is one word-like segment with "/" as punctuation boundary
        assert find_word_backward(text, 7) == 5
        assert find_word_backward(text, 5) == 4
        assert find_word_backward(text, 4) == 0

    def test_cjk_mixed(self):
        # NOTE: the upstream TS test relies on Intl.Segmenter's ICU
        # dictionary-based word breaking, which recognizes "你好" and "世界"
        # as two-character dictionary words and groups each pair into a
        # single word-like segment (verified against Node's V8
        # implementation). This port's `iter_word_segments` (see
        # `pi_tui.utils`) treats each CJK character as its own word-like
        # segment instead, since replicating ICU's dictionary segmentation
        # would require embedding large per-language dictionaries (Chinese,
        # Japanese, Thai, Khmer, Lao, Myanmar), which conflicts with this
        # project's minimal-dependency policy. The assertions below reflect
        # this port's actual (documented) per-character behavior rather than
        # the upstream dictionary-aware result.
        text = "你好世界 test"
        assert find_word_backward(text, len(text)) == 5
        assert find_word_backward(text, 5) == 3
        assert find_word_backward(text, 3) == 2
        assert find_word_backward(text, 2) == 1
        assert find_word_backward(text, 1) == 0

    def test_whitespace_at_boundaries(self):
        text = "  hello  "
        assert find_word_backward(text, 9) == 2
        assert find_word_backward(text, 2) == 0

    def test_punctuation_run_foo_dot_dot_dot_bar(self):
        text = "foo...bar"
        assert find_word_backward(text, 9) == 6
        assert find_word_backward(text, 6) == 3
        assert find_word_backward(text, 3) == 0

    def test_cursor_at_0_returns_0(self):
        assert find_word_backward("hello", 0) == 0


class TestFindWordForward:
    def test_basic_words_hello_world(self):
        text = "hello world"
        assert find_word_forward(text, 0) == 5
        assert find_word_forward(text, 5) == 11

    def test_dotted_foo_bar(self):
        text = "foo.bar"
        assert find_word_forward(text, 0) == 3
        assert find_word_forward(text, 3) == 4
        assert find_word_forward(text, 4) == 7

    def test_colon_foo_bar(self):
        text = "foo:bar"
        assert find_word_forward(text, 0) == 3
        assert find_word_forward(text, 3) == 4
        assert find_word_forward(text, 4) == 7

    def test_path_path_to_file(self):
        text = "path/to/file"
        assert find_word_forward(text, 0) == 4
        assert find_word_forward(text, 4) == 5
        assert find_word_forward(text, 5) == 7
        assert find_word_forward(text, 7) == 8
        assert find_word_forward(text, 8) == 12

    def test_cjk_mixed(self):
        text = "你好世界 test"
        first_end = find_word_forward(text, 0)
        assert first_end > 0
        assert first_end <= 4

        pos = 0
        while pos < len(text):
            next_pos = find_word_forward(text, pos)
            if next_pos == pos:
                break
            pos = next_pos
        assert pos == len(text)

    def test_whitespace_at_boundaries(self):
        text = "  hello  "
        assert find_word_forward(text, 0) == 7
        assert find_word_forward(text, 7) == 9

    def test_punctuation_run_foo_dot_dot_dot_bar(self):
        text = "foo...bar"
        assert find_word_forward(text, 0) == 3
        assert find_word_forward(text, 3) == 6
        assert find_word_forward(text, 6) == 9

    def test_cursor_at_end_returns_end(self):
        assert find_word_forward("hello", 5) == 5


class TestAtomicSegments:
    marker = "[paste #1 +5 lines]"

    def _text(self) -> str:
        return f"hello {self.marker} world"

    def _is_atomic(self, s: str) -> bool:
        return s == self.marker

    def _segment_map(self, text: str) -> dict[str, list[WordSegment]]:
        marker = self.marker
        return {
            text: [
                WordSegment("hello", True),
                WordSegment(" ", False),
                WordSegment(marker, True),
                WordSegment(" ", False),
                WordSegment("world", True),
            ],
            text[:26]: [
                WordSegment("hello", True),
                WordSegment(" ", False),
                WordSegment(marker, True),
                WordSegment(" ", False),
            ],
            text[6:]: [
                WordSegment(marker, True),
                WordSegment(" ", False),
                WordSegment("world", True),
            ],
        }

    def _options(self, text: str) -> WordNavigationOptions:
        segment_map = self._segment_map(text)
        return WordNavigationOptions(
            segment=lambda input_text: segment_map.get(input_text, []),
            is_atomic_segment=self._is_atomic,
        )

    def test_backward_skips_word_then_stops_before_atomic_marker(self):
        text = self._text()
        opts = self._options(text)
        assert find_word_backward(text, len(text), opts) == 26

    def test_backward_skips_whitespace_then_atomic_marker_as_one_unit(self):
        text = self._text()
        opts = self._options(text)
        assert find_word_backward(text, 26, opts) == 6

    def test_forward_skips_atomic_marker_as_one_unit(self):
        text = self._text()
        opts = self._options(text)
        assert find_word_forward(text, 6, opts) == 6 + len(self.marker)
