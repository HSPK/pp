"""Tests ported from packages/tui/test/input.test.ts."""

from pi_tui.components.input import Input
from pi_tui.utils import visible_width


class TestInputComponent:
    def test_submits_value_including_backslash_on_enter(self):
        input_ = Input()
        submitted = []

        input_.on_submit = submitted.append

        input_.handle_input("h")
        input_.handle_input("e")
        input_.handle_input("l")
        input_.handle_input("l")
        input_.handle_input("o")
        input_.handle_input("\\")
        input_.handle_input("\r")

        assert submitted == ["hello\\"]

    def test_inserts_backslash_as_regular_character(self):
        input_ = Input()

        input_.handle_input("\\")
        input_.handle_input("x")

        assert input_.get_value() == "\\x"


class TestInputRender:
    def test_does_not_overflow_with_wide_cjk_and_fullwidth_text(self):
        width = 93
        cases = [
            "가나다라마바사아자차카타파하 한글 텍스트가 터미널 너비를 초과하면 크래시가 발생합니다 이것은 재현용 테스트입니다",
            "これはテスト文章です。日本語のテキストが正しく表示されるかどうかを確認するためのサンプルテキストです。あいうえお",
            "这是一段测试文本，用于验证中文字符在终端中的显示宽度是否被正确计算，如果不正确就会导致用户界面崩溃的问题",
            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍ",
        ]

        def move_start(_input):
            pass

        def move_middle(input_):
            for _ in range(10):
                input_.handle_input("\x1b[C")

        def move_end(input_):
            input_.handle_input("\x05")

        cursor_positions = [
            ("start", move_start),
            ("middle", move_middle),
            ("end", move_end),
        ]

        for text in cases:
            for label, move in cursor_positions:
                input_ = Input()
                input_.set_value(text)
                input_.focused = True
                move(input_)

                lines = input_.render(width)
                line = lines[0]
                assert line
                assert visible_width(line) <= width, f"rendered line overflowed for {text} at {label}"

    def test_keeps_cursor_visible_when_horizontally_scrolling_wide_text(self):
        input_ = Input()
        width = 20
        text = "가나다라마바사아자차카타파하"
        input_.set_value(text)
        input_.focused = True
        input_.handle_input("\x01")
        for _ in range(5):
            input_.handle_input("\x1b[C")

        lines = input_.render(width)
        line = lines[0]
        assert line
        assert visible_width(line) <= width


class TestInputKillRing:
    def test_ctrl_w_saves_to_kill_ring_and_ctrl_y_yanks(self):
        input_ = Input()

        input_.set_value("foo bar baz")
        input_.handle_input("\x05")  # Ctrl+E

        input_.handle_input("\x17")  # Ctrl+W - deletes "baz"
        assert input_.get_value() == "foo bar "

        input_.handle_input("\x01")  # Ctrl+A
        input_.handle_input("\x19")  # Ctrl+Y
        assert input_.get_value() == "bazfoo bar "

    def test_ctrl_w_preserves_ascii_punctuation_boundaries(self):
        input_ = Input()

        input_.set_value("foo.bar")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        assert input_.get_value() == "foo."

        input_.set_value("foo:bar")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        assert input_.get_value() == "foo:"

    def test_ctrl_w_handles_unicode_word_boundaries(self):
        # Upstream TS relies on Intl.Segmenter's ICU *dictionary*-based word
        # segmentation, which groups "你好世界。你好，世界" into
        # 你好|世界|。|你好|，|世界 and deletes two-character chunks at a
        # time. This port's `iter_word_segments` treats each CJK character as
        # its own word-like segment (dictionary-free approximation, see
        # `utils.py`), so Ctrl+W here deletes one character/punctuation mark
        # at a time instead. See word_navigation.py's documented limitation.
        input_ = Input()

        input_.set_value("你好世界。你好，世界")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        assert input_.get_value() == "你好世界。你好，世"
        input_.handle_input("\x17")
        assert input_.get_value() == "你好世界。你好，"
        input_.handle_input("\x17")
        assert input_.get_value() == "你好世界。你好"
        input_.handle_input("\x17")
        assert input_.get_value() == "你好世界。你"
        input_.handle_input("\x17")
        assert input_.get_value() == "你好世界。"
        input_.handle_input("\x17")
        assert input_.get_value() == "你好世界"

    def test_ctrl_u_saves_deleted_text_to_kill_ring(self):
        input_ = Input()

        input_.set_value("hello world")
        input_.handle_input("\x01")
        for _ in range(6):
            input_.handle_input("\x1b[C")

        input_.handle_input("\x15")  # Ctrl+U
        assert input_.get_value() == "world"

        input_.handle_input("\x19")
        assert input_.get_value() == "hello world"

    def test_ctrl_k_saves_deleted_text_to_kill_ring(self):
        input_ = Input()

        input_.set_value("hello world")
        input_.handle_input("\x01")
        input_.handle_input("\x0b")  # Ctrl+K

        assert input_.get_value() == ""

        input_.handle_input("\x19")
        assert input_.get_value() == "hello world"

    def test_ctrl_y_does_nothing_when_kill_ring_empty(self):
        input_ = Input()

        input_.set_value("test")
        input_.handle_input("\x05")
        input_.handle_input("\x19")
        assert input_.get_value() == "test"

    def test_alt_y_cycles_through_kill_ring_after_ctrl_y(self):
        input_ = Input()

        input_.set_value("first")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        input_.set_value("second")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        input_.set_value("third")
        input_.handle_input("\x05")
        input_.handle_input("\x17")

        assert input_.get_value() == ""

        input_.handle_input("\x19")
        assert input_.get_value() == "third"

        input_.handle_input("\x1by")
        assert input_.get_value() == "second"

        input_.handle_input("\x1by")
        assert input_.get_value() == "first"

        input_.handle_input("\x1by")
        assert input_.get_value() == "third"

    def test_alt_y_does_nothing_if_not_preceded_by_yank(self):
        input_ = Input()

        input_.set_value("test")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        input_.set_value("other")
        input_.handle_input("\x05")

        input_.handle_input("x")
        assert input_.get_value() == "otherx"

        input_.handle_input("\x1by")
        assert input_.get_value() == "otherx"

    def test_alt_y_does_nothing_if_kill_ring_has_one_entry(self):
        input_ = Input()

        input_.set_value("only")
        input_.handle_input("\x05")
        input_.handle_input("\x17")

        input_.handle_input("\x19")
        assert input_.get_value() == "only"

        input_.handle_input("\x1by")
        assert input_.get_value() == "only"

    def test_consecutive_ctrl_w_accumulates_into_one_entry(self):
        input_ = Input()

        input_.set_value("one two three")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        input_.handle_input("\x17")
        input_.handle_input("\x17")

        assert input_.get_value() == ""

        input_.handle_input("\x19")
        assert input_.get_value() == "one two three"

    def test_non_delete_actions_break_kill_accumulation(self):
        input_ = Input()

        input_.set_value("foo bar baz")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        assert input_.get_value() == "foo bar "

        input_.handle_input("x")
        assert input_.get_value() == "foo bar x"

        input_.handle_input("\x17")
        assert input_.get_value() == "foo bar "

        input_.handle_input("\x19")
        assert input_.get_value() == "foo bar x"

        input_.handle_input("\x1by")
        assert input_.get_value() == "foo bar baz"

    def test_non_yank_actions_break_alt_y_chain(self):
        input_ = Input()

        input_.set_value("first")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        input_.set_value("second")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        input_.set_value("")

        input_.handle_input("\x19")
        assert input_.get_value() == "second"

        input_.handle_input("x")
        assert input_.get_value() == "secondx"

        input_.handle_input("\x1by")
        assert input_.get_value() == "secondx"

    def test_kill_ring_rotation_persists_after_cycling(self):
        input_ = Input()

        input_.set_value("first")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        input_.set_value("second")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        input_.set_value("third")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        input_.set_value("")

        input_.handle_input("\x19")
        input_.handle_input("\x1by")
        assert input_.get_value() == "second"

        input_.handle_input("x")
        input_.set_value("")

        input_.handle_input("\x19")
        assert input_.get_value() == "second"

    def test_backward_and_forward_deletions_accumulate_correctly(self):
        input_ = Input()

        input_.set_value("prefix|suffix")
        input_.handle_input("\x01")
        for _ in range(6):
            input_.handle_input("\x1b[C")

        input_.handle_input("\x0b")  # Ctrl+K
        assert input_.get_value() == "prefix"

        input_.handle_input("\x19")
        assert input_.get_value() == "prefix|suffix"

    def test_alt_d_deletes_word_forward_and_saves_to_kill_ring(self):
        input_ = Input()

        input_.set_value("hello world test")
        input_.handle_input("\x01")

        input_.handle_input("\x1bd")
        assert input_.get_value() == " world test"

        input_.handle_input("\x1bd")
        assert input_.get_value() == " test"

        input_.handle_input("\x19")
        assert input_.get_value() == "hello world test"

    def test_alt_d_preserves_ascii_punctuation_boundaries(self):
        input_ = Input()

        input_.set_value("foo.bar baz")
        input_.handle_input("\x01")
        input_.handle_input("\x1bd")
        assert input_.get_value() == ".bar baz"
        input_.handle_input("\x1bd")
        assert input_.get_value() == "bar baz"
        input_.handle_input("\x1bd")
        assert input_.get_value() == " baz"

    def test_alt_d_handles_unicode_word_boundaries(self):
        # See test_ctrl_w_handles_unicode_word_boundaries: this port's
        # dictionary-free word segmentation deletes one CJK
        # character/punctuation mark at a time rather than upstream's
        # ICU dictionary-segmented two-character words.
        input_ = Input()

        input_.set_value("你好世界。你好，世界")
        input_.handle_input("\x01")
        input_.handle_input("\x1bd")
        assert input_.get_value() == "好世界。你好，世界"
        input_.handle_input("\x1bd")
        assert input_.get_value() == "世界。你好，世界"
        input_.handle_input("\x1bd")
        assert input_.get_value() == "界。你好，世界"
        input_.handle_input("\x1bd")
        assert input_.get_value() == "。你好，世界"
        input_.handle_input("\x1bd")
        assert input_.get_value() == "你好，世界"
        input_.handle_input("\x1bd")
        assert input_.get_value() == "好，世界"

    def test_handles_yank_in_middle_of_text(self):
        input_ = Input()

        input_.set_value("word")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        input_.set_value("hello world")
        input_.handle_input("\x01")
        for _ in range(6):
            input_.handle_input("\x1b[C")

        input_.handle_input("\x19")
        assert input_.get_value() == "hello wordworld"

    def test_handles_yank_pop_in_middle_of_text(self):
        input_ = Input()

        input_.set_value("FIRST")
        input_.handle_input("\x05")
        input_.handle_input("\x17")
        input_.set_value("SECOND")
        input_.handle_input("\x05")
        input_.handle_input("\x17")

        input_.set_value("hello world")
        input_.handle_input("\x01")
        for _ in range(6):
            input_.handle_input("\x1b[C")

        input_.handle_input("\x19")
        assert input_.get_value() == "hello SECONDworld"

        input_.handle_input("\x1by")
        assert input_.get_value() == "hello FIRSTworld"


class TestInputUndo:
    def test_does_nothing_when_undo_stack_empty(self):
        input_ = Input()

        input_.handle_input("\x1b[45;5u")  # Ctrl+- (undo)
        assert input_.get_value() == ""

    def test_coalesces_consecutive_word_characters_into_one_undo_unit(self):
        input_ = Input()

        for ch in "hello world":
            input_.handle_input(ch)
        assert input_.get_value() == "hello world"

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == "hello"

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == ""

    def test_undoes_spaces_one_at_a_time(self):
        input_ = Input()

        for ch in "hello  ":
            input_.handle_input(ch)
        assert input_.get_value() == "hello  "

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == "hello "

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == "hello"

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == ""

    def test_undoes_backspace(self):
        input_ = Input()

        for ch in "hello":
            input_.handle_input(ch)
        input_.handle_input("\x7f")  # Backspace
        assert input_.get_value() == "hell"

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == "hello"

    def test_undoes_forward_delete(self):
        input_ = Input()

        for ch in "hello":
            input_.handle_input(ch)
        input_.handle_input("\x01")
        input_.handle_input("\x1b[C")
        input_.handle_input("\x1b[3~")  # Delete key
        assert input_.get_value() == "hllo"

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == "hello"

    def test_undoes_ctrl_w(self):
        input_ = Input()

        for ch in "hello world":
            input_.handle_input(ch)
        assert input_.get_value() == "hello world"

        input_.handle_input("\x17")
        assert input_.get_value() == "hello "

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == "hello world"

    def test_undoes_ctrl_k(self):
        input_ = Input()

        for ch in "hello world":
            input_.handle_input(ch)
        input_.handle_input("\x01")
        for _ in range(6):
            input_.handle_input("\x1b[C")

        input_.handle_input("\x0b")
        assert input_.get_value() == "hello "

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == "hello world"

    def test_undoes_ctrl_u(self):
        input_ = Input()

        for ch in "hello world":
            input_.handle_input(ch)
        input_.handle_input("\x01")
        for _ in range(6):
            input_.handle_input("\x1b[C")

        input_.handle_input("\x15")
        assert input_.get_value() == "world"

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == "hello world"

    def test_undoes_yank(self):
        input_ = Input()

        for ch in "hello ":
            input_.handle_input(ch)
        input_.handle_input("\x17")
        input_.handle_input("\x19")
        assert input_.get_value() == "hello "

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == ""

    def test_undoes_paste_atomically(self):
        input_ = Input()

        input_.set_value("hello world")
        input_.handle_input("\x01")
        for _ in range(5):
            input_.handle_input("\x1b[C")

        input_.handle_input("\x1b[200~beep boop\x1b[201~")
        assert input_.get_value() == "hellobeep boop world"

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == "hello world"

    def test_undoes_alt_d(self):
        input_ = Input()

        input_.set_value("hello world")
        input_.handle_input("\x01")

        input_.handle_input("\x1bd")
        assert input_.get_value() == " world"

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == "hello world"

    def test_cursor_movement_starts_new_undo_unit(self):
        input_ = Input()

        input_.handle_input("a")
        input_.handle_input("b")
        input_.handle_input("c")
        input_.handle_input("\x01")
        input_.handle_input("\x05")
        input_.handle_input("d")
        input_.handle_input("e")
        assert input_.get_value() == "abcde"

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == "abc"

        input_.handle_input("\x1b[45;5u")
        assert input_.get_value() == ""
