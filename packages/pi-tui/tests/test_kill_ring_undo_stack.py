"""Tests for kill_ring.py and undo_stack.py, ported from
packages/tui/src/kill-ring.ts and undo-stack.ts (no dedicated upstream test
files exist; these cover the documented behavior directly).
"""

from pi_tui.kill_ring import KillRing
from pi_tui.undo_stack import UndoStack


class TestKillRing:
    def test_push_and_peek(self):
        ring = KillRing()
        assert ring.peek() is None
        ring.push("hello", prepend=False)
        assert ring.peek() == "hello"
        assert ring.length == 1

    def test_ignores_empty_text(self):
        ring = KillRing()
        ring.push("", prepend=False)
        assert ring.length == 0
        assert ring.peek() is None

    def test_push_without_accumulate_creates_new_entry(self):
        ring = KillRing()
        ring.push("a", prepend=False)
        ring.push("b", prepend=False)
        assert ring.length == 2
        assert ring.peek() == "b"

    def test_accumulate_append_forward_deletion(self):
        ring = KillRing()
        ring.push("hello", prepend=False)
        ring.push(" world", prepend=False, accumulate=True)
        assert ring.length == 1
        assert ring.peek() == "hello world"

    def test_accumulate_prepend_backward_deletion(self):
        ring = KillRing()
        ring.push("world", prepend=True)
        ring.push("hello ", prepend=True, accumulate=True)
        assert ring.length == 1
        assert ring.peek() == "hello world"

    def test_accumulate_with_empty_ring_creates_new_entry(self):
        ring = KillRing()
        ring.push("first", prepend=False, accumulate=True)
        assert ring.length == 1
        assert ring.peek() == "first"

    def test_rotate_cycles_entries(self):
        ring = KillRing()
        ring.push("a", prepend=False)
        ring.push("b", prepend=False)
        ring.push("c", prepend=False)
        assert ring.peek() == "c"

        ring.rotate()
        assert ring.peek() == "b"

        ring.rotate()
        assert ring.peek() == "a"

    def test_rotate_noop_with_one_or_zero_entries(self):
        ring = KillRing()
        ring.rotate()
        assert ring.length == 0

        ring.push("only", prepend=False)
        ring.rotate()
        assert ring.peek() == "only"
        assert ring.length == 1


class TestUndoStack:
    def test_push_and_pop(self):
        stack: UndoStack[dict] = UndoStack()
        assert stack.pop() is None

        stack.push({"value": 1})
        assert stack.length == 1

        popped = stack.pop()
        assert popped == {"value": 1}
        assert stack.length == 0

    def test_push_deep_copies_state(self):
        stack: UndoStack[dict] = UndoStack()
        original = {"nested": {"value": 1}}
        stack.push(original)

        original["nested"]["value"] = 2

        popped = stack.pop()
        assert popped == {"nested": {"value": 1}}

    def test_pop_order_is_lifo(self):
        stack: UndoStack[int] = UndoStack()
        stack.push(1)
        stack.push(2)
        stack.push(3)

        assert stack.pop() == 3
        assert stack.pop() == 2
        assert stack.pop() == 1
        assert stack.pop() is None

    def test_clear_empties_stack(self):
        stack: UndoStack[int] = UndoStack()
        stack.push(1)
        stack.push(2)
        stack.clear()

        assert stack.length == 0
        assert stack.pop() is None
