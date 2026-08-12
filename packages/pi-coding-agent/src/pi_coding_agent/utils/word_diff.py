"""Word-level text diff.

Python port of the `diff` npm package's ``diffWords`` (jsdiff 8.0.4), which the
TypeScript CLI uses in ``modes/interactive/components/diff.ts`` for
intra-line change highlighting.

Python's ``difflib`` is not a substitute: it uses a different matching
algorithm, so it groups changes differently and would highlight different
tokens than the TypeScript CLI does for the same edit. This module therefore
ports jsdiff's own Myers implementation (``diff/base.js``), its word tokenizer
and its whitespace-dedupe post-processing (``diff/word.js``) so the rendered
diff matches byte for byte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# jsdiff's `extendedWordChars`, covering ASCII word characters plus the Latin
# supplement/extended blocks (minus the multiplication and division signs).
_EXTENDED_WORD_CHARS = "a-zA-Z0-9_\u00ad\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02c6\u02c8-\u02d7\u02de-\u02ff\u1e00-\u1eff"
_TOKENIZE_RE = re.compile(f"[{_EXTENDED_WORD_CHARS}]+|\\s+|[^{_EXTENDED_WORD_CHARS}]")
_WHITESPACE_RE = re.compile(r"\s")
_LEADING_WS_RE = re.compile(r"^\s*")


@dataclass
class Change:
    """One run of tokens, mirroring a jsdiff change object."""

    value: str
    count: int = 0
    added: bool = False
    removed: bool = False


@dataclass
class _Component:
    count: int
    added: bool
    removed: bool
    previous: _Component | None = None
    value: str = ""


@dataclass
class _Path:
    old_pos: int
    last_component: _Component | None


# --------------------------------------------------------------------------
# string helpers (jsdiff `util/string.js`)
# --------------------------------------------------------------------------


def _longest_common_prefix(str1: str, str2: str) -> str:
    limit = min(len(str1), len(str2))
    index = 0
    while index < limit and str1[index] == str2[index]:
        index += 1
    return str1[:index]


def _longest_common_suffix(str1: str, str2: str) -> str:
    if not str1 or not str2 or str1[-1] != str2[-1]:
        return ""
    limit = min(len(str1), len(str2))
    index = 0
    while index < limit and str1[len(str1) - (index + 1)] == str2[len(str2) - (index + 1)]:
        index += 1
    return str1[len(str1) - index :]


def _replace_prefix(string: str, old_prefix: str, new_prefix: str) -> str:
    if string[: len(old_prefix)] != old_prefix:
        raise ValueError(f"string {string!r} doesn't start with prefix {old_prefix!r}; this is a bug")
    return new_prefix + string[len(old_prefix) :]


def _replace_suffix(string: str, old_suffix: str, new_suffix: str) -> str:
    if not old_suffix:
        return string + new_suffix
    if string[-len(old_suffix) :] != old_suffix:
        raise ValueError(f"string {string!r} doesn't end with suffix {old_suffix!r}; this is a bug")
    return string[: -len(old_suffix)] + new_suffix


def _remove_prefix(string: str, old_prefix: str) -> str:
    return _replace_prefix(string, old_prefix, "")


def _remove_suffix(string: str, old_suffix: str) -> str:
    return _replace_suffix(string, old_suffix, "")


def _overlap_count(a: str, b: str) -> int:
    """Length of the longest suffix of ``a`` that is a prefix of ``b`` (KMP)."""
    start_a = len(a) - len(b) if len(a) > len(b) else 0
    end_b = len(a) if len(a) < len(b) else len(b)
    if end_b == 0:
        return 0

    table = [0] * end_b
    k = 0
    for j in range(1, end_b):
        if b[j] == b[k]:
            table[j] = table[k]
        else:
            table[j] = k
        while k > 0 and b[j] != b[k]:
            k = table[k]
        if b[j] == b[k]:
            k += 1

    k = 0
    for index in range(start_a, len(a)):
        while k > 0 and a[index] != b[k]:
            k = table[k]
        if a[index] == b[k]:
            k += 1
    return k


def _maximum_overlap(string1: str, string2: str) -> str:
    return string2[: _overlap_count(string1, string2)]


def _leading_ws(string: str) -> str:
    match = _LEADING_WS_RE.match(string)
    return match.group(0) if match else ""


def _trailing_ws(string: str) -> str:
    index = len(string) - 1
    while index >= 0 and _WHITESPACE_RE.match(string[index]):
        index -= 1
    return string[index + 1 :]


def _leading_and_trailing_ws(string: str) -> tuple[str, str]:
    return _leading_ws(string), _trailing_ws(string)


# --------------------------------------------------------------------------
# tokenization (jsdiff `WordDiff.tokenize`)
# --------------------------------------------------------------------------


def tokenize_words(value: str) -> list[str]:
    """Split into word/punctuation tokens with their surrounding whitespace."""
    parts = _TOKENIZE_RE.findall(value)
    tokens: list[str] = []
    prev_part: str | None = None
    for part in parts:
        if _WHITESPACE_RE.match(part):
            if prev_part is None:
                tokens.append(part)
            else:
                tokens.append(tokens.pop() + part)
        elif prev_part is not None and _WHITESPACE_RE.match(prev_part):
            if tokens and tokens[-1] == prev_part:
                tokens.append(tokens.pop() + part)
            else:
                tokens.append(prev_part + part)
        else:
            tokens.append(part)
        prev_part = part
    return tokens


def _join_tokens(tokens: list[str]) -> str:
    """Tokens carry their leading whitespace, so drop it on all but the first."""
    return "".join(
        token if index == 0 else _LEADING_WS_RE.sub("", token, count=1) for index, token in enumerate(tokens)
    )


def _equals(left: str, right: str) -> bool:
    return left.strip() == right.strip()


# --------------------------------------------------------------------------
# Myers diff core (jsdiff `Diff.prototype.diff`)
# --------------------------------------------------------------------------


def _add_to_path(path: _Path, added: bool, removed: bool, old_pos_inc: int) -> _Path:
    last = path.last_component
    if last is not None and last.added == added and last.removed == removed:
        return _Path(
            old_pos=path.old_pos + old_pos_inc,
            last_component=_Component(last.count + 1, added, removed, last.previous),
        )
    return _Path(
        old_pos=path.old_pos + old_pos_inc,
        last_component=_Component(1, added, removed, last),
    )


def _extract_common(base_path: _Path, new_tokens: list[str], old_tokens: list[str], diagonal_path: int) -> int:
    new_len, old_len = len(new_tokens), len(old_tokens)
    old_pos = base_path.old_pos
    new_pos = old_pos - diagonal_path
    common_count = 0
    while new_pos + 1 < new_len and old_pos + 1 < old_len and _equals(old_tokens[old_pos + 1], new_tokens[new_pos + 1]):
        new_pos += 1
        old_pos += 1
        common_count += 1
    if common_count:
        base_path.last_component = _Component(common_count, False, False, base_path.last_component)
    base_path.old_pos = old_pos
    return new_pos


def _build_values(last_component: _Component | None, new_tokens: list[str], old_tokens: list[str]) -> list[Change]:
    components: list[_Component] = []
    cursor = last_component
    while cursor is not None:
        components.append(cursor)
        cursor = cursor.previous
    components.reverse()

    changes: list[Change] = []
    new_pos = 0
    old_pos = 0
    for component in components:
        if not component.removed:
            component.value = _join_tokens(new_tokens[new_pos : new_pos + component.count])
            new_pos += component.count
            if not component.added:
                old_pos += component.count
        else:
            component.value = _join_tokens(old_tokens[old_pos : old_pos + component.count])
            old_pos += component.count
        changes.append(
            Change(value=component.value, count=component.count, added=component.added, removed=component.removed)
        )
    return changes


def _diff_tokens(old_tokens: list[str], new_tokens: list[str]) -> list[Change]:
    new_len, old_len = len(new_tokens), len(old_tokens)
    edit_length = 1
    max_edit_length = new_len + old_len

    best_path: dict[int, _Path | None] = {0: _Path(old_pos=-1, last_component=None)}
    new_pos = _extract_common(best_path[0], new_tokens, old_tokens, 0)  # type: ignore[arg-type]
    if best_path[0].old_pos + 1 >= old_len and new_pos + 1 >= new_len:  # type: ignore[union-attr]
        return _build_values(best_path[0].last_component, new_tokens, old_tokens)  # type: ignore[union-attr]

    # Once a diagonal reaches an edge of the edit graph there is no point
    # extending past it; jsdiff records that with these bounds.
    min_diagonal: float = -float("inf")
    max_diagonal: float = float("inf")

    while edit_length <= max_edit_length:
        diagonal = int(max(min_diagonal, -edit_length))
        upper = int(min(max_diagonal, edit_length))
        while diagonal <= upper:
            remove_path = best_path.get(diagonal - 1)
            add_path = best_path.get(diagonal + 1)
            if remove_path is not None:
                best_path[diagonal - 1] = None

            can_add = False
            if add_path is not None:
                add_path_new_pos = add_path.old_pos - diagonal
                can_add = 0 <= add_path_new_pos < new_len
            can_remove = remove_path is not None and remove_path.old_pos + 1 < old_len

            if not can_add and not can_remove:
                best_path[diagonal] = None
                diagonal += 2
                continue

            if not can_remove or (can_add and remove_path.old_pos < add_path.old_pos):  # type: ignore[union-attr]
                base_path = _add_to_path(add_path, True, False, 0)  # type: ignore[arg-type]
            else:
                base_path = _add_to_path(remove_path, False, True, 1)  # type: ignore[arg-type]

            new_pos = _extract_common(base_path, new_tokens, old_tokens, diagonal)
            if base_path.old_pos + 1 >= old_len and new_pos + 1 >= new_len:
                return _build_values(base_path.last_component, new_tokens, old_tokens)

            best_path[diagonal] = base_path
            if base_path.old_pos + 1 >= old_len:
                max_diagonal = min(max_diagonal, diagonal - 1)
            if new_pos + 1 >= new_len:
                min_diagonal = max(min_diagonal, diagonal + 1)
            diagonal += 2
        edit_length += 1

    return []


# --------------------------------------------------------------------------
# whitespace dedupe (jsdiff `WordDiff.postProcess`)
# --------------------------------------------------------------------------


def _dedupe_whitespace(
    start_keep: Change | None,
    deletion: Change | None,
    insertion: Change | None,
    end_keep: Change | None,
) -> None:
    """Stop trailing whitespace in one change repeating as leading whitespace
    in the next, so rendered insertions/deletions line up with the source."""
    if deletion is not None and insertion is not None:
        old_ws_prefix, old_ws_suffix = _leading_and_trailing_ws(deletion.value)
        new_ws_prefix, new_ws_suffix = _leading_and_trailing_ws(insertion.value)
        if start_keep is not None:
            common_ws_prefix = _longest_common_prefix(old_ws_prefix, new_ws_prefix)
            start_keep.value = _replace_suffix(start_keep.value, new_ws_prefix, common_ws_prefix)
            deletion.value = _remove_prefix(deletion.value, common_ws_prefix)
            insertion.value = _remove_prefix(insertion.value, common_ws_prefix)
        if end_keep is not None:
            common_ws_suffix = _longest_common_suffix(old_ws_suffix, new_ws_suffix)
            end_keep.value = _replace_prefix(end_keep.value, new_ws_suffix, common_ws_suffix)
            deletion.value = _remove_suffix(deletion.value, common_ws_suffix)
            insertion.value = _remove_suffix(insertion.value, common_ws_suffix)
    elif insertion is not None:
        # All the whitespace reflects the new text, so there is nothing to
        # attribute; just drop the duplicated leading whitespace.
        if start_keep is not None:
            insertion.value = insertion.value[len(_leading_ws(insertion.value)) :]
        if end_keep is not None:
            end_keep.value = end_keep.value[len(_leading_ws(end_keep.value)) :]
    elif deletion is None:
        return
    elif start_keep is not None and end_keep is not None:
        new_ws_full = _leading_ws(end_keep.value)
        del_ws_start, del_ws_end = _leading_and_trailing_ws(deletion.value)

        new_ws_start = _longest_common_prefix(new_ws_full, del_ws_start)
        deletion.value = _remove_prefix(deletion.value, new_ws_start)

        new_ws_end = _longest_common_suffix(_remove_prefix(new_ws_full, new_ws_start), del_ws_end)
        deletion.value = _remove_suffix(deletion.value, new_ws_end)
        end_keep.value = _replace_prefix(end_keep.value, new_ws_full, new_ws_end)

        start_keep.value = _replace_suffix(
            start_keep.value, new_ws_full, new_ws_full[: len(new_ws_full) - len(new_ws_end)]
        )
    elif end_keep is not None:
        # Start of the text: keep all of end_keep's whitespace and only trim
        # the deletion where it overlaps.
        overlap = _maximum_overlap(_trailing_ws(deletion.value), _leading_ws(end_keep.value))
        deletion.value = _remove_suffix(deletion.value, overlap)
    elif start_keep is not None:
        # End of the text: mirror image of the branch above.
        overlap = _maximum_overlap(_trailing_ws(start_keep.value), _leading_ws(deletion.value))
        deletion.value = _remove_prefix(deletion.value, overlap)


def _post_process(changes: list[Change]) -> list[Change]:
    last_keep: Change | None = None
    insertion: Change | None = None
    deletion: Change | None = None

    for change in changes:
        if change.added:
            insertion = change
        elif change.removed:
            deletion = change
        else:
            if insertion is not None or deletion is not None:
                _dedupe_whitespace(last_keep, deletion, insertion, change)
            last_keep = change
            insertion = None
            deletion = None

    if insertion is not None or deletion is not None:
        _dedupe_whitespace(last_keep, deletion, insertion, None)

    return changes


def diff_words(old_str: str, new_str: str) -> list[Change]:
    """Word-level diff of two strings, matching jsdiff's ``diffWords``."""
    old_tokens = [token for token in tokenize_words(old_str) if token]
    new_tokens = [token for token in tokenize_words(new_str) if token]
    return _post_process(_diff_tokens(old_tokens, new_tokens))


__all__ = ["Change", "diff_words", "tokenize_words"]
