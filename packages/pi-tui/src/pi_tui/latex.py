"""Render basic LaTeX math expressions as terminal-friendly Unicode text.

Python port of `packages/tui/src/latex.ts`.

Behaviour notes and adaptations:

* Like the rest of `pi_tui`, this port indexes strings by Python code point
  rather than by UTF-16 code unit. The upstream TypeScript uses spread and
  regex `u` flag, which count astral characters as one; the difference is not
  observable for the LaTeX syntax this module accepts.
* Unicode letter/number classes (`\\p{L}`, `\\p{N}`) in the TypeScript regexes
  are approximated with Python's `str.isalpha()` / `str.isnumeric()` /
  `str.isalnum()`. For the sets these regexes are matched against (rendered
  math output) the coverage is equivalent in practice; superscript letters
  (`Lm`) and superscript digits (`No`) all classify as alphanumeric in Python.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .utils import visible_width

SYMBOLS: dict[str, str] = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ϵ",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "varkappa": "ϰ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "ϖ",
    "rho": "ρ",
    "varrho": "ϱ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "ϕ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Upsilon": "Υ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "pm": "±",
    "mp": "∓",
    "times": "×",
    "div": "÷",
    "cdot": "·",
    "ast": "∗",
    "star": "⋆",
    "circ": "∘",
    "bullet": "•",
    "oplus": "⊕",
    "ominus": "⊖",
    "otimes": "⊗",
    "oslash": "⊘",
    "odot": "⊙",
    "bigcirc": "○",
    "dagger": "†",
    "ddagger": "‡",
    "amalg": "⨿",
    "uplus": "⊎",
    "sqcap": "⊓",
    "sqcup": "⊔",
    "triangleleft": "◁",
    "triangleright": "▷",
    "wr": "≀",
    "cap": "∩",
    "cup": "∪",
    "bigcap": "⋂",
    "bigcup": "⋃",
    "bigwedge": "⋀",
    "bigvee": "⋁",
    "bigsqcup": "⨆",
    "biguplus": "⨄",
    "bigoplus": "⨁",
    "bigotimes": "⨂",
    "bigodot": "⨀",
    "setminus": "∖",
    "in": "∈",
    "notin": "∉",
    "ni": "∋",
    "subset": "⊂",
    "supset": "⊃",
    "subseteq": "⊆",
    "supseteq": "⊇",
    "sqsubset": "⊏",
    "sqsupset": "⊐",
    "sqsubseteq": "⊑",
    "sqsupseteq": "⊒",
    "prec": "≺",
    "preceq": "≼",
    "succ": "≻",
    "succeq": "≽",
    "ll": "≪",
    "gg": "≫",
    "le": "≤",
    "leq": "≤",
    "leqslant": "≤",
    "ge": "≥",
    "geq": "≥",
    "geqslant": "≥",
    "ne": "≠",
    "neq": "≠",
    "equiv": "≡",
    "approx": "≈",
    "sim": "∼",
    "simeq": "≃",
    "cong": "≅",
    "asymp": "≍",
    "doteq": "≐",
    "propto": "∝",
    "parallel": "∥",
    "perp": "⊥",
    "mid": "∣",
    "vdash": "⊢",
    "dashv": "⊣",
    "models": "⊨",
    "Vdash": "⊩",
    "Vvdash": "⊪",
    "nvdash": "⊬",
    "nvDash": "⊭",
    "forall": "∀",
    "exists": "∃",
    "nexists": "∄",
    "neg": "¬",
    "land": "∧",
    "wedge": "∧",
    "lor": "∨",
    "vee": "∨",
    "to": "→",
    "rightarrow": "→",
    "longrightarrow": "→",
    "leftarrow": "←",
    "longleftarrow": "←",
    "gets": "←",
    "leftrightarrow": "↔",
    "longleftrightarrow": "↔",
    "hookleftarrow": "↩",
    "hookrightarrow": "↪",
    "twoheadleftarrow": "↞",
    "twoheadrightarrow": "↠",
    "leftharpoonup": "↼",
    "leftharpoondown": "↽",
    "rightharpoonup": "⇀",
    "rightharpoondown": "⇁",
    "rightleftharpoons": "⇌",
    "leftrightharpoons": "⇋",
    "nearrow": "↗",
    "searrow": "↘",
    "swarrow": "↙",
    "nwarrow": "↖",
    "rightsquigarrow": "⇝",
    "leadsto": "⇝",
    "Rightarrow": "⇒",
    "Longrightarrow": "⇒",
    "Leftarrow": "⇐",
    "Longleftarrow": "⇐",
    "Leftrightarrow": "⇔",
    "Longleftrightarrow": "⇔",
    "implies": "⇒",
    "iff": "⇔",
    "mapsto": "↦",
    "longmapsto": "↦",
    "uparrow": "↑",
    "downarrow": "↓",
    "partial": "∂",
    "nabla": "∇",
    "int": "∫",
    "iint": "∬",
    "iiint": "∭",
    "oint": "∮",
    "sum": "∑",
    "prod": "∏",
    "coprod": "∐",
    "infty": "∞",
    "emptyset": "∅",
    "varnothing": "∅",
    "angle": "∠",
    "therefore": "∴",
    "because": "∵",
    "aleph": "ℵ",
    "beth": "ℶ",
    "gimel": "ℷ",
    "daleth": "ℸ",
    "top": "⊤",
    "bot": "⊥",
    "triangle": "△",
    "square": "□",
    "lozenge": "◊",
    "checkmark": "✓",
    "complement": "∁",
    "wp": "℘",
    "prime": "′",
    "ldots": "…",
    "dots": "…",
    "cdots": "⋯",
    "vdots": "⋮",
    "ddots": "⋱",
    "ell": "ℓ",
    "hbar": "ℏ",
    "Im": "ℑ",
    "Re": "ℜ",
    "langle": "⟨",
    "rangle": "⟩",
    "vert": "|",
    "lvert": "|",
    "rvert": "|",
    "Vert": "‖",
    "lVert": "‖",
    "rVert": "‖",
    "lbrace": "{",
    "rbrace": "}",
    "backslash": "\\",
    "lfloor": "⌊",
    "rfloor": "⌋",
    "lceil": "⌈",
    "rceil": "⌉",
    "colon": ":",
}

NAMED_OPERATORS: frozenset[str] = frozenset(
    {
        "arccos",
        "arcsin",
        "arctan",
        "arg",
        "cos",
        "cosh",
        "cot",
        "coth",
        "csc",
        "deg",
        "det",
        "dim",
        "exp",
        "gcd",
        "hom",
        "inf",
        "ker",
        "lg",
        "lim",
        "liminf",
        "limsup",
        "ln",
        "log",
        "max",
        "min",
        "Pr",
        "sec",
        "sin",
        "sinh",
        "sup",
        "tan",
        "tanh",
    }
)

LIMIT_OPERATORS: frozenset[str] = frozenset(
    {
        "argmax",
        "argmin",
        "inf",
        "injlim",
        "lim",
        "liminf",
        "limsup",
        "max",
        "min",
        "projlim",
        "sup",
    }
)

DISPLAY_LIMIT_SYMBOLS: frozenset[str] = frozenset(
    {
        "bigcap",
        "bigcup",
        "bigodot",
        "bigoplus",
        "bigotimes",
        "bigsqcup",
        "biguplus",
        "bigvee",
        "bigwedge",
        "coprod",
        "int",
        "iint",
        "iiint",
        "oint",
        "prod",
        "sum",
    }
)

RELATION_COMMANDS: frozenset[str] = frozenset(
    {
        "Leftarrow",
        "Leftrightarrow",
        "Longleftarrow",
        "Longleftrightarrow",
        "Longrightarrow",
        "Rightarrow",
        "Vdash",
        "Vvdash",
        "approx",
        "asymp",
        "cong",
        "dashv",
        "doteq",
        "downarrow",
        "equiv",
        "ge",
        "geq",
        "geqslant",
        "gets",
        "gg",
        "hookleftarrow",
        "hookrightarrow",
        "iff",
        "implies",
        "in",
        "leadsto",
        "le",
        "leftarrow",
        "leftharpoondown",
        "leftharpoonup",
        "leftrightarrow",
        "leftrightharpoons",
        "leq",
        "leqslant",
        "ll",
        "longleftarrow",
        "longleftrightarrow",
        "longmapsto",
        "longrightarrow",
        "mapsto",
        "mid",
        "models",
        "ne",
        "nearrow",
        "neq",
        "ni",
        "notin",
        "nvdash",
        "nvDash",
        "nwarrow",
        "parallel",
        "perp",
        "prec",
        "preceq",
        "propto",
        "rightharpoondown",
        "rightharpoonup",
        "rightleftharpoons",
        "rightarrow",
        "rightsquigarrow",
        "searrow",
        "sim",
        "simeq",
        "sqsubset",
        "sqsubseteq",
        "sqsupset",
        "sqsupseteq",
        "subset",
        "subseteq",
        "succ",
        "succeq",
        "supset",
        "supseteq",
        "swarrow",
        "to",
        "triangleleft",
        "triangleright",
        "twoheadleftarrow",
        "twoheadrightarrow",
        "uparrow",
        "vdash",
    }
)

NEGATED_SYMBOLS: dict[str, str] = {
    "<": "≮",
    ">": "≯",
    "=": "≠",
    "∈": "∉",
    "∋": "∌",
    "∣": "∤",
    "∥": "∦",
    "∼": "≁",
    "≃": "≄",
    "≅": "≇",
    "≈": "≉",
    "≡": "≢",
    "≤": "≰",
    "≥": "≱",
    "≺": "⊀",
    "≻": "⊁",
    "⊂": "⊄",
    "⊃": "⊅",
    "⊆": "⊈",
    "⊇": "⊉",
    "⊢": "⊬",
    "⊨": "⊭",
    "↔": "↮",
    "←": "↚",
    "→": "↛",
    "⇒": "⇏",
    "⇐": "⇍",
    "⇔": "⇎",
    "≼": "⋠",
    "≽": "⋡",
}

BLACKBOARD: dict[str, str] = {
    "C": "ℂ",
    "H": "ℍ",
    "N": "ℕ",
    "P": "ℙ",
    "Q": "ℚ",
    "R": "ℝ",
    "Z": "ℤ",
}

SUPERSCRIPTS: dict[str, str] = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
}

SUBSCRIPTS: dict[str, str] = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
}

SPACING_COMMANDS: frozenset[str] = frozenset(
    {
        ",",
        ":",
        ";",
        " ",
        ">",
        "enspace",
        "enskip",
        "medspace",
        "quad",
        "qquad",
        "thickspace",
        "thinspace",
    }
)
NEGATIVE_SPACING_COMMANDS: frozenset[str] = frozenset({"!", "negmedspace", "negthickspace", "negthinspace"})
NEGATIVE_SPACE = "\x00"
IGNORED_COMMANDS: frozenset[str] = frozenset(
    {
        "displaystyle",
        "limits",
        "nolimits",
        "scriptstyle",
        "scriptscriptstyle",
        "textstyle",
    }
)
SIZE_COMMANDS: frozenset[str] = frozenset(
    {
        "big",
        "Big",
        "bigg",
        "Bigg",
        "bigl",
        "Bigl",
        "biggl",
        "Biggl",
        "bigr",
        "Bigr",
        "biggr",
        "Biggr",
    }
)
PLAIN_WRAPPERS: frozenset[str] = frozenset(
    {
        "emph",
        "mathcal",
        "mathbf",
        "mathfrak",
        "mathit",
        "mathrm",
        "mathnormal",
        "mathscr",
        "mathsf",
        "mathtt",
        "mathup",
        "mbox",
        "overbrace",
        "pmb",
        "smash",
        "substack",
        "text",
        "textbf",
        "textit",
        "textmd",
        "textnormal",
        "textrm",
        "textsc",
        "textsf",
        "textsl",
        "texttt",
        "textup",
        "underbrace",
        "bm",
        "boldsymbol",
    }
)
ACCENTS: dict[str, str] = {
    "acute": "\u0301",
    "bar": "\u0305",
    "breve": "\u0306",
    "check": "\u030c",
    "ddot": "\u0308",
    "dot": "\u0307",
    "grave": "\u0300",
    "hat": "\u0302",
    "mathring": "\u030a",
    "overleftarrow": "\u20d6",
    "overleftrightarrow": "\u20e1",
    "overline": "\u0305",
    "overrightarrow": "\u20d7",
    "tilde": "\u0303",
    "underline": "\u0332",
    "vec": "\u20d7",
    "widehat": "\u0302",
    "widetilde": "\u0303",
}


NAMED_OPERATOR_START = "\U000f0004"
NAMED_OPERATOR_END = "\U000f0005"
LAYOUT_MARKER_START = "\U000f0000"
LAYOUT_MARKER_END = "\U000f0001"
PROTECTED_SPACE = "\U000f0002"

_LAYOUT_MARKER_PATTERN = re.compile(f"{LAYOUT_MARKER_START}(\\d+){LAYOUT_MARKER_END}")
_TRAILING_LAYOUT_MARKER_PATTERN = re.compile(f"{LAYOUT_MARKER_START}(\\d+){LAYOUT_MARKER_END}$")
_SCRIPT_SIGN_PATTERN = re.compile(r"\s*([=+\-])\s*")
_LIMITS_MODIFIER_PATTERN = re.compile(r"^\\(limits|nolimits)(?![A-Za-z])")


def _replace_characters(value: str, replacements: dict[str, str]) -> str | None:
    result: list[str] = []
    for character in value:
        replacement = replacements.get(character)
        if replacement is None:
            return None
        result.append(replacement)
    return "".join(result)


def _is_simple_alnum_dot(value: str) -> bool:
    if not value:
        return False
    return all(character.isalnum() or character == "." for character in value)


def _is_simple_numeric_dot(value: str) -> bool:
    if not value:
        return False
    return all(character.isnumeric() or character == "." for character in value)


def _format_script(value: str, kind: Literal["sub", "sup"]) -> str:
    value = value.strip()
    replacements = SUBSCRIPTS if kind == "sub" else SUPERSCRIPTS
    unicode_value = _replace_characters(_SCRIPT_SIGN_PATTERN.sub(r"\1", value), replacements)
    if unicode_value is not None:
        return unicode_value

    prefix = "_" if kind == "sub" else "^"
    if len(value) == 1 or (kind == "sub" and value and all(c.isascii() and c.isalpha() for c in value)):
        return f"{prefix}{value}"
    return f"{prefix}({value})"


def _format_fraction(numerator: str, denominator: str) -> str:
    numerator = numerator.strip()
    denominator = denominator.strip()
    simple_numerator = _is_simple_alnum_dot(numerator)
    simple_denominator = _is_simple_numeric_dot(denominator) or len(denominator) == 1
    left = numerator if simple_numerator else f"({numerator})"
    right = denominator if simple_denominator else f"({denominator})"
    return f"{left}/{right}"


def _format_root(value: str, symbol: str = "√") -> str:
    value = value.strip()
    if _is_simple_alnum_dot(value):
        return f"{symbol}{value}"
    return f"{symbol}({value})"


def _is_named_left_context(character: str) -> bool:
    return character.isalpha() or character.isnumeric() or character in (")", "]", "}", LAYOUT_MARKER_END)


def _is_named_right_context(character: str) -> bool:
    return character.isalpha() or character.isnumeric() or character == "√" or character == LAYOUT_MARKER_START


def _normalize_named_operators(value: str) -> str:
    result: list[str] = []
    length = len(value)
    for index, character in enumerate(value):
        if character == NAMED_OPERATOR_START:
            if index > 0 and _is_named_left_context(value[index - 1]):
                result.append(" ")
            continue
        if character == NAMED_OPERATOR_END:
            if index + 1 < length and _is_named_right_context(value[index + 1]):
                result.append(" ")
            continue
        result.append(character)
    return "".join(result)


def _normalize_output(value: str) -> str:
    value = _normalize_named_operators(value)
    lines = value.split("\n")
    collapsed = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    filtered: list[str] = []
    for index, line in enumerate(collapsed):
        if line or (0 < index < len(collapsed) - 1):
            filtered.append(line)
    return "\n".join(filtered).strip()


@dataclass
class FractionNode:
    numerator: str
    denominator: str
    type: Literal["fraction"] = "fraction"


@dataclass
class OperatorNode:
    operator: str
    lower: str | None = None
    upper: str | None = None
    type: Literal["operator"] = "operator"


@dataclass
class MatrixNode:
    lines: list[str]
    baseline: int = 0
    type: Literal["matrix"] = "matrix"


LayoutNode = FractionNode | OperatorNode | MatrixNode


@dataclass
class Layout:
    lines: list[str]
    width: int
    baseline: int


def _pad_layout_line(line: str, width: int, centered: bool = False) -> str:
    padding = max(0, width - visible_width(line))
    left = padding // 2 if centered else 0
    return f"{' ' * left}{line}{' ' * (padding - left)}"


def _join_layouts(layouts: list[Layout]) -> Layout:
    if not layouts:
        return Layout(lines=[""], width=0, baseline=0)
    baseline = max(layout.baseline for layout in layouts)
    below = max(len(layout.lines) - layout.baseline - 1 for layout in layouts)
    lines: list[str] = []
    for row in range(baseline + below + 1):
        line = ""
        for layout in layouts:
            source_row = row - baseline + layout.baseline
            if 0 <= source_row < len(layout.lines):
                line += _pad_layout_line(layout.lines[source_row], layout.width)
            else:
                line += " " * layout.width
        lines.append(line.rstrip())
    total_width = sum(layout.width for layout in layouts)
    return Layout(lines=lines, width=total_width, baseline=baseline)


def _render_layout(source: str, nodes: list[LayoutNode]) -> Layout:
    rendered_lines: list[str] = []
    first_baseline = 0
    for source_line in source.split("\n"):
        layouts: list[Layout] = []
        position = 0
        previous_node: LayoutNode | None = None
        for match in _LAYOUT_MARKER_PATTERN.finditer(source_line):
            index = match.start()
            node_index = int(match.group(1))
            if node_index >= len(nodes):
                continue
            node = nodes[node_index]
            if index > position:
                sliced = source_line[position:index]
                trimmed = (sliced.lstrip() if previous_node else sliced).rstrip()
                preserve_leading_space = isinstance(previous_node, MatrixNode) and sliced[:1].isspace()
                preserve_trailing_space = isinstance(node, MatrixNode) and sliced[-1:].isspace()
                if trimmed:
                    text = f"{' ' if preserve_leading_space else ''}{trimmed}{' ' if preserve_trailing_space else ''}"
                elif preserve_leading_space or preserve_trailing_space:
                    text = " "
                else:
                    text = ""
                layouts.append(Layout(lines=[text], width=visible_width(text), baseline=0))
            if isinstance(node, FractionNode):
                numerator = _render_layout(node.numerator, nodes)
                denominator = _render_layout(node.denominator, nodes)
                content_width = max(numerator.width, denominator.width, 1)
                width = content_width + 2
                frac_lines = [_pad_layout_line(line, width, True) for line in numerator.lines]
                frac_lines.append(f" {'─' * content_width} ")
                frac_lines.extend(_pad_layout_line(line, width, True) for line in denominator.lines)
                layouts.append(
                    Layout(
                        lines=frac_lines,
                        width=width,
                        baseline=len(numerator.lines),
                    )
                )
            elif isinstance(node, OperatorNode):
                content_width = max(
                    visible_width(node.operator),
                    0 if node.lower is None else visible_width(node.lower),
                    0 if node.upper is None else visible_width(node.upper),
                )
                op_lines: list[str] = []
                if node.upper is not None:
                    op_lines.append(f"{_pad_layout_line(node.upper, content_width, True)} ")
                op_lines.append(f"{_pad_layout_line(node.operator, content_width, True)} ")
                if node.lower is not None:
                    op_lines.append(f"{_pad_layout_line(node.lower, content_width, True)} ")
                layouts.append(
                    Layout(
                        lines=op_lines,
                        width=content_width + 1,
                        baseline=0 if node.upper is None else 1,
                    )
                )
            else:
                width = max((visible_width(line) for line in node.lines), default=0)
                layouts.append(
                    Layout(
                        lines=[_pad_layout_line(line, width) for line in node.lines],
                        width=width,
                        baseline=node.baseline,
                    )
                )
            position = match.end()
            previous_node = node
        if position < len(source_line):
            sliced = source_line[position:]
            trimmed = sliced.lstrip() if previous_node else sliced
            if isinstance(previous_node, MatrixNode) and sliced[:1].isspace():
                text = f" {trimmed}"
            else:
                text = trimmed
            layouts.append(Layout(lines=[text], width=visible_width(text), baseline=0))
        line_layout = _join_layouts(layouts)
        if not rendered_lines:
            first_baseline = line_layout.baseline
        rendered_lines.extend(line_layout.lines)
    return Layout(
        lines=rendered_lines,
        width=max((visible_width(line) for line in rendered_lines), default=0),
        baseline=first_baseline,
    )


class LatexParser:
    """Recursive-descent LaTeX parser that emits terminal-friendly text.

    Ported from the `LatexParser` class in `packages/tui/src/latex.ts`.
    """

    def __init__(self, source: str, layout_nodes: list[LayoutNode], display: bool) -> None:
        self.source = source
        self.layout_nodes = layout_nodes
        self.display = display
        self.position = 0
        self.supported = True
        self.stack_fractions = True

    def render(self) -> str | None:
        rendered = self._parse_sequence()
        if not self.supported or self.position != len(self.source):
            return None
        return _normalize_output(rendered)

    def _parse_sequence(self, end_character: str | None = None) -> str:
        result = ""
        while self.position < len(self.source):
            character = self.source[self.position]
            if end_character is not None and character == end_character:
                self.position += 1
                return result

            if character == "}":
                self.supported = False
                return result

            if character == "{":
                self.position += 1
                result += self._parse_sequence("}")
                continue

            if character == "\\":
                command = self._parse_command()
                if command == NEGATIVE_SPACE:
                    result = result.rstrip()
                    if result.endswith(NAMED_OPERATOR_END):
                        result = result[: -len(NAMED_OPERATOR_END)]
                else:
                    result += command
                continue

            if character == "^" or character == "_":
                self.position += 1
                result = result.rstrip()
                script = _format_script(
                    self._parse_required_argument(False),
                    "sub" if character == "_" else "sup",
                )
                if result.endswith(NAMED_OPERATOR_END):
                    result = f"{result[: -len(NAMED_OPERATOR_END)]}{script}{NAMED_OPERATOR_END}"
                else:
                    result += script
                continue

            if character in " \t\n\r\f\v":
                result += self._parse_whitespace()
                continue

            if character == "=" or character == "<" or character == ">":
                result = f"{result.rstrip()} {character} "
                self.position += 1
                continue

            if character == "&":
                self.position += 1
                continue

            if character == "~":
                self.position += 1
                result += " "
                continue

            if character == ".":
                marker = _TRAILING_LAYOUT_MARKER_PATTERN.search(result)
                node = self.layout_nodes[int(marker.group(1))] if marker else None
                if isinstance(node, MatrixNode):
                    last_line = len(node.lines) - 1
                    node.lines[last_line] = f"{node.lines[last_line]}{character}"
                    self.position += 1
                    continue

            result += character
            self.position += 1

        if end_character is not None:
            self.supported = False
        return result

    def _parse_whitespace(self) -> str:
        while self.position < len(self.source) and self.source[self.position] in " \t\n\r\f\v":
            self.position += 1
        return " "

    def _parse_command(self) -> str:
        self.position += 1
        if self.position >= len(self.source):
            self.supported = False
            return ""

        first = self.source[self.position]
        if first in ("\n", "\r"):
            self.position += 1
            if first == "\r" and self.position < len(self.source) and self.source[self.position] == "\n":
                self.position += 1
            return " "
        if first.isascii() and first.isalpha():
            start = self.position
            while (
                self.position < len(self.source)
                and self.source[self.position].isascii()
                and self.source[self.position].isalpha()
            ):
                self.position += 1
            command = self.source[start : self.position]
        else:
            command = first
            self.position += 1

        if command == "\\":
            return "\n"
        if command in SPACING_COMMANDS:
            return " "
        if command in NEGATIVE_SPACING_COMMANDS:
            return NEGATIVE_SPACE
        if command in IGNORED_COMMANDS:
            return ""
        if command in ("{", "}", "$", "%", "#", "_", "&"):
            return command
        if command == "|":
            return "‖"
        if command == "not":
            value = self._parse_required_argument(False).strip()
            negated = NEGATED_SYMBOLS.get(value)
            if negated is not None:
                return f" {negated} "
            if not value:
                self.supported = False
                return ""
            return f" {value[0]}\u0338{value[1:]} "
        if command in LIMIT_OPERATORS:
            return self._parse_operator(command, "bracket", True, True)

        symbol = SYMBOLS.get(command)
        if symbol is not None:
            if command in DISPLAY_LIMIT_SYMBOLS:
                return self._parse_operator(symbol, "script", True)
            if command == "cdot" or command == "times" or command in RELATION_COMMANDS:
                return f" {symbol} "
            return symbol
        if command in NAMED_OPERATORS:
            return f"{NAMED_OPERATOR_START}{command}{NAMED_OPERATOR_END}"
        if command in SIZE_COMMANDS:
            return ""
        if command in ("left", "middle", "right"):
            if self.position < len(self.source) and self.source[self.position] == ".":
                self.position += 1
            return ""
        if command in ("frac", "dfrac", "tfrac"):
            should_stack = self.display and self.stack_fractions and command != "tfrac"
            numerator = self._parse_required_argument(not should_stack)
            denominator = self._parse_required_argument(not should_stack)
            if should_stack:
                self.layout_nodes.append(
                    FractionNode(
                        numerator=_normalize_output(numerator),
                        denominator=_normalize_output(denominator),
                    )
                )
                index = len(self.layout_nodes) - 1
                return f"{LAYOUT_MARKER_START}{index}{LAYOUT_MARKER_END}"
            return _format_fraction(numerator, denominator)
        if command == "sqrt":
            degree_raw = self._parse_optional_argument()
            degree = degree_raw.strip() if degree_raw is not None else None
            value = self._parse_required_argument()
            if degree is None or degree == "2":
                return _format_root(value)
            if degree == "3":
                return _format_root(value, "∛")
            if degree == "4":
                return _format_root(value, "∜")
            return f"{_format_script(degree, 'sup')}{_format_root(value)}"
        if command in ("boxed", "fbox"):
            return f"[{self._parse_required_argument().strip()}]"
        if command in ("binom", "dbinom", "tbinom"):
            return f"({self._parse_required_argument()} choose {self._parse_required_argument()})"
        accent = ACCENTS.get(command)
        if accent is not None:
            value = self._parse_required_argument()
            return f"{value}{accent}" if len(value) == 1 else f"{command}({value})"
        if command == "mathbb":
            value = self._parse_required_argument()
            return "".join(BLACKBOARD.get(character, character) for character in value)
        if command == "operatorname":
            starred = self.position < len(self.source) and self.source[self.position] == "*"
            if starred:
                self.position += 1
            operator = _normalize_output(self._parse_required_argument()).strip()
            return self._parse_operator(operator, "bracket", starred, True)
        if command in ("mod", "bmod"):
            return " mod "
        if command in ("pmod", "pod"):
            value = self._parse_required_argument().strip()
            return f" (mod {value})" if command == "pmod" else f" ({value})"
        if command in ("overset", "stackrel"):
            upper = self._parse_required_argument()
            value = self._parse_required_argument().strip()
            return f"{value}{_format_script(upper, 'sup')}"
        if command == "underset":
            lower = self._parse_required_argument()
            value = self._parse_required_argument().strip()
            return f"{value}{_format_script(lower, 'sub')}"
        if command in PLAIN_WRAPPERS:
            value = self._parse_required_argument()
            return value if command.startswith("text") or command == "mbox" else value.strip()
        if command == "begin":
            return self._parse_environment()
        if command == "end":
            self.supported = False
            return ""

        self.supported = False
        return f"\\{command}"

    def _parse_operator(
        self,
        operator: str,
        inline_lower_style: Literal["bracket", "script"],
        display_limits: bool,
        spaced: bool = False,
    ) -> str:
        use_display_limits = display_limits
        modifier_position = self.position
        while modifier_position < len(self.source) and self.source[modifier_position] in " \t":
            modifier_position += 1
        modifier = _LIMITS_MODIFIER_PATTERN.match(self.source[modifier_position:])
        if modifier:
            use_display_limits = modifier.group(1) == "limits"
            self.position = modifier_position + len(modifier.group(0))

        lower: str | None = None
        upper: str | None = None
        while True:
            script_position = self.position
            while script_position < len(self.source) and self.source[script_position] in " \t":
                script_position += 1
            kind = self.source[script_position] if script_position < len(self.source) else ""
            if kind != "_" and kind != "^":
                break
            self.position = script_position + 1
            value = _normalize_output(self._parse_required_argument(False)).replace(" ", "")
            if kind == "_":
                if lower is not None:
                    self.supported = False
                lower = value
            else:
                if upper is not None:
                    self.supported = False
                upper = value

        if self.display and use_display_limits and (lower is not None or upper is not None):
            self.layout_nodes.append(OperatorNode(operator=operator, lower=lower, upper=upper))
            index = len(self.layout_nodes) - 1
            return f"{LAYOUT_MARKER_START}{index}{LAYOUT_MARKER_END}"

        rendered = operator
        if lower is not None:
            rendered += f"[{lower}]" if inline_lower_style == "bracket" else _format_script(lower, "sub")
        if upper is not None:
            rendered += _format_script(upper, "sup")
        return f" {rendered} " if spaced else rendered

    def _parse_required_argument(self, stack_fractions: bool = True) -> str:
        previous_stack_fractions = self.stack_fractions
        self.stack_fractions = previous_stack_fractions and stack_fractions
        try:
            return self._parse_required_argument_value()
        finally:
            self.stack_fractions = previous_stack_fractions

    def _parse_required_argument_value(self) -> str:
        while self.position < len(self.source) and self.source[self.position] in " \t\n\r\f\v":
            self.position += 1
        if self.position >= len(self.source):
            self.supported = False
            return ""
        if self.source[self.position] == "{":
            self.position += 1
            return self._parse_sequence("}")
        if self.source[self.position] == "\\":
            return self._parse_command()
        value = self.source[self.position]
        self.position += 1
        return value

    def _parse_optional_argument(self) -> str | None:
        while self.position < len(self.source) and self.source[self.position] in " \t":
            self.position += 1
        if self.position >= len(self.source) or self.source[self.position] != "[":
            return None
        end = self.source.find("]", self.position + 1)
        if end < 0:
            self.supported = False
            return None
        value = self.source[self.position + 1 : end]
        self.position = end + 1
        return self._render_nested(value)

    def _read_raw_group(self) -> str | None:
        while self.position < len(self.source) and self.source[self.position] in " \t":
            self.position += 1
        if self.position >= len(self.source) or self.source[self.position] != "{":
            self.supported = False
            return None

        self.position += 1
        start = self.position
        depth = 1
        while self.position < len(self.source):
            character = self.source[self.position]
            if character == "\\":
                self.position += 2
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            if depth == 0:
                value = self.source[start : self.position]
                self.position += 1
                return value
            self.position += 1
        self.supported = False
        return None

    def _split_environment_rows(self, body: str) -> list[str]:
        return re.split(r"\\\\(?:\[[^\]\n]*\])?", body)

    def _parse_environment(self) -> str:
        environment = self._read_raw_group()
        if environment is None:
            return ""
        end_marker = f"\\end{{{environment}}}"
        end = self.source.find(end_marker, self.position)
        if end < 0:
            self.supported = False
            return ""
        body = self.source[self.position : end]
        self.position = end + len(end_marker)

        if environment in ("equation", "equation*", "displaymath"):
            return self._render_nested(body).strip()

        if environment in (
            "aligned",
            "align",
            "align*",
            "alignedat",
            "alignat",
            "alignat*",
            "gather",
            "gathered",
            "multline",
            "multline*",
            "split",
        ):
            aligned_at = environment in ("alignedat", "alignat", "alignat*")
            aligned_body = re.sub(r"^\s*\{[^}]*\}", "", body) if aligned_at else body
            rendered_rows: list[str] = []
            for row in self._split_environment_rows(aligned_body):
                cells = row.split("&")
                if aligned_at:
                    grouped: list[str] = []
                    pairs = (len(cells) + 1) // 2
                    for pair_index in range(pairs):
                        grouped.append("".join(cells[pair_index * 2 : pair_index * 2 + 2]))
                    source = " ".join(grouped)
                else:
                    source = "".join(cells)
                rendered = self._render_nested(source).strip()
                if rendered:
                    rendered_rows.append(rendered)
            return "\n".join(rendered_rows)

        if environment in ("cases", "cases*"):
            rows: list[list[str]] = []
            for row in self._split_environment_rows(body):
                cells = [self._render_nested(cell, False).strip() for cell in row.split("&")]
                if any(cells):
                    rows.append(cells)
            output_rows: list[str] = []
            for index, row in enumerate(rows):
                value = re.sub(r",\s*$", "", row[0] if row else "")
                condition = row[1] if len(row) > 1 else ""
                if index == 0:
                    delimiter = "⎧"
                elif index == len(rows) - 1:
                    delimiter = "⎩"
                else:
                    delimiter = "⎨"
                if re.match(r"(?i)^(?:if|when|for|otherwise)\b", condition):
                    condition_prefix = " "
                else:
                    condition_prefix = " if "
                if condition:
                    output_rows.append(f"{delimiter} {value}{condition_prefix}{condition}")
                else:
                    output_rows.append(f"{delimiter} {value}")
            return "\n".join(output_rows)

        if environment in (
            "array",
            "matrix",
            "smallmatrix",
            "pmatrix",
            "bmatrix",
            "Bmatrix",
            "vmatrix",
            "Vmatrix",
        ):
            matrix_body = re.sub(r"^\s*\{[^}]*\}", "", body) if environment == "array" else body
            return self._render_matrix(environment, matrix_body)

        self.supported = False
        return body

    def _render_matrix(self, environment: str, body: str) -> str:
        matrix: list[list[str]] = []
        for row in self._split_environment_rows(body):
            cells = [self._render_nested(cell, False).strip() for cell in row.split("&")]
            if any(cells):
                matrix.append(cells)
        column_count = max((len(row) for row in matrix), default=0)
        column_widths = [
            max((visible_width(row[column] if column < len(row) else "") for row in matrix), default=0)
            for column in range(column_count)
        ]
        rows = [
            " │ ".join(
                (
                    f"{row[column] if column < len(row) else ''}"
                    f"{PROTECTED_SPACE * max(0, column_widths[column] - visible_width(row[column] if column < len(row) else ''))}"
                )
                for column in range(column_count)
            )
            for row in matrix
        ]

        if environment in ("array", "matrix", "smallmatrix"):
            lines = rows
        else:
            delimiters: dict[str, tuple[str, str, str, str, str, str]] = {
                "pmatrix": ("⎛", "⎞", "⎜", "⎟", "⎝", "⎠"),
                "bmatrix": ("⎡", "⎤", "⎢", "⎥", "⎣", "⎦"),
                "Bmatrix": ("⎧", "⎫", "⎨", "⎬", "⎩", "⎭"),
                "vmatrix": ("│", "│", "│", "│", "│", "│"),
                "Vmatrix": ("║", "║", "║", "║", "║", "║"),
            }
            delimiter = delimiters.get(environment)
            if delimiter is None:
                self.supported = False
                return "\n".join(rows)
            lines = []
            for index, row_line in enumerate(rows):
                if index == 0:
                    left, right = delimiter[0], delimiter[1]
                elif index == len(rows) - 1:
                    left, right = delimiter[4], delimiter[5]
                else:
                    left, right = delimiter[2], delimiter[3]
                lines.append(f"{left} {row_line} {right}")

        if len(lines) <= 1:
            return lines[0] if lines else ""
        self.layout_nodes.append(MatrixNode(lines=lines, baseline=0))
        index = len(self.layout_nodes) - 1
        return f"{LAYOUT_MARKER_START}{index}{LAYOUT_MARKER_END}"

    def _render_nested(self, source: str, stack_fractions: bool = True) -> str:
        rendered = LatexParser(source, self.layout_nodes, self.display and stack_fractions).render()
        if rendered is None:
            self.supported = False
            return source
        return rendered


@dataclass
class RenderLatexOptions:
    """Options for :func:`render_latex`.

    ``display``: stack fractions and operator limits vertically for display
    math (default ``False``).
    """

    display: bool = False


def render_latex(source: str, options: RenderLatexOptions | None = None) -> str | None:
    """Render a basic LaTeX math expression as terminal-friendly Unicode.

    Returns ``None`` when the expression contains unsupported or malformed
    syntax. Ported from the ``renderLatex`` function in
    ``packages/tui/src/latex.ts``.
    """
    display = options.display if options is not None else False
    layout_nodes: list[LayoutNode] = []
    rendered = LatexParser(source, layout_nodes, display).render()
    if rendered is None:
        return None
    if not layout_nodes:
        return rendered.replace(PROTECTED_SPACE, " ")
    lines = _render_layout(rendered, layout_nodes).lines
    non_empty = [line for line in lines if line.strip()]
    if non_empty:
        indentation = min(len(line) - len(line.lstrip(" ")) for line in non_empty)
    else:
        indentation = 0
    return "\n".join(line[indentation:].rstrip() for line in lines).rstrip().replace(PROTECTED_SPACE, " ")
