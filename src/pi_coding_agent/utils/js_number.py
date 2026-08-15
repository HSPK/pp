"""JavaScript number-formatting semantics.

Python and JavaScript disagree on rounding, and the port has to match JS
byte-for-byte wherever a number reaches the screen.

Concretely: ``(1.25).toFixed(1)`` is ``"1.3"`` in JS but ``format(1.25, ".1f")``
is ``"1.2"`` in Python, because JS breaks ties away from zero while Python
breaks ties to even. Likewise ``Math.round(-0.5)`` is ``-0`` in JS (ties go
towards positive infinity) but Python's ``round(-0.5)`` is ``0`` and
``round(2.5)`` is ``2``.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal


def to_fixed(value: float, digits: int) -> str:
    """JS ``Number.prototype.toFixed``.

    JS rounds the *exact* binary value of the double, breaking ties away from
    zero. ``Decimal(float)`` reproduces the exact binary value and
    ``ROUND_HALF_UP`` reproduces the JS tie rule.
    """
    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(value).quantize(quantum, rounding=ROUND_HALF_UP))


def js_round(value: float) -> int:
    """JS ``Math.round``: ties go towards positive infinity."""
    return math.floor(value + 0.5)


__all__ = ["js_round", "to_fixed"]
