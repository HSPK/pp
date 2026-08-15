"""Make `harness.py` importable from tests in `tests/suite/regressions/`."""

from __future__ import annotations

import sys
from pathlib import Path

_SUITE_DIR = str(Path(__file__).parent)
if _SUITE_DIR not in sys.path:
    sys.path.insert(0, _SUITE_DIR)
