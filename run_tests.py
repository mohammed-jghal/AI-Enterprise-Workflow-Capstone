"""Run the test suite from any working directory."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pytest

raise SystemExit(pytest.main(["-q", str(ROOT / "tests")]))
