from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stage2_guard import guard


if __name__ == "__main__":
    guard("extract FV cells and facelets")

