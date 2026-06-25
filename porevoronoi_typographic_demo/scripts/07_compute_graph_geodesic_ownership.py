from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geodesic_ownership import write_ownership


if __name__ == "__main__":
    audit = write_ownership()
    print("graph-geodesic ownership saved")
    print(audit)
