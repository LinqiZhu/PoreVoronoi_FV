from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.flux_projection import write_flux_projection


if __name__ == "__main__":
    audit = write_flux_projection()
    print("conservative flux projection saved")
    print(audit)
