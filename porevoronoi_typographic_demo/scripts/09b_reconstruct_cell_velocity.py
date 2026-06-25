from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cell_velocity import write_cell_velocity


if __name__ == "__main__":
    audit = write_cell_velocity()
    print("Voronoi-FV cell velocity saved")
    print(audit)
