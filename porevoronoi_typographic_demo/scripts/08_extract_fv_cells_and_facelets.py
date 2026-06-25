from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.facelets import write_facelets
from src.fv_cells import write_fv_cells


if __name__ == "__main__":
    cell_audit = write_fv_cells()
    face_audit = write_facelets()
    print("FV cells saved")
    print(cell_audit)
    print("facelets and FV graph saved")
    print(face_audit)
