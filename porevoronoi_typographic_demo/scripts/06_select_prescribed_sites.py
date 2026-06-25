from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit import stage1_ready
from src.io_utils import DEMO_ROOT
from src.prescribed_sites import write_prescribed_sites


if __name__ == "__main__":
    if not stage1_ready(DEMO_ROOT):
        raise SystemExit("Run scripts/05_audit_outputs.py first and pass Stage 1 audit.")
    audit = write_prescribed_sites()
    print("prescribed sites saved")
    print(audit)
