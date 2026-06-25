from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit import audit_stage2
from src.io_utils import DEMO_ROOT


if __name__ == "__main__":
    report = audit_stage2(DEMO_ROOT)
    print(report)
    if not report["passed"]:
        raise SystemExit(1)
