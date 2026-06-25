from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audit import audit_stage1
from src.io_utils import DEMO_ROOT


def main() -> None:
    report = audit_stage1(DEMO_ROOT)
    print(report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

