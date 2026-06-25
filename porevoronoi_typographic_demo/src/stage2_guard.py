from __future__ import annotations

from pathlib import Path

from src.audit import stage1_ready
from src.io_utils import DEMO_ROOT


def guard(stage_name: str) -> None:
    if not stage1_ready(DEMO_ROOT):
        raise SystemExit(
            f"{stage_name} is a Stage 2 step. Run scripts/05_audit_outputs.py first and pass Stage 1 audit."
        )
    raise SystemExit(
        f"{stage_name} is intentionally gated for Stage 2. Stage 1 is ready; implement this step only after the Stage 2 prompt is active."
    )
