from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEMO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = DEMO_ROOT / "outputs"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def out_dir(*parts: str) -> Path:
    return ensure_dir(OUTPUT_ROOT.joinpath(*parts))


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_file(path: str | Path, message: str | None = None) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(message or f"Required file is missing: {p}")
    return p

