from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(os.environ.get("GEOVORONOI_FV_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUTPUTS_ROOT = Path(os.environ.get("GEOVORONOI_FV_OUTPUTS", REPO_ROOT / "outputs")).resolve()
DATA_ROOT = Path(os.environ.get("GEOVORONOI_FV_DATA", REPO_ROOT / "data")).resolve()
EXAMPLES_ROOT = Path(os.environ.get("GEOVORONOI_FV_EXAMPLES", REPO_ROOT / "examples")).resolve()
SRC_ROOT = REPO_ROOT / "src" / "geovoronoi_fv"


def add_source_path() -> None:
    for path in (SRC_ROOT, REPO_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
