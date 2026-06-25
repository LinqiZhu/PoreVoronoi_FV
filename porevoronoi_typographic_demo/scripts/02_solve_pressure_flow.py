from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.io_utils import out_dir, save_json
from src.pressure_flow import solve_pressure


def main() -> None:
    pore = np.load(out_dir("masks") / "pore_mask_3d.npz")["mask"]
    pressure, speed, audit = solve_pressure(pore)
    out = out_dir("flow")
    np.savez_compressed(out / "pressure_flow.npz", pressure=pressure, speed=speed)
    save_json(out / "pressure_flow_audit.json", audit)
    print(f"pressure flow saved: {out / 'pressure_flow.npz'}")
    print(audit)


if __name__ == "__main__":
    main()

