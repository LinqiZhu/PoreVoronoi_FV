from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.io_utils import out_dir, save_json
from src.streamlines import sample_particles_from_pore, trace_streamlines


def main() -> None:
    pore = np.load(out_dir("masks") / "pore_mask_3d.npz")["mask"]
    flow = np.load(out_dir("flow") / "pressure_flow.npz")
    points, offsets, audit_lines = trace_streamlines(pore, flow["pressure"])
    particles, audit_particles = sample_particles_from_pore(pore, flow["pressure"])
    stream_dir = out_dir("streamlines")
    particle_dir = out_dir("particles")
    np.savez_compressed(stream_dir / "streamlines.npz", points=points, offsets=offsets)
    np.savez_compressed(particle_dir / "particles.npz", particles=particles)
    save_json(stream_dir / "streamlines_audit.json", audit_lines)
    save_json(particle_dir / "particles_audit.json", audit_particles)
    print(f"streamlines saved: {stream_dir / 'streamlines.npz'}")
    print(audit_lines)
    print(audit_particles)


if __name__ == "__main__":
    main()


