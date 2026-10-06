from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(r".")
DEFAULT_REF = (
    ROOT
    / "outputs"
    / "reference_data"
    / "reference_data"
    / "public_berea_figshare_128__reference.npz"
)
DEFAULT_OUT_DIR = ROOT / "outputs" / "particle_tracks"


def parse_frame_list(text: str) -> list[int]:
    frames: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            frames.append(int(item))
            continue
        parts = [int(v) for v in item.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        elif len(parts) == 3:
            start, stop, step = parts
        else:
            raise ValueError(f"Bad frame range: {item!r}")
        frames.extend(range(start, stop + 1, step))
    return sorted(set(frames))


def weighted_without_replacement(
    rng: np.random.Generator,
    weights: np.ndarray,
    n: int,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if int(np.count_nonzero(weights)) == 0:
        weights = np.ones_like(weights, dtype=np.float64)
    weights = weights / float(np.sum(weights))
    n = min(int(n), int(weights.size))
    return rng.choice(weights.size, size=n, replace=False, p=weights)


def select_initial_indices(
    rng: np.random.Generator,
    pore_cells: np.ndarray,
    velocities: np.ndarray,
    n_particles: int,
    mode: str,
) -> np.ndarray:
    mode_key = str(mode).lower().strip()
    speed = np.linalg.norm(velocities[pore_cells], axis=1)
    ux = np.abs(velocities[pore_cells, 0])
    if mode_key == "uniform":
        local = rng.choice(pore_cells.size, size=min(int(n_particles), pore_cells.size), replace=False)
    elif mode_key == "speed_weighted":
        local = weighted_without_replacement(rng, speed + 1.0e-12 * float(np.nanmax(speed) + 1.0), int(n_particles))
    elif mode_key == "flux_weighted":
        local = weighted_without_replacement(rng, ux + 1.0e-12 * float(np.nanmax(ux) + 1.0), int(n_particles))
    else:
        raise ValueError(f"Unknown sample mode: {mode!r}")
    return pore_cells[local].astype(np.int64, copy=False)


def snap_to_pore(
    zyx: np.ndarray,
    labels: np.ndarray,
    tree: cKDTree,
    pore_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    d, h, w = labels.shape
    rounded = np.rint(zyx).astype(np.int64)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, d - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, h - 1)
    rounded[:, 2] = np.mod(rounded[:, 2], w)
    ok = labels[rounded[:, 0], rounded[:, 1], rounded[:, 2]] >= 0
    snapped = rounded.copy()
    disp = np.zeros(rounded.shape[0], dtype=np.float64)
    if not bool(np.all(ok)):
        bad = np.flatnonzero(~ok)
        dist, idx = tree.query(rounded[bad].astype(np.float64), k=1)
        snapped[bad] = pore_zyx[np.asarray(idx, dtype=np.int64)]
        disp[bad] = np.asarray(dist, dtype=np.float64)
    return snapped, disp


def first_impermeable_face_fraction(
    position_zyx_unwrapped: np.ndarray,
    displacement_zyx: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, int, bool]:
    """Return the traversable segment fraction using exact six-neighbour face crossings."""
    position = np.asarray(position_zyx_unwrapped, dtype=np.float64)
    displacement = np.asarray(displacement_zyx, dtype=np.float64)
    if position.shape != (3,) or displacement.shape != (3,):
        raise ValueError("Particle position and displacement must have three components")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(displacement)):
        raise ValueError("Particle position and displacement must be finite")

    d, h, w = mask.shape
    # Match the CSV-to-voxel convention used by the transfer runner. Using
    # floor(x + 0.5) is not equivalent near a half-integer because the addition
    # itself can round to the next integer.
    cell = np.rint(position).astype(np.int64)
    start_index = (int(cell[0]), int(cell[1]), int(cell[2] % w))
    if not (0 <= cell[0] < d and 0 <= cell[1] < h and bool(mask[start_index])):
        raise RuntimeError(
            "Particle starts outside the pore space: "
            f"position={position.tolist()}, cell={start_index}"
        )

    step = np.sign(displacement).astype(np.int64)
    t_max = np.full(3, np.inf, dtype=np.float64)
    t_delta = np.full(3, np.inf, dtype=np.float64)
    for axis in range(3):
        value = float(displacement[axis])
        if value > 0.0:
            boundary = float(cell[axis]) + 0.5
            t_max[axis] = max((boundary - float(position[axis])) / value, 0.0)
            t_delta[axis] = 1.0 / value
        elif value < 0.0:
            boundary = float(cell[axis]) - 0.5
            t_max[axis] = max((boundary - float(position[axis])) / value, 0.0)
            t_delta[axis] = -1.0 / value

    face_crossings = 0
    while True:
        axis = int(np.argmin(t_max))
        crossing_fraction = float(t_max[axis])
        if not np.isfinite(crossing_fraction) or crossing_fraction > 1.0:
            return 1.0, face_crossings, False

        next_cell = cell.copy()
        next_cell[axis] += step[axis]
        outside_nonperiodic = not (0 <= next_cell[0] < d and 0 <= next_cell[1] < h)
        next_index = (int(next_cell[0]), int(next_cell[1]), int(next_cell[2] % w))
        next_is_pore = not outside_nonperiodic and bool(mask[next_index])
        if not next_is_pore:
            accepted = float(np.nextafter(max(crossing_fraction, 0.0), 0.0))
            return max(accepted, 0.0), face_crossings, True

        cell = next_cell
        face_crossings += 1
        t_max[axis] += t_delta[axis]


def advance_particles_pore_safe(
    zyx: np.ndarray,
    x_unwrapped: np.ndarray,
    velocity_xyz: np.ndarray,
    *,
    displacement_scale: float,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Advance particles along velocity and stop at the first impermeable voxel face."""
    positions = np.asarray(zyx, dtype=np.float64)
    unwrapped_x = np.asarray(x_unwrapped, dtype=np.float64)
    velocity = np.asarray(velocity_xyz, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected particle positions of shape (N, 3), got {positions.shape}")
    if velocity.shape != positions.shape or unwrapped_x.shape != (positions.shape[0],):
        raise ValueError("Particle position, unwrapped-x, and velocity arrays are inconsistent")

    start = positions.copy()
    start[:, 2] = unwrapped_x
    displacement = float(displacement_scale) * velocity[:, [2, 1, 0]]
    accepted_fraction = np.ones(positions.shape[0], dtype=np.float64)
    crossing_count = 0
    collision_count = 0
    for particle_index in range(positions.shape[0]):
        fraction, crossings, collided = first_impermeable_face_fraction(
            start[particle_index],
            displacement[particle_index],
            mask,
        )
        accepted_fraction[particle_index] = fraction
        crossing_count += int(crossings)
        collision_count += int(collided)

    advanced = start + accepted_fraction[:, None] * displacement
    # A scalar nextafter on the segment fraction can still round a Cartesian
    # endpoint exactly onto a voxel face. Give every moved endpoint a unique
    # pore-side face ownership by one representable coordinate value toward
    # its start position; this changes no model-scale trajectory parameter.
    moved_particle = np.any(displacement != 0.0, axis=1)
    advanced[moved_particle] = np.nextafter(
        advanced[moved_particle],
        start[moved_particle],
    )
    new_x_unwrapped = advanced[:, 2].copy()
    advanced[:, 2] = np.mod(advanced[:, 2], mask.shape[2])
    rounded = np.rint(advanced).astype(np.int64)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, mask.shape[0] - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, mask.shape[1] - 1)
    rounded[:, 2] = np.mod(rounded[:, 2], mask.shape[2])
    endpoint_is_pore = mask[rounded[:, 0], rounded[:, 1], rounded[:, 2]]
    if not bool(np.all(endpoint_is_pore)):
        bad = np.flatnonzero(~endpoint_is_pore)
        first = int(bad[0])
        raise RuntimeError(
            "Pore-safe particle advance produced a solid-voxel endpoint: "
            f"particle={first}, start={start[first].tolist()}, "
            f"displacement={displacement[first].tolist()}, "
            f"accepted_fraction={accepted_fraction[first]:.17g}, "
            f"advanced={advanced[first].tolist()}, rounded={rounded[first].tolist()}, "
            f"bad_count={bad.size}"
        )
    return advanced, new_x_unwrapped, {
        "segment_face_crossings": int(crossing_count),
        "wall_collision_events": int(collision_count),
        "minimum_accepted_fraction": float(np.min(accepted_fraction)),
    }


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Berea PTV-like particle-state window from a voxel-FV reference.")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--frame-list", default="0:99:1")
    parser.add_argument("--n-particles", type=int, default=80000)
    parser.add_argument("--sample-mode", choices=["speed_weighted", "flux_weighted", "uniform"], default="speed_weighted")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--step-vox-per-frame", type=float, default=0.45)
    parser.add_argument("--jitter", type=float, default=0.30)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    t0 = time.perf_counter()
    ref_path = Path(args.reference)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = parse_frame_list(str(args.frame_list))
    if not frames:
        raise ValueError("No frames requested")
    tag = str(args.tag).strip() or f"n{int(args.n_particles)}_frames{frames[0]:03d}_{frames[-1]:03d}_{args.sample_mode}"
    out_csv = out_dir / f"berea_particle_window_{tag}.csv.gz"
    out_summary = out_dir / f"berea_particle_window_{tag}_summary.json"

    data = np.load(ref_path, allow_pickle=True)
    labels = np.asarray(data["labels"], dtype=np.int32)
    mask = np.asarray(data["mask"], dtype=bool)
    velocities = np.asarray(data["U"], dtype=np.float64)
    pressure = np.asarray(data["p"], dtype=np.float64)
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        raise RuntimeError(f"Unexpected velocity shape: {velocities.shape}")
    pore_zyx = np.argwhere(labels >= 0).astype(np.float64)
    pore_cells = labels[labels >= 0].astype(np.int64, copy=False)
    if pore_cells.size != velocities.shape[0]:
        raise RuntimeError(f"Label/cell mismatch: labels={pore_cells.size}, U={velocities.shape[0]}")
    rng = np.random.default_rng(int(args.seed))

    initial_cells = select_initial_indices(
        rng,
        pore_cells=np.arange(velocities.shape[0], dtype=np.int64),
        velocities=velocities,
        n_particles=int(args.n_particles),
        mode=str(args.sample_mode),
    )
    # Convert cell ids back to voxel coordinates via the one-to-one voxel-FV label map.
    cell_to_zyx = np.empty((velocities.shape[0], 3), dtype=np.float64)
    cell_to_zyx[pore_cells] = pore_zyx
    zyx = cell_to_zyx[initial_cells].astype(np.float64, copy=True)
    if float(args.jitter) > 0.0:
        zyx += rng.uniform(-float(args.jitter), float(args.jitter), size=zyx.shape)
    d, h, w = labels.shape
    zyx[:, 0] = np.clip(zyx[:, 0], 0.0, d - 1.0)
    zyx[:, 1] = np.clip(zyx[:, 1], 0.0, h - 1.0)
    x_unwrapped = np.mod(zyx[:, 2], w).astype(np.float64)
    zyx[:, 2] = np.mod(zyx[:, 2], w)

    speed = np.linalg.norm(velocities, axis=1)
    speed_scale = float(np.percentile(speed[np.isfinite(speed)], 75))
    speed_scale = max(speed_scale, 1.0e-12)

    fields = [
        "frame_id",
        "frame_absolute_time_s",
        "frame_pv",
        "particle_id",
        "batch_id",
        "x_unwrapped_voxel",
        "x_mod_voxel",
        "y_voxel",
        "z_voxel",
        "ux_dns_frame_interp",
        "uy_dns_frame_interp",
        "uz_dns_frame_interp",
        "p_dns_frame_interp",
        "track_interp_alpha",
        "track_status_before",
        "track_status_after",
    ]

    unique_site_flats: set[int] = set()
    snap_all: list[float] = []
    speed_samples: list[float] = []
    n_rows = 0
    total_face_crossings = 0
    total_wall_collisions = 0
    minimum_accepted_fraction = 1.0
    with gzip.open(out_csv, "wt", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for frame_pos, frame_id in enumerate(frames):
            snapped = np.rint(zyx).astype(np.int64)
            snapped[:, 0] = np.clip(snapped[:, 0], 0, d - 1)
            snapped[:, 1] = np.clip(snapped[:, 1], 0, h - 1)
            snapped[:, 2] = np.mod(snapped[:, 2], w)
            disp = np.zeros(snapped.shape[0], dtype=np.float64)
            cell_ids = labels[snapped[:, 0], snapped[:, 1], snapped[:, 2]].astype(np.int64, copy=False)
            valid = cell_ids >= 0
            if not bool(np.all(valid)):
                raise RuntimeError("Snapping failed to map every particle to a pore cell")
            vel = velocities[cell_ids]
            p = pressure[cell_ids]
            x_mod = np.mod(x_unwrapped, w)
            frame_pv = 0.0 if len(frames) == 1 else frame_pos / float(len(frames) - 1)
            rows = []
            for i in range(cell_ids.size):
                zz = float(zyx[i, 0])
                yy = float(zyx[i, 1])
                xx_mod = float(x_mod[i])
                rows.append(
                    {
                        "frame_id": int(frame_id),
                        "frame_absolute_time_s": float(frame_pos),
                        "frame_pv": float(frame_pv),
                        "particle_id": int(i),
                        "batch_id": 0,
                        "x_unwrapped_voxel": float(x_unwrapped[i]),
                        "x_mod_voxel": xx_mod,
                        "y_voxel": yy,
                        "z_voxel": zz,
                        "ux_dns_frame_interp": float(vel[i, 0]),
                        "uy_dns_frame_interp": float(vel[i, 1]),
                        "uz_dns_frame_interp": float(vel[i, 2]),
                        "p_dns_frame_interp": float(p[i]),
                        "track_interp_alpha": 0.0,
                        "track_status_before": "reference_rollout",
                        "track_status_after": "reference_rollout",
                    }
                )
            writer.writerows(rows)
            n_rows += len(rows)
            unique_site_flats.update(np.ravel_multi_index((snapped[:, 0], snapped[:, 1], snapped[:, 2]), labels.shape).tolist())
            snap_all.extend(float(v) for v in disp)
            speed_samples.extend(float(v) for v in np.linalg.norm(vel, axis=1))

            if frame_pos != len(frames) - 1:
                zyx, x_unwrapped, advance_metadata = advance_particles_pore_safe(
                    zyx,
                    x_unwrapped,
                    vel,
                    displacement_scale=float(args.step_vox_per_frame) / speed_scale,
                    mask=mask,
                )
                total_face_crossings += int(advance_metadata["segment_face_crossings"])
                total_wall_collisions += int(advance_metadata["wall_collision_events"])
                minimum_accepted_fraction = min(
                    minimum_accepted_fraction,
                    float(advance_metadata["minimum_accepted_fraction"]),
                )

    snap_arr = np.asarray(snap_all, dtype=np.float64)
    sampled_speed = np.asarray(speed_samples, dtype=np.float64)
    summary = {
        "reference": str(ref_path),
        "output_csv": str(out_csv),
        "frame_list": ",".join(str(v) for v in frames),
        "n_frames": int(len(frames)),
        "n_particles": int(initial_cells.size),
        "rows": int(n_rows),
        "unique_snapped_sites": int(len(unique_site_flats)),
        "pore_voxels": int(mask.sum()),
        "unique_site_fraction_of_pores": float(len(unique_site_flats) / max(int(mask.sum()), 1)),
        "sample_mode": str(args.sample_mode),
        "seed": int(args.seed),
        "step_vox_per_frame": float(args.step_vox_per_frame),
        "speed_scale_p75": float(speed_scale),
        "trajectory_integrator": "explicit_euler_exact_voxel_face_stop",
        "trajectory_connectivity": 6,
        "trajectory_wall_rule": "stop_at_first_impermeable_voxel_face",
        "trajectory_face_crossings": int(total_face_crossings),
        "trajectory_wall_collision_events": int(total_wall_collisions),
        "trajectory_minimum_accepted_fraction": float(minimum_accepted_fraction),
        "snap_disp_median_vox": float(np.median(snap_arr)) if snap_arr.size else 0.0,
        "snap_disp_p95_vox": float(np.percentile(snap_arr, 95)) if snap_arr.size else 0.0,
        "snap_disp_max_vox": float(np.max(snap_arr)) if snap_arr.size else 0.0,
        "sampled_speed_mean": float(np.mean(sampled_speed)) if sampled_speed.size else 0.0,
        "sampled_speed_p50": float(np.percentile(sampled_speed, 50)) if sampled_speed.size else 0.0,
        "sampled_speed_p95": float(np.percentile(sampled_speed, 95)) if sampled_speed.size else 0.0,
        "K_eff_x_reference": float(data["K_eff_x"]),
        "elapsed_s": float(time.perf_counter() - t0),
    }
    write_summary(out_summary, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
