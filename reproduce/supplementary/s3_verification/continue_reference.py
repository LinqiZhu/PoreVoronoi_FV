"""Continue an image-resolved reference to steadiness, and re-score against it.

Two of the three refinement cases carry archived reference fields that did not
reach their own steady criterion: the skewed duct stopped at a steady momentum
residual of 1.9e-05 after 900 steps and the Bentheimer crop at 3.2e-05, while the
orthogonal duct reached 1.0e-09 after 400.  Those are exactly the two cases whose
error against the reference stalls while their error against their own finest
level keeps falling, so the reference is the leading candidate for the floor.

This module tests that directly rather than arguing it.  It warm-starts the
image-resolved reference solver from the archived velocity field, continues it, and
re-evaluates the finished refinement levels against the continued reference.  If
the floor is the reference, the gap closes; if it does not, the floor belongs to
the method and is reported as such.

Nothing is overwritten: the continued reference is written to a new file and the
archived one is left in place.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
STUDIES_DIR = Path(
    os.environ.get(
        "PVFV_STUDIES_DIR",
        "gpu/studies",
    )
)
if str(STUDIES_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIES_DIR))

import study_common as ec  # noqa: E402

PROTOCOL_ID = "pvfv_reference_continuation"


def continue_case(case, steps, out_dir, boot, dt=None, steady_tol=None):
    ns, cfg, cp = boot["ns"], boot["cfg"], boot["cp"]
    forward = boot["forward"]
    paths = ec.case_paths(case)
    geom, archived = forward.segmented.load_reference_npz(ns, paths["reference"], cfg)
    if geom is None:
        raise FileNotFoundError(paths["reference"])

    clone = ns["pvfv_clone_cfg"](cfg)
    clone.n_steps = int(steps)
    if dt is not None:
        clone.dt = float(dt)
    if steady_tol is not None:
        clone.steady_tol = float(steady_tol)
    clone.stop_on_steady = True

    started = time.perf_counter()
    result = dict(
        ns["run_velocity_pressure_projection_gpu"](
            geom, clone, initial_U=archived["U"], run_label="continue_" + case
        )
    )
    elapsed = float(time.perf_counter() - started)

    record = {
        "protocol_id": PROTOCOL_ID,
        "case": case,
        "archived_reference": str(paths["reference"]),
        "archived_reference_sha256": ec.sha256_file(paths["reference"]),
        "continuation_steps_requested": int(steps),
        "continuation_steps_completed": int(result.get("steps_completed", -1)),
        "continuation_dt": float(clone.dt),
        "continuation_steady_tol": float(getattr(clone, "steady_tol", float("nan"))),
        "steady_converged": bool(result.get("steady_converged", False)),
        "steady_momentum_inf": float(result.get("steady_momentum_inf", float("nan"))),
        "mass_inf_per_volume": float(result.get("mass_inf_per_volume", float("nan"))),
        "K_eff_x_archived": float(archived["K_eff_x"]),
        "K_eff_x_continued": float(result.get("K_eff_x", float("nan"))),
        "wall_seconds": elapsed,
        "solver": "run_velocity_pressure_projection_gpu, warm started from the archived U",
    }
    U_old = cp.asnumpy(archived["U"]).astype(np.float64)
    U_new = cp.asnumpy(result["U"]).astype(np.float64)
    volume = cp.asnumpy(geom.volume).astype(np.float64)

    def weighted(a, b):
        num = float(np.sqrt(np.sum(volume * np.sum((a - b) ** 2, axis=1))))
        den = float(np.sqrt(np.sum(volume * np.sum(b ** 2, axis=1))))
        return 100.0 * num / max(den, 1e-300)

    record["reference_moved_percent"] = weighted(U_new, U_old)
    record["K_eff_moved_percent"] = 100.0 * abs(
        record["K_eff_x_continued"] - record["K_eff_x_archived"]
    ) / max(abs(record["K_eff_x_archived"]), 1e-300)

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / (case + "__reference_continued.npz")
    with np.load(paths["reference"], allow_pickle=True) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["U"] = U_new
    payload["p"] = cp.asnumpy(result["p"]).astype(np.float64)
    payload["phi"] = cp.asnumpy(result["phi"]).astype(np.float64)
    payload["div"] = cp.asnumpy(result["div"]).astype(np.float64)
    payload["K_eff_x"] = np.asarray(record["K_eff_x_continued"])
    payload["metadata"] = np.asarray(json.dumps({
        "continued_from": str(paths["reference"]),
        "steady_converged": record["steady_converged"],
        "steady_momentum_inf": record["steady_momentum_inf"],
        "steps_completed": record["continuation_steps_completed"],
        "continuation_protocol": PROTOCOL_ID,
    }))
    np.savez_compressed(target, **payload)
    record["continued_reference"] = str(target)
    record["continued_reference_sha256"] = ec.sha256_file(target)
    print(
        "[cont] %-16s steps=%s steady=%s momentum_inf=%.3e  reference moved %.4f%%  "
        "K moved %.4f%%  (%.1fs)"
        % (case, record["continuation_steps_completed"], record["steady_converged"],
           record["steady_momentum_inf"], record["reference_moved_percent"],
           record["K_eff_moved_percent"], elapsed),
        flush=True,
    )
    del geom
    ec.free_gpu()
    return record


def rescore(case, continued_path, ladder_root, boot):
    """Re-evaluate every finished refinement level against the continued reference."""
    ns, cfg, cp = boot["ns"], boot["cfg"], boot["cp"]
    forward = boot["forward"]
    reference_geom, reference_result = forward.segmented.load_reference_npz(
        ns, Path(continued_path), cfg
    )
    rows = []
    case_dir = Path(ladder_root) / "runs" / case
    if not case_dir.exists():
        return rows
    for level in sorted(case_dir.iterdir(), key=lambda d: d.name):
        state = level / "state.npz"
        partition = level / "sites_and_partition.npz"
        if not (state.exists() and partition.exists()):
            continue
        with np.load(partition) as data:
            sites = np.asarray(data["seed_flat"], dtype=np.int64)
        geom, _meta, _t = ec.build_geometry_from_seed_flat(
            sites, mask_path=ec.case_paths(case)["mask"],
            seed_spec="rescore:" + level.name,
        )
        U_ref_gpu, p_ref_gpu = ns["coarsen_voxel_reference_to_coarse_gpu"](
            reference_geom, reference_result, geom
        )
        U_ref = cp.asnumpy(U_ref_gpu).astype(np.float64)
        p_ref = cp.asnumpy(p_ref_gpu).astype(np.float64)
        volume = cp.asnumpy(geom.volume).astype(np.float64)
        dvec = cp.asnumpy(geom.dvec).astype(np.float64)
        with np.load(state) as data:
            U = np.asarray(data["U"], dtype=np.float64)
            p = np.asarray(data["p"], dtype=np.float64)
            phi = np.asarray(data["phi"], dtype=np.float64)
        e_u = 100.0 * float(
            np.sqrt(np.sum(volume * np.sum((U - U_ref) ** 2, axis=1)))
            / max(np.sqrt(np.sum(volume * np.sum(U_ref ** 2, axis=1))), 1e-300)
        )
        e_p, _detail = ec.gauge_invariant_pressure_error(p, p_ref, volume)
        mean_flux = np.sum(phi[:, None] * dvec, axis=0) / float(volume.sum())
        force_x = float(ec.FORWARD_ARGS["body_force"][0])
        K_eff = float(cfg.nu) * float(mean_flux[0]) / force_x
        K_ref = float(reference_result["K_eff_x"])
        old = json.loads((level / "row.json").read_text(encoding="utf-8"))
        rows.append({
            "protocol_id": PROTOCOL_ID, "case": case, "level_label": level.name,
            "N_cv": int(old["N_cv"]),
            "e_u_archived_reference_percent": old["e_u_percent"],
            "e_u_continued_reference_percent": e_u,
            "e_p_archived_reference_percent": old["e_p_percent"],
            "e_p_continued_reference_percent": e_p,
            "e_K_archived_reference_percent": old["e_K_percent"],
            "e_K_continued_reference_percent":
                100.0 * abs(K_eff - K_ref) / max(abs(K_ref), 1e-300),
        })
        print(
            "[score] %-16s %-10s e_u %8.4f%% -> %8.4f%%   e_K %8.4f%% -> %8.4f%%"
            % (case, level.name, old["e_u_percent"], e_u,
               old["e_K_percent"], rows[-1]["e_K_continued_reference_percent"]),
            flush=True,
        )
        del geom
        ec.free_gpu()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="skewed_duct")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--steady-tol", type=float, default=None)
    parser.add_argument("--ladder-root", default="reproduce/figure_06/refinement")
    parser.add_argument("--out", default="reproduce/supplementary/s3_verification")
    parser.add_argument("--no-rescore", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    boot = ec.bootstrap(out / "bootstrap")
    records, scores = [], []
    for case in [c for c in args.cases.split(",") if c]:
        record = continue_case(case, args.steps, out, boot,
                               dt=args.dt, steady_tol=args.steady_tol)
        records.append(record)
        if not args.no_rescore:
            scores.extend(rescore(case, record["continued_reference"],
                                  args.ladder_root, boot))
    ec.write_json(out / "reference_continuation.json", records)
    if scores:
        ec.write_csv(out / "rescored_refinement.csv", scores,
                     sorted({k for r in scores for k in r}))
        print("[write] " + str(out / "rescored_refinement.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
