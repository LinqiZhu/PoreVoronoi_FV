"""Library of the assisted calculations on the trajectory partition: paths, hashes, the partition built from every
particle record, the two record selectors, one solve with its readouts, checkpoint I/O.

All three arms use the SAME partition: the paper's published site set, built from every window record of every frame
(case_loader.select_records(case, 'pub')).
  PN  Stokes-only (the configuration of Table 5)
  AN  assisted, one record per cell (selector S-A)  -> N_obs = N_cells
  MN  assisted, every record (selector S-M)         -> N_obs = number of window records

The modules and config.json one folder up are imported unchanged and never written.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import os
import socket
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import config_io  # noqa: E402
config_io.thread_env(1)
import numpy as np  # noqa: E402

import case_loader  # noqa: E402
import cell_complex  # noqa: E402
import readouts as paper_readouts  # noqa: E402  (alias: this module defines a function readouts())
import observations  # noqa: E402
import stokes_solve  # noqa: E402
import velocity_recovery as rp  # noqa: E402

PARENT_RUNTIME = ("config_io.py", "case_loader.py", "cell_complex.py", "readouts.py", "observations.py",
                  "stokes_solve.py", "geodesic_face_operator.py", "hybrid_voronoi_trace.py", "fast_assembly.py",
                  "pore_ownership.py", "velocity_recovery.py", "point_location.py", "config.json")
D_RUNTIME = ("assisted_arms.py", "run_assisted_arms.py", "ud.py", "jd.sh", "assisted_protocol.json")
ARMS = ("PN", "AN", "MN")


class Stop(Exception):
    """Ends an arm with a declared outcome (solve_failed, partition_invalid)."""

    def __init__(self, outcome, n=None, message=""):
        super().__init__(f"{outcome} n={n} {message}")
        self.outcome, self.n, self.message = outcome, n, message


# ----------------------------------------------------------------------------------------------------------- config
def load_cfg():
    cfg = config_io.load_cfg(os.path.join(PARENT, "config.json"))
    cfg.pop("single_frame", None)  # single-frame test parameter; the arms use every record
    cfg["out_dir"] = os.path.abspath(os.environ.get("PVFV_ASSISTED_OUT")
                                     or os.path.join(HERE, "../../outputs/assisted"))
    assert cfg["trace_basis"] == "connected_p1" and cfg["viscous_form"] == "symmetric_gradient"
    return cfg


def protocol():
    return config_io.load_json(os.path.join(HERE, "assisted_protocol.json"))


def code_hashes():
    h = {name: config_io.sha(os.path.join(HERE, name)) for name in D_RUNTIME
         if os.path.exists(os.path.join(HERE, name))}
    h.update({"../" + name: config_io.sha(os.path.join(PARENT, name)) for name in PARENT_RUNTIME})
    return h


def input_hashes(cfg, code):
    p = config_io.case_paths(cfg, code)
    return {k: config_io.sha(p[k]) for k in ("reference", "window", "manifest")}


def cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor()


def attempt():
    e = os.environ.get
    return dict(job=e("SLURM_JOB_ID"), array_job=e("SLURM_ARRAY_JOB_ID"), array_task=e("SLURM_ARRAY_TASK_ID"),
                restart=e("SLURM_RESTART_COUNT"), qos=e("SLURM_JOB_QOS"), qos_scontrol=e("PVFV_QOS"),
                partition=e("SLURM_JOB_PARTITION"), host=socket.gethostname(), pid=os.getpid(),
                threads={k: e(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                affinity=(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None), cpu=cpu_model(),
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def stamp(obj):
    return dict(obj, attempt=attempt(), code=code_hashes(),
                protocol_sha256=config_io.sha(os.path.join(HERE, "assisted_protocol.json")))


# ------------------------------------------------------------------------------------------ published partition
def published_records(case):
    """Every window record, all frames, all particles (protocol.partition.rule)."""
    return case_loader.select_records(case, "pub")


def published_partition(cfg, case):
    rec = published_records(case)
    seeds = cell_complex.sites_of(case, rec)
    part = cell_complex.build(case, seeds, order="paper", basis=cfg["trace_basis"])
    return rec, seeds, part


def check_partition(case, rec, seeds, part):
    """protocol.gates.D_part: cell_complex checks pass AND n_cells equals the published N_c of the paper row."""
    geom = part["geom"]
    published_N_c = int(case.manifest["row"]["N_cv"])
    out = dict(n_cells=int(geom.n_cells), n_sites=int(seeds.size), published_N_c=published_N_c,
               n_records=int(rec.size), n_frames=int(np.unique(case.records["frame"][rec]).size),
               n_particles=int(np.unique(case.records["particle"][rec]).size),
               partition_checks=part["checks"])
    out["n_cells_equals_published"] = bool(out["n_cells"] == published_N_c == out["n_sites"])
    out["ok"] = bool(out["n_cells_equals_published"] and part["checks"]["pass_"])
    return out


# --------------------------------------------------------------------------------------------------- selectors
def owners_of(case, part, rec):
    owner = part["geom"].labels.ravel()[case.records["flat"][rec]].astype(np.int64)
    if np.any(owner < 0):
        raise Stop("partition_invalid", int(rec.size), "record voxel without owner")
    return owner


def selection_one_per_cell(case, part, rec):
    """S-A: the declared nearest-centroid rule, exactly one record per cell (observations asserts it)."""
    return observations.nearest_centroid_selection(case, part, rec)


def selection_all(case, part, rec):
    """S-M: every record is an observation. No per-cell reduction; a cell may receive many, or none."""
    rec = np.asarray(rec, dtype=np.int64)
    owner = owners_of(case, part, rec)
    return dict(record=rec, cell=owner, counts=np.bincount(owner, minlength=part["geom"].n_cells))


def obs_stats(part, selection):
    counts = selection.get("counts")
    if counts is None:
        counts = np.bincount(np.asarray(selection["cell"], np.int64), minlength=part["geom"].n_cells)
    return dict(n_obs=int(np.asarray(selection["record"]).size),
                records_per_cell_min=int(counts.min()), records_per_cell_max=int(counts.max()),
                records_per_cell_median=float(np.median(counts)),
                cells_with_zero_records=int(np.sum(counts == 0)))


# ----------------------------------------------------------------------------------- readouts and declared checks
def field_operator(part, R, G):
    return rp.point_operator(part["xyz"], part["geom"].labels.ravel()[part["pore"]], part["trace"], R, G)


def kkt_residual(A, D, b, z, p):
    K = stokes_solve.reduced_kkt(A, D)
    rhs = np.concatenate([np.asarray(b, np.float64), np.zeros(D.shape[0] - 1)])
    r = K @ stokes_solve.state_vector(z, p) - rhs
    return float(np.linalg.norm(r) / max(float(np.linalg.norm(rhs)), 1e-300))


def leave_out_e_ff(case, part, Hfield, z, observed_flats):
    """protocol.readouts.leave_out_check: e_ff with the directly observed pore voxels removed from the error norm."""
    pore = np.asarray(case.pore, np.int64)
    obs = np.unique(np.asarray(observed_flats, np.int64))
    pos = np.searchsorted(pore, obs)
    hit = (pos < pore.size) & (pore[np.minimum(pos, pore.size - 1)] == obs)
    w = np.ones(pore.size, dtype=np.float64)
    w[pos[hit]] = 0.0
    value = (Hfield @ z).reshape(-1, 3)
    return dict(e_ff_leave_out=float(rp.normerr(value, case.u_ref_vox, weight=w)),
                observed_voxels=int(hit.sum()), pore_voxels=int(pore.size),
                observed_voxel_fraction=float(hit.sum()) / float(pore.size))


def naive_eK(case, part, selection):
    """protocol.readouts.naive_baseline: per-cell mean of the observed velocities, then volume-weighted, no solve."""
    geom = part["geom"]
    cell = np.asarray(selection["cell"], np.int64)
    vel = np.asarray(case.records["vel"][np.asarray(selection["record"], np.int64)], np.float64)
    n = int(geom.n_cells)
    cnt = np.bincount(cell, minlength=n).astype(np.float64)
    acc = np.zeros((n, 3), dtype=np.float64)
    for j in range(3):
        acc[:, j] = np.bincount(cell, weights=vel[:, j], minlength=n)
    have = cnt > 0
    mean_u = np.zeros((n, 3), dtype=np.float64)
    mean_u[have] = acc[have] / cnt[have, None]
    vol = np.asarray(geom.volume, np.float64)
    ux = float(np.sum(vol[have] * mean_u[have, 0]) / np.sum(vol[have]))
    K = case.nu * ux / float(case.force[0])
    return dict(naive_K=K, naive_eK_arch_s=float(paper_readouts.signed_percent(K, case.k_ref)),
                cells_used=int(have.sum()), cells_without_record=int((~have).sum()))


def readouts(case, part, system, A_used, b_used, result, Hfield, H=None, y=None, assisted=False):
    z = np.asarray(result["trace_coefficients"], np.float64).ravel()
    p = np.asarray(result["p"], np.float64)
    D = system.divergence_matrix
    row, _, _ = paper_readouts.table_row(case, part, result["phi"], result["U"], z=z, D=D)
    out = dict(table_row=row, e_ff=float(rp.normerr((Hfield @ z).reshape(-1, 3), case.u_ref_vox)),
               kkt_relative_residual_recomputed=kkt_residual(A_used, D, b_used, z, p),
               eta_m=row["eta_m"], r_inf_m=row["r_inf_m"],
               U_equals_Rz_maxabs=float(np.max(np.abs(paper_readouts.cell_velocity(z, system)
                                                      - np.asarray(result["U"])))),
               phi_equals_face_aggregate_maxabs=float(np.max(np.abs(paper_readouts.face_fields(z, part["trace"])[2]
                                                                    - np.asarray(result["phi"])))),
               n_trace_modes=int(part["trace"].n_trace_modes), trace_basis=str(part["trace"].trace_basis))
    if H is not None:
        yy = np.asarray(y, np.float64).ravel()
        out["observation_misfit"] = float(np.linalg.norm(H @ z - yy) / max(float(np.linalg.norm(yy)), 1e-300))
    if assisted:
        rhs = system.rhs_trace
        mom = system.velocity_matrix @ z + D.T @ p - rhs
        out["plain_stokes_momentum_relative_residual"] = float(np.linalg.norm(mom) / np.linalg.norm(rhs))
    return out


# ------------------------------------------------------------------------------------------------ solve one arm
def solve_arm(cfg, case, seeds, rec, *, arm, label="solve", check_identity=True):
    """One arm on the published partition. arm in ('PN', 'AN', 'MN')."""
    if arm not in ARMS:
        raise ValueError(arm)
    assisted = arm != "PN"
    t0 = time.perf_counter()
    part = cell_complex.build(case, seeds, order="paper", basis=cfg["trace_basis"])
    if not part["checks"]["pass_"]:
        raise Stop("partition_invalid", int(seeds.size), repr(part["checks"]))
    system, _ = stokes_solve.assemble(case, part["trace"], "fast", cfg["viscous_form"])
    R, G, gdef = rp.recovery_operators(part["trace"], system)
    Hfield = field_operator(part, R, G)
    selection = H = y = xyz = None
    if assisted:
        selection = selection_one_per_cell(case, part, rec) if arm == "AN" else selection_all(case, part, rec)
        H, y, xyz = observations.observations(case, part, R, G, selection)
    t_setup = time.perf_counter() - t0
    try:
        result, receipt, fac, solved = stokes_solve.solve_entry(case, part, system,
                                                                alpha=float(cfg["alpha"]) if assisted else 0.0,
                                                                selection=selection, R=R, G=G, solver_cfg=cfg["solver"],
                                                                label=label, trace_basis=cfg["trace_basis"],
                                                                check_identity=check_identity)
    except RuntimeError as exc:
        raise Stop("solve_failed", int(seeds.size), str(exc)) from exc
    A_used, b_used = solved.velocity_matrix, solved.body_force_matrix @ solved.body_force
    diag = readouts(case, part, system, A_used, b_used, result, Hfield, H=H, y=y, assisted=assisted)
    del fac
    z = np.asarray(result["trace_coefficients"], np.float64).ravel()
    info = dict(arm=arm, n_cells=int(part["geom"].n_cells), n_sites=int(seeds.size),
                partition_checks=part["checks"], recovery_gradient_divergence_defect=gdef, solver=receipt,
                setup_s=t_setup, n_dofs=int(system.velocity_matrix.shape[0]), A_nnz=int(A_used.nnz), **diag)
    state = dict(z=z, p=np.asarray(result["p"], np.float64), U=np.asarray(result["U"], np.float64),
                 phi=np.asarray(result["phi"], np.float64), seeds=seeds)
    if assisted:
        info.update(gamma=float(receipt["gamma"]), weight=float(receipt["weight"]),
                    mu_times_w=float(receipt["gamma"]) * float(receipt["weight"]), H_shape=list(H.shape),
                    point_operator_defect=float(observations.point_operator_defect(H, xyz, selection["cell"],
                                                                                   part["trace"], R, G)),
                    **obs_stats(part, selection))
        info.update(leave_out_e_ff(case, part, Hfield, z,
                                   case.records["flat"][np.asarray(selection["record"], np.int64)]))
        info.update(naive_eK(case, part, selection))
        state.update(selected_records=np.asarray(selection["record"], np.int64), obs_xyz=xyz,
                     obs_u=np.asarray(y, np.float64))
    return info, state


# --------------------------------------------------------------------------------------------------- checkpoints
def checkpoint_valid(path):
    js = config_io.load_json(path)
    if js.get("status") == "failed":
        return
    if config_io.sha(path[:-5] + ".npz") != js["npz_sha256"]:
        raise RuntimeError("npz hash mismatch")


def load_or_run(path, fn):
    if config_io.done(path, checkpoint_valid):
        js = config_io.load_json(path)
        js["_from_checkpoint"] = True
        return js
    try:
        info, state = fn()
    except Stop as stop:
        js = stamp(dict(status="failed", outcome=stop.outcome, n=stop.n, message=stop.message))
        config_io.save_json(path, js)
        return js
    npz = path[:-5] + ".npz"
    config_io.save_npz(npz, **state)
    js = stamp(dict(info, status="ok", npz_sha256=config_io.sha(npz)))
    config_io.save_json(path, js)
    js["_from_checkpoint"] = False
    return js
