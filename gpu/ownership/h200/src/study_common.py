"""Shared helpers of the GPU studies.

This module never modifies the pipeline code in gpu/code.  It imports the
pipeline entry points and re-uses them:

* ``run_hybrid_voronoi_forward``  - runner argument contract and error metrics
* ``run_segmented_selector_geodesic_rows``  - namespace / cfg / reference loader
* ``hybrid_voronoi_trace``  - trace geometry, assembly, factorization, solve
* ``hybrid_site_sources``  - particle-window loader and seed validation

The only new capability added here is a forward entry point that accepts an
explicit validated ``seed_flat`` array (or an explicit admissible owner-label
partition), which the command-line runner deliberately refuses.
Every numerical operator is the pipeline one.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

FORMULATION_ID = "moment_constrained_hybrid_voronoi_connected_p1_symmetric_gradient_v1"
PROTOCOL_ID = "pvfv_forward_studies_v1"

# Locked forward arguments: the forward formulation of the paper, the same as
# FORWARD_ARGS in reproduce/supplementary/s3_verification/forward_resolution_ducts.py.
FORWARD_ARGS: dict[str, Any] = {
    "label_backend": "exact_frontier_gpu",
    "trace_basis": "connected_p1",
    "viscous_form": "symmetric_gradient",
    "body_force": [0.002, 0.0, 0.0],
    "linear_solver": "minres",
    "linear_rtol": 1.0e-14,
    "linear_maxiter": 50000,
    "linear_refinement_steps": 1,
    "repeats": 1,
}

REQUIRED_CODE_FILES = (
    "run_hybrid_voronoi_forward.py",
    "hybrid_voronoi_trace.py",
    "hybrid_site_sources.py",
    "run_segmented_selector_geodesic_rows.py",
    "geodesic_face_operator.py",
    "flow_runner.py",
    "roi_jfa_backend.py",
)


def compute_root() -> Path:
    for name in ("PVFV_GPU_ROOT", "POREVORONOI_COMPUTE_ROOT", "PVFV_COMPUTE_ROOT"):
        value = os.environ.get(name)
        if value:
            return Path(value).resolve()
    return Path(__file__).resolve().parents[1]


COMPUTE_ROOT = compute_root()
CODE_DIR = COMPUTE_ROOT / "code"


def delivery_root() -> Path:
    value = os.environ.get("PVFV_OUTPUT_ROOT")
    if value:
        return Path(value).resolve()
    return COMPUTE_ROOT / "../outputs/forward_studies"


DELIVERY_ROOT = delivery_root()


def protocol_root() -> Path:
    value = os.environ.get("PVFV_PROTOCOL_ROOT")
    if value:
        return Path(value).resolve()
    return Path("outputs/protocol")


PROTOCOL_ROOT = protocol_root()


# --------------------------------------------------------------------------
# hashing / io helpers
# --------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return target


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> Path:
    import csv as _csv

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if columns is None:
            raise ValueError("Cannot write an empty CSV without an explicit column list")
        with target.open("w", newline="", encoding="utf-8") as handle:
            _csv.DictWriter(handle, fieldnames=columns).writeheader()
        return target
    fieldnames = columns if columns is not None else list(rows[0])
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return target


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
def _package_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "cpu_logical_cores": os.cpu_count(),
        "packages": {
            name: _package_version(name)
            for name in (
                "numpy",
                "scipy",
                "pandas",
                "matplotlib",
                "cupy",
                "numba",
                "sksparse",
                "networkx",
            )
        },
        "compute_root": str(COMPUTE_ROOT),
        "delivery_root": str(DELIVERY_ROOT),
        "protocol_root": str(PROTOCOL_ROOT),
    }
    try:
        import cupy as cp

        record["cuda"] = {
            "cupy_version": cp.__version__,
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "device_count": int(cp.cuda.runtime.getDeviceCount()),
            "devices": [
                {
                    "index": index,
                    "name": cp.cuda.runtime.getDeviceProperties(index)["name"].decode("utf-8"),
                    "total_global_mem_bytes": int(
                        cp.cuda.runtime.getDeviceProperties(index)["totalGlobalMem"]
                    ),
                }
                for index in range(int(cp.cuda.runtime.getDeviceCount()))
            ],
        }
    except Exception as error:  # pragma: no cover - reported honestly
        record["cuda"] = {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}
    try:
        total_ram = int(
            subprocess.run(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.split()[-1]
        )
        record["total_ram_bytes"] = total_ram
    except Exception:
        record["total_ram_bytes"] = None
    record["git"] = _git_state(COMPUTE_ROOT)
    record["code_sha256"] = {
        name: (sha256_file(CODE_DIR / name) if (CODE_DIR / name).is_file() else None)
        for name in REQUIRED_CODE_FILES
    }
    return record


def _git_state(root: Path) -> dict[str, Any]:
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if top.returncode != 0:
            return {"is_git_repository": False, "reason": top.stderr.strip()[:400]}
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True, timeout=60
        ).stdout
        return {
            "is_git_repository": True,
            "toplevel": top.stdout.strip(),
            "commit": commit,
            "branch": branch,
            "dirty": bool(status.strip()),
            "status_lines": len([line for line in status.splitlines() if line.strip()]),
        }
    except Exception as error:
        return {"is_git_repository": False, "reason": f"{type(error).__name__}: {error}"}


# --------------------------------------------------------------------------
# production namespace bootstrap (loaded once per process)
# --------------------------------------------------------------------------
_BOOTSTRAP: dict[str, Any] | None = None


def bootstrap(out_dir: str | Path | None = None) -> dict[str, Any]:
    """Load the production runner namespace exactly as the paper runner does."""
    global _BOOTSTRAP
    if _BOOTSTRAP is not None:
        return _BOOTSTRAP
    if str(CODE_DIR) in sys.path:
        sys.path.remove(str(CODE_DIR))
    sys.path.insert(0, str(CODE_DIR))
    import run_hybrid_voronoi_forward as forward  # noqa: E402
    from hybrid_voronoi_trace import (  # noqa: E402
        assemble_moment_constrained_hybrid_stokes,
        build_hybrid_trace_factorization,
        build_hybrid_trace_geometry,
        solve_moment_constrained_hybrid_stokes,
    )

    scratch = Path(out_dir) if out_dir is not None else (DELIVERY_ROOT / "runs" / "_scratch")
    scratch.mkdir(parents=True, exist_ok=True)

    forward.segmented.configure_roi_environment()
    os.environ["PVFV_LABEL_BACKEND"] = str(FORWARD_ARGS["label_backend"])
    runner = forward.segmented.load_runner()
    runner_args = forward._runner_args(types.SimpleNamespace())
    ns, _fixed, _global = forward.segmented.install_namespace(runner, runner_args)
    cfg = forward.segmented.make_cfg(runner, ns, scratch, runner_args)
    cfg.body_force = tuple(float(value) for value in FORWARD_ARGS["body_force"])

    _BOOTSTRAP = {
        "forward": forward,
        "runner": runner,
        "runner_args": runner_args,
        "ns": ns,
        "cfg": cfg,
        "cp": ns["cp"],
        "build_hybrid_trace_geometry": build_hybrid_trace_geometry,
        "assemble_moment_constrained_hybrid_stokes": assemble_moment_constrained_hybrid_stokes,
        "build_hybrid_trace_factorization": build_hybrid_trace_factorization,
        "solve_moment_constrained_hybrid_stokes": solve_moment_constrained_hybrid_stokes,
    }
    return _BOOTSTRAP


def free_gpu() -> None:
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# --------------------------------------------------------------------------
# case registry
# --------------------------------------------------------------------------
CASES: dict[str, dict[str, Any]] = {
    "orthogonal_duct": {
        "display": "Orthogonal duct",
        "mask": "../data/controlled_cases/orthogonal_duct/reference_flow.npz",
        "reference": "../data/controlled_cases/orthogonal_duct/reference_flow.npz",
        "particle_window": (
            "../data/controlled_cases/orthogonal_duct/particle_tracks.csv.gz"
        ),
    },
    "skewed_duct": {
        "display": "Skewed duct",
        "mask": "../data/controlled_cases/skewed_duct/reference_flow.npz",
        "reference": "../data/controlled_cases/skewed_duct/reference_flow.npz",
        "particle_window": (
            "../data/controlled_cases/skewed_duct/particle_tracks.csv.gz"
        ),
    },
    "bentheimer_crop": {
        "display": "Bentheimer crop",
        "mask": "../data/controlled_cases/bentheimer_crop/reference_flow.npz",
        "reference": "../data/controlled_cases/bentheimer_crop/reference_flow.npz",
        "particle_window": (
            "../data/controlled_cases/bentheimer_crop/particle_tracks.csv.gz"
        ),
    },
}


def case_paths(case: str) -> dict[str, Path]:
    spec = CASES[case]
    return {
        "mask": COMPUTE_ROOT / spec["mask"],
        "reference": COMPUTE_ROOT / spec["reference"],
        "particle_window": COMPUTE_ROOT / spec["particle_window"],
    }


def load_mask(case_or_path: str | Path) -> np.ndarray:
    """Load a boolean pore mask through the production loader."""
    boot = bootstrap()
    ns = boot["ns"]
    runner = boot["runner"]
    cp = boot["cp"]
    path = Path(case_or_path) if not isinstance(case_or_path, str) or os.sep in str(case_or_path) or "/" in str(case_or_path) else case_paths(str(case_or_path))["mask"]
    mask_gpu = runner.load_mask_npz(ns, Path(path))
    return cp.asnumpy(mask_gpu).astype(bool, copy=False)


def full_trajectory_seed_flat(case: str) -> tuple[np.ndarray, dict[str, Any]]:
    """The frozen zero-snap candidate pool: unique pore voxels visited by the window."""
    boot = bootstrap()
    sys.path.insert(0, str(CODE_DIR))
    from hybrid_site_sources import load_particle_window_seed_flat  # noqa: E402

    paths = case_paths(case)
    mask = load_mask(paths["mask"])
    seed_flat, metadata = load_particle_window_seed_flat(
        mask,
        paths["particle_window"],
        frame_selection="",
        particle_id_selection="",
        max_snap_displacement_vox=0.0,
    )
    return np.asarray(seed_flat, dtype=np.int64), dict(metadata)


# --------------------------------------------------------------------------
# forward evaluation from an explicit site set or explicit partition
# --------------------------------------------------------------------------
def build_geometry_from_seed_flat(
    seed_flat: np.ndarray,
    *,
    mask_path: Path,
    seed_spec: str,
) -> tuple[Any, dict[str, Any], float]:
    """Production ownership build; identical call to run_hybrid_voronoi_forward._build_geometry."""
    boot = bootstrap()
    ns, cfg, cp = boot["ns"], boot["cfg"], boot["cp"]
    runner = boot["runner"]
    sys.path.insert(0, str(CODE_DIR))
    from hybrid_site_sources import validate_seed_flat  # noqa: E402

    mask_gpu = runner.load_mask_npz(ns, Path(mask_path))
    mask_np = cp.asnumpy(mask_gpu).astype(bool, copy=False)
    validated = validate_seed_flat(mask_np, np.asarray(seed_flat, dtype=np.int64))
    seed_gpu = cp.asarray(validated, dtype=cp.int64)
    start = time.perf_counter()
    geom, meta = ns["pvfv_build_geometry_from_seed_flat_timed"](
        mask_gpu,
        seed_gpu,
        cfg,
        split_face_components=True,
        label_mode=str(FORWARD_ARGS["label_backend"]),
        seed_spec=seed_spec,
    )
    build_time = float(time.perf_counter() - start)
    return geom, dict(meta or {}), build_time


def build_geometry_from_partition(
    partition_labels: np.ndarray,
    *,
    mask_path: Path,
    seed_spec: str,
) -> tuple[Any, dict[str, Any], float, np.ndarray]:
    """Production geometry from an explicit admissible owner-label partition.

    The partition is first passed through the production six-connected
    component split and reindex (``pvfv_face_connected_reindex_cpu``), so the
    resulting control volumes obey exactly the production connectivity
    convention, then handed to the production moment/face builder
    ``pvfv_build_geometry_from_labels_ncells_gpu`` - the same builder that the
    site-based path calls.

    ``geom.dist`` carries no weight in the forward matrix.
    For a non-site partition it is defined as the within-cell six-neighbour
    graph distance to the cell's minimum-flattened-index voxel, which is
    deterministic and is reported as a descriptor only.
    """
    boot = bootstrap()
    ns, cfg, cp = boot["ns"], boot["cfg"], boot["cp"]
    runner = boot["runner"]
    mask_gpu = runner.load_mask_npz(ns, Path(mask_path))
    mask = cp.asnumpy(mask_gpu).astype(bool, copy=False)
    labels = np.asarray(partition_labels, dtype=np.int32).reshape(mask.shape)
    if np.any(labels[mask] < 0):
        raise ValueError("Every pore voxel must be assigned exactly once")
    if np.any(labels[~mask] >= 0):
        raise ValueError("A solid voxel carries a partition label")

    start = time.perf_counter()
    reindexed_gpu, split_info = ns["pvfv_face_connected_reindex_cpu"](
        mask_gpu, cp.asarray(labels, dtype=cp.int32)
    )
    reindexed = cp.asnumpy(reindexed_gpu).astype(np.int32)
    n_cells = int(split_info["n_cv"])
    distance = within_cell_reference_distance(mask, reindexed, n_cells)
    geom = ns["pvfv_build_geometry_from_labels_ncells_gpu"](
        mask_gpu,
        cp.asarray(reindexed, dtype=cp.int32),
        cp.asarray(distance, dtype=cp.float64),
        n_cells,
        cfg,
    )
    build_time = float(time.perf_counter() - start)
    meta = {
        "seed_spec": seed_spec,
        "S": int(np.unique(labels[mask]).size),
        "N_cv": int(geom.n_cells),
        "N_fl": int(np.count_nonzero(mask)),
        "N_split": int(split_info["n_split_extra"]),
        "N_face_disconnected_labels": int(split_info["n_face_disconnected_labels"]),
        "face_connected_fraction": float(split_info["face_connected_fraction"]),
        "label_mode": "explicit_partition_with_production_six_connected_reindex",
        "distance_connectivity": int(cfg.distance_connectivity),
        "dist_field_definition": (
            "within-cell six-neighbour graph distance to the minimum flattened "
            "voxel index of the cell; inactive in the forward matrix"
        ),
    }
    return geom, meta, build_time, reindexed


def within_cell_reference_distance(
    mask: np.ndarray, labels: np.ndarray, n_cells: int
) -> np.ndarray:
    """Deterministic in-cell graph distance field for a non-site partition."""
    pore, neighbours = pore_neighbour_table(mask, periodic_x=True)
    node_label = labels.reshape(-1)[pore].astype(np.int64)
    order = np.argsort(node_label * (pore.size + 1) + np.arange(pore.size), kind="stable")
    first_node = np.zeros(n_cells, dtype=np.int64)
    seen = np.zeros(n_cells, dtype=bool)
    sorted_labels = node_label[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    for boundary in boundaries:
        cell = int(sorted_labels[boundary])
        if cell >= 0 and not seen[cell]:
            seen[cell] = True
            first_node[cell] = int(order[boundary])
    distance = np.full(pore.size, _INF, dtype=np.int32)
    # Restrict the expansion to within-cell edges by masking cross-cell links.
    same_cell = np.where(
        neighbours >= 0, node_label[np.clip(neighbours, 0, None)] == node_label[:, None], False
    )
    restricted = np.where(same_cell, neighbours, -1)
    multi_source_bfs(restricted, first_node[seen], distance=distance)
    field = np.zeros(mask.shape, dtype=np.float64).reshape(-1)
    field[:] = 0.0
    field[pore] = distance.astype(np.float64)
    unreachable = field[pore] >= float(_INF)
    if np.any(unreachable):
        raise RuntimeError("A partition cell is not six-connected after the production reindex")
    return field.reshape(mask.shape)


def solve_forward(
    geom: Any,
    *,
    reference_path: Path,
    repeats: int = 1,
    body_force: list[float] | None = None,
    keep_system: bool = False,
) -> dict[str, Any]:
    """Trace geometry + assembly + factorization + solve + reference errors.

    Every numerical step is the retained production function.
    """
    boot = bootstrap()
    ns, cfg, cp = boot["ns"], boot["cfg"], boot["cp"]
    forward = boot["forward"]
    force = np.asarray(body_force if body_force is not None else FORWARD_ARGS["body_force"], dtype=np.float64)

    trace_start = time.perf_counter()
    trace = boot["build_hybrid_trace_geometry"](ns, geom, cfg, trace_basis=str(FORWARD_ARGS["trace_basis"]))
    trace_time = float(time.perf_counter() - trace_start)

    system = boot["assemble_moment_constrained_hybrid_stokes"](
        trace,
        viscosity=float(cfg.nu),
        body_force=force,
        viscous_form=str(FORWARD_ARGS["viscous_form"]),
    )
    factorization = boot["build_hybrid_trace_factorization"](
        system,
        solver=str(FORWARD_ARGS["linear_solver"]),
        iterative_rtol=float(FORWARD_ARGS["linear_rtol"]),
        iterative_maxiter=int(FORWARD_ARGS["linear_maxiter"]),
        iterative_refinement_steps=int(FORWARD_ARGS["linear_refinement_steps"]),
        velocity_lu_ordering="COLAMD",
    )
    results = [
        boot["solve_moment_constrained_hybrid_stokes"](system, factorization=factorization)
        for _ in range(max(int(repeats), 1))
    ]
    result = results[-1]
    cached_times = [float(item["cached_call_time_s"]) for item in results]

    reference_geom, reference_result = boot["forward"].segmented.load_reference_npz(
        ns, Path(reference_path), cfg
    )
    if reference_geom is None or reference_result is None:
        raise FileNotFoundError(reference_path)
    U_ref_gpu, p_ref_gpu = ns["coarsen_voxel_reference_to_coarse_gpu"](
        reference_geom, reference_result, geom
    )
    U_ref = cp.asnumpy(U_ref_gpu).astype(np.float64)
    p_ref = cp.asnumpy(p_ref_gpu).astype(np.float64)
    volume = cp.asnumpy(geom.volume).astype(np.float64)
    dvec = cp.asnumpy(geom.dvec).astype(np.float64)

    velocity_metrics = forward._velocity_errors(np.asarray(result["U"]), U_ref, volume)
    e_phi = 100.0 * float(
        ns["pvfv_flux_rel_error"](
            reference_geom,
            reference_result,
            geom,
            {"phi": cp.asarray(result["phi"]), "K_eff_x": 0.0},
        )
    )
    e_p, pressure_detail = gauge_invariant_pressure_error(
        np.asarray(result["p"], dtype=np.float64), p_ref, volume
    )
    mean_flux = np.sum(np.asarray(result["phi"])[:, None] * dvec, axis=0) / np.sum(volume)
    mean_velocity = np.sum(volume[:, None] * np.asarray(result["U"]), axis=0) / np.sum(volume)
    force_x = float(force[0])
    if force_x == 0.0:
        K_eff = float("nan")
        e_K = float("nan")
    else:
        K_eff = float(cfg.nu) * float(mean_flux[0]) / force_x
        K_ref = float(reference_result["K_eff_x"])
        e_K = 100.0 * abs(K_eff - K_ref) / abs(K_ref)

    payload: dict[str, Any] = {
        "trace": trace,
        "result": result,
        "U_ref": U_ref,
        "p_ref": p_ref,
        "volume": volume,
        "t_trace_geometry_s": trace_time,
        "t_matrix_assembly_s": float(system.assembly_time_s),
        "t_factorization_s": float(factorization.factorization_time_s),
        "t_cached_call_median_s": float(statistics.median(cached_times)),
        "cached_call_times_s": cached_times,
        "K_eff_parallel": K_eff,
        "K_ref_parallel": float(reference_result["K_eff_x"]),
        "e_K_percent": e_K,
        "e_phi_percent": e_phi,
        "e_p_percent": e_p,
        "pressure_error_detail": pressure_detail,
        "mean_velocity": mean_velocity,
        "mean_flux": mean_flux,
        "reference_metadata": _reference_metadata(Path(reference_path)),
        **velocity_metrics,
    }
    if keep_system:
        payload["system"] = system
        payload["factorization"] = factorization
    else:
        del system, factorization
    return payload


def gauge_invariant_pressure_error(
    p_computed: np.ndarray, p_reference: np.ndarray, volume: np.ndarray
) -> tuple[float, dict[str, Any]]:
    """e_p_percent: volume-weighted, gauge-invariant.

    ``p_computed`` is the pressure unknown returned by the retained solver.  That
    unknown is the Lagrange multiplier of the discrete constraint ``D u = 0`` in
    the KKT convention actually assembled by the production code,

        [ A    D^T ] [u]   [f]
        [ D    0   ] [p] = [0]      (hybrid_voronoi_trace.py, solve path)

    where ``D`` is the *outward* discrete divergence, ``D u = sum_f A_f (u_f.n_f)``
    with ``n_f`` the outward facelet normal (hybrid_voronoi_trace.py, the local
    assembly line ``divergence[block] += area * value * normal``).  The physical
    Stokes momentum equation ``-mu.Lap(u) + grad(p) = f`` has the weak form
    ``a(u,v) - (p, div v) = (f,v)``, i.e. ``A u - D^T p_physical = f``.  Comparing
    that with the assembled row ``A u + D^T p = f`` gives

        p_physical = -p_computed .

    The metric differences the computed pressure against an *external* voxel
    reference pressure, which is stored in the physical convention, so the
    multiplier must be mapped to the physical convention before the difference is
    taken.  Confirmed numerically on every non-degenerate run in this study: the
    volume-weighted cosine between the centred multiplier and the centred
    reference pressure is -0.980 to -1.0000, while the velocity cosine between the
    same two solvers is +0.969 to +0.9998.

    Both values are returned in ``detail``: the reconciled one, which is the
    metric, and the raw multiplier-convention one, which is retained so that the
    correction is auditable and nothing is hidden.
    """
    total_volume = float(np.sum(volume))
    multiplier = np.asarray(p_computed, dtype=np.float64)
    p_r0 = np.asarray(p_reference, dtype=np.float64)
    p_r0 = p_r0 - float(np.sum(volume * p_r0) / max(total_volume, 1.0e-300))

    def _centred(field: np.ndarray) -> np.ndarray:
        return field - float(np.sum(volume * field) / max(total_volume, 1.0e-300))

    def _norm(field: np.ndarray) -> float:
        return float(np.sqrt(np.sum(volume * field**2)))

    multiplier0 = _centred(multiplier)
    p_h0 = -multiplier0  # multiplier -> physical pressure convention
    denominator = max(_norm(p_r0), 1.0e-300)
    numerator = _norm(p_h0 - p_r0)
    numerator_multiplier_convention = _norm(multiplier0 - p_r0)
    cosine = float(np.sum(volume * multiplier0 * p_r0)) / max(
        _norm(multiplier0) * _norm(p_r0), 1.0e-300
    )
    detail = {
        "aggregation": "coarsen_voxel_reference_to_coarse_gpu volume mean over the coarse labels",
        "gauge_operation": "volume-weighted mean removed from both computed and reference cell pressure",
        "norm": "M_p = diag(cell volume)",
        "sign_convention": (
            "the solver pressure unknown is the constraint multiplier of the assembled "
            "[[A, D^T], [D, 0]] system with D the outward divergence, hence "
            "p_physical = -p_solver; the multiplier is mapped to the physical convention "
            "before differencing (protocol amendment 7)"
        ),
        "sign_reconciliation_applied": True,
        "multiplier_vs_reference_cosine": cosine,
        "numerator_Mp_norm": numerator,
        "numerator_Mp_norm_multiplier_convention": numerator_multiplier_convention,
        "denominator_Mp_norm": denominator,
        "e_p_percent_multiplier_convention": 100.0 * numerator_multiplier_convention / denominator,
    }
    return 100.0 * numerator / denominator, detail


def _reference_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        if "metadata" not in data.files:
            return {}
        raw = data["metadata"].item()
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    return {}


# --------------------------------------------------------------------------
# geometry / topology descriptors
# --------------------------------------------------------------------------
def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    """Deterministic weighted quantile, identical to the one of forward_resolution_ducts.py."""
    order = np.argsort(values, kind="stable")
    ordered_values = np.asarray(values)[order]
    ordered_weights = np.asarray(weights)[order]
    cumulative = np.cumsum(ordered_weights)
    target = probability * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def detect_ownership_graph_convention(
    mask: np.ndarray, distance: np.ndarray, site_flat: np.ndarray
) -> dict[str, Any]:
    """Measure which six-neighbour boundary convention reproduces ``geom.dist``.

    ``cfg.periodic_x`` is True for the physics, but the ownership frontier is a
    separate object.  The convention is measured, never assumed.
    """
    findings: dict[str, Any] = {}
    for periodic in (False, True):
        pore, neighbours = pore_neighbour_table(mask, periodic_x=periodic)
        lookup = np.full(mask.size, -1, dtype=np.int64)
        lookup[pore] = np.arange(pore.size, dtype=np.int64)
        nodes = lookup[np.unique(np.asarray(site_flat, dtype=np.int64))]
        computed, _labels = multi_source_bfs(neighbours, nodes)
        production = distance.reshape(-1)[pore]
        findings[periodic] = {
            "identical": bool(np.array_equal(computed.astype(np.float64), production)),
            "max_absolute_difference": float(
                np.max(np.abs(computed.astype(np.float64) - production))
            ),
        }
    matched = [key for key, value in findings.items() if value["identical"]]
    return {
        "matched_periodic_x": matched[0] if matched else None,
        "reproduced": bool(matched),
        "findings": {str(key): value for key, value in findings.items()},
    }


def geometry_descriptors(
    geom: Any,
    trace: Any,
    *,
    site_flat: np.ndarray | None,
    graph_periodic_x: bool | None = None,
) -> dict[str, Any]:
    """Every geometry/topology metric required by controlled_refinement.csv."""
    boot = bootstrap()
    cp = boot["cp"]
    labels = cp.asnumpy(geom.labels).astype(np.int64, copy=False)
    distance = cp.asnumpy(geom.dist).astype(np.float64, copy=False)
    mask = cp.asnumpy(geom.mask).astype(bool, copy=False)
    valid = labels >= 0
    valid_labels = labels[valid]
    valid_distance = distance[valid]
    n_cells = int(geom.n_cells)

    radii = np.zeros(n_cells, dtype=np.float64)
    np.maximum.at(radii, valid_labels, valid_distance)

    volume = cp.asnumpy(geom.volume).astype(np.float64, copy=False)
    positive = volume[volume > 0.0]

    parent = np.asarray(trace.face_parent_edge, dtype=np.int64)
    normals = np.asarray(trace.face_normal, dtype=np.float64)
    n_edges = int(geom.owner.size)
    facelet_count = np.bincount(parent, minlength=n_edges).astype(np.float64)
    vector_sum = np.zeros((n_edges, 3), dtype=np.float64)
    np.add.at(vector_sum, parent, normals)
    edge_valid = facelet_count > 0
    chi = np.linalg.norm(vector_sum[edge_valid], axis=1) / facelet_count[edge_valid]
    chi_weight = facelet_count[edge_valid]

    patch = np.asarray(trace.face_patch, dtype=np.int64)
    edge_patch_pairs = np.unique(np.stack([parent, patch], axis=1), axis=0)
    patches_per_edge = np.bincount(edge_patch_pairs[:, 0], minlength=n_edges)
    fragmented = int(np.sum(patches_per_edge > 1))
    represented = int(np.sum(patches_per_edge > 0))

    tensor = np.einsum("fi,fj->ij", normals, normals) / max(float(normals.shape[0]), 1.0e-300)
    eigenvalues = np.sort(np.linalg.eigvalsh(tensor))[::-1]
    anisotropy = float(
        (eigenvalues[0] - eigenvalues[2]) / max(float(eigenvalues[0]), 1.0e-300)
    )

    h_S = float(np.max(valid_distance))
    convention: dict[str, Any] = {"matched_periodic_x": graph_periodic_x, "reproduced": None, "findings": {}}
    if site_flat is not None and graph_periodic_x is None:
        convention = detect_ownership_graph_convention(mask, distance, site_flat)
        graph_periodic_x = bool(convention["matched_periodic_x"]) if convention["reproduced"] else False
    if site_flat is not None:
        q_S = separation_distance(mask, site_flat, periodic_x=bool(graph_periodic_x))
    else:
        q_S = float("nan")

    return {
        "graph_metric_periodic_x": bool(graph_periodic_x) if graph_periodic_x is not None else None,
        "ownership_distance_reproduced_by_bfs": convention["reproduced"],
        "cfg_periodic_x": bool(trace.periodic_x),
        "N_f": int(np.count_nonzero(mask)),
        "N_cv": n_cells,
        "N_edges": n_edges,
        "N_interface_facelets": int(trace.n_facelets),
        "N_connected_patches": int(trace.n_patches),
        "N_trace_modes": int(trace.n_trace_modes),
        "N_trace_vector_dofs": 3 * int(trace.n_trace_modes),
        "N_pressure_dofs_after_gauge": n_cells - 1,
        "N_system": 3 * int(trace.n_trace_modes) + n_cells - 1,
        "h_S_over_h": h_S,
        "q_S_over_h": q_S,
        "mesh_ratio_hS_over_qS": float(h_S / q_S) if q_S and np.isfinite(q_S) and q_S > 0 else float("nan"),
        "cell_graph_radius_mean": float(np.mean(radii)),
        "cell_graph_radius_p95": float(np.quantile(radii, 0.95)),
        "cell_graph_radius_max": float(np.max(radii)),
        "cell_volume_mean": float(np.mean(positive)),
        "cell_volume_cv": float(np.std(positive) / max(float(np.mean(positive)), 1.0e-300)),
        "cell_volume_min": float(np.min(positive)),
        "cell_volume_max": float(np.max(positive)),
        "patches_per_edge_mean": float(int(trace.n_patches) / max(n_edges, 1)),
        "fragmented_edge_fraction": float(fragmented / max(represented, 1)),
        "area_weighted_chi_p05": weighted_quantile(chi, chi_weight, 0.05),
        "area_weighted_chi_median": weighted_quantile(chi, chi_weight, 0.50),
        "chi_min": float(np.min(chi)),
        "orientation_anisotropy": anisotropy,
        "n_cells_with_zero_volume": int(np.count_nonzero(volume <= 0.0)),
    }


# --------------------------------------------------------------------------
# six-neighbour pore graph utilities
# --------------------------------------------------------------------------
_NEIGHBOUR_OFFSETS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
_INF = np.int32(np.iinfo(np.int32).max // 4)


def pore_neighbour_table(mask: np.ndarray, *, periodic_x: bool) -> tuple[np.ndarray, np.ndarray]:
    """CSR-like six-neighbour adjacency over pore voxels, in flattened index space.

    ``mask`` is indexed (z, y, x).  ``periodic_x`` wraps the LAST axis, matching
    the production ``periodic_x`` / ``domain_length_x`` convention used by
    ``_minimum_image_x`` in ``hybrid_voronoi_trace``.
    """
    shape = mask.shape
    index = np.full(shape, -1, dtype=np.int64)
    pore = np.flatnonzero(mask.reshape(-1))
    index.reshape(-1)[pore] = np.arange(pore.size, dtype=np.int64)
    coords = np.stack(np.unravel_index(pore, shape), axis=1).astype(np.int64)

    neighbour_lists = []
    for offset in _NEIGHBOUR_OFFSETS:
        shifted = coords + np.asarray(offset, dtype=np.int64)
        ok = np.ones(shifted.shape[0], dtype=bool)
        for axis in range(3):
            if axis == 2 and periodic_x:
                shifted[:, axis] = np.mod(shifted[:, axis], shape[axis])
            else:
                ok &= (shifted[:, axis] >= 0) & (shifted[:, axis] < shape[axis])
        clipped = np.clip(shifted, 0, np.asarray(shape, dtype=np.int64) - 1)
        candidate = index[clipped[:, 0], clipped[:, 1], clipped[:, 2]]
        ok &= candidate >= 0
        neighbour = np.where(ok, candidate, -1)
        neighbour_lists.append(neighbour)
    neighbours = np.stack(neighbour_lists, axis=1)
    return pore, neighbours


def multi_source_bfs(
    neighbours: np.ndarray,
    sources: np.ndarray,
    *,
    distance: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    source_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Unit-weight multi-source BFS on the compact pore-node graph."""
    n = neighbours.shape[0]
    if distance is None:
        distance = np.full(n, _INF, dtype=np.int32)
    if labels is None and source_labels is not None:
        labels = np.full(n, -1, dtype=np.int64)
    frontier = np.asarray(sources, dtype=np.int64)
    if source_labels is not None:
        frontier_labels = np.asarray(source_labels, dtype=np.int64)
    else:
        frontier_labels = None
    improve = distance[frontier] > 0
    frontier = frontier[improve]
    if frontier_labels is not None:
        frontier_labels = frontier_labels[improve]
    distance[frontier] = 0
    if labels is not None and frontier_labels is not None:
        labels[frontier] = frontier_labels
    level = 0
    while frontier.size:
        level += 1
        candidate = neighbours[frontier].reshape(-1)
        if frontier_labels is not None:
            candidate_labels = np.repeat(frontier_labels, neighbours.shape[1])
        valid = candidate >= 0
        candidate = candidate[valid]
        if frontier_labels is not None:
            candidate_labels = candidate_labels[valid]
        better = distance[candidate] > level
        candidate = candidate[better]
        if frontier_labels is not None:
            candidate_labels = candidate_labels[better]
        if candidate.size == 0:
            break
        order = np.argsort(candidate, kind="stable")
        candidate = candidate[order]
        if frontier_labels is not None:
            candidate_labels = candidate_labels[order]
        unique_mask = np.ones(candidate.size, dtype=bool)
        unique_mask[1:] = candidate[1:] != candidate[:-1]
        candidate = candidate[unique_mask]
        if frontier_labels is not None:
            candidate_labels = candidate_labels[unique_mask]
        distance[candidate] = level
        if labels is not None and frontier_labels is not None:
            labels[candidate] = candidate_labels
        frontier = candidate
        frontier_labels = candidate_labels if frontier_labels is not None else None
    return distance, labels


def separation_distance(mask: np.ndarray, site_flat: np.ndarray, *, periodic_x: bool) -> float:
    """q_S/h: half the minimum six-neighbour pore-graph distance between two sites.

    The minimum over adjacent, differently-owned pore-voxel pairs of
    ``d(u) + 1 + d(v)`` equals the minimum pairwise site distance; both
    inequalities are elementary for unit-weight graphs.
    """
    pore, neighbours = pore_neighbour_table(mask, periodic_x=periodic_x)
    lookup = np.full(mask.size, -1, dtype=np.int64)
    lookup[pore] = np.arange(pore.size, dtype=np.int64)
    sites = np.unique(np.asarray(site_flat, dtype=np.int64))
    nodes = lookup[sites]
    if np.any(nodes < 0):
        raise ValueError("A prescribed site is not a pore voxel")
    if nodes.size < 2:
        return float("inf")
    distance, labels = multi_source_bfs(
        neighbours, nodes, source_labels=np.arange(nodes.size, dtype=np.int64)
    )
    best = np.iinfo(np.int64).max
    for column in range(neighbours.shape[1]):
        other = neighbours[:, column]
        ok = other >= 0
        left = np.flatnonzero(ok)
        right = other[ok]
        different = labels[left] != labels[right]
        if not np.any(different):
            continue
        total = distance[left[different]].astype(np.int64) + 1 + distance[right[different]].astype(np.int64)
        best = min(best, int(np.min(total)))
    if best == np.iinfo(np.int64).max:
        return float("inf")
    return 0.5 * float(best)


def graph_farthest_point_order(
    mask: np.ndarray,
    candidate_flat: np.ndarray,
    *,
    periodic_x: bool,
    count: int,
    progress: Any = None,
) -> np.ndarray:
    """Deterministic nested graph-farthest-point ordering on a frozen candidate pool.

    Initial candidate: minimum Euclidean distance to the pore-domain centroid,
    ties broken by the smallest flattened voxel index.
    Subsequent candidate: maximum six-neighbour pore-graph distance to the
    already-selected set, ties broken by the smallest flattened voxel index.
    """
    pore, neighbours = pore_neighbour_table(mask, periodic_x=periodic_x)
    lookup = np.full(mask.size, -1, dtype=np.int64)
    lookup[pore] = np.arange(pore.size, dtype=np.int64)
    candidates = np.unique(np.asarray(candidate_flat, dtype=np.int64))
    candidate_nodes = lookup[candidates]
    if np.any(candidate_nodes < 0):
        raise ValueError("A candidate site is not a pore voxel")

    coords = np.stack(np.unravel_index(pore, mask.shape), axis=1).astype(np.float64)
    centroid = coords.mean(axis=0)
    candidate_coords = coords[candidate_nodes]
    squared = np.sum((candidate_coords - centroid[None, :]) ** 2, axis=1)
    first = int(np.argmin(squared))  # candidates ascending => smallest flat index wins ties

    order = np.empty(int(count), dtype=np.int64)
    order[0] = candidates[first]
    distance = np.full(pore.size, _INF, dtype=np.int32)
    multi_source_bfs(neighbours, np.asarray([candidate_nodes[first]], dtype=np.int64), distance=distance)
    selected = np.zeros(candidates.size, dtype=bool)
    selected[first] = True

    for step in range(1, int(count)):
        candidate_distance = distance[candidate_nodes].astype(np.int64)
        candidate_distance[selected] = -1
        pick = int(np.argmax(candidate_distance))
        if candidate_distance[pick] < 0:
            raise RuntimeError("Exhausted the candidate pool before reaching the requested count")
        selected[pick] = True
        order[step] = candidates[pick]
        multi_source_bfs(neighbours, np.asarray([candidate_nodes[pick]], dtype=np.int64), distance=distance)
        if progress is not None and step % max(int(count) // 20, 1) == 0:
            progress(step, int(count))
    return order


__all__ = [name for name in dir() if not name.startswith("_")]
