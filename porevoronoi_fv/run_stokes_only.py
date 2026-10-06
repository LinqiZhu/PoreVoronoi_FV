"""Cluster stage runner for one case: G1 (fast vs production assembly) and E0 (pure solve, published partition).

Requeue-safe: each stage writes its outputs atomically and is skipped when its checkpoint validates. No assisted
solve is run here.
  python -B run_stokes_only.py --case c1 --stage g1
  python -B run_stokes_only.py --case c1 --stage e0
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import argparse
import os
import time

import config_io
config_io.thread_env(1)
import numpy as np

import case_loader
import cell_complex
import stokes_solve


def build_published(cfg, code):
    case = case_loader.load_case(cfg, code, with_state=False)
    t = time.perf_counter()
    rec = case_loader.select_records(case, "pub")
    seeds = cell_complex.sites_of(case, rec)
    site_s = time.perf_counter() - t
    part = cell_complex.build(case, seeds, order="paper", basis=cfg["trace_basis"])
    man = case.manifest["row"]
    counts = dict(N_cv=(part["geom"].n_cells, man["N_cv"]), N_edges=(part["geom"].owner.size, man["N_edges"]),
                  N_trace_modes=(part["trace"].n_trace_modes, man["N_trace_modes"]),
                  N_connected_patches=(part["trace"].n_patches, man["N_connected_patches"]),
                  N_interface_facelets=(part["trace"].n_facelets, man["N_interface_facelets"]))
    bad = {k: v for k, v in counts.items() if int(v[0]) != int(v[1])}
    if bad or not part["checks"]["pass_"]:
        raise RuntimeError(f"{code}: published partition does not rebuild the paper complex: {bad} {part['checks']}")
    return case, part, site_s


def stage_g1(cfg, code, out):
    path = os.path.join(out, "g1.json")
    if config_io.done(path, config_io.load_json):
        print(dict(case=code, stage="g1", status="checkpoint"), flush=True)
        return
    case, part, _ = build_published(cfg, code)
    print(dict(case=code, stage="g1", status="start", cells=part["geom"].n_cells), flush=True)
    res, _, _ = stokes_solve.g1(case, part["trace"], cfg["viscous_form"])
    res.update(case=code, environment=config_io.environment(), code=config_io.code_hashes(), partition_checks=part["checks"])
    config_io.save_json(path, res)
    print(dict(case=code, stage="g1", pass_=res["pass_"], A_rel=res["gate"]["A"]["max_rel_diff"],
               D_rel=res["gate"]["D"]["max_rel_diff"], t_prod=res["t_prod_s"], t_fast=res["t_fast_s"]), flush=True)


def _validate_e0(path):
    receipt = config_io.load_json(path)
    npz = os.path.join(os.path.dirname(path), "e0.npz")
    if config_io.sha(npz) != receipt["npz_sha256"]:
        raise RuntimeError("e0.npz hash mismatch")


def stage_e0(cfg, code, out):
    path = os.path.join(out, "e0.json")
    if config_io.done(path, _validate_e0):
        print(dict(case=code, stage="e0", status="checkpoint"), flush=True)
        return
    t0 = time.perf_counter()
    case, part, site_s = build_published(cfg, code)
    t = time.perf_counter()
    system, asm = stokes_solve.assemble(case, part["trace"], "fast", cfg["viscous_form"])
    asm_s = time.perf_counter() - t
    print(dict(case=code, stage="e0", status="solve", cells=part["geom"].n_cells,
               dofs=int(system.velocity_matrix.shape[0]), A_nnz=int(system.velocity_matrix.nnz)), flush=True)
    result, receipt, fac, _ = stokes_solve.solve_entry(case, part, system, alpha=0.0, solver_cfg=cfg["solver"],
                                                       label=f"{code}-e0", trace_basis=cfg["trace_basis"])
    npz = os.path.join(out, "e0.npz")
    config_io.save_npz(npz, z=np.asarray(result["trace_coefficients"]).ravel(), p=np.asarray(result["p"]),
                       U=np.asarray(result["U"]), phi=np.asarray(result["phi"]),
                       face_flux_sorted=np.asarray(result["face_flux_sorted"]), volume=np.asarray(part["geom"].volume),
                       seeds=part["seeds"])
    receipt.update(case=code, stage="e0", assembly=asm, timing=dict(site_input_s=site_s, **part["timing"],
                   assembly_s=asm_s, total_s=time.perf_counter() - t0), partition_checks=part["checks"],
                   npz_sha256=config_io.sha(npz), environment=config_io.environment(), code=config_io.code_hashes(),
                   inputs=dict(reference=config_io.sha(case.paths["reference"]), window=config_io.sha(case.paths["window"])))
    config_io.save_json(path, receipt)
    print(dict(case=code, stage="e0", status="done", iterations=receipt["linear_solver_iterations"],
               residual=receipt["linear_solver_relative_residual"], solve_s=receipt["solve_call_s"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg")
    ap.add_argument("--case", required=True)
    ap.add_argument("--stage", choices=["g1", "e0"], required=True)
    a = ap.parse_args()
    cfg = config_io.load_cfg(a.cfg)
    out = config_io.ensure_dir(os.path.join(cfg["out_dir"], a.case))
    {"g1": stage_g1, "e0": stage_e0}[a.stage](cfg, a.case, out)


if __name__ == "__main__":
    main()
