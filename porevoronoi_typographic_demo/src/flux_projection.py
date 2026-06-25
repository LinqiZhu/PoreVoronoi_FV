from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .io_utils import out_dir, save_json


def conservative_flux_projection(
    cell_site: np.ndarray,
    site_pressure: np.ndarray,
    graph: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Project site-state edge fluxes onto a cell-conservative FV graph field."""
    n_cells = int(len(cell_site))
    p_cell = site_pressure[cell_site].astype(np.float64)

    owners: list[int] = []
    neighs: list[int] = []
    conductance: list[float] = []
    q_target: list[float] = []
    kind: list[str] = []

    for a, b, g in zip(graph["edge_owner"], graph["edge_neighbor"], graph["conductance"]):
        aa, bb = int(a), int(b)
        gg = float(g)
        owners.append(aa)
        neighs.append(bb)
        conductance.append(gg)
        q_target.append(gg * (p_cell[aa] - p_cell[bb]))
        kind.append("internal")

    for cid, area in enumerate(graph["inlet_area"]):
        if area <= 0:
            continue
        gg = float(area)
        owners.append(cid)
        neighs.append(-1)
        conductance.append(gg)
        q_target.append(gg * (p_cell[cid] - 1.0))
        kind.append("inlet")

    for cid, area in enumerate(graph["outlet_area"]):
        if area <= 0:
            continue
        gg = float(area)
        owners.append(cid)
        neighs.append(-2)
        conductance.append(gg)
        q_target.append(gg * p_cell[cid])
        kind.append("outlet")

    n_edges = len(owners)
    row: list[int] = []
    col: list[int] = []
    data: list[float] = []
    for eid, (a, b) in enumerate(zip(owners, neighs)):
        row.append(a)
        col.append(eid)
        data.append(1.0)
        if b >= 0:
            row.append(b)
            col.append(eid)
            data.append(-1.0)
    bmat = sparse.csr_matrix((data, (row, col)), shape=(n_cells, n_edges))
    q0 = np.asarray(q_target, dtype=np.float64)
    div0 = bmat @ q0
    lhs = bmat @ bmat.T + sparse.eye(n_cells, format="csr") * 1.0e-10
    lam = spla.spsolve(lhs, div0)
    q_proj = q0 - bmat.T @ lam
    div_proj = bmat @ q_proj

    out = {
        "edge_owner": np.asarray(owners, dtype=np.int32),
        "edge_neighbor": np.asarray(neighs, dtype=np.int32),
        "edge_kind": np.asarray(kind, dtype="U8"),
        "conductance": np.asarray(conductance, dtype=np.float32),
        "q_target": q0.astype(np.float32),
        "q_conservative": q_proj.astype(np.float32),
        "divergence_target": div0.astype(np.float32),
        "divergence_conservative": div_proj.astype(np.float32),
        "cell_pressure_state": p_cell.astype(np.float32),
    }
    audit = {
        "cells": int(n_cells),
        "projection_edges": int(n_edges),
        "internal_edges": int(np.sum(out["edge_kind"] == "internal")),
        "inlet_edges": int(np.sum(out["edge_kind"] == "inlet")),
        "outlet_edges": int(np.sum(out["edge_kind"] == "outlet")),
        "target_max_abs_divergence": float(np.max(np.abs(div0))) if len(div0) else 0.0,
        "conservative_max_abs_divergence": float(np.max(np.abs(div_proj))) if len(div_proj) else 0.0,
        "target_l2_divergence": float(np.linalg.norm(div0)),
        "conservative_l2_divergence": float(np.linalg.norm(div_proj)),
        "projection_rule": "Euclidean closest edge field subject to zero cell balance on the FV graph",
    }
    return out, audit


def write_flux_projection(root: str | Path | None = None) -> dict[str, object]:
    cells = np.load(out_dir("fv") / "fv_cells.npz")
    sites = np.load(out_dir("sites") / "prescribed_sites.npz")
    graph_npz = np.load(out_dir("fv") / "fv_graph.npz")
    graph = {k: graph_npz[k] for k in graph_npz.files}
    flux, audit = conservative_flux_projection(
        cells["cell_site"], sites["state"]["pressure"], graph
    )
    out = out_dir("flux")
    np.savez_compressed(out / "conservative_flux_projection.npz", **flux)
    save_json(out / "conservative_flux_projection_audit.json", audit)
    return audit
