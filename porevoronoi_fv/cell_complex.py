"""Complex builder in the paper's cell order.

Ownership: pore_ownership.exact_owners (non-periodic 6-neighbour graph distance, lower site-id ties; site ids are
the sorted unique flats, the site order of the paper).
Paper cell order: cells are numbered by the rank of their smallest flat voxel index (the cell order of the runs
behind the paper; N_split = 0 in every run record, so no cell is split). Edges are the sorted unique (lo, hi) cell
pairs, so edge order follows the cell order.
Facelets, patches and trace modes depend on labels only through label inequality and key orientation, so with the
paper cell order every archived per-cell, per-edge, per-facelet and per-mode array is directly comparable.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import time
from types import SimpleNamespace

import numpy as np

import hybrid_voronoi_trace as hvt
import pore_ownership as core
import velocity_recovery as rp


def edges_from_labels(labels, periodic_x):
    nc = int(labels.max()) + 1
    keys = []
    for axis in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        a, b = labels[tuple(lo)], labels[tuple(hi)]
        ok = (a >= 0) & (b >= 0) & (a != b)
        keys.append(np.minimum(a[ok], b[ok]).astype(np.int64) * nc + np.maximum(a[ok], b[ok]))
    if periodic_x:
        a, b = labels[:, :, -1], labels[:, :, 0]
        ok = (a >= 0) & (b >= 0) & (a != b)
        keys.append(np.minimum(a[ok], b[ok]).astype(np.int64) * nc + np.maximum(a[ok], b[ok]))
    edge = np.unique(np.concatenate(keys))
    return edge // nc, edge % nc


def paper_order(geom, pore):
    own = geom.labels.ravel()[pore]
    _, first_pos = np.unique(own, return_index=True)
    first_flat = pore[first_pos]
    rank = np.empty(geom.n_cells, dtype=np.int64)
    rank[np.argsort(first_flat, kind="stable")] = np.arange(geom.n_cells)
    return rank


def build(case, seeds, *, order="paper", basis="connected_p1"):
    """Sites -> ownership -> (paper-order) geometry -> hybrid trace. No assembly, no solve."""
    t = time.perf_counter()
    seeds = np.asarray(seeds, dtype=np.int64)
    if not np.array_equal(seeds, np.unique(seeds)):
        raise ValueError("sites must be sorted unique flats")
    site_geom, pore, xyz = core.geometry(case.mask, seeds, case.h, case.periodic_x)
    t_own = time.perf_counter() - t
    if order == "paper":
        rank = paper_order(site_geom, pore)
        labels = np.full(case.shape, -1, dtype=np.int32)
        labels.ravel()[pore] = rank[site_geom.labels.ravel()[pore]]
    elif order == "site":
        rank = np.arange(site_geom.n_cells)
        labels = site_geom.labels
    else:
        raise ValueError(order)
    nc = site_geom.n_cells
    own = labels.ravel()[pore]
    counts = np.bincount(own, minlength=nc)
    centroid = np.column_stack([np.bincount(own, weights=xyz[:, a], minlength=nc) / counts for a in range(3)])
    owner, neigh = edges_from_labels(labels, case.periodic_x)
    geom = SimpleNamespace(mask=case.mask, labels=labels, dist=site_geom.dist, n_cells=nc, volume=counts * case.h ** 3,
                           centroid=centroid, owner=owner, neigh=neigh)
    t = time.perf_counter()
    trace = hvt.build_hybrid_trace_geometry({"cp": rp.HostAPI()}, geom,
                                            SimpleNamespace(voxel_size=case.h, periodic_x=case.periodic_x),
                                            trace_basis=basis)
    t_trace = time.perf_counter() - t
    site_cell = labels.ravel()[seeds]
    checks = partition_checks(case, geom, pore, seeds)
    return dict(geom=geom, pore=pore, xyz=xyz, trace=trace, seeds=seeds, rank_from_site_id=rank,
                site_cell=site_cell, checks=checks, timing=dict(ownership_s=t_own, trace_geometry_s=t_trace))


def partition_checks(case, geom, pore, seeds):
    labels = geom.labels
    wrap = case.mask[:, :, -1] & case.mask[:, :, 0]
    wrap_same = int(np.sum(wrap & (labels[:, :, -1] == labels[:, :, 0])))
    own = labels.ravel()[pore]
    x = np.unravel_index(pore, case.shape)[2]
    xmin = np.full(geom.n_cells, np.iinfo(np.int64).max)
    xmax = np.full(geom.n_cells, -1)
    np.minimum.at(xmin, own, x)
    np.maximum.at(xmax, own, x)
    extent = (xmax - xmin + 1) * case.h
    site_owned = bool(np.array_equal(np.sort(labels.ravel()[seeds]), np.arange(geom.n_cells)))
    return dict(n_cells=int(geom.n_cells), n_sites=int(seeds.size), wrap_same=wrap_same,
                max_cell_x_extent=float(extent.max()), half_length_x=0.5 * case.length_x,
                x_extent_ok=bool(extent.max() < 0.5 * case.length_x), each_site_owns_its_cell=site_owned,
                pass_=bool(wrap_same == 0 and extent.max() < 0.5 * case.length_x and site_owned))


def sites_of(case, rec_idx):
    return np.unique(case.records["flat"][rec_idx])
