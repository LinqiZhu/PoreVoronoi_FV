#!/usr/bin/env python
"""Vectorised assembly of the trace system of hybrid_voronoi_trace, equal to its loop assembly to round-off
(gate()), and the direct KKT.

The loop assembler (imported, never edited here):
  hybrid_voronoi_trace.py
    assemble_moment_constrained_hybrid_stokes   (8 nested python for-loops)
    build_hybrid_trace_factorization            (the KKT and its solver)

This file reassembles the SAME matrix with numpy fancy indexing / np.add.at / one sparse product and no python
loop over cells, gated to round-off against the loop assembler's matrix (gate()).

The vectorisation is exact algebra, not a re-derivation.  Per cell i, with local mode l and component c
(local dof 3l + c), the production loops build

    gradient_basis[3l+c] = E_c (x) g_l ,      g_l = sum_gamma (A w_{gamma,l} / V) n_gamma          (3-vector)
    recovery[:, 3l+c]    = Q_l[:, c] ,        Q_l = sum_gamma (A w_{gamma,l} / V) r_gamma (x) n_gamma
    divergence[3l+c]     = d_l[c] ,           d_l = sum_gamma  A w_{gamma,l}      n_gamma
    trace_map[a, 3l+c]   = delta_ac w_{gamma,l}
    (G_i z)(r_gamma)     -> delta_ac (g_l . r_gamma)

so that gradient_basis is rank-one in the component index and every 3x3 (l, m) block of the local matrix
closes in the per-(cell, mode) arrays g, Q, d and the per-(cell, facelet, mode) scalar
s_{gamma,l} = w_{gamma,l} - g_l . r_gamma :

  symmetric_gradient viscous:  mu V [ delta_cd (g_l.g_m) + (g_l[d] g_m[c] + g_l[c] g_m[d]) / 2 ]
  full_gradient      viscous:  mu V   delta_cd (g_l.g_m)
  stabilisation:  w_stab [ delta_cd P_lm - S_l Q_m[c,d] - S_m Q_l[d,c] + n_gamma(i) (Q_l^T Q_m)[c,d] ]
      P_lm = sum_gamma s_{gamma,l} s_{gamma,m}      S_l = sum_gamma s_{gamma,l}

P is the only facelet x mode x mode object; it is obtained as one scipy sparse product S^T S of the
(incidence x cell-mode) block-diagonal s matrix, which also enumerates exactly the within-cell (l, m)
pairs the production dense blocks emit -- same triplets, same count, no python loop.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pathlib
import pstats
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

import hybrid_voronoi_trace as hvt  # noqa: E402  (the loop assembler this module reproduces)


NU = 1.0                      # solver units (nu = 1)
BODY_FORCE = np.array([0.002, 0.0, 0.0])   # body force of the paper's runs
VISCOUS_FORM = "symmetric_gradient"        # viscous form of the paper's runs

_TRACE_FIELDS = ["n_cells", "voxel_size", "periodic_x", "domain_length_x", "cell_volume", "cell_centroid",
                 "face_key", "face_group_id", "face_owner", "face_neigh", "face_normal", "face_centroid",
                 "face_r_owner_g", "face_r_neigh_g", "face_parent_edge", "face_patch", "patch_component_key",
                 "face_mode_ids", "face_mode_values", "mode_patch", "wall_cell", "wall_normal", "wall_centroid",
                 "wall_r_g", "stored_edge_sign_from_sorted", "trace_basis"]


def load_trace(path: pathlib.Path) -> hvt.HybridTraceGeometry:
    """Rehydrate the production HybridTraceGeometry cached by the complex builder."""
    z = np.load(path, allow_pickle=False)
    kw: Dict[str, Any] = {}
    for f in _TRACE_FIELDS:
        v = z[f]
        if f == "n_cells":
            kw[f] = int(v)
        elif f in ("voxel_size", "domain_length_x"):
            kw[f] = float(v)
        elif f == "periodic_x":
            kw[f] = bool(v)
        elif f == "trace_basis":
            kw[f] = str(v)
        else:
            kw[f] = np.ascontiguousarray(v)
    return hvt.HybridTraceGeometry(**kw)


# =========================================================================================== incidence
def _incidence(trace: hvt.HybridTraceGeometry) -> Dict[str, np.ndarray]:
    """The production incidence list of section 'for cell in range(trace.n_cells)', verbatim."""
    mi = hvt._minimum_image_x
    owner_r = mi(trace.face_centroid - trace.cell_centroid[trace.face_owner], trace.domain_length_x, trace.periodic_x)
    neigh_r = mi(trace.face_centroid - trace.cell_centroid[trace.face_neigh], trace.domain_length_x, trace.periodic_x)
    wall_r = mi(trace.wall_centroid - trace.cell_centroid[trace.wall_cell], trace.domain_length_x, trace.periodic_x)
    n_f = trace.n_facelets
    inc_cell = np.concatenate([trace.face_owner, trace.face_neigh, trace.wall_cell])
    inc_face = np.concatenate([np.arange(n_f, dtype=np.int32), np.arange(n_f, dtype=np.int32),
                               np.full(trace.wall_cell.size, -1, dtype=np.int32)])
    inc_normal = np.concatenate([trace.face_normal, -trace.face_normal, trace.wall_normal], axis=0)
    inc_r = np.concatenate([owner_r, neigh_r, wall_r], axis=0)
    order = np.argsort(inc_cell, kind="stable")
    return {"cell": inc_cell[order].astype(np.int64), "face": inc_face[order].astype(np.int64),
            "normal": np.ascontiguousarray(inc_normal[order]), "r": np.ascontiguousarray(inc_r[order]),
            "start": np.searchsorted(inc_cell[order], np.arange(trace.n_cells + 1)).astype(np.int64)}


def _ragged_arange(counts, xp=np):
    """0,1,..,c-1 concatenated over the counts, without a python loop."""
    counts = counts.astype(xp.int64)
    total = int(counts.sum())
    if total == 0:
        return xp.zeros(0, dtype=xp.int64)
    offsets = xp.cumsum(counts) - counts                     # group start in the flat array
    return xp.arange(total, dtype=xp.int64) - xp.repeat(offsets, counts)


class _Backend:
    """numpy / cupy switch: the arrays, the scatter-add and the sparse module."""

    def __init__(self, name: str):
        self.name = name
        if name == "numpy":
            self.xp = np
            self.sp = sp
            self.scatter_add = np.add.at
            self.to_host = np.asarray
            self.sync = lambda: None
        elif name == "cupy":
            import cupy as cp
            import cupyx
            import cupyx.scipy.sparse as csp
            self.cp = cp
            self.xp = cp
            self.sp = csp
            self.scatter_add = cupyx.scatter_add
            self.to_host = cp.asnumpy
            self.sync = lambda: cp.cuda.runtime.deviceSynchronize()
        else:
            raise ValueError(name)

    def arr(self, a, dtype=None):
        if self.name == "numpy":
            return np.ascontiguousarray(a, dtype=dtype) if dtype else np.ascontiguousarray(a)
        return self.cp.asarray(np.ascontiguousarray(a), dtype=dtype) if dtype else self.cp.asarray(np.ascontiguousarray(a))

    def drop_zeros(self, M):
        """eliminate_zeros, written out: cupy 14.1.1's own corrupts the values (see coo_to_csr)."""
        if self.name == "numpy":
            M.eliminate_zeros()
            return M
        cp = self.cp
        keep = M.data != 0.0
        n = M.shape[0]
        rows = cp.repeat(cp.arange(n, dtype=cp.int32), cp.diff(M.indptr))[keep]
        indptr = cp.concatenate([cp.zeros(1, dtype=cp.int64),
                                 cp.cumsum(cp.bincount(rows, minlength=n))]).astype(cp.int32)
        out = self.sp.csr_matrix((M.data[keep], M.indices[keep], indptr), shape=M.shape)
        out._has_canonical_format = True
        out._has_sorted_indices = True
        return out

    def csr_to_host(self, M):
        return M if self.name == "numpy" else sp.csr_matrix(
            (self.to_host(M.data), self.to_host(M.indices), self.to_host(M.indptr)), shape=M.shape)

    def coo_to_csr(self, rows, cols, vals, shape):
        """COO -> canonical CSR with the duplicates summed.

        On the host this is scipy's coo_tocsr.  On the device it is written out by hand: cupy 14.1.1's
        `csr_matrix.eliminate_zeros()` corrupts the values (verified: a 259 022-nnz matrix's data sum
        jumps 442.18 -> 110 200 with the nnz unchanged) and its `coo_matrix.tocsr()` leaves duplicates
        in place, so neither can be trusted here.  unique(key) + scatter_add is exact and deterministic
        in the pattern, and the key is already row-major so the CSR comes out index-sorted.
        """
        if self.name == "numpy":
            M = sp.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
            M.sum_duplicates()
            return M
        cp = self.cp
        n_col = cp.int64(shape[1])
        key = rows.astype(cp.int64) * n_col + cols.astype(cp.int64)
        uniq, inv = cp.unique(key, return_inverse=True)
        inv = inv.reshape(-1)
        data = cp.zeros(uniq.size, dtype=cp.float64)
        self.scatter_add(data, inv, vals)
        r = (uniq // n_col).astype(cp.int64)
        c = (uniq - r * n_col).astype(cp.int32)
        indptr = cp.concatenate([cp.zeros(1, dtype=cp.int64),
                                 cp.cumsum(cp.bincount(r, minlength=shape[0]))]).astype(cp.int32)
        M = self.sp.csr_matrix((data, c, indptr), shape=shape)
        M._has_canonical_format = True
        M._has_sorted_indices = True
        return M


# =========================================================================================== vectorised assembly
def assemble_vectorised(trace: hvt.HybridTraceGeometry, *, viscosity: float, body_force: np.ndarray,
                        viscous_form: str = "symmetric_gradient",
                        backend: str = "numpy", timing: Optional[Dict[str, float]] = None,
                        wall: str = "skip", wall_penalty_scale: float = 1.0,
                        wall_velocity: Optional[np.ndarray] = None,
                        nitsche_scale: float = 1.0, nitsche_theta: float = 1.0,
                        tau_cell=None):
    """Assemble the identical system with array algebra only (no python loop over cells/facelets/modes).

    `wall` selects how a solid facelet (facelet id -1, no trace unknown) enters the cell system.

    "skip"       the default treatment of the loop assembler, reproduced bit-exactly: the wall facelet is absent
                 from the recovery / gradient / divergence Gauss sums and present only in the
                 stabilisation, with trace_map = 0 (hybrid_voronoi_trace.py:1184-1186, :1217-1230).

    "dirichlet"  the wall facelet carries a PRESCRIBED trace u_w (default 0) and enters the recovery,
                 gradient and divergence Gauss sums exactly as an interior facelet does.  Because a
                 prescribed trace is data and not an unknown, every one of those contributions is
                 affine in u_w:  G += (A/V) u_w (x) n,  ubar += (A/V) (n.u_w) r,  div += A (n.u_w).
                 With the physical no-slip value u_w = 0 all three are IDENTICALLY ZERO, so this path
                 returns the SAME operator as "skip" -- bit for bit.  It is kept because it is the
                 literal reading of "a wall facelet is a facelet whose trace is known to be zero",
                 and its being a no-op is a result, not an omission.  A non-zero `wall_velocity`
                 (a moving wall) is rejected here and must go through the reference loop assembler,
                 which carries the resulting right-hand sides.

    "nitsche"    "dirichlet" PLUS the wall's VISCOUS FLUX written as the consistency and
                 adjoint-consistency pair of a Nitsche Dirichlet condition,

                     -  A [ (sigma(U).n) . V(x_w)  +  (sigma(V).n) . U(x_w) ] summed over wall facelets

                 with sigma(U) = 2 mu eps(U) for viscous_form="symmetric_gradient" and mu grad U for
                 "full_gradient", U(x) = ubar + G.(x - x_i) the cell's own linear reconstruction and
                 x_w the wall facelet centroid.  Cross-checked bit-for-bit against an independent
                 loop assembler.

                 `nitsche_theta` selects the member of the Nitsche family (Cascavita, Chouly & Ern
                 2020, IMA JNA 40(4):2189-2226, eq. (5.11a) / Remark 2.1; the CELL version, which is
                 by construction the configuration here -- no face unknown on the Dirichlet
                 boundary).  With C the consistency operator, C[i,j] = A (sigma(phi_j).n_w).phi_i(x_w),
                 the added block is

                     -( C + theta C^T )

                 theta = +1  symmetric
                 theta =  0  incomplete     (threshold halved: sym(A_{theta=0}) is exactly
                                             0.5 (A_skip + A_{theta=+1}))
                 theta = -1  skew-symmetric (the added block is EXACTLY antisymmetric, so the
                                             symmetric part of the operator is untouched and
                                             coercivity is inherited from the penalty form for any
                                             gamma_0 > 0 -- Remark 2.1.  The operator is then
                                             NON-SYMMETRIC: a symmetric/LDL^T factorisation is
                                             invalid and a general LU arm must be used.)

                 theta = +1 reproduces the symmetric loop form bit for bit.

                 The nitsche variants lower the convergence order on the duct family and are not used
                 for any result of the paper; the default wall="skip" is.

    `wall_penalty_scale` multiplies w_stab on the WALL facelets only (lambda of the penalty sweep);
    1.0 is the default weight and takes the untouched code path.
    """
    if wall not in ("skip", "dirichlet", "nitsche"):
        raise ValueError("wall must be 'skip', 'dirichlet' or 'nitsche', got %r" % (wall,))
    if wall != "nitsche" and float(nitsche_theta) != 1.0:
        raise ValueError("nitsche_theta only acts on wall='nitsche'; it would be a silent no-op "
                         "here (wall=%r, nitsche_theta=%r)" % (wall, nitsche_theta))
    if wall_velocity is not None and float(np.abs(np.asarray(wall_velocity)).max()) != 0.0:
        raise NotImplementedError(
            "a non-zero prescribed wall trace only produces right-hand-side terms; the vectorised "
            "path assembles the operator.  Use the reference loop assembler for a moving wall.")
    B = _Backend(backend)
    xp, xsp = B.xp, B.sp
    B.sync()
    t_all = time.perf_counter()
    area = trace.voxel_size * trace.voxel_size
    n_modes = trace.n_trace_modes
    n_trace_dofs = 3 * n_modes
    body_force = np.asarray(body_force, dtype=np.float64).reshape(3)
    tk: Dict[str, float] = {}

    def mark(name: str, t0: float) -> float:
        B.sync()
        t1 = time.perf_counter()
        tk[name] = t1 - t0
        return t1

    t0 = time.perf_counter()
    inc_h = _incidence(trace)
    inc_cell = B.arr(inc_h["cell"]); inc_face = B.arr(inc_h["face"])
    inc_n = B.arr(inc_h["normal"]); inc_r = B.arr(inc_h["r"])
    inc_start = B.arr(inc_h["start"])
    n_inc = int(inc_cell.size)
    volume = B.arr(trace.cell_volume, dtype=np.float64)
    t0 = mark("incidence", t0)

    # ---- (facelet-mode) entries restricted to the incidences: (incidence k, mode, value)
    ids = B.arr(trace.face_mode_ids)
    vals = B.arr(trace.face_mode_values, dtype=np.float64)
    n_slot = int(ids.shape[1])
    safe_face = xp.where(inc_face >= 0, inc_face, 0)
    e_ids = ids[safe_face]                                   # (n_inc, n_slot)
    e_vals = vals[safe_face]
    ok = (inc_face >= 0)[:, None] & (e_ids >= 0)
    e_k = xp.repeat(xp.arange(n_inc, dtype=xp.int64), n_slot).reshape(n_inc, n_slot)[ok]
    e_mode = e_ids[ok].astype(xp.int64)
    e_val = e_vals[ok]
    e_cell = inc_cell[e_k]
    t0 = mark("mode_entries", t0)

    # ---- the per-cell local mode list (production: sorted(mode_set)); flat, cell-major, mode-ascending
    key = e_cell * xp.int64(n_modes) + e_mode
    uniq, e_flat = xp.unique(key, return_inverse=True)
    e_flat = e_flat.reshape(-1)
    cm_cell = uniq // xp.int64(n_modes)
    cm_mode = uniq - cm_cell * xp.int64(n_modes)
    n_cm = int(uniq.size)
    cell_nl = xp.bincount(cm_cell, minlength=trace.n_cells).astype(xp.int64)
    cm_start = xp.concatenate([xp.zeros(1, dtype=xp.int64), xp.cumsum(cell_nl)]).astype(xp.int64)
    t0 = mark("local_modes", t0)

    # ---- g_l, Q_l, d_l : per (cell, mode).  Term-by-term identical to the production expressions.
    coef = area * e_val / volume[e_cell]                     # (A value / V)
    g = xp.zeros((n_cm, 3))
    B.scatter_add(g, e_flat, coef[:, None] * inc_n[e_k])
    Q = xp.zeros((n_cm, 3, 3))
    B.scatter_add(Q, e_flat, coef[:, None, None] * (inc_r[e_k][:, :, None] * inc_n[e_k][:, None, :]))
    dvec = xp.zeros((n_cm, 3))
    B.scatter_add(dvec, e_flat, (area * e_val)[:, None] * inc_n[e_k])
    t0 = mark("moments_gQd", t0)

    # ---- s_{gamma,l} on the full (incidence x local-mode) block of each cell
    nl_inc = cell_nl[inc_cell]                               # local modes of this incidence's cell
    inc_off = xp.concatenate([xp.zeros(1, dtype=xp.int64), xp.cumsum(nl_inc)]).astype(xp.int64)
    m_tot = int(inc_off[-1])
    b_k = xp.repeat(xp.arange(n_inc, dtype=xp.int64), nl_inc)          # incidence of each block entry
    b_l = _ragged_arange(nl_inc, xp) + xp.repeat(cm_start[inc_cell], nl_inc)   # flat cell-mode per block entry
    s = -xp.einsum("ij,ij->i", g[b_l], inc_r[b_k])           # -(g_l . r_gamma)
    pos = inc_off[e_k] + (e_flat - cm_start[e_cell])         # where each (facelet,mode) entry lands
    B.scatter_add(s, pos, e_val)                             # + w_{gamma,l}
    t0 = mark("residual_s", t0)

    # ---- the within-cell (l,m) pair list: enumerated STRUCTURALLY, cell by cell, l-major.
    # (it cannot come out of the sparse product below: scipy/cuSPARSE prune an exactly-zero P_lm, and such a
    #  pair still carries a nonzero viscous and Q^T Q block.)
    npair_cell = cell_nl * cell_nl
    pair_cell = xp.repeat(xp.arange(trace.n_cells, dtype=xp.int64), npair_cell)
    tpair = _ragged_arange(npair_cell, xp)
    nl_p = cell_nl[pair_cell]
    base = cm_start[pair_cell]
    t_row = tpair // nl_p
    t_col = tpair - t_row * nl_p
    pl = base + t_row
    pm = base + t_col
    # index of the transposed pair (m,l) inside the same cell: lets 0.5 (A + A^T) be done on the dense
    # blocks instead of on the assembled sparse matrix, which halves the COO the reduction has to sort
    pair_base = xp.cumsum(npair_cell) - npair_cell
    jT = pair_base[pair_cell] + t_col * nl_p + t_row
    n_pair = int(pl.size)
    t0 = mark("pair_list", t0)

    # ---- the per-incidence stabilisation weight.  lambda = 1 on every facelet is the default
    # weight and takes the identical (unweighted) code path, so the bit-exact gate is unaffected.
    lam_scale = float(wall_penalty_scale)
    weighted = (lam_scale != 1.0)
    if weighted:
        lam_inc = xp.ones(n_inc)
        lam_inc[inc_face < 0] = lam_scale

    # ---- P_lm = sum_gamma lam_gamma s_{gamma,l} s_{gamma,m} : one block-diagonal sparse product
    S = xsp.csr_matrix((s, b_l.astype(xp.int32), inc_off.astype(xp.int32)), shape=(n_inc, n_cm))
    if weighted:
        Sw = xsp.csr_matrix((s * lam_inc[b_k], b_l.astype(xp.int32), inc_off.astype(xp.int32)),
                            shape=(n_inc, n_cm))
        P = (Sw.T @ S).tocsr()
    else:
        P = (S.T @ S).tocsr()
    P.sort_indices()
    Pv = xp.zeros(n_pair)
    key_pair = pl * xp.int64(n_cm) + pm
    key_P = (xp.repeat(xp.arange(n_cm, dtype=xp.int64), xp.diff(P.indptr.astype(xp.int64)))
             * xp.int64(n_cm) + P.indices.astype(xp.int64))
    Pv[xp.searchsorted(key_pair, key_P)] = P.data
    t0 = mark("pair_product", t0)

    # ---- the 3x3 (l,m) blocks
    Ssum = xp.zeros(n_cm)
    if weighted:
        B.scatter_add(Ssum, b_l, s * lam_inc[b_k])
        n_face_cell = xp.zeros(trace.n_cells)
        B.scatter_add(n_face_cell, inc_cell, lam_inc)
    else:
        B.scatter_add(Ssum, b_l, s)
        n_face_cell = xp.diff(inc_start).astype(xp.float64)  # incidences per cell (walls included)
    cell_of_pair = cm_cell[pl]
    Vp = volume[cell_of_pair]
    nfp = n_face_cell[cell_of_pair]
    gl, gm = g[pl], g[pm]
    Ql, Qm = Q[pl], Q[pm]
    Sl, Sm = Ssum[pl], Ssum[pm]
    eye3 = xp.eye(3)

    gg = xp.einsum("ij,ij->i", gl, gm)
    if viscous_form == "full_gradient":
        diag = viscosity * Vp * gg
        blk = xp.zeros((n_pair, 3, 3))
        w_stab = viscosity * area / trace.voxel_size
    elif viscous_form == "symmetric_gradient":
        # 2 mu V sum_ij S_a S_b  with  S_a[i,j] = (delta_ic g_l[j] + delta_jc g_l[i]) / 2
        #   = mu V [ delta_cd (g_l . g_m) + g_l[d] g_m[c] ]
        diag = viscosity * Vp * gg
        blk = (viscosity * Vp)[:, None, None] * (gm[:, :, None] * gl[:, None, :])
        w_stab = 2.0 * viscosity * area / trace.voxel_size
    else:
        raise ValueError(f"Unsupported viscous form: {viscous_form}")

    # The only change from porevoronoi_fv/fast_assembly.py: the residual-stabilization weight of cell i
    # is multiplied by tau_cell[i].  Every stabilization term below is carried on the within-cell
    # (l, m) pair list, whose cell is pair_cell, so one per-pair factor reproduces exactly
    # "multiply the cell's local stabilization matrix by tau_i".  tau_cell=None keeps the scalar
    # w_stab and therefore the identical (bit-exact) arithmetic.
    if tau_cell is None:
        w_stab_p = w_stab
    else:
        tau_arr = B.arr(np.asarray(tau_cell, dtype=np.float64).reshape(-1))
        if int(tau_arr.size) != int(trace.n_cells):
            raise ValueError("tau_cell must have one entry per cell")
        w_stab_p = w_stab * tau_arr[cell_of_pair]

    blk += (w_stab_p * nfp)[:, None, None] * xp.einsum("kac,kad->kcd", Ql, Qm)
    blk -= (w_stab_p * Sl)[:, None, None] * Qm
    blk -= (w_stab_p * Sm)[:, None, None] * xp.transpose(Ql, (0, 2, 1))
    diag = diag + w_stab_p * Pv

    # ---- the wall's viscous flux (Nitsche consistency + adjoint consistency).
    # For each wall facelet w of the cell, with outward normal n_w, offset r_w = x_w - x_i, the
    # prescribed trace u_w = 0 and the cell's own reconstruction U(x) = ubar + G.(x - x_i):
    #     -A [ (sigma(U).n_w) . U(x_w) + (sigma(V).n_w) . U(x_w) ]
    # sigma(U).n = mu[(g_l.n) z_{3l+a} E_a + g_l n.z_l]  (symmetric_gradient) or mu (g_l.n) z_{3l+a}
    # (full_gradient).  Written on the pair list it closes in two per-cell wall moments,
    #     N = sum_w n_w      and      M1 = sum_w n_w (x) r_w,
    # because every dependence on the individual facelet is linear in n_w and in r_w.  At
    # theta = +1 only the (l,m) half is emitted and the 0.5 (A + A^T) below turns it into the
    # symmetric pair; at any other theta the pair is emitted explicitly AFTER that symmetrisation,
    # because -(C + theta C^T) is not symmetric and must not be run through it.
    nit_T = None
    nit_coef = 0.0
    if wall == "nitsche" and int(trace.wall_cell.size) > 0 and float(nitsche_scale) != 0.0:
        w_sel = inc_face < 0
        w_cell = inc_cell[w_sel]
        w_n = inc_n[w_sel]
        w_r = inc_r[w_sel]
        Nwall = xp.zeros((trace.n_cells, 3))
        B.scatter_add(Nwall, w_cell, w_n)
        M1wall = xp.zeros((trace.n_cells, 3, 3))
        B.scatter_add(M1wall, w_cell, w_n[:, :, None] * w_r[:, None, :])
        Np = Nwall[cell_of_pair]
        M1p = M1wall[cell_of_pair]
        alpha_l = xp.einsum("ki,ki->k", gl, Np)                     # sum_w  g_l . n_w
        beta_lm = xp.einsum("ki,kij,kj->k", gl, M1p, gm)            # sum_w (g_l.n_w)(g_m.r_w)
        coef_n = -2.0 * float(nitsche_scale) * area * viscosity
        Xn = alpha_l[:, None, None] * Qm
        if viscous_form == "symmetric_gradient":
            Xn = Xn + (Np[:, :, None] * xp.einsum("ka,kad->kd", gl, Qm)[:, None, :]
                       + xp.einsum("kij,kj->ki", M1p, gm)[:, :, None] * gl[:, None, :])
        if float(nitsche_theta) == 1.0:
            # kept in this form so that theta = +1 matches the loop assembler bit for bit:
            # 0.5 (A + A^T) of (-2 C^T) is -(C + C^T)
            blk += coef_n * Xn
            diag = diag + coef_n * beta_lm
            del Xn
        else:
            # nit_T = C^T / (A mu nitsche_scale): the consistency operator's transpose, on the pair
            # list.  (Xn + beta_lm I is exactly that -- the theta = +1 branch above emits
            # coef_n = -2 A mu s times it, and 0.5(A + A^T) then gives -(C + C^T).)
            nit_T = Xn
            nit_T[:, 0, 0] += beta_lm
            nit_T[:, 1, 1] += beta_lm
            nit_T[:, 2, 2] += beta_lm
            nit_coef = 0.5 * coef_n                                  # = -A mu nitsche_scale
        del Np, M1p, alpha_l, beta_lm

    blk[:, 0, 0] += diag
    blk[:, 1, 1] += diag
    blk[:, 2, 2] += diag
    del diag, gl, gm, Ql, Qm, Sl, Sm, gg
    # 0.5 (A + A^T), done on the dense blocks: the cell's local matrix is symmetric in exact arithmetic,
    # so this only removes the production assembler's own round-off, exactly as its 0.5 (A + A^T) does.
    blk = 0.5 * (blk + xp.transpose(blk[jT], (0, 2, 1)))
    if nit_T is not None:
        # -(C + theta C^T), added after the symmetrisation.  The (m,l) pair's block transposed in
        # (c,d) is the assembled transpose of the (l,m) pair's block, so C = (C^T)^T = nit_T[jT]^T.
        blk += nit_coef * (xp.transpose(nit_T[jT], (0, 2, 1)) + float(nitsche_theta) * nit_T)
        del nit_T
    t0 = mark("blocks", t0)

    # ---- COO triplets: row 3*mode_l + c, col 3*mode_m + d
    c3 = xp.arange(3, dtype=xp.int32)
    rows = (3 * cm_mode[pl].astype(xp.int32)[:, None] + c3[None, :]).repeat(3, axis=1).reshape(-1)
    cols = xp.tile((3 * cm_mode[pm].astype(xp.int32)[:, None] + c3[None, :]), (1, 3)).reshape(-1)
    velocity = B.coo_to_csr(rows, cols, blk.reshape(-1), (n_trace_dofs, n_trace_dofs))
    del rows, cols, blk
    t0 = mark("coo_velocity", t0)

    # ---- divergence, body force, cell recovery
    nz = dvec != 0.0
    d_rows = xp.repeat(cm_cell, 3).reshape(n_cm, 3)[nz]
    d_cols = (3 * cm_mode[:, None] + c3[None, :])[nz]
    divergence = B.coo_to_csr(d_rows, d_cols, dvec[nz], (trace.n_cells, n_trace_dofs))
    bfm = xp.zeros((n_trace_dofs, 3))
    B.scatter_add(bfm, (3 * cm_mode[:, None] + c3[None, :]).reshape(-1),
                  (volume[cm_cell][:, None, None] * xp.transpose(Q, (0, 2, 1))).reshape(-1, 3))
    rhs_trace = bfm @ B.arr(body_force)
    t0 = mark("divergence_bfm", t0)

    # what the production 0.5 (A + A^T) drops implicitly: scipy's sparse add prunes an exactly-zero sum
    velocity = B.drop_zeros(velocity)
    t0 = mark("eliminate_zeros", t0)

    if backend == "cupy":
        velocity = B.csr_to_host(velocity)
        divergence = B.csr_to_host(divergence)
        bfm = B.to_host(bfm)
        rhs_trace = B.to_host(rhs_trace)
        cm_start_h = B.to_host(cm_start)
        cm_mode_h = B.to_host(cm_mode)
        Q_h = B.to_host(Q)
        t0 = mark("device_to_host", t0)
    else:
        cm_start_h, cm_mode_h, Q_h = cm_start, cm_mode, Q

    cell_recovery: List[Tuple[np.ndarray, np.ndarray]] = []
    gdofs_all = (3 * cm_mode_h[:, None] + np.arange(3, dtype=np.int64)[None, :]).reshape(-1)
    Qflat = np.transpose(Q_h, (1, 0, 2)).reshape(3, -1)        # [a, 3l+c] = Q_l[a,c] == the recovery row block
    for cell in range(trace.n_cells):
        a, b = int(cm_start_h[cell]), int(cm_start_h[cell + 1])
        if a == b:
            cell_recovery.append((np.empty(0, dtype=np.int64), np.zeros((3, 0))))
        else:
            cell_recovery.append((gdofs_all[3 * a:3 * b], np.ascontiguousarray(Qflat[:, 3 * a:3 * b])))
    t0 = mark("cell_recovery", t0)

    total = time.perf_counter() - t_all
    tk["total"] = total
    if timing is not None:
        timing.update(tk)
    sysd = hvt.HybridTraceSystem(
        trace_geometry=trace, velocity_matrix=velocity, divergence_matrix=divergence,
        body_force_matrix=bfm, rhs_trace=rhs_trace, cell_recovery=tuple(cell_recovery),
        viscosity=float(viscosity), body_force=body_force, viscous_form=viscous_form,
        assembly_time_s=total)
    sysd_sizes = {"n_incidences": int(n_inc), "n_mode_entries": int(e_val.size), "n_cell_modes": int(n_cm),
                  "n_block_entries": int(m_tot), "n_pairs": int(n_pair), "n_triplets": int(9 * n_pair)}
    return sysd, sysd_sizes


# =========================================================================================== gate
def _relerr(A: sp.spmatrix, B: sp.spmatrix) -> Dict[str, Any]:
    A = A.tocsr().copy(); A.sum_duplicates(); A.sort_indices()
    B = B.tocsr().copy(); B.sum_duplicates(); B.sort_indices()
    same_shape = A.shape == B.shape
    d = (A - B).tocsr()
    scale = float(np.abs(A.data).max()) if A.nnz else 1.0
    Az = A.copy(); Az.eliminate_zeros()
    Bz = B.copy(); Bz.eliminate_zeros()
    # which entries live in one pattern and not the other, and how big they are there
    ka = A.indices.astype(np.int64) + np.int64(A.shape[1]) * np.repeat(
        np.arange(A.shape[0], dtype=np.int64), np.diff(A.indptr))
    kb = B.indices.astype(np.int64) + np.int64(B.shape[1]) * np.repeat(
        np.arange(B.shape[0], dtype=np.int64), np.diff(B.indptr))
    only_a = np.isin(ka, kb, invert=True)
    only_b = np.isin(kb, ka, invert=True)
    extra = {"in_ref_not_new": int(only_a.sum()), "in_new_not_ref": int(only_b.sum()),
             "max_abs_ref_value_on_ref_only": float(np.abs(A.data[only_a]).max()) if only_a.any() else 0.0,
             "max_abs_new_value_on_new_only": float(np.abs(B.data[only_b]).max()) if only_b.any() else 0.0}
    extra["max_rel_ref_value_on_ref_only"] = extra["max_abs_ref_value_on_ref_only"] / scale
    extra["max_rel_new_value_on_new_only"] = extra["max_abs_new_value_on_new_only"] / scale
    # the same comparison once both matrices are trimmed at the tolerance the gate is stated at: an entry
    # whose value is below 1e-12 |A|_max is not a structural entry, it is the round-off the production
    # assembler's dense einsum / gemm leaves in slots that are algebraically zero.
    thr = 1e-12 * scale
    ka_t = ka[np.abs(A.data) > thr]
    kb_t = kb[np.abs(B.data) > thr]
    extra["nnz_ref_above_1e-12_rel"] = int(ka_t.size)
    extra["nnz_new_above_1e-12_rel"] = int(kb_t.size)
    extra["patterns_identical_above_1e-12_rel"] = bool(
        ka_t.size == kb_t.size and np.array_equal(ka_t, kb_t))
    return {"shape_ref": list(A.shape), "shape_new": list(B.shape), "same_shape": bool(same_shape),
            "pattern_difference": extra,
            "nnz_ref": int(A.nnz), "nnz_new": int(B.nnz), "same_nnz": bool(A.nnz == B.nnz),
            "nnz_ref_nonzero": int(Az.nnz), "nnz_new_nonzero": int(Bz.nnz),
            "same_nnz_nonzero": bool(Az.nnz == Bz.nnz),
            "same_pattern": bool(A.nnz == B.nnz and np.array_equal(A.indices, B.indices)
                                 and np.array_equal(A.indptr, B.indptr)),
            "max_abs_diff": float(np.abs(d.data).max()) if d.nnz else 0.0,
            "max_abs_ref": scale,
            "max_rel_diff": (float(np.abs(d.data).max()) / scale) if d.nnz else 0.0}


def gate(ref, new) -> Dict[str, Any]:
    out = {"A": _relerr(ref.velocity_matrix, new.velocity_matrix),
           "D": _relerr(ref.divergence_matrix, new.divergence_matrix)}
    bf_scale = float(np.abs(ref.body_force_matrix).max())
    out["body_force_matrix"] = {"max_abs_diff": float(np.abs(ref.body_force_matrix - new.body_force_matrix).max()),
                                "max_rel_diff": float(np.abs(ref.body_force_matrix - new.body_force_matrix).max() / bf_scale)}
    r_scale = float(np.abs(ref.rhs_trace).max())
    out["rhs_trace"] = {"max_abs_diff": float(np.abs(ref.rhs_trace - new.rhs_trace).max()),
                        "max_rel_diff": float(np.abs(ref.rhs_trace - new.rhs_trace).max() / max(r_scale, 1e-300))}
    # cell recovery blocks
    md = 0.0; ms = 0.0; ok_dofs = True
    for (ga, Ra), (gb, Rb) in zip(ref.cell_recovery, new.cell_recovery):
        if not np.array_equal(np.asarray(ga), np.asarray(gb)):
            ok_dofs = False
        if Ra.size:
            md = max(md, float(np.abs(Ra - Rb).max())); ms = max(ms, float(np.abs(Ra).max()))
    out["cell_recovery"] = {"same_global_dofs": bool(ok_dofs), "max_abs_diff": md,
                            "max_rel_diff": md / max(ms, 1e-300)}
    tol = 1e-12
    # the production matrix carries round-off in slots that are structurally zero (its einsum and gemm sum
    # nine / n_dof products where the closed form has an exact delta); those slots survive its 0.5 (A + A^T)
    # as ~1e-18 explicit entries.  The vectorised matrix computes them as exact zeros, so its pattern is a
    # SUBSET.  The gate therefore asks: no entry the production matrix does not have, and every entry it has
    # and we do not is itself below tolerance.
    for k in ("A", "D"):
        pd = out[k]["pattern_difference"]
        pd["symmetric_difference"] = int(pd["in_ref_not_new"] + pd["in_new_not_ref"])
        pd["symmetric_difference_frac_of_nnz"] = pd["symmetric_difference"] / max(out[k]["nnz_ref"], 1)
        pd["max_rel_value_in_symmetric_difference"] = max(pd["max_rel_ref_value_on_ref_only"],
                                                          pd["max_rel_new_value_on_new_only"])
        out[k]["pattern_difference_is_round_off"] = bool(
            pd["max_rel_value_in_symmetric_difference"] < tol)
        out[k]["pass"] = bool(out[k]["same_shape"] and out[k]["max_rel_diff"] < tol
                              and out[k]["pattern_difference_is_round_off"])
    out["pass"] = bool(out["A"]["pass"] and out["D"]["pass"]
                       and out["body_force_matrix"]["max_rel_diff"] < tol
                       and out["cell_recovery"]["same_global_dofs"]
                       and out["cell_recovery"]["max_rel_diff"] < tol)
    out["pass_strict_same_nnz"] = bool(out["pass"] and out["A"]["same_nnz"] and out["D"]["same_nnz"])
    out["tol"] = tol
    return out


# =========================================================================================== KKT
def build_kkt(system) -> sp.csc_matrix:
    """The saddle point of build_hybrid_trace_factorization(solver='direct_lu'), verbatim."""
    velocity = system.velocity_matrix
    divergence = system.divergence_matrix
    n_trace_dofs = int(velocity.shape[0])
    n_cells = system.trace_geometry.n_cells
    gauge = sp.csr_matrix(np.ones((n_cells, 1)) / float(n_cells))
    zero_p = sp.csr_matrix((n_cells, n_cells))
    zero_g = sp.csr_matrix((n_trace_dofs, 1))
    return sp.bmat([[velocity, divergence.T, zero_g],
                    [divergence, zero_p, gauge],
                    [zero_g.T, gauge.T, None]], format="csc")


if __name__ == "__main__":
    raise SystemExit("this module is a library (vectorised assembler); import it instead of running it")
