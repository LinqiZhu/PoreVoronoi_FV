"""Readouts of the paper (Table 5 and Section on validation), from the archived or a fresh solution.

  e_u,cell      100 * sqrt(sum_i V_i |U_i - U_ref,i|^2 / sum_i V_i |U_ref,i|^2), with U_ref,i the volume average of
                the reference voxel velocity over cell i.
  e_phi         100 * ||phi - phi_ref|| / ||phi_ref||, with phi_ref the reference voxel-face fluxes summed over the
                facelets of each coarse edge (owner = min label -> neigh = max label).
  e_K^s         K = nu * sum_e phi_e dvec_e,x / sum_i V_i / f_x, dvec = centroid[neigh] - centroid[owner] folded
                periodically in x; signed error 100 * (K - K_ref) / |K_ref| against
                stored: K_ref = K_eff_x of the reference npz (the loader passes it through);
                flux: K_ref = nu * sum_f phi_f dvec_f,x / sum V / f_x over the reference voxel faces.
  r_inf^m       max_i |(D z)_i| / V_i (D z as hybrid_voronoi_trace.solve_moment_constrained_hybrid_stokes forms it);
                eta_m = max |D z| / (max |D| |z| + eps * max(1, max |D| |z|)), the mass backward error of the paper.
  full field    rp.normerr(Hfield z, U_ref at every pore voxel): relative L2 error of the voxel velocities.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import math

import numpy as np

import velocity_recovery as rp


def u_ref_cells(case, geom, pore):
    lab = geom.labels.ravel()[pore]
    Uvox = case.reference["U"][case.reference["labels"].ravel()[pore]]
    n = geom.n_cells
    cnt = np.bincount(lab, minlength=n).astype(np.float64)
    return np.stack([np.bincount(lab, weights=Uvox[:, c], minlength=n) / np.maximum(cnt, 1e-300) for c in range(3)],
                    axis=1)


def e_u_cell(U, U_ref, volume):
    num = float(np.sum(volume[:, None] * (U - U_ref) ** 2))
    den = max(float(np.sum(volume[:, None] * U_ref ** 2)), 1e-300)
    return 100.0 * math.sqrt(num / den)


def phi_ref_edges(case, geom):
    mask = case.mask
    D, H, W = mask.shape
    c_lab = geom.labels
    r_lab = case.reference["labels"]
    idx = np.arange(mask.size, dtype=np.int64).reshape(mask.shape)
    parts = []
    # accumulation order of the reference implementation: +x (periodic), +y, +z; voxels in C order
    q = np.roll(idx, -1, axis=2)
    parts.append((idx.ravel(), q.ravel(), 0))
    parts.append((idx[:, :-1, :].ravel(), idx[:, 1:, :].ravel(), 1))
    parts.append((idx[:-1, :, :].ravel(), idx[1:, :, :].ravel(), 2))
    P = np.concatenate([p[0] for p in parts])
    Q = np.concatenate([p[1] for p in parts])
    A = np.concatenate([np.full(p[0].size, p[2], np.int64) for p in parts])
    m = mask.ravel()
    li = c_lab.ravel()[P].astype(np.int64)
    lj = c_lab.ravel()[Q].astype(np.int64)
    ok = m[P] & m[Q] & (li >= 0) & (lj >= 0) & (li != lj)
    ri = r_lab.ravel()[P].astype(np.int64)
    rj = r_lab.ravel()[Q].astype(np.int64)
    ok &= (ri >= 0) & (rj >= 0) & (ri != rj)
    P, A, li, lj, ri, rj = P[ok], A[ok], li[ok], lj[ok], ri[ok], rj[ok]
    order = np.argsort(P * 3 + A, kind="stable")
    li, lj, ri, rj = li[order], lj[order], ri[order], rj[order]
    nref = int(case.reference["U"].shape[0])
    rkey_store = case.reference["owner"] * nref + case.reference["neigh"]
    rorder = np.argsort(rkey_store, kind="stable")
    rkeys = rkey_store[rorder]
    rlo, rhi = np.minimum(ri, rj), np.maximum(ri, rj)
    want = rlo * nref + rhi
    pos = np.searchsorted(rkeys, want)
    posc = np.minimum(pos, rkeys.size - 1)
    found = rkeys[posc] == want
    val = np.where(found, case.reference["phi"][rorder[posc]], 0.0)
    sign_ref = np.where(ri == rlo, 1.0, -1.0)
    clo, chi = np.minimum(li, lj), np.maximum(li, lj)
    sign_coarse = np.where(li == clo, 1.0, -1.0)
    contrib = sign_coarse * sign_ref * val
    nc = geom.n_cells
    ekey = np.asarray(geom.owner, np.int64) * nc + np.asarray(geom.neigh, np.int64)
    if not np.all(ekey[1:] > ekey[:-1]):
        raise RuntimeError("coarse edges are not sorted (owner, neigh) keys")
    ck = clo * nc + chi
    epos = np.searchsorted(ekey, ck)
    if not np.array_equal(ekey[np.minimum(epos, ekey.size - 1)], ck):
        raise RuntimeError("a coarse facelet pair has no edge")
    out = np.zeros(ekey.size, dtype=np.float64)
    np.add.at(out, epos, contrib)  # sequential, in that order (bitwise reproducible)
    return out, dict(voxel_faces=int(ck.size), reference_faces_missing=int(np.sum(~found)))


def e_phi(phi, phi_ref):
    return 100.0 * float(np.linalg.norm(phi - phi_ref) / max(np.linalg.norm(phi_ref), 1e-300))


def dvec_periodic_fold(geom, length_x, periodic_x=True):
    ci = geom.centroid[geom.owner]
    cj = geom.centroid[geom.neigh]
    d = cj - ci
    if periodic_x:
        dx = d[:, 0]
        d[:, 0] = np.where(dx > 0.5 * length_x, dx - length_x, np.where(dx < -0.5 * length_x, dx + length_x, dx))
    return d


def k_method(phi, geom, case):
    dvec = dvec_periodic_fold(geom, case.length_x, case.periodic_x)
    volume = np.asarray(geom.volume, np.float64)
    mean_flux = np.sum(np.asarray(phi)[:, None] * dvec, axis=0) / np.sum(volume)
    return case.nu * float(mean_flux[0]) / float(case.force[0]), mean_flux


def k_ref_flux(case):
    r = case.reference
    total = float(r["volume"].sum())
    flux = float(np.sum(r["phi"][:, None] * r["dvec"], axis=0)[0] / total)
    return case.nu * flux / float(case.force[0])


def signed_percent(K, K_ref):
    return 100.0 * (K - K_ref) / abs(K_ref)


def mass(D, z, volume):
    r = np.asarray(D @ z).reshape(-1)
    r_inf = float(np.max(np.abs(r) / np.maximum(volume, 1e-300)))
    absdz = np.asarray(abs(D) @ np.abs(z)).reshape(-1)
    denom_base = float(np.max(absdz, initial=0.0))
    eta = float(np.max(np.abs(r), initial=0.0)) / (denom_base + np.finfo(float).eps * max(1.0, denom_base))
    return dict(r_inf_m=r_inf, eta_m=eta, Dz_inf=float(np.max(np.abs(r), initial=0.0)), absD_absz_inf=denom_base)


def face_fields(z, trace):
    """Facelet velocity and flux exactly as hybrid_voronoi_trace.solve_moment_constrained_hybrid_stokes forms them."""
    coeff = np.asarray(z).reshape(trace.n_trace_modes, 3)
    face_velocity = np.zeros((trace.n_facelets, 3), dtype=np.float64)
    for column in range(trace.face_mode_ids.shape[1]):
        mode = trace.face_mode_ids[:, column]
        valid = mode >= 0
        if np.any(valid):
            face_velocity[valid] += trace.face_mode_values[valid, column, None] * coeff[mode[valid]]
    face_flux_sorted = (trace.voxel_size * trace.voxel_size) * np.sum(face_velocity * trace.face_normal, axis=1)
    aggregate = np.bincount(trace.face_parent_edge, weights=face_flux_sorted,
                            minlength=trace.stored_edge_sign_from_sorted.size)
    return face_velocity, face_flux_sorted, aggregate * trace.stored_edge_sign_from_sorted


def cell_velocity(z, system):
    U = np.zeros((system.trace_geometry.n_cells, 3))
    for cell, (dofs, recovery) in enumerate(system.cell_recovery):
        if dofs.size:
            U[cell] = recovery @ z[dofs]
    return U


def full_field_error(z, Hfield, case):
    return rp.normerr((Hfield @ z).reshape(-1, 3), case.u_ref_vox)


def table_row(case, part, phi, U, z=None, D=None, kref_flux=None):
    geom = part["geom"]
    U_ref = u_ref_cells(case, geom, part["pore"])
    phi_ref, phi_info = phi_ref_edges(case, geom)
    K, mean_flux = k_method(phi, geom, case)
    kref_flux = k_ref_flux(case) if kref_flux is None else kref_flux
    row = dict(N_c=int(geom.n_cells), K_method=K, K_ref_arch=case.k_ref, K_ref_flux=kref_flux,
               eK_arch_s=signed_percent(K, case.k_ref), eK_flux_s=signed_percent(K, kref_flux),
               e_phi=e_phi(phi, phi_ref), e_u_cell=e_u_cell(U, U_ref, np.asarray(geom.volume, np.float64)),
               mean_flux_x=float(mean_flux[0]), phi_ref_info=phi_info)
    row["eK_arch_abs"] = abs(row["eK_arch_s"])
    row["eK_flux_abs"] = abs(row["eK_flux_s"])
    if z is not None and D is not None:
        row.update(mass(D, z, np.asarray(geom.volume, np.float64)))
    return row, U_ref, phi_ref


def literal_fixed(value, literal):
    literal = literal.strip()
    decimals = len(literal.split(".", 1)[1]) if "." in literal else 0
    text = f"{value:.{decimals}f}"
    if text == "-" + "0." + "0" * decimals:
        text = text[1:]
    return dict(value=value, printed=literal, formatted=text, match=text == literal)


def literal_sci(value, mantissa, exponent):
    digits = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    text = f"{value:.{digits}e}"
    m, e = text.split("e")
    return dict(value=value, printed=f"{mantissa}e{exponent}", formatted=text,
                match=(m == mantissa and int(e) == int(exponent)))


def sig_digits(value, ref):
    if value == ref:
        return 17.0
    if ref == 0:
        return 0.0
    return float(-math.log10(abs(value - ref) / abs(ref)))
