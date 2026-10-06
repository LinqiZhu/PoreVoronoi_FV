"""Velocity recovery operators (R, G), the point operator H, the relative L2 error norm, a NumPy adapter for the
trace-geometry builder and the Cartesian-lattice geometry used by the tests.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

import hybrid_voronoi_trace as hvt  # minimum-image offsets used by point_operator


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HostAPI:
    """NumPy stand-in for the CuPy namespace expected by the face enumeration (provides scalar .get())."""
    def __getattr__(self, name):
        return getattr(np, name)

    def any(self, value):
        return SimpleNamespace(get=lambda: np.any(value))

    def asnumpy(self, value):
        return np.asarray(value)


def cube_geometry(n, m, length=16.0):
    """Exact L1 Voronoi ownership on a Cartesian product lattice, lower-id ties."""
    h = length / n
    site_index = np.rint((np.arange(m) + 0.5) * n / m - 0.5).astype(int)
    if len(np.unique(site_index)) != m:
        raise ValueError("Site lattice is degenerate")
    axis_distance = np.abs(np.arange(n)[:, None] - site_index[None, :])
    owner_axis = np.argmin(axis_distance, axis=1)
    dist_axis = np.min(axis_distance, axis=1)
    zz, yy, xx = np.indices((n, n, n))
    labels = ((owner_axis[zz] * m + owner_axis[yy]) * m + owner_axis[xx]).astype(np.int32)
    dist = (dist_axis[zz] + dist_axis[yy] + dist_axis[xx]) * h
    xyz = np.column_stack([xx.ravel() + .5, yy.ravel() + .5, zz.ravel() + .5]) * h
    nc = m ** 3
    counts = np.bincount(labels.ravel(), minlength=nc)
    centroid = np.column_stack([np.bincount(labels.ravel(), weights=xyz[:, a], minlength=nc)
                                / counts for a in range(3)])
    keys = []
    for axis in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        lo, hi = labels[tuple(a)], labels[tuple(b)]
        sel = lo != hi
        keys.append(np.minimum(lo[sel], hi[sel]) * nc + np.maximum(lo[sel], hi[sel]))
    edges = np.unique(np.concatenate(keys))
    geom = SimpleNamespace(mask=np.ones(labels.shape, bool), labels=labels, dist=dist,
                           n_cells=nc, volume=counts * h**3, centroid=centroid,
                           owner=edges // nc, neigh=edges % nc)
    sites = np.array(np.meshgrid(site_index, site_index, site_index, indexing="ij")).reshape(3, -1).T
    # Independent all-site distance check; limited to small cases to avoid a large dense matrix.
    if nc <= 64:
        q = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()])
        exact = np.abs(q[:, None, :] - sites[None, :, :]).sum(axis=2)
        assert np.array_equal(np.argmin(exact, axis=1), labels.ravel())
        assert np.allclose(np.min(exact, axis=1) * h, dist.ravel())
    return geom, xyz


def recovery_operators(trace, system):
    nz, nc = system.velocity_matrix.shape[0], trace.n_cells
    rr, rc, rv = [], [], []
    for i, (dofs, recovery) in enumerate(system.cell_recovery):
        rr.extend(np.repeat(3*i + np.arange(3), len(dofs)))
        rc.extend(np.tile(dofs, 3))
        rv.extend(recovery.ravel())
    R = sparse.coo_matrix((rv, (rr, rc)), shape=(3*nc, nz)).tocsr()
    gr, gc, gv = [], [], []
    for cells, sign in [(trace.face_owner, 1.), (trace.face_neigh, -1.)]:
        for mode_col in range(3):
            modes = trace.face_mode_ids[:, mode_col]
            valid = modes >= 0
            c, modes = cells[valid], modes[valid]
            val = trace.face_mode_values[valid, mode_col] * trace.voxel_size**2 / trace.cell_volume[c]
            normals = sign * trace.face_normal[valid]
            for component in range(3):
                for axis in range(3):
                    gr.append(9*c + 3*component + axis)
                    gc.append(3*modes + component)
                    gv.append(val * normals[:, axis])
    G = sparse.coo_matrix((np.concatenate(gv), (np.concatenate(gr), np.concatenate(gc))),
                          shape=(9*nc, nz)).tocsr()
    G.eliminate_zeros()
    d_from_g = (G[9*np.arange(nc)] + G[9*np.arange(nc)+4] + G[9*np.arange(nc)+8]).multiply(trace.cell_volume[:, None])
    delta = (d_from_g - system.divergence_matrix).tocsr()
    defect = float(np.max(np.abs(delta.data), initial=0))
    assert defect < 1e-11
    return R, G, defect


def point_operator(xyz, owners, trace, R, G):
    delta = xyz - trace.cell_centroid[owners]
    delta = hvt._minimum_image_x(delta, trace.domain_length_x, trace.periodic_x)
    rows = (3*owners[:, None] + np.arange(3)).ravel()
    H = R[rows].copy()
    for axis in range(3):
        grows = (9*owners[:, None] + 3*np.arange(3) + axis).ravel()
        H = H + G[grows].multiply(np.repeat(delta[:, axis], 3)[:, None])
    H = H.tocsr()
    H.eliminate_zeros()
    return H


def normerr(value, reference, weight=None):
    d = (value - reference)**2
    q = reference**2
    if weight is not None:
        if d.ndim > 1:
            weight = weight[:, None]
        d, q = weight*d, weight*q
    return float(np.sqrt(d.sum() / max(q.sum(), 1e-30)))


def face_flux(z, trace):
    zv = z.reshape(-1, 3)
    ids = trace.face_mode_ids
    field = (zv[np.maximum(ids, 0)] * (trace.face_mode_values*(ids >= 0))[:, :, None]).sum(axis=1)
    return np.einsum("ij,ij->i", field, trace.face_normal) * trace.voxel_size**2

