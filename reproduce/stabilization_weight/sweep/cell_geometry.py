"""cell_geometry: per-cell geometry a user can measure without a reference solution, for c1..c6.

Written because the `aspect` diagnostic recorded by run_sweep.py is unusable: it took the second-moment
tensor of the cell's voxel CENTRES, which is exactly singular for a one-voxel cell (and c1..c6 have
many), so the regularised ratio blew up to ~1e15.  NO reported accuracy number depends on it: aspect
is a descriptive column only, and run_sweep.py is unchanged.  This script recomputes it correctly and adds
the other a-priori quantities.  It performs no solve and reads nothing but the partition.

DEFINITIONS (fixed before running)
  H_i^V   = V_i^(1/3), V_i the cell volume (voxel count times h^3).
  H_i^F   = h sqrt(n_facelet,i / 6), n_facelet,i the incidences of cell i (interior + wall facelets),
            i.e. the count the stabilization sum runs over.
  aspect_i= sqrt(lambda_max / lambda_min) of M_i = (1/n) sum_v (x_v - c_i)(x_v - c_i)^T + (h^2/12) I,
            the voxel-resolved second moment of the cell (the h^2/12 term is the exact contribution of
            a voxel's own extent, so a one-voxel cell has aspect exactly 1).
  wall_fraction_i = wall facelets / all facelets of cell i.
Percentiles are unweighted over cells.

  python -B cell_geometry.py
"""
import sys, os, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = "../../../porevoronoi_fv"
sys.path.insert(0, HERE); sys.path.insert(0, PACKAGE_DIR); sys.path.insert(0, os.path.join(PACKAGE_DIR, "assisted"))
import config_io
config_io.thread_env(2)
import numpy as np
import assisted_arms, case_loader

OUT = os.path.join(HERE, "../../../outputs/stabilization_weight/sweep")


def pct(a, n):
    a = np.asarray(a, np.float64)
    return {n + "_min": float(a.min()), n + "_p10": float(np.percentile(a, 10)),
            n + "_med": float(np.median(a)), n + "_p90": float(np.percentile(a, 90)),
            n + "_max": float(a.max()), n + "_mean": float(a.mean())}


def main():
    cfg = assisted_arms.load_cfg()
    res = {}
    for c in ["c1", "c2", "c3", "c4", "c5", "c6"]:
        case = case_loader.load_case(cfg, c, with_state=False)
        rec, seeds, part = assisted_arms.published_partition(cfg, case)
        tr, geom = part["trace"], part["geom"]
        h = float(tr.voxel_size)
        nc = int(tr.n_cells)
        V = np.asarray(tr.cell_volume, np.float64)
        inc = np.concatenate([tr.face_owner, tr.face_neigh, tr.wall_cell])
        nfac = np.bincount(inc, minlength=nc).astype(np.float64)
        nwall = np.bincount(tr.wall_cell, minlength=nc).astype(np.float64)
        pore = np.asarray(case.pore, np.int64)
        own = np.asarray(geom.labels).ravel()[pore]
        zyx = np.column_stack(np.unravel_index(pore, case.shape)).astype(np.float64)
        xyz = (zyx[:, ::-1] + 0.5) * h
        cnt = np.bincount(own, minlength=nc).astype(np.float64)
        mean = np.column_stack([np.bincount(own, weights=xyz[:, a], minlength=nc) / np.maximum(cnt, 1)
                                for a in range(3)])
        d = xyz - mean[own]
        M = np.zeros((nc, 3, 3))
        for a in range(3):
            for b in range(3):
                M[:, a, b] = (np.bincount(own, weights=d[:, a] * d[:, b], minlength=nc)
                              / np.maximum(cnt, 1))
        M += (h * h / 12.0) * np.eye(3)
        ev = np.linalg.eigvalsh(M)
        aspect = np.sqrt(ev[:, 2] / ev[:, 0])
        row = dict(case=c, tag=case.tag, n_cells=nc, h=h,
                   pore_voxels=int(pore.size), mean_voxels_per_cell=float(pore.size / nc),
                   one_voxel_cells=int((V / h ** 3 < 1.5).sum()),
                   frac_cells_Hv_over_h_below_2=float((V ** (1 / 3) / h < 2).mean()))
        row.update(pct(V ** (1 / 3) / h, "Hv_over_h"))
        row.update(pct(h * np.sqrt(nfac / 6.0) / h, "Hf_over_h"))
        row.update(pct(nfac, "nfacelet"))
        row.update(pct(aspect, "aspect"))
        row.update(pct(nwall / np.maximum(nfac, 1), "wall_fraction"))
        res[c] = row
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in row.items()}), flush=True)
    config_io.save_json(os.path.join(OUT, "geom.json"), res)


if __name__ == "__main__":
    main()
