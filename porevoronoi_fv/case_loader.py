"""Paper-case loader: mask, reference field, trajectory-window records, archived connected-P1 state.

Coordinate conventions:
  * track positions in the window CSV are voxel-centre-at-integer;
  * site rule of the paper: rounded = rint([z, y, x_mod]); z and y clipped to the image; x taken modulo the
    image length W; flat = ravel_multi_index; sites are the sorted unique flats; the snap displacement must be
    zero (run record);
  * physical positions used by the observation operator put the voxel centre at (index + 0.5) h, as
    pore_ownership.geometry does, so xyz = (x_mod + 0.5, y + 0.5, z + 0.5) h with x wrapped modulo L.
The loader asserts that floor(xyz / h) equals the site-rule voxel for every record.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import gzip
import json
from types import SimpleNamespace

import numpy as np

import config_io

WINDOW_COLUMNS = ("frame_id", "particle_id", "x_mod_voxel", "y_voxel", "z_voxel",
                  "ux_dns_frame_interp", "uy_dns_frame_interp", "uz_dns_frame_interp")


def read_window(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as g:
        header = g.readline().strip().split(",")
    col = {name: i for i, name in enumerate(header)}
    missing = [c for c in WINDOW_COLUMNS if c not in col]
    if missing:
        raise ValueError(f"window lacks columns {missing}")
    arr = np.loadtxt(path, delimiter=",", skiprows=1, usecols=[col[c] for c in WINDOW_COLUMNS], dtype=np.float64,
                     ndmin=2)
    frame = arr[:, 0]
    particle = arr[:, 1]
    if not (np.all(frame == np.rint(frame)) and np.all(particle == np.rint(particle))):
        raise ValueError("non-integer frame or particle id")
    return dict(frame=frame.astype(np.int64), particle=particle.astype(np.int64),
                zyx=np.column_stack([arr[:, 4], arr[:, 3], arr[:, 2]]),
                vel=np.ascontiguousarray(arr[:, 5:8]), row=np.arange(arr.shape[0], dtype=np.int64))


def site_voxels(zyx, shape):
    """The paper's per-record voxel rule: rint, clip z/y, periodic x."""
    rounded = np.rint(zyx).astype(np.int64)
    mapped = rounded.copy()
    mapped[:, 0] = np.clip(mapped[:, 0], 0, shape[0] - 1)
    mapped[:, 1] = np.clip(mapped[:, 1], 0, shape[1] - 1)
    mapped[:, 2] = np.mod(mapped[:, 2], shape[2])
    clipped = int(np.sum(np.any(rounded[:, :2] != mapped[:, :2], axis=1)))
    wrapped = int(np.sum(rounded[:, 2] != mapped[:, 2]))
    return np.ravel_multi_index(mapped.T, shape).astype(np.int64), mapped, clipped, wrapped


def load_case(cfg, code, with_state=True):
    paths = config_io.case_paths(cfg, code)
    tag = cfg["cases"][code]["tag"]
    man = config_io.load_json(paths["manifest"])
    for key, name in (("reference", "reference_npz"), ("window", "particle_window")):
        got = config_io.sha(paths[key])
        want = man["input_sha256"][name]
        if got != want:
            raise RuntimeError(f"{code}: {key} sha256 {got} != manifest {want}")
    ref = np.load(paths["reference"], allow_pickle=False)
    mask = np.asarray(ref["mask"], dtype=bool)
    shape = mask.shape
    h = float(cfg["h"])
    pore = np.flatnonzero(mask.ravel())
    labels_ref = np.asarray(ref["labels"], dtype=np.int64)
    if not np.array_equal(labels_ref.ravel()[pore], np.arange(pore.size)):
        raise RuntimeError(f"{code}: reference labels are not the raster pore-voxel identity")
    U = np.asarray(ref["U"], dtype=np.float64)
    reference = dict(labels=labels_ref, U=U, phi=np.asarray(ref["phi"], dtype=np.float64),
                     owner=np.asarray(ref["owner"], dtype=np.int64), neigh=np.asarray(ref["neigh"], dtype=np.int64),
                     dvec=np.asarray(ref["dvec"], dtype=np.float64), volume=np.asarray(ref["volume"], dtype=np.float64),
                     K_eff_x=float(ref["K_eff_x"]), metadata=json.loads(str(ref["metadata"])))
    w = read_window(paths["window"])
    flat, mapped, clipped, wrapped = site_voxels(w["zyx"], shape)
    solid = int(np.sum(~mask.ravel()[flat]))
    if solid:
        raise RuntimeError(f"{code}: {solid} records round into solid (the paper requires zero snap)")
    L = shape[2] * h
    xyz = np.column_stack([w["zyx"][:, 2] + 0.5, w["zyx"][:, 1] + 0.5, w["zyx"][:, 0] + 0.5]) * h
    xyz[:, 0] = np.mod(xyz[:, 0], L)
    floor_zyx = np.floor(xyz[:, ::-1] / h).astype(np.int64)
    floor_ok = np.all((floor_zyx >= 0) & (floor_zyx < np.asarray(shape)), axis=1)
    floor_flat = np.full(flat.shape, -1, np.int64)
    floor_flat[floor_ok] = np.ravel_multi_index(floor_zyx[floor_ok].T, shape)
    convention_mismatch = int(np.sum(floor_flat != flat))
    records = dict(w, flat=flat, xyz=xyz, clipped_rows=clipped, wrapped_rows=wrapped,
                   convention_mismatch=convention_mismatch)
    case = SimpleNamespace(code=code, tag=tag, paths=paths, manifest=man, mask=mask, shape=shape, h=h,
                           nu=float(cfg["nu"]), force=np.asarray(cfg["force"], dtype=np.float64),
                           periodic_x=bool(cfg["periodic_x"]), pore=pore, reference=reference,
                           u_ref_vox=U[labels_ref.ravel()[pore]], records=records, length_x=L)
    case.k_ref = reference["K_eff_x"]
    case.velocity = lambda p: voxel_velocity(case, p)
    if with_state:
        st = np.load(paths["state"], allow_pickle=False)
        case.state = {k: np.asarray(st[k]) for k in st.files}
        case.state_sha256 = config_io.sha(paths["state"])
    return case


def voxel_velocity(case, xyz):
    """Nearest-voxel reference velocity at physical points (the window builder's 'interp' lookup)."""
    p = np.asarray(xyz, dtype=np.float64).copy()
    p[..., 0] = np.mod(p[..., 0], case.length_x)
    ijk = np.floor(p[..., ::-1] / case.h).astype(np.int64)
    flat = np.ravel_multi_index(ijk.reshape(-1, 3).T, case.shape)
    if np.any(~case.mask.ravel()[flat]):
        raise ValueError("point in solid")
    lab = case.reference["labels"].ravel()[flat]
    return case.reference["U"][lab].reshape(p.shape)


def select_records(case, spec):
    """Record subset of a partition spec: 'pub' (all), 'pfx:K' (particle ids 0..K-1, the paper runner's
    --particle-ids 0:K-1 inclusive convention), 'frm:F' (single frame F); terms joined with '+' intersect."""
    r = case.records
    keep = np.ones(r["row"].size, dtype=bool)
    for term in str(spec).split("+"):
        term = term.strip()
        if term in ("pub", ""):
            continue
        kind, _, value = term.partition(":")
        if kind == "pfx":
            keep &= r["particle"] < int(value)
        elif kind == "frm":
            keep &= r["frame"] == int(value)
        else:
            raise ValueError(f"unknown partition term {term!r}")
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        raise ValueError(f"partition {spec!r} selects no record")
    return idx
