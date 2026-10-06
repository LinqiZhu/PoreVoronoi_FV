"""Reproduce the ROI--JFA proposal / certification / closure fields on the CPU.

The GPU implementation (gpu/ownership/code/ptv_ownership_gpu6.py) requires CuPy and a
CUDA device.  This module re-implements the same three stages in NumPy so that the
spatial fields behind Figure 3(a)-(c) can be drawn on a CPU:

  1. free-space lexicographic lower pair  zbar = lexmin_i (h*||p-s_i||_1, kappa_i)
     via three separable one-dimensional min-plus passes (Eq. main-l1-lower-pair);
  2. the sound axis-monotone pore-path certificate: the six straight axis-order
     paths followed by the six greedy monotone walks, exactly as in the GPU
     kernel ptv_bitset_compact_path_certificate6_64;
  3. exact single-edge relaxation restricted to the unresolved region with the
     certified pairs held fixed (Eq. roi-unresolved-update).

Full-domain propagation is computed independently as the exact reference and the
three verification measures are evaluated:
  n_L (owner mismatches), e_D^inf (max distance error), n_imp (improving directed edges).

The certified / unresolved voxel counts produced here are compared against the GPU
timing records of Table 2 (reproduce/table_02/records/); agreement shows that the drawn
fields are the same fields the paper reports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHARED_CODE = ROOT / "../gpu" / "code"
if str(SHARED_CODE) not in sys.path:
    sys.path.insert(0, str(SHARED_CODE))

from hybrid_site_sources import load_particle_window_seed_flat  # noqa: E402

MASK_NPZ = Path(
    "data/berea64/mask.npz"
)
PARTICLE_WINDOW = Path(
    "data/berea64/particle_tracks.csv.gz"
)

# The five nested trajectory prefixes of Table 2 (particle ids 0:N-1, all frames).
PREFIXES = [10, 50, 100, 200, 500]

INF = np.int64(1) << np.int64(40)


def load_mask() -> np.ndarray:
    with np.load(MASK_NPZ, allow_pickle=False) as data:
        mask = np.asarray(data["mask"], dtype=bool)
    if mask.ndim != 3:
        raise ValueError("expected a 3-D mask, got %s" % (mask.shape,))
    return mask


def load_sites(mask: np.ndarray, particle_count: int) -> np.ndarray:
    """Sorted unique flat site ids for the first `particle_count` particles."""
    seed_flat, _meta = load_particle_window_seed_flat(
        mask,
        PARTICLE_WINDOW,
        particle_id_selection="0:%d" % (particle_count - 1),
        x_boundary_mode="periodic_wrap",
    )
    return np.asarray(seed_flat, dtype=np.int64)


# ------------------------------------------------------------------ stage 1
def free_space_lower_pair(shape, seeds_zyx: np.ndarray, n_sites: int) -> np.ndarray:
    """Exact free-space lexicographic lower pair, packed as distance * n_sites + key.

    The packing is order preserving for the lexicographic pair because every key is
    strictly below n_sites, so a minimum over packed values is a lexicographic minimum
    over (distance, key).  Adding a constant c to the distance adds c * n_sites to the
    pack, which is why the separable min-plus scan stays exact: inside one scan every
    competing site shares the same remaining-axis displacement, so a common constant is
    added to all of them and the lexicographic order is preserved.
    """
    ns = np.int64(n_sites)
    packed = np.full(shape, INF, dtype=np.int64)
    flat = np.ravel_multi_index(seeds_zyx.T, shape)
    packed.reshape(-1)[flat] = np.arange(n_sites, dtype=np.int64)  # distance 0, key i

    for axis in (2, 1, 0):
        view = np.moveaxis(packed, axis, -1)
        n = view.shape[-1]
        for i in range(1, n):
            np.minimum(view[..., i], view[..., i - 1] + ns, out=view[..., i])
        for i in range(n - 2, -1, -1):
            np.minimum(view[..., i], view[..., i + 1] + ns, out=view[..., i])
    return packed


# ------------------------------------------------------------------ stage 2
def _solid_prefix_sums(mask: np.ndarray):
    solid = (~mask).astype(np.int32)
    cz = np.zeros((mask.shape[0] + 1,) + mask.shape[1:], dtype=np.int32)
    cy = np.zeros((mask.shape[0], mask.shape[1] + 1, mask.shape[2]), dtype=np.int32)
    cx = np.zeros(mask.shape[:2] + (mask.shape[2] + 1,), dtype=np.int32)
    np.cumsum(solid, axis=0, out=cz[1:])
    np.cumsum(solid, axis=1, out=cy[:, 1:])
    np.cumsum(solid, axis=2, out=cx[:, :, 1:])
    return cz, cy, cx


def _segment_clear(cum, axis, z, y, x, target):
    """True where the closed axis-aligned segment to `target` contains no solid voxel."""
    if axis == 0:
        lo, hi = np.minimum(z, target), np.maximum(z, target)
        return (cum[hi + 1, y, x] - cum[lo, y, x]) == 0
    if axis == 1:
        lo, hi = np.minimum(y, target), np.maximum(y, target)
        return (cum[z, hi + 1, x] - cum[z, lo, x]) == 0
    lo, hi = np.minimum(x, target), np.maximum(x, target)
    return (cum[z, y, hi + 1] - cum[z, y, lo]) == 0


def axis_order_clear(cums, seed_zyx, point_zyx, order) -> np.ndarray:
    """Vectorised ptv_pore_bitset_axis_order_clear: three straight segments in `order`."""
    cz, cy, cx = cums
    z = seed_zyx[:, 0].copy()
    y = seed_zyx[:, 1].copy()
    x = seed_zyx[:, 2].copy()
    targets = (point_zyx[:, 0], point_zyx[:, 1], point_zyx[:, 2])
    ok = np.ones(z.shape[0], dtype=bool)
    for axis in order:
        cum = (cz, cy, cx)[axis]
        target = targets[axis]
        ok &= _segment_clear(cum, axis, z, y, x, target)
        if axis == 0:
            z = np.where(ok, target, z)
        elif axis == 1:
            y = np.where(ok, target, y)
        else:
            x = np.where(ok, target, x)
    return ok


def greedy_monotone_clear(mask, seed_zyx, point_zyx, order) -> np.ndarray:
    """Vectorised ptv_pore_bitset_greedy_monotone_clear for one axis preference order."""
    cur = seed_zyx.copy()
    tgt = point_zyx
    remaining = np.abs(tgt - cur).sum(axis=1)
    alive = np.ones(cur.shape[0], dtype=bool)
    max_steps = int(remaining.max()) if remaining.size else 0
    for step in range(max_steps):
        active = alive & (step < remaining)
        if not active.any():
            break
        idx = np.flatnonzero(active)
        moved = np.zeros(idx.size, dtype=bool)
        for axis in order:
            todo = np.flatnonzero(~moved)
            if todo.size == 0:
                break
            rows = idx[todo]
            delta = tgt[rows, axis] - cur[rows, axis]
            can = delta != 0
            if not can.any():
                continue
            cand = cur[rows].copy()
            cand[:, axis] += np.sign(delta)
            pore = np.zeros(rows.size, dtype=bool)
            pore[can] = mask[cand[can, 0], cand[can, 1], cand[can, 2]]
            take = can & pore
            if take.any():
                cur[rows[take]] = cand[take]
                moved[todo[take]] = True
        alive[idx[~moved]] = False
    return alive & np.all(cur == tgt, axis=1)


def certify(mask, pore_zyx, proposed_owner, seeds_zyx):
    """Sound axis-monotone pore-path certificate."""
    cums = _solid_prefix_sums(mask)
    seed_of_voxel = seeds_zyx[proposed_owner]
    orders = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

    certified = np.zeros(pore_zyx.shape[0], dtype=bool)
    for order in orders:
        todo = np.flatnonzero(~certified)
        if todo.size == 0:
            break
        certified[todo] = axis_order_clear(cums, seed_of_voxel[todo], pore_zyx[todo], order)
    straight_only = int(certified.sum())
    for order in orders:
        todo = np.flatnonzero(~certified)
        if todo.size == 0:
            break
        certified[todo] = greedy_monotone_clear(mask, seed_of_voxel[todo], pore_zyx[todo], order)
    return certified, straight_only


# ------------------------------------------------------------------ stage 3
def _shift_views(axis, sign):
    sl_dst = [slice(None)] * 3
    sl_src = [slice(None)] * 3
    if sign > 0:
        sl_dst[axis] = slice(1, None)
        sl_src[axis] = slice(0, -1)
    else:
        sl_dst[axis] = slice(0, -1)
        sl_src[axis] = slice(1, None)
    return tuple(sl_dst), tuple(sl_src)


def _neighbour_min(packed, mask, ns):
    """Lexicographic minimum over the six face neighbours of (D(u) + 1, L(u))."""
    out = np.full(packed.shape, INF, dtype=np.int64)
    src = np.where(mask, packed, INF)
    for axis in (0, 1, 2):
        for sign in (+1, -1):
            dst_sl, src_sl = _shift_views(axis, sign)
            shifted = np.full(packed.shape, INF, dtype=np.int64)
            shifted[dst_sl] = src[src_sl]
            np.minimum(out, shifted, out=out)
    finite = out < INF
    out[finite] += ns
    return out


def restricted_closure(mask, packed_lower, certified_mask, ns):
    """Fix the certified pairs and relax only the unresolved targets."""
    packed = np.where(certified_mask, packed_lower, INF).astype(np.int64)
    packed = np.where(mask, packed, INF)
    unresolved = mask & ~certified_mask
    iterations = 0
    while True:
        proposal = _neighbour_min(packed, mask, ns)
        updated = np.where(unresolved & (proposal < packed), proposal, packed)
        iterations += 1
        if np.array_equal(updated, packed):
            return packed, iterations
        packed = updated
        if iterations > 10000:
            raise RuntimeError("restricted closure did not converge")


def full_propagation(mask, seeds_zyx, ns):
    """Exact full-domain propagation: BFS distance with lexicographic owner ties."""
    packed = np.full(mask.shape, INF, dtype=np.int64)
    flat = np.ravel_multi_index(seeds_zyx.T, mask.shape)
    packed.reshape(-1)[flat] = np.arange(int(ns), dtype=np.int64)
    packed = np.where(mask, packed, INF)
    while True:
        proposal = _neighbour_min(packed, mask, ns)
        updated = np.where(mask & (proposal < packed), proposal, packed)
        if np.array_equal(updated, packed):
            return packed
        packed = updated


def improving_edges(mask, packed, ns) -> int:
    """Directed pore-graph edges whose single-edge proposal would still improve."""
    total = 0
    current = np.where(mask, packed, INF)
    for axis in (0, 1, 2):
        for sign in (+1, -1):
            dst_sl, src_sl = _shift_views(axis, sign)
            shifted = np.full(packed.shape, INF, dtype=np.int64)
            shifted[dst_sl] = current[src_sl]
            finite = shifted < INF
            proposal = np.where(finite, shifted + ns, INF)
            total += int(np.count_nonzero(mask & (proposal < current)))
    return total


def run_prefix(mask, particle_count):
    seed_flat = load_sites(mask, particle_count)
    ns = np.int64(seed_flat.size)
    seeds_zyx = np.stack(np.unravel_index(seed_flat, mask.shape), axis=1).astype(np.int64)

    packed_lower = free_space_lower_pair(mask.shape, seeds_zyx, int(ns))
    pore_zyx = np.argwhere(mask).astype(np.int64)
    proposed_owner = (packed_lower[mask] % ns).astype(np.int64)

    certified_flags, straight_only = certify(mask, pore_zyx, proposed_owner, seeds_zyx)
    certified_mask = np.zeros(mask.shape, dtype=bool)
    certified_mask[mask] = certified_flags

    packed_roi, iterations = restricted_closure(mask, packed_lower, certified_mask, ns)
    packed_exact = full_propagation(mask, seeds_zyx, ns)

    d_roi = np.where(mask, packed_roi // ns, -1)
    l_roi = np.where(mask, packed_roi % ns, -1)
    d_exact = np.where(mask, packed_exact // ns, -1)
    l_exact = np.where(mask, packed_exact % ns, -1)

    # Shortest-path depth of the full-domain problem: the number of fair relaxation
    # sweeps exact propagation must perform before its fixed point is reached.
    full_depth = int(d_exact[mask].max())

    n_l = int(np.count_nonzero(mask & (l_roi != l_exact)))
    e_d = int(np.abs(d_roi[mask] - d_exact[mask]).max())
    n_imp = improving_edges(mask, packed_roi, ns)

    pore_voxels = int(mask.sum())
    certified_voxels = int(certified_flags.sum())
    record = {
        "particle_count": int(particle_count),
        "site_count": int(ns),
        "pore_voxels": pore_voxels,
        "certified_voxels": certified_voxels,
        "roi_voxels": pore_voxels - certified_voxels,
        "certified_percent": 100.0 * certified_voxels / pore_voxels,
        "roi_percent": 100.0 * (pore_voxels - certified_voxels) / pore_voxels,
        "straight_path_certified_voxels": straight_only,
        "closure_sweeps": int(iterations),
        "full_propagation_depth": full_depth,
        "label_mismatch_voxels": n_l,
        "distance_mismatch_max": e_d,
        "improving_directed_edges": n_imp,
    }
    fields = {
        "seeds_zyx": seeds_zyx,
        "certified_mask": certified_mask,
        "d_roi": d_roi.astype(np.int32),
        "l_roi": l_roi.astype(np.int32),
        "d_exact": d_exact.astype(np.int32),
        "l_exact": l_exact.astype(np.int32),
        "lower_distance": np.where(mask, packed_lower // ns, -1).astype(np.int32),
        "lower_owner": np.where(mask, packed_lower % ns, -1).astype(np.int32),
        "n_sites": int(ns),
    }
    return record, fields


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "figure_03" / "."))
    parser.add_argument(
        "--field-prefix",
        type=int,
        default=50,
        help="particle prefix whose spatial fields are stored for the figure",
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = load_mask()
    print("mask %s  pore voxels %d" % (mask.shape, int(mask.sum())))

    summary = []
    for particle_count in PREFIXES:
        record, fields = run_prefix(mask, particle_count)
        summary.append(record)
        print(json.dumps(record))
        if particle_count == args.field_prefix:
            np.savez_compressed(
                out_dir / "roi_jfa_fields.npz", mask=mask,
                particle_count=particle_count, **fields
            )

    (out_dir / "roi_jfa_fields_check.json").write_text(
        json.dumps(
            {
                "rows": summary,
                "mask_npz": str(MASK_NPZ),
                "particle_window": str(PARTICLE_WINDOW),
            },
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
