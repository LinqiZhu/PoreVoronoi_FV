from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import platform
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage


CODE_DIR = Path(__file__).resolve().parent
SHARED_CODE_DIR = CODE_DIR.parents[1] / "code"
for path in (CODE_DIR, SHARED_CODE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ptv_ownership_gpu6 as backend  # noqa: E402
from hybrid_site_sources import load_particle_window_seed_flat  # noqa: E402


SIX_CONNECTED_STRUCTURE = np.zeros((3, 3, 3), dtype=np.uint8)
SIX_CONNECTED_STRUCTURE[1, 1, 1] = 1
for offset in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
    SIX_CONNECTED_STRUCTURE[1 + offset[0], 1 + offset[1], 1 + offset[2]] = 1
    SIX_CONNECTED_STRUCTURE[1 - offset[0], 1 - offset[1], 1 - offset[2]] = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpu_lexicographic_dijkstra(mask: np.ndarray, seeds_zyx: np.ndarray):
    D, H, W = (int(value) for value in mask.shape)
    nvox = int(mask.size)
    distance = np.full(nvox, np.iinfo(np.int32).max, dtype=np.int32)
    label = np.full(nvox, np.iinfo(np.int32).max, dtype=np.int32)
    queue: list[tuple[int, int, int]] = []
    for owner, seed in enumerate(seeds_zyx):
        idx = int(np.ravel_multi_index(tuple(seed), mask.shape))
        distance[idx] = 0
        label[idx] = owner
        heapq.heappush(queue, (0, owner, idx))

    while queue:
        current_distance, current_label, idx = heapq.heappop(queue)
        if current_distance != int(distance[idx]) or current_label != int(label[idx]):
            continue
        z = idx // (H * W)
        remainder = idx - z * H * W
        y = remainder // W
        x = remainder - y * W
        for nz, ny, nx in (
            (z - 1, y, x),
            (z + 1, y, x),
            (z, y - 1, x),
            (z, y + 1, x),
            (z, y, x - 1),
            (z, y, x + 1),
        ):
            if not (0 <= nz < D and 0 <= ny < H and 0 <= nx < W):
                continue
            if not mask[nz, ny, nx]:
                continue
            neighbor = int((nz * H + ny) * W + nx)
            candidate = (current_distance + 1, current_label)
            if candidate < (int(distance[neighbor]), int(label[neighbor])):
                distance[neighbor] = candidate[0]
                label[neighbor] = candidate[1]
                heapq.heappush(queue, (candidate[0], candidate[1], neighbor))
    return label.reshape(mask.shape), distance.reshape(mask.shape)


def _component_safe_sites(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    components, count = ndimage.label(mask, structure=SIX_CONNECTED_STRUCTURE)
    seed_flat: list[int] = []
    for component in range(1, count + 1):
        members = np.flatnonzero(components.ravel() == component)
        seed_flat.append(int(rng.choice(members)))
    pore = np.flatnonzero(mask.ravel())
    extra_count = min(int(pore.size), int(rng.integers(1, 12)))
    seed_flat.extend(int(value) for value in rng.choice(pore, size=extra_count, replace=False))
    unique_flat = np.asarray(sorted(set(seed_flat)), dtype=np.int64)
    return np.column_stack(np.unravel_index(unique_flat, mask.shape)).astype(np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GPU six-neighbour ownership against an independent CPU heap Dijkstra."
    )
    parser.add_argument("--random-cases", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--ptv-record",
        type=Path,
        action="append",
        default=[],
        help="Prior PTV audit JSON supplying the frozen mask, particle file, and selection.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.random_cases < 1:
        raise ValueError("--random-cases must be positive")

    import cupy as cp

    rng = np.random.default_rng(args.seed)
    totals = {
        "exact_gpu_propagation_label_mismatches": 0,
        "exact_gpu_propagation_distance_mismatches": 0,
        "roi_jfa_label_mismatches": 0,
        "roi_jfa_distance_mismatches": 0,
    }
    max_roi_fraction = 0.0

    for _ in range(args.random_cases):
        mask = rng.random((7, 8, 9)) > 0.40
        if not np.any(mask):
            mask[3, 4, 4] = True
        seeds = _component_safe_sites(mask, rng)
        cpu_label, cpu_distance = _cpu_lexicographic_dijkstra(mask, seeds)
        mask_cp = cp.asarray(mask, dtype=cp.uint8)
        seeds_cp = cp.asarray(seeds, dtype=cp.int32)
        pore = mask

        exact_label, exact_distance = backend.exact_frontier_dijkstra_gpu_6(
            mask_cp, seeds_cp, validate=True
        )
        exact_label_np = cp.asnumpy(exact_label)
        exact_distance_np = cp.asnumpy(exact_distance)
        totals["exact_gpu_propagation_label_mismatches"] += int(
            np.count_nonzero(exact_label_np[pore] != cpu_label[pore])
        )
        totals["exact_gpu_propagation_distance_mismatches"] += int(
            np.count_nonzero(exact_distance_np[pore] != cpu_distance[pore])
        )

        line_bits = backend.prepare_mask_line_bits64(mask_cp)
        label, distance, stats = backend.certified_roi_jfa_gpu_6(
            mask_cp,
            seeds_cp,
            validate=True,
            return_stats=True,
            certificate_line_bits=line_bits,
            pore_voxel_count=int(np.count_nonzero(mask)),
        )
        label_np = cp.asnumpy(label)
        distance_np = cp.asnumpy(distance)
        totals["roi_jfa_label_mismatches"] += int(
            np.count_nonzero(label_np[pore] != cpu_label[pore])
        )
        totals["roi_jfa_distance_mismatches"] += int(
            np.count_nonzero(distance_np[pore] != cpu_distance[pore])
        )
        max_roi_fraction = max(max_roi_fraction, float(stats["roi_fraction"]))

    ptv_cases: list[dict[str, object]] = []
    for record_path in args.ptv_record:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        input_record = record["input"]
        mask_path = Path(input_record["mask_npz"])
        particle_path = Path(input_record["particle_window"])
        with np.load(mask_path, allow_pickle=False) as data:
            mask = np.asarray(data["mask"], dtype=bool)
        frame_selection = str(input_record.get("particle_frames", ""))
        particle_selection = str(input_record.get("particle_ids", ""))
        if frame_selection.strip().lower() == "all":
            frame_selection = ""
        if particle_selection.strip().lower() == "all":
            particle_selection = ""
        seed_flat, metadata = load_particle_window_seed_flat(
            mask,
            particle_path,
            frame_selection=frame_selection,
            particle_id_selection=particle_selection,
        )
        seeds = np.column_stack(np.unravel_index(seed_flat, mask.shape)).astype(np.int32)
        cpu_label, cpu_distance = _cpu_lexicographic_dijkstra(mask, seeds)
        mask_cp = cp.asarray(mask, dtype=cp.uint8)
        seeds_cp = cp.asarray(seeds, dtype=cp.int32)
        case_result: dict[str, object] = {
            "source_record": str(record_path.resolve()),
            "particle_ids": metadata["particle_ids"],
            "particle_rows_selected": int(metadata["particle_rows_selected"]),
            "particle_site_count": int(seed_flat.size),
            "pore_voxels": int(np.count_nonzero(mask)),
        }
        exact_label, exact_distance = backend.exact_frontier_dijkstra_gpu_6(
            mask_cp, seeds_cp, validate=True
        )
        exact_label_np = cp.asnumpy(exact_label)
        exact_distance_np = cp.asnumpy(exact_distance)
        case_result["exact_gpu_propagation_label_mismatches"] = int(
            np.count_nonzero(exact_label_np[mask] != cpu_label[mask])
        )
        case_result["exact_gpu_propagation_distance_mismatches"] = int(
            np.count_nonzero(exact_distance_np[mask] != cpu_distance[mask])
        )
        line_bits = backend.prepare_mask_line_bits64(mask_cp)
        label, distance, stats = backend.certified_roi_jfa_gpu_6(
            mask_cp,
            seeds_cp,
            validate=True,
            return_stats=True,
            certificate_line_bits=line_bits,
            pore_voxel_count=int(np.count_nonzero(mask)),
        )
        label_np = cp.asnumpy(label)
        distance_np = cp.asnumpy(distance)
        case_result["roi_jfa_label_mismatches"] = int(
            np.count_nonzero(label_np[mask] != cpu_label[mask])
        )
        case_result["roi_jfa_distance_mismatches"] = int(
            np.count_nonzero(distance_np[mask] != cpu_distance[mask])
        )
        case_result["roi_jfa_voxels"] = int(stats["roi_voxels"])
        case_result["roi_jfa_fixed_point_residuals"] = int(
            stats["fixed_point_residuals"]
        )
        ptv_cases.append(case_result)

    ptv_mismatch_fields = [
        value
        for case in ptv_cases
        for key, value in case.items()
        if key.endswith("_mismatches") or key.endswith("_residuals")
    ]
    passed = all(value == 0 for value in totals.values()) and all(
        int(value) == 0 for value in ptv_mismatch_fields
    )
    module_path = CODE_DIR / "ptv_ownership_gpu6.py"
    payload = {
        "protocol": {
            "reference": "independent CPU heap Dijkstra on the unit six-neighbour pore graph",
            "gpu_backends": [
                "exact six-neighbour GPU propagation",
                "ROI-JFA",
            ],
            "graph_connectivity": 6,
            "diagonal_moves": False,
            "tie_rule": "minimum prescribed-site id at equal distance",
            "random_mask_shape": [7, 8, 9],
            "random_pore_probability": 0.60,
            "at_least_one_site_per_six_connected_component": True,
            "random_cases": int(args.random_cases),
            "random_seed": int(args.seed),
        },
        "result": {
            "passed": bool(passed),
            **totals,
            "maximum_roi_fraction": float(max_roi_fraction),
            "ptv_cases": ptv_cases,
        },
        "code_sha256": {
            "ptv_ownership_gpu6.py": _sha256(module_path),
            "verify_ptv_ownership_gpu6.py": _sha256(Path(__file__)),
            "hybrid_site_sources.py": _sha256(SHARED_CODE_DIR / "hybrid_site_sources.py"),
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "cupy": cp.__version__,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["result"], indent=2, sort_keys=True))
    print(args.out)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
