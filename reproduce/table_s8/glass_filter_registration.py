"""Can the archive's simulated velocity fields be registered to its released mask?

The measured X-ray PTV archive ships a segmented pore mask, a tracking output,
and three resliced simulated velocity components.  If the fields could be placed
on the mask, the reconstruction built from the tracked particles could be scored
against a reference field on measured geometry, which is the strongest
validation the dataset could support.  This script asks whether they can, and
records the answer as an artefact rather than an assertion.

The test is a one-dimensional porosity-profile cross-correlation, run per axis.
It is chosen because it is cheap, it is invariant to everything except the
alignment being tested, and because it isolates *which* axis fails rather than
returning a single pass or fail.  Three progressively weaker hypotheses are
tried, and each is given a null control so that a correlation can be read
against something:

1. rigid alignment: every integer offset, with and without reflection;
2. uniform rescaling: every start and length in the mask, resampled to the
   field's length, with and without reflection;
3. the null control: the identical search run against a mask axis the field
   axis should not match.

A hypothesis is accepted only if it beats its own control by a clear margin.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tifffile

DATASET = {
    "name": "porous_glass",
    "root": Path("data/glass_filter"),
    "mask_raw": "2_segmentedImage/000_6_HQ_sample2_poreMask.raw",
    "mask_shape": (656, 657, 657),
    "fields": ("3_simulatedVelocityFields/Ufx_resliced.tif",
               "3_simulatedVelocityFields/Ufy_resliced.tif",
               "3_simulatedVelocityFields/Ufz_resliced.tif"),
    "mask_header_crop_zyx": ((20, 519), (160, 498), (175, 498)),
    "citation": "Bultreys et al. 2022, Zenodo 10.5281/zenodo.6010490",
}
AXIS_NAMES = ("z", "y", "x")


def normalized(v):
    centred = v - v.mean()
    return centred, float(np.linalg.norm(centred))


def best_rigid(field_profile, mask_profile):
    """Best correlation over every integer offset and both reflections."""
    n, N = len(field_profile), len(mask_profile)
    if n > N:
        return None
    best = None
    for reflected in (False, True):
        f = field_profile[::-1] if reflected else field_profile
        fc, fn = normalized(f)
        for offset in range(N - n + 1):
            wc, wn = normalized(mask_profile[offset:offset + n])
            if fn * wn <= 0:
                continue
            r = float(fc @ wc / (fn * wn))
            if best is None or r > best["r"]:
                best = {"r": r, "offset": int(offset), "reflected": bool(reflected)}
    return best


def best_rescaled(field_profile, mask_profile, start_step=1, length_step=1,
                  min_length=200):
    """Best correlation allowing an arbitrary uniform resampling of the axis."""
    n, N = len(field_profile), len(mask_profile)
    grid = np.arange(N, dtype=np.float64)
    best = None
    for reflected in (False, True):
        f = field_profile[::-1] if reflected else field_profile
        fc, fn = normalized(f)
        for start in range(0, min(300, N - min_length), start_step):
            for length in range(min_length, N - start, length_step):
                sample = np.interp(np.linspace(start, start + length - 1, n),
                                   grid, mask_profile)
                wc, wn = normalized(sample)
                if fn * wn <= 0:
                    continue
                r = float(fc @ wc / (fn * wn))
                if best is None or r > best["r"]:
                    best = {"r": r, "start": int(start), "length": int(length),
                            "reflected": bool(reflected)}
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reproduce/table_s8")
    parser.add_argument("--accept-margin", type=float, default=0.20,
                        help="a hypothesis must beat its null control by this much")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = DATASET

    raw = spec["root"] / spec["mask_raw"]
    expected = int(np.prod(spec["mask_shape"]))
    if raw.stat().st_size != expected:
        raise RuntimeError("released mask is %d bytes, not %d"
                           % (raw.stat().st_size, expected))
    mask = np.fromfile(raw, dtype=np.uint8).reshape(spec["mask_shape"]) == 255

    magnitude = None
    for relative in spec["fields"]:
        component = np.asarray(tifffile.memmap(spec["root"] / relative),
                               dtype=np.float32)
        magnitude = np.abs(component) if magnitude is None else magnitude + np.abs(component)
        del component
    fluid = magnitude > 0
    del magnitude

    (z0, z1), (y0, y1), (x0, x1) = spec["mask_header_crop_zyx"]
    crop = mask[z0:z1, y0:y1, x0:x1]
    overlap = float((crop & fluid).sum()) / max(float(fluid.sum()), 1.0)
    record = {
        "protocol_id": "pvfv_glass_filter_registration",
        "citation": spec["citation"],
        "mask_shape_zyx": list(spec["mask_shape"]),
        "mask_bytes": int(raw.stat().st_size),
        "field_shape_zyx": [int(v) for v in fluid.shape],
        "mask_header_crop_zyx": [list(pair) for pair in spec["mask_header_crop_zyx"]],
        "crop_shape_matches_field": bool(crop.shape == fluid.shape),
        "mask_crop_porosity": float(crop.mean()),
        "field_fluid_fraction": float(fluid.mean()),
        "voxelwise_overlap_ratio_over_chance": overlap / max(float(crop.mean()), 1e-300),
    }

    mask_profiles = [mask.mean(axis=tuple(j for j in range(3) if j != i))
                     for i in range(3)]
    field_profiles = [fluid.mean(axis=tuple(j for j in range(3) if j != i))
                      for i in range(3)]
    del mask, fluid, crop

    per_axis = []
    for fi, field_profile in enumerate(field_profiles):
        rigid = [
            dict(best_rigid(field_profile, mask_profiles[mi]) or {},
                 mask_axis=AXIS_NAMES[mi])
            for mi in range(3) if len(field_profile) <= len(mask_profiles[mi])
        ]
        rigid.sort(key=lambda d: -d["r"])
        entry = {"field_axis": fi, "field_length": int(len(field_profile)),
                 "rigid_best": rigid[0], "rigid_all": rigid}
        # the matching mask axis is the best rigid one; the control is the next
        matched = AXIS_NAMES.index(rigid[0]["mask_axis"])
        control = AXIS_NAMES.index(rigid[1]["mask_axis"]) if len(rigid) > 1 else None
        entry["rescaled_best"] = best_rescaled(field_profile, mask_profiles[matched])
        entry["rescaled_best"]["mask_axis"] = AXIS_NAMES[matched]
        if control is not None:
            entry["rescaled_control"] = best_rescaled(field_profile,
                                                      mask_profiles[control])
            entry["rescaled_control"]["mask_axis"] = AXIS_NAMES[control]
            entry["margin_over_control"] = (
                entry["rescaled_best"]["r"] - entry["rescaled_control"]["r"])
            entry["registers"] = bool(entry["margin_over_control"] >= args.accept_margin)
        per_axis.append(entry)
        print("[field axis %d, n=%d] rigid best r=%+.4f on mask %s; rescaled r=%+.4f "
              "vs control r=%+.4f -> %s"
              % (fi, len(field_profile), entry["rigid_best"]["r"],
                 entry["rigid_best"]["mask_axis"], entry["rescaled_best"]["r"],
                 entry.get("rescaled_control", {}).get("r", float("nan")),
                 "REGISTERS" if entry.get("registers") else "does not register"),
              flush=True)

    record["per_axis"] = per_axis
    record["axes_that_register"] = int(sum(1 for e in per_axis if e.get("registers")))
    record["all_axes_register"] = bool(record["axes_that_register"] == 3)
    record["accept_margin"] = args.accept_margin
    record["conclusion"] = (
        "the released simulated fields can be placed on the released mask"
        if record["all_axes_register"] else
        "the released simulated fields cannot be placed on the released mask; "
        "%d of 3 axes register" % record["axes_that_register"]
    )
    path = out / "xptv_field_registration.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("[write] " + str(path))
    print(record["conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
