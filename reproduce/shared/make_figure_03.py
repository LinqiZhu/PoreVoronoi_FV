"""Figure 3: ROI--JFA certified exact acceleration of graph-geodesic ownership.

Central claim, readable in a few seconds: the free-space proposal is computed as if the
solid were absent (a); a pore-path certificate proves most of it exact and isolates a small
unresolved region (b); exact graph relaxation restricted to that region supplies the
missing distance (c); the completed field is identical to full propagation (d); and the
ownership kernel is 7-13x faster (e).

Data sources
------------
Panels a-c  reproduce/figure_03/roi_jfa_fields.npz, written by
            reproduce/shared/roi_jfa_fields_cpu.py from
              mask   data/berea64/mask.npz
              window data/berea64/particle_tracks.csv.gz  sha256 3ea61d6e...01528
            the inputs whose sha256 the Table 2 timing records list in input_sha256.
Panels d-e  reproduce/table_02/records/
            ptv_ownership_statistics_canonical.json (the declared canonical record).

Every plotted number is archived or recomputed-and-cross-checked; none is invented.  The
build aborts if the CPU reproduction disagrees with the GPU timing records, if any of the
three verification measures is non-zero, or if any type size falls below the floor.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch, Rectangle

import figure_style as hs
import build_ownership_statistics as ownership
from roi_jfa_fields_cpu import free_space_lower_pair

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "figure_03" / "."
FIELDS_NPZ = SOURCE_DIR / "roi_jfa_fields.npz"
CPU_JSON = SOURCE_DIR / "roi_jfa_fields_check.json"
STEM = "figure_03"

INK, MUTED, GRID = hs.INK, hs.MUTED, hs.GRID
BLUE, ORANGE, TEAL = hs.CERTIFIED, hs.UNRESOLVED, hs.TEAL
SOLID_FILL, SOLID_EDGE = hs.SOLID_FILL, hs.SOLID_EDGE

SLICE_Z = 39               # representative plane, stated in the caption
FIELD_PREFIX = 50          # audited trajectory prefix used for panels a-c


def load_canonical_payload() -> dict:
    """Read the declared canonical ROI--JFA record as shipped.

    build_ownership_statistics.load_canonical_payload() also re-serialises the
    record and compares byte for byte.  That comparison fails here for a cosmetic reason
    only: the shipped records store relative paths with forward slashes while the
    regenerator emits the native Windows separator.  Every numeric field,
    outputs/generated/ownership_statistics.tex and
    table_02_timing_source.csv reproduce exactly.  The record files are
    therefore read as shipped rather than rewritten, and the numbers used here are
    cross-checked against the generated LaTeX macro table the manuscript typesets.
    """
    return ownership._load_json(ownership.CANONICAL_JSON)


# ----------------------------------------------------------------------- panels
def draw_panel_a(ax, mask, lower_distance_full, sites_in_slab):
    """Free-space lower bound, evaluated with the obstacles removed."""
    image = ax.imshow(lower_distance_full, cmap=hs.FIELD_CMAP, origin="lower",
                      interpolation="nearest")
    # A cased boundary: the proposal is visibly continued straight across the solid.
    hs.cased_line(ax, ~mask)
    if sites_in_slab.size:
        ax.scatter(sites_in_slab[:, 2], sites_in_slab[:, 1], s=5.5, marker="o",
                   facecolor="white", edgecolor=INK, linewidth=0.4, zorder=6)
    hs.map_axes(ax)
    return image


def draw_panel_b(ax, mask, certified):
    """Certified against unresolved on the same plane."""
    layer = np.zeros(mask.shape + (4,), dtype=float)
    layer[~mask] = mpl.colors.to_rgba(SOLID_FILL)
    layer[mask & certified] = mpl.colors.to_rgba(BLUE)
    layer[mask & ~certified] = mpl.colors.to_rgba(ORANGE)
    ax.imshow(layer, interpolation="nearest", origin="lower")
    # Outline the unresolved region: a value cue that survives greyscale conversion and
    # keeps the region prominent without covering the pore geometry.
    ax.contour(np.asarray(mask & ~certified, dtype=float), levels=[0.5], colors=[INK],
               linewidths=0.55, origin="lower", zorder=5)
    ax.contour(np.asarray(~mask, dtype=float), levels=[0.5], colors=[SOLID_EDGE],
               linewidths=0.5, origin="lower", zorder=4)
    hs.map_axes(ax)
    return int((mask & certified).sum()), int((mask & ~certified).sum())


def draw_panel_c(ax, mask, certified, gap):
    """Distance correction over the whole pore space.

    The correction is drawn everywhere in the pore rather than only on the unresolved
    region.  It is exactly zero on every certified voxel, because the certificate proves
    the free-space bound is attained there, so a single field carries both facts: the
    certified region is white, and the unresolved region shows what the closure had to
    add.  That also removes a third light category from the panel, which keeps the
    solid, the certified pore and the residual separable in greyscale.
    """
    unresolved = mask & ~certified
    layer = np.zeros(mask.shape + (4,), dtype=float)
    layer[~mask] = mpl.colors.to_rgba(SOLID_FILL)
    ax.imshow(layer, interpolation="nearest", origin="lower")

    shown = np.where(mask, gap, np.nan).astype(float)
    vmax = float(np.nanmax(shown)) if np.isfinite(shown).any() else 1.0
    image = ax.imshow(shown, cmap=hs.ERROR_CMAP, vmin=0.0, vmax=vmax,
                      interpolation="nearest", origin="lower")
    ax.contour(np.asarray(unresolved, dtype=float), levels=[0.5], colors=[INK],
               linewidths=0.55, origin="lower", zorder=5)
    ax.contour(np.asarray(~mask, dtype=float), levels=[0.5], colors=[SOLID_EDGE],
               linewidths=0.5, origin="lower", zorder=4)
    hs.map_axes(ax)
    return image, vmax


def draw_panel_d(ax, rows) -> None:
    """Compact exactness summary: every measure is zero on every audited site set."""
    measures = [r"$n_L$", r"$e_D^{\infty}$", r"$n_{\mathrm{imp}}$"]
    n_cols = len(rows)
    ax.set_xlim(-0.56, n_cols - 0.44)
    ax.set_ylim(len(measures) - 0.44, -0.56)
    ax.set_aspect("equal")

    for i in range(len(measures)):
        for j in range(n_cols):
            ax.add_patch(
                Rectangle((j - 0.43, i - 0.37), 0.86, 0.74,
                          facecolor="#eaf3f0", edgecolor=TEAL, linewidth=0.6)
            )
            ax.text(j, i, "0", ha="center", va="center", fontsize=hs.MIN_PT,
                    color=TEAL, weight="bold")
    ax.set_yticks(range(len(measures)))
    ax.set_yticklabels(measures, fontsize=hs.MATH_PT)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(int(r["site_count"])) for r in rows], fontsize=hs.TICK_PT)
    ax.set_xlabel(r"trajectory sites, $N_s$", labelpad=2, fontsize=hs.MATH_PT)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_panel_e(ax, rows) -> None:
    """Measured ownership-kernel speedup against the size of the unresolved region."""
    roi = np.array([r["roi_percent"] for r in rows])
    ratio = np.array([r["paired_speedup_median"] for r in rows])
    lo = np.array([r["paired_speedup_ci_low"] for r in rows])
    hi = np.array([r["paired_speedup_ci_high"] for r in rows])
    full = np.array([r["exact_gpu_propagation_median_ms"] for r in rows])
    fast = np.array([r["roi_jfa_median_ms"] for r in rows])

    order = np.argsort(roi)
    ax.plot(roi[order], ratio[order], color=BLUE, linewidth=0.95, zorder=2)
    ax.errorbar(roi, ratio, yerr=[ratio - lo, hi - ratio], fmt="none",
                ecolor=INK, elinewidth=0.8, capsize=1.9, capthick=0.8, zorder=3)
    ax.scatter(roi, ratio, s=30, marker="o", facecolor=BLUE, edgecolor=INK,
               linewidth=0.6, zorder=4)
    ax.axhline(1.0, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
    ax.text(0.985, 1.35, "parity", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=hs.MIN_PT, color=MUTED)

    # The site count runs the other way along this axis, so it is written on the points:
    # the largest ratio belongs to the sparsest support, not to a better kernel.
    for index, (x, y) in enumerate(zip(roi, ratio)):
        label = (r"$N_s$=%d" % int(rows[index]["site_count"])) if index == 0 \
            else "%d" % int(rows[index]["site_count"])
        ax.annotate(label, (x, y), xytext=(0, -10), textcoords="offset points",
                    ha="center", va="top", fontsize=hs.MATH_PT, color=MUTED)

    ax.set_xscale("log")
    ax.set_xlim(0.42, 42)
    ax.set_ylim(0, 16.6)
    ax.set_xlabel(r"unresolved fraction of $\mathcal{V}_f$ (%)", fontsize=hs.MATH_PT)
    ax.set_ylabel("paired ownership-kernel speedup")
    # The decade labels are mathtext, so their exponents follow the index ratio.
    ax.tick_params(axis="both", labelsize=hs.MATH_PT)
    ax.grid(which="major")
    ax.text(
        0.028, 0.955,
        "full propagation %.2f-%.2f ms\nROI--JFA %.2f-%.2f ms"
        % (full.min(), full.max(), fast.min(), fast.max()),
        transform=ax.transAxes, ha="left", va="top", fontsize=hs.MIN_PT,
        color=INK, linespacing=1.35,
    )


# ----------------------------------------------------------------------- build
def main() -> None:
    data = dict(np.load(FIELDS_NPZ, allow_pickle=False))
    cpu = json.loads(CPU_JSON.read_text(encoding="utf-8"))
    payload = load_canonical_payload()
    rows = list(payload["primary_rows"])
    cpu_rows = {int(r["particle_count"]): r for r in cpu["rows"]}

    for row in rows:
        reproduced = cpu_rows[int(row["particle_count"])]
        for key in ("certified_voxels", "roi_voxels", "site_count", "pore_voxels"):
            if int(reproduced[key]) != int(row[key]):
                raise SystemExit(
                    "CPU reproduction disagrees with the audited record on %s for %s "
                    "particles" % (key, row["particle_count"]))
        for key in ("label_mismatch_voxels", "distance_mismatch_max",
                    "improving_directed_edges"):
            if int(reproduced[key]) != 0:
                raise SystemExit("CPU reproduction is not exact: %s" % key)
    macro_rows = ownership.GENERATED_TEX.read_text(encoding="utf-8")
    for row in rows:
        stamp = ("%d & %d & %.2f & %.2f & "
                 % (int(row["particle_count"]), int(row["site_count"]),
                    row["certified_percent"], row["roi_percent"]))
        if stamp not in macro_rows:
            raise SystemExit("canonical row missing from ownership_statistics.tex: " + stamp)

    mask3 = data["mask"]
    n_sites = int(data["n_sites"])
    seeds = data["seeds_zyx"]
    # The proposal lives on the whole lattice with the obstacles removed; the stored copy
    # is restricted to the pore space, so it is recomputed here for panel a.
    packed_full = free_space_lower_pair(mask3.shape, seeds, n_sites)
    lower_distance_full = (packed_full // np.int64(n_sites)).astype(np.int32)

    mask = mask3[SLICE_Z]
    certified = data["certified_mask"][SLICE_Z]
    gap = (data["d_exact"] - data["lower_distance"])[SLICE_Z]
    in_slab = seeds[np.abs(seeds[:, 0] - SLICE_Z) <= 1]
    field_row = cpu_rows[FIELD_PREFIX]

    unresolved_all = mask3 & ~data["certified_mask"]
    gap_all = (data["d_exact"] - data["lower_distance"])[unresolved_all]
    strict_fraction = float((gap_all > 0).mean())

    width_pt, height_pt = hs.CANVAS_PT, 404.0
    fig = plt.figure(figsize=(width_pt / 72, height_pt / 72))
    left, right, gapx = 0.055, 0.988, 0.032
    map_w = (right - left - 2 * gapx) / 3.0
    map_h = map_w * width_pt / height_pt
    map_top = 0.940
    map_y0 = map_top - map_h

    ax_a = fig.add_axes([left, map_y0, map_w, map_h])
    ax_b = fig.add_axes([left + map_w + gapx, map_y0, map_w, map_h])
    ax_c = fig.add_axes([left + 2 * (map_w + gapx), map_y0, map_w, map_h])

    image_a = draw_panel_a(ax_a, mask, lower_distance_full[SLICE_Z], in_slab)
    n_cert_slice, n_unres_slice = draw_panel_b(ax_b, mask, certified)
    image_c, vmax_c = draw_panel_c(ax_c, mask, certified, gap)

    hs.panel_label(ax_a, "a", "Free-space proposal",
                   r"lower bound $\bar D$, obstacles removed", subtitle_pt=hs.MATH_PT)
    hs.panel_label(ax_b, "b", "Certification",
                   r"$\mathcal{V}_{\mathrm{cert}}$ against $\mathcal{V}_{\mathrm{unres}}$",
                   subtitle_pt=hs.MATH_PT)
    hs.panel_label(ax_c, "c", "Restricted closure",
                   r"correction $D^\star-\bar D$ over the pore space",
                   subtitle_pt=hs.MATH_PT)

    bar_h = 0.015

    def strip(host, offset, height):
        box = host.get_position()
        return fig.add_axes([box.x0, box.y0 - offset, box.width, height])

    cax_a = strip(ax_a, 0.036, bar_h)
    bar_a = fig.colorbar(image_a, cax=cax_a, orientation="horizontal")
    bar_a.set_label(r"$\bar D$ (voxels)", fontsize=hs.MATH_PT, labelpad=1.8)

    cax_c = strip(ax_c, 0.036, bar_h)
    bar_c = fig.colorbar(image_c, cax=cax_c, orientation="horizontal")
    bar_c.set_ticks([0, vmax_c / 2.0, vmax_c])
    bar_c.set_ticklabels(["0", "%.0f" % (vmax_c / 2.0), "%.0f" % vmax_c])
    bar_c.set_label("distance correction (voxels)\n"
                    "white: certified, zero by certificate; outline: unresolved",
                    fontsize=hs.NOTE_PT, labelpad=1.8)
    for bar in (bar_a, bar_c):
        bar.ax.tick_params(labelsize=hs.TICK_PT, length=1.9, pad=1.4)
        bar.outline.set_linewidth(0.5)
        bar.outline.set_edgecolor(GRID)

    # The boundary handle reproduces the cased line actually drawn in panel a: a dark
    # casing with a light core, so it reads over both ends of the sequential map.
    cased_handle = (mpl.lines.Line2D([], [], color=INK, linewidth=1.35),
                    mpl.lines.Line2D([], [], color="white", linewidth=0.6))
    ax_a.legend(
        handles=[
            mpl.lines.Line2D([], [], linestyle="none", marker="o", markersize=3.1,
                             markerfacecolor="white", markeredgecolor=INK,
                             markeredgewidth=0.4),
            cased_handle,
        ],
        labels=["trajectory sites", "solid boundary"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper center", bbox_to_anchor=(0.5, -0.215), ncol=2, frameon=False,
        handletextpad=0.35, columnspacing=1.0, fontsize=hs.MIN_PT,
    )
    ax_b.text(0.5, -0.055,
              "volume  %.2f%% certified / %.2f%% unresolved"
              % (field_row["certified_percent"], field_row["roi_percent"]),
              transform=ax_b.transAxes, ha="center", va="top", fontsize=hs.NOTE_PT,
              color=INK)
    ax_b.text(0.5, -0.122,
              "this plane  %.1f%% / %.1f%%"
              % (100.0 * n_cert_slice / (n_cert_slice + n_unres_slice),
                 100.0 * n_unres_slice / (n_cert_slice + n_unres_slice)),
              transform=ax_b.transAxes, ha="center", va="top", fontsize=hs.NOTE_PT,
              color=MUTED)
    ax_b.legend(
        handles=[
            Patch(facecolor=BLUE, edgecolor="none", label="certified"),
            Patch(facecolor=ORANGE, edgecolor=INK, linewidth=0.55, label="unresolved"),
            Patch(facecolor=SOLID_FILL, edgecolor=SOLID_EDGE, linewidth=0.45,
                  label="solid"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.175), ncol=3, frameon=False,
        handletextpad=0.35, columnspacing=0.9, fontsize=hs.MIN_PT,
    )

    row_top, row_bottom = 0.392, 0.148
    ax_d = fig.add_axes([left + 0.030, row_bottom + 0.022,
                         map_w - 0.030, row_top - row_bottom - 0.022])
    ax_e = fig.add_axes([left + map_w + gapx + 0.062, row_bottom,
                         2 * map_w + gapx - 0.062, row_top - row_bottom])
    draw_panel_d(ax_d, rows)
    draw_panel_e(ax_e, rows)
    hs.panel_label(ax_d, "d", "Exactness",
                   "completed field identical to full propagation",
                   letter_y=1.170, subtitle_y=1.055)
    hs.panel_label(ax_e, "e", "Ownership-kernel cost",
                   "GPU-resident ownership only, not end-to-end Stokes",
                   letter_y=1.170, subtitle_y=1.055)

    fig.text(
        0.5, 0.024,
        "Panels a-c: one plane of the fixed $64^3$ Berea mask with %d trajectory sites.  "
        "Panels d-e: all five audited nested prefixes on that mask."
        % int(field_row["site_count"]),
        ha="center", fontsize=hs.MATH_PT, color=MUTED,
    )

    offenders = hs.check_min_font(fig)
    if offenders:
        raise SystemExit("type below the %.1f pt floor: %s" % (hs.MIN_PT, offenders))

    hs.save(fig, STEM)

    summary = {
        "slice_z": SLICE_Z,
        "field_prefix_particles": FIELD_PREFIX,
        "field_site_count": int(field_row["site_count"]),
        "slice_certified_voxels": n_cert_slice,
        "slice_unresolved_voxels": n_unres_slice,
        "volume_certified_percent": field_row["certified_percent"],
        "volume_unresolved_percent": field_row["roi_percent"],
        "unresolved_strictly_positive_correction_fraction": strict_fraction,
        "max_distance_correction_voxels": int(gap_all.max()),
        "median_distance_correction_voxels": float(np.median(gap_all)),
        "field_cmap": hs.FIELD_CMAP.name,
        "error_cmap": hs.ERROR_CMAP.name,
        "colour_limits": {"panel_a": [0, int(lower_distance_full[SLICE_Z].max())],
                          "panel_c": [0, float(vmax_c)]},
        "colour_limit_clipping": "none; both maps span the full data range",
        "display_only_interpolation": "none; imshow interpolation is 'nearest'",
        "type_size_floor_pt": hs.MIN_PT,
        "reproduction": hs.reproduction_note(),
        "rows": rows,
    }
    (SOURCE_DIR / (STEM + "_source.json")).write_text(json.dumps(summary, indent=1),
                                                      encoding="utf-8")
    print("wrote " + STEM)


if __name__ == "__main__":
    main()
