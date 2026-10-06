from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter

from build_ownership_statistics import load_canonical_payload


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "../outputs" / "figures"
FORWARD_CSV = ROOT / "table_05" / "." / "." / "forward_rows.csv"
REFERENCE_CSV = ROOT / "table_05" / "reference_repeat.csv"

INK = "#243039"
MUTED = "#66737d"
GRID = "#d5dde2"
TEAL = "#159a88"
BLUE = "#4c78a8"
ORANGE = "#d08f17"
PURPLE = "#7b66b3"
MAGENTA = "#c45f91"
RED = "#c84d43"

CASE_ORDER = [
    "orthogonal_duct",
    "skewed_duct",
    "A_thin_wall",
    "B_narrow_throat",
    "C_maze",
    "bentheimer_crop",
]
CASE_LABEL = {
    "orthogonal_duct": "Orthogonal",
    "skewed_duct": "Skewed",
    "A_thin_wall": "Thin wall",
    "B_narrow_throat": "Narrow throat",
    "C_maze": "Maze",
    "bentheimer_crop": "Bentheimer crop",
}
CASE_COLOR = {
    "orthogonal_duct": BLUE,
    "skewed_duct": "#55b5c5",
    "A_thin_wall": ORANGE,
    "B_narrow_throat": "#4f9d69",
    "C_maze": PURPLE,
    "bentheimer_crop": "#d06c4f",
}
DRIFT_CASE = {"bentheimer_crop": "bentheimer_sandstone_crop"}

CASE_MARKER = {
    "orthogonal_duct": "o",
    "skewed_duct": "D",
    "A_thin_wall": "s",
    "B_narrow_throat": "^",
    "C_maze": "P",
    "bentheimer_crop": "v",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.labelsize": 7.4,
        "axes.titlesize": 8.0,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.6,
        "grid.color": GRID,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.85,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def load_forward() -> pd.DataFrame:
    df = pd.read_csv(FORWARD_CSV).rename(
        columns={"pore_voxels": "N_f"}
    )
    numeric = [
        "trajectory_unique_sites",
        "N_f",
        "pressure_unknown_compression",
        "e_K_percent",
        "e_phi_percent",
        "e_u_percent",
        "mass_inf_per_volume",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["N_c"] = df["trajectory_unique_sites"]
    df["R_cell"] = df["N_f"] / df["N_c"]
    if not np.allclose(df["pressure_unknown_compression"], df["R_cell"]):
        raise ValueError("Stored aggregation ratio does not equal N_f/N_c")
    df = df.drop(columns=["pressure_unknown_compression"])
    df = df.loc[df["case"].isin(CASE_ORDER)].copy()
    df["case"] = pd.Categorical(df["case"], CASE_ORDER, ordered=True)
    return df.sort_values("case").reset_index(drop=True)


def save(fig: plt.Figure, stem: str, dpi: int = 400) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches=None, pad_inches=0)
    svg_path = FIG_DIR / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches=None, pad_inches=0)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, weight="bold", fontsize=8.3, va="bottom")
    ax.text(0.02, 1.06, title, transform=ax.transAxes, fontsize=8.0, va="bottom")


def make_figure_04() -> None:
    payload = load_canonical_payload()
    df = pd.DataFrame.from_records(payload["primary_rows"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(518.74 / 72, 215.433 / 72))
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.24, top=0.76, wspace=0.34)

    x = np.arange(len(df))
    width = 0.62
    bars_certified = ax_a.bar(
        x,
        df["certified_percent"],
        width,
        color=TEAL,
        edgecolor=INK,
        linewidth=0.55,
        label="certified",
    )
    ax_a.bar(
        x,
        df["roi_percent"],
        width,
        bottom=df["certified_percent"],
        color=ORANGE,
        edgecolor=INK,
        linewidth=0.55,
        label="unresolved ROI",
    )
    ax_a.set_xticks(x, [str(value) for value in df["particle_count"]])
    ax_a.set_xlabel("trajectory particles in nested prefix")
    ax_a.set_ylabel("pore voxels (%)")
    ax_a.set_ylim(0, 109)
    ax_a.grid(axis="y")
    ax_a.legend(frameon=False, loc="lower left")
    for bar, roi_percent, site_count in zip(
        bars_certified, df["roi_percent"], df["site_count"]
    ):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            92.5,
            f"{roi_percent:.2f}%",
            ha="center",
            va="center",
            fontsize=5.6,
            color="white",
            weight="bold",
        )
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            102.5,
            f"{site_count:,}",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color=MUTED,
        )
    panel_label(ax_a, "a", "Certified support on one fixed mask")
    ax_a.text(
        0.0,
        1.20,
        "zero label/distance mismatches; unique sites above bars",
        transform=ax_a.transAxes,
        fontsize=6.1,
        color=MUTED,
    )

    timing_width = 0.30
    bars_exact = ax_b.bar(
        x - timing_width / 2,
        df["exact_gpu_propagation_median_ms"],
        timing_width,
        color=INK,
        edgecolor=INK,
        linewidth=0.55,
        label="exact GPU propagation",
    )
    bars_roi = ax_b.bar(
        x + timing_width / 2,
        df["roi_jfa_median_ms"],
        timing_width,
        color=BLUE,
        edgecolor=INK,
        linewidth=0.55,
        label="ROI-JFA",
    )
    ax_b.set_xticks(x, [str(value) for value in df["particle_count"]])
    ax_b.set_xlabel("trajectory particles in nested prefix")
    ax_b.set_ylabel("median GPU-resident call (ms)")
    timing_limit = 1.32 * float(
        df[
            [
                "exact_gpu_propagation_median_ms",
                "roi_jfa_median_ms",
            ]
        ]
        .to_numpy()
        .max()
    )
    ax_b.set_ylim(0.0, timing_limit)
    ax_b.grid(axis="y")
    ax_b.legend(frameon=False, loc="upper right", ncol=1)
    panel_label(ax_b, "b", "Ownership-kernel timing on the same mask")
    ax_b.text(
        0.0,
        1.20,
        "paired exact/ROI-JFA: "
        + ", ".join(
            f"{value:.2f}" for value in df["paired_speedup_median"]
        )
        + "x",
        transform=ax_b.transAxes,
        fontsize=6.1,
        color=MUTED,
    )

    save(fig, "Figure_04_ownership_speed_audit")

def make_figure_09() -> None:
    df = load_forward().iloc[:6].copy()
    source_dir = ROOT / "../outputs" / "figure_source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(source_dir / "table_05_source.csv", index=False)
    reference = pd.read_csv(REFERENCE_CSV)
    reference["relative_K_ref_change_percent"] = pd.to_numeric(
        reference["relative_K_ref_change_percent"], errors="coerce"
    )
    drift = reference.set_index("case")["relative_K_ref_change_percent"]

    fig, axs = plt.subplots(2, 2, figsize=(518.74 / 72, 436.535 / 72))
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.15, top=0.90, wspace=0.42, hspace=0.55)
    ax_a, ax_b, ax_c, ax_d = axs.ravel()
    x = np.arange(len(df))
    labels = [CASE_LABEL[str(case)] for case in df["case"]]
    colors = [CASE_COLOR[str(case)] for case in df["case"]]

    bars = ax_a.bar(x, df["e_K_percent"], color=colors, edgecolor=INK, linewidth=0.55, width=0.62)
    ax_a.axhline(12.5, color=RED, linestyle=(0, (3, 2)), linewidth=0.8)
    ax_a.set_ylabel(r"permeability error, $e_K$ (%)")
    ax_a.set_xticks(x, labels, rotation=25, ha="right")
    ax_a.set_ylim(0, 13.2)
    ax_a.grid(axis="y")
    for bar, value in zip(bars, df["e_K_percent"]):
        ax_a.text(bar.get_x() + bar.get_width() / 2, value + 0.22, f"{value:.2f}", ha="center", va="bottom", fontsize=6.2)
    panel_label(ax_a, "a", "Integrated response")

    label_offsets = {
        "orthogonal_duct": (5, -12),
        "skewed_duct": (-33, -3),
        "A_thin_wall": (5, 6),
        "B_narrow_throat": (-34, -2),
        "C_maze": (5, -11),
        "bentheimer_crop": (-34, 1),
    }
    for _, row in df.iterrows():
        case = str(row["case"])
        ax_b.scatter(
            row["e_phi_percent"],
            row["e_u_percent"],
            s=40,
            marker=CASE_MARKER[case],
            color=CASE_COLOR[case],
            edgecolor=INK,
            linewidth=0.55,
            zorder=3,
        )
        ax_b.annotate(
            f"{row['e_K_percent']:.2f}%",
            (row["e_phi_percent"], row["e_u_percent"]),
            xytext=label_offsets[case],
            textcoords="offset points",
            fontsize=6.1,
            color=INK,
        )
    ax_b.axvline(15, color="#8b98a1", linestyle=(0, (3, 2)), linewidth=0.75)
    ax_b.axhline(15, color="#8b98a1", linestyle=(0, (3, 2)), linewidth=0.75)
    ax_b.set_xlim(0, 15)
    ax_b.set_ylim(0, 15)
    ax_b.set_xlabel(r"interface-flux error, $e_\phi$ (%)")
    ax_b.set_ylabel(r"recovered-cell velocity error, $e_u$ (%)")
    ax_b.grid()
    panel_label(ax_b, "b", "Local-field errors")

    for _, row in df.iterrows():
        case = str(row["case"])
        ax_c.scatter(
            row["N_c"],
            row["mass_inf_per_volume"],
            s=43,
            marker=CASE_MARKER[case],
            color=CASE_COLOR[case],
            edgecolor=INK,
            linewidth=0.55,
            zorder=3,
        )
    ax_c.set_yscale("log")
    ax_c.set_xlim(4500, 7100)
    ax_c.set_ylim(3e-18, 5e-16)
    ax_c.set_xlabel(r"trajectory-induced control volumes, $N_c$")
    ax_c.set_ylabel(r"mass residual, $r_\infty^m$")
    ax_c.grid(which="major")
    panel_label(ax_c, "c", "Conservation across cell counts")

    ratio_rows = []
    for _, row in df.iterrows():
        case = str(row["case"])
        drift_case = DRIFT_CASE.get(case, case)
        if (drift_case not in drift.index or not np.isfinite(drift.loc[drift_case])
                or drift.loc[drift_case] <= 0):
            continue
        ratio_rows.append((case, float(row["e_K_percent"]) / float(drift.loc[drift_case])))
    rx = np.arange(len(ratio_rows))
    for index, (case, value) in enumerate(ratio_rows):
        ax_d.scatter(index, value, s=43, marker=CASE_MARKER[case], color=CASE_COLOR[case], edgecolor=INK, linewidth=0.55, zorder=3)
        ax_d.text(index, value * 1.18, f"{value:.1f}x", ha="center", va="bottom", fontsize=6.1)
    ax_d.axhline(1, color="#8b98a1", linestyle=(0, (3, 2)), linewidth=0.75)
    ax_d.set_yscale("log")
    ax_d.set_ylim(0.7, 10000)
    ax_d.set_xticks(rx, [CASE_LABEL[case] for case, _ in ratio_rows], rotation=25, ha="right")
    ax_d.set_ylabel(r"$e_K$ / reference-repeat drift")
    ax_d.grid(which="major", axis="y")
    panel_label(ax_d, "d", "Reference-repeat margin")

    legend_handles = [
        mpl.lines.Line2D(
            [],
            [],
            linestyle="none",
            marker=CASE_MARKER[case],
            markersize=4.8,
            markerfacecolor=CASE_COLOR[case],
            markeredgecolor=INK,
            markeredgewidth=0.55,
            label=CASE_LABEL[case],
        )
        for case in CASE_ORDER[:6]
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.052),
        ncol=6,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.9,
    )
    fig.text(
        0.5,
        0.018,
        "Dashed 12.5% and 15% lines are reporting guides; each case uses its separately selected operating point.",
        ha="center",
        fontsize=6.5,
        color=MUTED,
    )
    save(fig, "Figure_09_canonical_benchmarks")


def marker_area(aggregation_ratio: float) -> float:
    return 38.0 + 4.8 * aggregation_ratio


def make_figure_11() -> None:
    df = load_forward()
    source_dir = ROOT / "../outputs" / "figure_source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(source_dir / "pareto_source.csv", index=False)
    fig = plt.figure(figsize=(529.2 / 72, 248.4 / 72))
    ax = fig.add_axes([0.095, 0.19, 0.67, 0.65])
    cax = fig.add_axes([0.79, 0.30, 0.022, 0.42])
    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(vmin=0, vmax=14)

    offsets = {
        "orthogonal_duct": (5, 5),
        "skewed_duct": (5, 5),
        "A_thin_wall": (5, -12),
        "B_narrow_throat": (-72, 4),
        "C_maze": (5, 5),
        "bentheimer_crop": (-76, -13),
    }
    for _, row in df.iterrows():
        case = str(row["case"])
        ax.scatter(
            row["e_phi_percent"],
            row["e_K_percent"],
            s=marker_area(float(row["R_cell"])),
            marker=CASE_MARKER[case],
            color=cmap(norm(float(row["e_u_percent"]))),
            edgecolor=INK,
            linewidth=0.7,
            alpha=0.92,
            zorder=3,
        )
        ax.annotate(
            CASE_LABEL[case],
            (row["e_phi_percent"], row["e_K_percent"]),
            xytext=offsets[case],
            textcoords="offset points",
            fontsize=6.7,
            color=INK,
        )
    ax.axvline(12.5, color=RED, linestyle=(0, (4, 2)), linewidth=0.85)
    ax.text(12.58, 11.95, "12.5% field guide", color=RED, fontsize=6.5, va="top")
    ax.set_xlim(3.5, 14.5)
    ax.set_ylim(3.5, 12.5)
    ax.set_xlabel(r"interface-flux error, $e_\phi$ (%)")
    ax.set_ylabel(r"permeability error, $e_K$ (%)")
    ax.grid()

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.set_title(r"$e_u$ (%)", fontsize=6.5, pad=4)
    cb.ax.tick_params(labelsize=6.1)

    handles = [
        ax.scatter([], [], s=marker_area(value), color="#b9c5cc", edgecolor="white", linewidth=0.6)
        for value in (6, 7, 8)
    ]
    fig.legend(
        handles,
        ["6", "7", "8"],
        title=r"area: $R_{\mathrm{cell}}=N_f/N_c$",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.855, 0.46),
        fontsize=6.3,
        title_fontsize=6.5,
        labelspacing=1.15,
    )
    fig.text(0.095, 0.925, "Forward-error landscape on trajectory-induced control volumes", fontsize=9.1, weight="bold", color=INK)
    fig.text(
        0.095,
        0.875,
        "marker area encodes voxel-to-cell aggregation; colour encodes recovered-cell velocity error",
        fontsize=7.0,
        color=MUTED,
    )
    fig.text(
        0.095,
        0.075,
        "All six points use exact-frontier ownership, zero wall trace, and the same fixed face-residual stabilization.",
        fontsize=6.4,
        color=MUTED,
    )
    save(fig, "Figure_11_synthetic_pareto")


def main() -> None:
    make_figure_04()
    make_figure_09()
    make_figure_11()


if __name__ == "__main__":
    main()
