from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "cmame_artifacts" / "figures"
DATA_DIR = ROOT / "cmame_artifacts" / "figure_source_data"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.dpi": 200,
            "savefig.dpi": 600,
        }
    )


def build_data() -> pd.DataFrame:
    data = pd.DataFrame(
        [
            {
                "case": "Bentheimer",
                "e_K_percent": 2.30,
                "e_phi_percent": 19.72,
                "e_u_percent": 19.14,
            },
            {
                "case": "Berea",
                "e_K_percent": 0.39,
                "e_phi_percent": 13.32,
                "e_u_percent": 17.24,
            },
        ]
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_DIR / "Figure_13_mainline_transfer_data.csv", index=False)
    return data


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.025,
        label,
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="#222222",
    )


def main() -> None:
    setup_style()
    data = build_data()

    colors = {
        "bentheimer": "#3F76AE",
        "berea": "#E27A3F",
        "e_phi": "#7A61A8",
        "e_u": "#2F9D8C",
        "guide": "#8E9AA7",
        "grid": "#D8DEE6",
        "text": "#222222",
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.52),
        gridspec_kw={"width_ratios": [1.0, 1.08], "wspace": 0.28},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.070, right=0.985, bottom=0.175, top=0.790, wspace=0.280)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Real-mask sampled-state transfer mainline",
        x=0.52,
        y=0.965,
        fontsize=8.8,
        fontweight="bold",
        color=colors["text"],
    )

    x = np.arange(len(data))
    case_labels = data["case"].tolist()

    ax = axes[0]
    bars = ax.bar(
        x,
        data["e_K_percent"],
        width=0.58,
        color=[colors["bentheimer"], colors["berea"]],
        edgecolor="white",
        linewidth=0.7,
    )
    for bar, val in zip(bars, data["e_K_percent"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.06,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=colors["text"],
        )
    ax.set_title("Permeability recovery", loc="left", pad=5)
    add_panel_label(ax, "a")
    ax.set_ylabel(r"$e_K$ (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels)
    ax.set_ylim(0, 2.65)
    ax.set_yticks(np.arange(0, 2.6, 0.5))
    ax.grid(axis="y", color=colors["grid"], linewidth=0.55)
    ax.tick_params(axis="x", length=0)

    ax = axes[1]
    width = 0.30
    ax.bar(
        x - width / 2,
        data["e_phi_percent"],
        width=width,
        color=colors["e_phi"],
        edgecolor="white",
        linewidth=0.7,
        label=r"$e_\phi$",
    )
    ax.bar(
        x + width / 2,
        data["e_u_percent"],
        width=width,
        color=colors["e_u"],
        edgecolor="white",
        linewidth=0.7,
        label=r"$e_u$",
    )
    ax.axhline(20.0, color=colors["guide"], linestyle=(0, (2.0, 2.0)), linewidth=0.75)
    for xpos, val in zip(x - width / 2, data["e_phi_percent"]):
        if val > 18.5:
            ax.text(xpos, val - 1.20, f"{val:.2f}", ha="center", va="top", fontsize=6.1, color="white")
        else:
            ax.text(xpos, val + 0.45, f"{val:.2f}", ha="center", va="bottom", fontsize=6.1)
    for xpos, val in zip(x + width / 2, data["e_u_percent"]):
        if val > 18.5:
            ax.text(xpos, val - 1.20, f"{val:.2f}", ha="center", va="top", fontsize=6.1, color="white")
        else:
            ax.text(xpos, val + 0.45, f"{val:.2f}", ha="center", va="bottom", fontsize=6.1)
    ax.set_title("State-to-flux field accuracy", loc="left", pad=5)
    add_panel_label(ax, "b")
    ax.set_ylabel("Field error (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels)
    ax.set_ylim(0, 21.8)
    ax.set_yticks(np.arange(0, 21, 5))
    ax.grid(axis="y", color=colors["grid"], linewidth=0.55)
    ax.tick_params(axis="x", length=0)
    ax.text(
        -0.47,
        20.35,
        "20% guide",
        ha="left",
        va="bottom",
        fontsize=6.1,
        color=colors["guide"],
    )
    ax.scatter([0.50], [-0.18], s=28, marker="s", color=colors["e_phi"], transform=ax.transAxes, clip_on=False)
    ax.text(0.535, -0.165, r"$e_\phi$", transform=ax.transAxes, ha="left", va="center", fontsize=6.4)
    ax.scatter([0.63], [-0.18], s=28, marker="s", color=colors["e_u"], transform=ax.transAxes, clip_on=False)
    ax.text(0.665, -0.165, r"$e_u$", transform=ax.transAxes, ha="left", va="center", fontsize=6.4)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = FIG_DIR / "Figure_13_mainline_transfer"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(base.with_suffix(".pdf"))
    print(base.with_suffix(".svg"))


if __name__ == "__main__":
    main()

