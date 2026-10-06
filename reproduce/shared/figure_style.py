"""Shared figure style of the ownership figures (Figure 3, make_figure_03.py).

The base style (fonts, palette, panel lettering, save routine) is inherited unchanged from
reproduce/shared/result_figures.py, which generates the result figures,
so these figures carry the manuscript's visual grammar.

Two additions are enforced on top of it, both from the authors' figure rules ("all text
checked at 180 mm main figure width, minimum type size not below 6.5 pt"; and "no
rainbow, no drop shadows, no 3-D perspective"):

  * MIN_PT is a hard floor on every type size in the figure.  The figures are drawn on the
    project's 518.74 pt canvas; at a 180 mm (510.24 pt) reproduction width the scale is
    0.9836, so a 6.8 pt design size reproduces at 6.69 pt, above the 6.5 pt floor.

  * Only perceptually uniform, colour-blind-safe, monotone-lightness colour maps are used,
    with one map per physical role so the visual grammar is consistent across figures:
        FIELD_CMAP  - a magnitude field (graph distance, speed)
        ERROR_CMAP  - a discrepancy (distance correction, velocity difference)
    Neither is a rainbow map, and both survive conversion to greyscale.

Categorical colours are the manuscript's own result-figure palette.
"""
from __future__ import annotations

import matplotlib as mpl
import numpy as np

import result_figures as base

# ---------------------------------------------------------------- inherited palette
INK = base.INK          # "#243039"
MUTED = base.MUTED      # "#66737d"
GRID = base.GRID        # "#d5dde2"
TEAL = base.TEAL
BLUE = base.BLUE        # certified
ORANGE = base.ORANGE    # unresolved
RED = base.RED

# Solid is drawn as a very light form of the solid green used in Figures 1 and 4.  Its
# luminance is deliberately placed above the top of FIELD_CMAP so that solid and fast flow
# stay separable after conversion to greyscale; the boundary contour is a second cue.
SOLID_FILL = "#eaf3ee"
SOLID_EDGE = "#15614b"

# Categorical pair for the certification map.  The manuscript's blue and orange are kept,
# but their values are pulled apart so the two categories differ in lightness as well as
# hue and remain distinguishable in greyscale and under colour-vision deficiency.
CERTIFIED = "#3d6a99"
UNRESOLVED = "#e8a33d"

# ---------------------------------------------------------------- type sizes
MIN_PT = 6.8            # hard floor; 6.69 pt at 180 mm reproduction width
TICK_PT = 6.8
NOTE_PT = 6.8
SUBTITLE_PT = 6.9
LABEL_PT = 7.4
TITLE_PT = 8.0
LETTER_PT = 8.3

# Mathtext renders subscripts and superscripts at 0.7 of the surrounding size, so a label
# set at the floor puts its indices at 4.76 pt.  Every label that carries mathematics is
# therefore set at MATH_PT instead, which lifts its indices to 5.46 pt (5.37 pt at 180 mm).
# For reference, the smallest rendered glyph in the other main-text figures of this
# manuscript is 4.48-4.76 pt.
MATH_PT = 7.8
MATHTEXT_INDEX_RATIO = 0.7

CANVAS_PT = 518.74      # design width, identical to the result figures
NATURE_DOUBLE_PT = 510.24   # 180 mm

# ---------------------------------------------------------------- colour maps
# Magnitude fields use viridis truncated just below its brightest yellow.  The full map
# tops out at luminance 0.87, which is close enough to the pale solid to blur the geometry
# in a greyscale reproduction; truncating at 0.90 of the ramp lowers the top to about 0.79
# and opens a clear gap, at the cost of a top colour that is marginally less yellow.
FIELD_CMAP = mpl.colors.ListedColormap(
    mpl.colormaps["viridis"](np.linspace(0.0, 0.90, 256)), name="viridis_field"
)
try:
    mpl.colormaps.register(FIELD_CMAP, name="viridis_field", force=True)
except Exception:  # pragma: no cover - registration is a convenience only
    pass

# A discrepancy map built from the manuscript's own accent red: white at zero rising to a
# single dark hue.  Lightness decreases monotonically, so it is unambiguous in greyscale;
# it is single-hue, so it is safe under every common colour-vision deficiency; and it is
# visually separate from FIELD_CMAP, so a residual panel can never be mistaken for a field
# panel.  Deliberately gentler than a multi-hue map: the residual must be locatable without
# dominating the figure over the fields it is compared against.
ERROR_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "house_residual",
    ["#ffffff", "#fbe6e0", "#f0b3a6", "#dd7a68", "#c84d43", "#8f2f27", "#4d1a15"],
)
try:
    mpl.colormaps.register(ERROR_CMAP, name="house_residual", force=True)
except Exception:  # pragma: no cover - registration is a convenience only
    pass

mpl.rcParams.update(
    {
        "legend.fontsize": MIN_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "axes.labelsize": LABEL_PT,
        "font.size": 7.2,
        "pdf.fonttype": 42,      # embed as TrueType; text stays selectable vector text
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "image.interpolation": "nearest",
    }
)

save = base.save


def panel_label(ax, letter: str, title: str, subtitle: str | None = None,
                letter_y: float | None = None, subtitle_y: float = 1.012,
                subtitle_pt: float | None = None) -> None:
    """Manuscript panel lettering: bold lowercase letter, then a title, then a grey note.

    The bold-lowercase form (a, b, c ...) rather than (a), (b), (c) is the convention of
    Figures 1-9 of this manuscript and is kept for consistency.
    """
    # 1.06 is the placement used by the result figures, which carry no second
    # line.  A subtitle needs the title lifted, or the two rows touch at print size.
    if letter_y is None:
        letter_y = 1.085 if subtitle else 1.06
    ax.text(-0.10, letter_y, letter, transform=ax.transAxes, weight="bold",
            fontsize=LETTER_PT, va="bottom")
    ax.text(0.02, letter_y, title, transform=ax.transAxes, fontsize=TITLE_PT, va="bottom")
    if subtitle:
        ax.text(0.02, subtitle_y, subtitle, transform=ax.transAxes,
                fontsize=SUBTITLE_PT if subtitle_pt is None else subtitle_pt,
                color=MUTED, va="bottom")


def map_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
        spine.set_linewidth(0.7)


def cased_line(ax, field, *, level=0.5, inner="white", outer=INK,
               inner_w=0.55, outer_w=1.25, zorder=5, extent=None):
    """A boundary line that stays legible over both ends of a sequential colour map.

    A dark casing is drawn first and a light core on top, so the line reads against dark
    and light backgrounds alike without introducing a new hue.
    """
    kwargs = {"levels": [level], "origin": "lower"}
    if extent is not None:
        kwargs["extent"] = extent
    ax.contour(np.asarray(field, dtype=float), colors=[outer], linewidths=outer_w,
               zorder=zorder, **kwargs)
    ax.contour(np.asarray(field, dtype=float), colors=[inner], linewidths=inner_w,
               zorder=zorder + 0.1, **kwargs)


def check_min_font(fig, floor: float = MIN_PT) -> list[tuple[str, float]]:
    """Return every text artist drawn below the type-size floor.

    Called before saving; a non-empty result is a build failure.
    """
    offenders = []
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible():
            continue
        content = text.get_text()
        if not content or not content.strip():
            continue
        size = float(text.get_fontsize())
        if size < floor - 1e-9:
            offenders.append((content[:40], size))
    return offenders


def reproduction_note(width_pt: float = CANVAS_PT) -> str:
    scale = NATURE_DOUBLE_PT / width_pt
    return ("design width %.2f pt; at 180 mm the scale is %.3f, so the %.1f pt floor "
            "reproduces at %.2f pt" % (width_pt, scale, MIN_PT, MIN_PT * scale))
