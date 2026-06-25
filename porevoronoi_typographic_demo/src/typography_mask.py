from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .connectivity import largest_component


WORD = "PoreVoronoi"
SEED = 20260625


FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\Arial.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def find_font() -> str:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No suitable bold sans-serif font was found.")


def _draw_thick_line(mask: np.ndarray, p0: tuple[int, int], p1: tuple[int, int], radius: int) -> None:
    y0, x0 = p0
    y1, x1 = p1
    n = max(abs(y1 - y0), abs(x1 - x0), 1) + 1
    yy = np.linspace(y0, y1, n)
    xx = np.linspace(x0, x1, n)
    h, w = mask.shape
    for y, x in zip(yy, xx):
        yi = int(round(y))
        xi = int(round(x))
        ylo = max(0, yi - radius)
        yhi = min(h, yi + radius + 1)
        xlo = max(0, xi - radius)
        xhi = min(w, xi + radius + 1)
        gy, gx = np.ogrid[ylo:yhi, xlo:xhi]
        disk = (gy - yi) ** 2 + (gx - xi) ** 2 <= radius**2
        mask[ylo:yhi, xlo:xhi] |= disk


def _bridge_components(mask: np.ndarray, radius: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels, nlab = ndimage.label(mask)
    if nlab <= 1:
        return mask

    comps = []
    for lab in range(1, nlab + 1):
        pts = np.argwhere(labels == lab)
        if pts.shape[0] < 20:
            continue
        cy, cx = pts.mean(axis=0)
        comps.append((cx, cy, pts))
    comps.sort(key=lambda item: item[0])

    out = mask.copy()
    for (_, _, left), (_, _, right) in zip(comps[:-1], comps[1:]):
        li = left[rng.integers(0, len(left), size=min(2000, len(left)))]
        ri = right[rng.integers(0, len(right), size=min(2000, len(right)))]
        dist = ((li[:, None, :] - ri[None, :, :]) ** 2).sum(axis=2)
        a, b = np.unravel_index(int(np.argmin(dist)), dist.shape)
        p0 = tuple(int(v) for v in li[a])
        p1 = tuple(int(v) for v in ri[b])
        _draw_thick_line(out, p0, p1, radius)
    return out


def make_word_mask(
    word: str = WORD,
    canvas: tuple[int, int] = (1600, 430),
    font_size: int = 245,
    bridge_radius: int = 5,
    seed: int = SEED,
) -> tuple[np.ndarray, dict[str, object]]:
    """Create a connected pore mask where the text itself is the pore."""
    width, height = canvas
    font_path = find_font()
    font = ImageFont.truetype(font_path, font_size)
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), word, font=font, stroke_width=0)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (width - tw) // 2 - bbox[0]
    y = (height - th) // 2 - bbox[1] - 5
    draw.text((x, y), word, fill=255, font=font)

    mask = np.asarray(img) > 64
    mask = ndimage.binary_closing(mask, iterations=1)
    mask = _bridge_components(mask, radius=bridge_radius, seed=seed)

    # Add inlet and outlet throats so the word pore is a through-domain.
    coords = np.argwhere(mask)
    cy = int(np.median(coords[:, 0]))
    left = coords[np.argmin(coords[:, 1])]
    right = coords[np.argmax(coords[:, 1])]
    _draw_thick_line(mask, (cy, 0), tuple(int(v) for v in left), bridge_radius + 1)
    _draw_thick_line(mask, tuple(int(v) for v in right), (cy, width - 1), bridge_radius + 1)

    mask = ndimage.binary_closing(mask, iterations=1)
    mask, stats = largest_component(mask, connectivity=1)
    audit = {
        "word": word,
        "seed": int(seed),
        "font_path": font_path,
        "canvas": [int(width), int(height)],
        "font_size": int(font_size),
        "bridge_radius_px": int(bridge_radius),
        "porosity_2d": float(mask.mean()),
        **stats,
    }
    return mask.astype(bool), audit


