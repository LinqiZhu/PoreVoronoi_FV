from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from src.io_utils import out_dir, save_json
from src.typography_mask import SEED, WORD, make_word_mask


def main() -> None:
    mask, audit = make_word_mask(word=WORD, seed=SEED)
    out = out_dir("masks")
    np.savez_compressed(out / "word_mask_2d.npz", mask=mask)
    Image.fromarray((mask * 255).astype("uint8")).save(out / "word_mask_2d.png")
    save_json(out / "word_mask_2d_audit.json", audit)
    print(f"word mask saved: {out / 'word_mask_2d.npz'}")
    print(audit)


if __name__ == "__main__":
    main()

