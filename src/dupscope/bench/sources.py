"""Source images for the benchmark.

Real microscopy: the 2018 Data Science Bowl nuclei images as packaged by StarDist
(fluorescence and brightfield), plus the histology, retina and cell samples that
ship with scikit-image. The DSB set is known to contain overlapping tiles, so the
pool is de-duplicated before use: any pair that shares pixels would turn a planted
"negative" into a real positive and make the detector look worse than it is.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2
import numpy as np

DSB_URL = "https://github.com/stardist/stardist/releases/download/0.1.0/dsb2018.zip"


def _gray(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3:
        a = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)
    if a.dtype != np.uint8:
        a = cv2.normalize(a.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return a


def load_pool(dsb_dir: str | Path) -> list[tuple[str, np.ndarray]]:
    """All usable source images as (name, grayscale uint8)."""
    from skimage import io
    import skimage.data

    pool = []
    for f in sorted(glob.glob(os.path.join(str(dsb_dir), "**", "images", "*.tif"), recursive=True)):
        a = _gray(io.imread(f))
        if min(a.shape) >= 200:
            pool.append((f"dsb/{Path(f).parent.parent.name}/{Path(f).stem}", a))
    base = os.path.dirname(skimage.data.__file__)
    for name in ("cell.png", "ihc.png", "retina.jpg", "microaneurysms.png"):
        a = _gray(io.imread(os.path.join(base, name)))
        # large samples are cut into tiles so each tile is its own source
        th, tw = 360, 360
        for i, y in enumerate(range(0, a.shape[0] - th + 1, th)):
            for j, x in enumerate(range(0, a.shape[1] - tw + 1, tw)):
                tile = a[y : y + th, x : x + tw]
                if tile.std() > 8:
                    pool.append((f"skimage/{name}/{i}-{j}", tile))
    return pool


def _thumb(a: np.ndarray) -> np.ndarray:
    t = cv2.resize(a, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (t - t.mean()) / (t.std() + 1e-6)


def dedupe(pool: list[tuple[str, np.ndarray]], workers: int = 8) -> tuple[list[tuple[str, np.ndarray]], list]:
    """Drop images that share content with an earlier one.

    Candidates come from the detector's own matcher; each is then confirmed
    independently by plain Pearson correlation of the aligned overlap (>= 0.9),
    so only genuine shared pixels remove an image."""
    from concurrent.futures import ProcessPoolExecutor

    names = [n for n, _ in pool]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        dupes = [d for chunk in ex.map(_dedupe_chunk, [(pool, i, workers) for i in range(workers)]) for d in chunk]
    drop = {b for a, b, _ in dupes}
    kept = [(n, a) for n, a in pool if n not in drop]
    return kept, [(names_a, names_b, c) for names_a, names_b, c in dupes]


def _dedupe_chunk(args):
    from ..match import compare_panels, load_panel

    pool, part, parts = args
    imgs = [load_panel(n, a) for n, a in pool]
    out = []
    for i in range(part, len(imgs), parts):
        for j in range(i + 1, len(imgs)):
            m = compare_panels(imgs[i], imgs[j], min_inliers=12, min_correlation=0.6, min_overlap=0.05)
            if m is None:
                continue
            M = np.array(m.transform, np.float64)
            target = imgs[j].gray[:, ::-1] if m.mirrored else imgs[j].gray
            h, w = target.shape
            warped = cv2.warpAffine(imgs[i].gray, M, (w, h))
            mask = cv2.warpAffine(np.ones_like(imgs[i].gray), M, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
            if mask.sum() < 500:
                continue
            c = np.corrcoef(warped[mask].astype(float), target[mask].astype(float))[0, 1]
            if c >= 0.9:
                out.append((pool[i][0], pool[j][0], round(float(c), 3)))
    return out
