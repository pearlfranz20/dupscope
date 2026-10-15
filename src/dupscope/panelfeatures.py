"""Features that separate photographs (micrographs, blots, gels) from graphics
(bar fills, hatching, text, plots) in a grayscale panel crop. Used by the learned
panel classifier (see classifier.json and bench/train_panel_classifier.py)."""

from __future__ import annotations

import cv2
import numpy as np

NAMES = [
    "log_w",
    "log_h",
    "aspect",
    "white",
    "black",
    "mid",
    "entropy",
    "top1",
    "top2",
    "top3",
    "levels_1pct",
    "flat",
    "flat_lit",
    "noise_med",
    "edge_density",
    "axis_edges",
    "border_dark",
    "row_var",
    "col_var",
    "std",
]


def features(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    s = min(1.0, 256 / max(h, w))
    g = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    n = g.size
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64) / n
    nz = hist[hist > 0]
    smooth = np.convolve(hist, np.ones(9), mode="same")
    rest, peaks = smooth.copy(), []
    for _ in range(3):
        p = int(np.argmax(rest))
        peaks.append(p)
        rest[max(0, p - 8) : p + 9] = 0

    def mass(ps):
        near = np.zeros(256, bool)
        for p in ps:
            near[max(0, p - 4) : p + 5] = True
        return float(hist[near].sum())

    f = g.astype(np.float32)
    m = cv2.blur(f, (5, 5))
    sd = np.sqrt(np.maximum(cv2.blur(f * f, (5, 5)) - m * m, 0))
    lit = g > 40
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    strong = mag > 40
    axis = (np.abs(gx) < 0.2 * np.abs(gy)) | (np.abs(gy) < 0.2 * np.abs(gx))
    ring = np.concatenate([g[:2].ravel(), g[-2:].ravel(), g[:, :2].ravel(), g[:, -2:].ravel()])
    return np.array(
        [
            np.log(w),
            np.log(h),
            np.log(w / h),
            float(hist[245:].sum()),
            float(hist[:11].sum()),
            float(hist[41:215].sum()),
            float(-(nz * np.log2(nz)).sum()),
            mass(peaks[:1]),
            mass(peaks[:2]),
            mass(peaks),
            float((hist > 0.01).sum()),
            float((sd < 1.0).mean()),
            float((sd[lit] < 1.0).mean()) if lit.any() else 0.0,
            float(np.median(sd)),
            float(strong.mean()),
            float((strong & axis).sum() / max(1, strong.sum())),
            float((ring < 60).mean()),
            float(np.std(f.mean(axis=1))),
            float(np.std(f.mean(axis=0))),
            float(f.std()),
        ],
        np.float64,
    )
