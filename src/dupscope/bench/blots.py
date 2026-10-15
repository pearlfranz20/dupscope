"""Procedural western blots.

Each blot is a strip of lanes with dark bands on a light background: bands at
shared molecular weights across lanes with varying intensity, a slight "smile",
uneven background and film grain. Two blots from different seeds look alike to a
person (that is the point of using them as hard negatives) but share no pixels.
"""

from __future__ import annotations

import cv2
import numpy as np


def make_blot(rng: np.random.Generator, n_lanes: int | None = None, lane_w: int = 34, height: int = 110) -> np.ndarray:
    n_lanes = n_lanes or int(rng.integers(5, 11))
    gap = int(rng.integers(6, 12))
    w = n_lanes * (lane_w + gap) + gap
    h = height
    img = np.zeros((h, w), np.float32)
    n_bands = int(rng.integers(1, 4))
    rows = sorted(rng.uniform(0.2, 0.8, n_bands) * h)
    smile = rng.uniform(-3, 3)
    for lane in range(n_lanes):
        x0 = gap + lane * (lane_w + gap)
        for r in rows:
            strength = rng.uniform(0.15, 1.0) if rng.random() > 0.1 else 0.0
            thickness = rng.uniform(3.0, 7.0)
            band = np.zeros((h, w), np.float32)
            cx = x0 + lane_w / 2
            curve = smile * (((lane + 0.5) / n_lanes) - 0.5) ** 2 * 4
            cy = r + curve + rng.normal(0, 0.6)
            cv2.ellipse(
                band, (int(cx), int(cy)), (int(lane_w / 2 - rng.uniform(0, 3)), int(thickness / 2) + 1),
                0, 0, 360, float(strength), -1, cv2.LINE_AA,
            )
            img = np.maximum(img, band)
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=rng.uniform(1.2, 2.2), sigmaY=rng.uniform(1.5, 3.0))
    # background: light, with a gentle gradient and grain
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = 0.9 + 0.05 * (xx / w - 0.5) * rng.uniform(-1, 1) + 0.03 * (yy / h - 0.5) * rng.uniform(-1, 1)
    grain = cv2.GaussianBlur(rng.normal(0, 0.035, (h, w)).astype(np.float32), (0, 0), 0.8)
    out = bg - 0.75 * img + grain
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def lane_bounds(blot: np.ndarray) -> list[tuple[int, int]]:
    """Approximate lane column ranges, from where the bands are."""
    dark = (blot < np.median(blot) - 20).any(axis=0)
    spans, start = [], None
    for x, d in enumerate(dark):
        if d and start is None:
            start = x
        elif not d and start is not None:
            if x - start >= 8:
                spans.append((start, x))
            start = None
    if start is not None:
        spans.append((start, len(dark)))
    return spans
