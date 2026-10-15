"""Split a figure into its image panels.

Scientific figures are panels laid out on a plain background (almost always white)
with gutters between them. We estimate the background colour from the border,
mark everything that differs from it, close small gaps (text inside a panel,
thin scale bars), and keep connected components large enough to be images.

Plots, diagrams and text blocks are also "panels" geometrically. `is_photographic`
separates them from photographic content (microscopy, blots, gels), which is the
only content the duplication checks run on.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Panel:
    figure_id: str
    index: int
    x: int
    y: int
    w: int
    h: int
    photographic: bool

    @property
    def panel_id(self) -> str:
        return f"{self.figure_id}#{self.index}"

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.y : self.y + self.h, self.x : self.x + self.w]


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _background(gray: np.ndarray) -> int:
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    return int(np.median(border))


def _trim(gray: np.ndarray, box: tuple[int, int, int, int], bg: int, tol: int) -> tuple[int, int, int, int]:
    """Shrink a box while its outermost row or column is plain background. Closing
    can glue a nearby label onto a panel; the straight panel-to-gutter edge that
    leaves behind looks identical at every shift along it."""
    x, y, w, h = box

    def blank(line: np.ndarray) -> bool:
        return float((np.abs(line.astype(np.int16) - bg) <= tol).mean()) > 0.9

    while h > 8 and blank(gray[y, x : x + w]):
        y, h = y + 1, h - 1
    while h > 8 and blank(gray[y + h - 1, x : x + w]):
        h -= 1
    while w > 8 and blank(gray[y : y + h, x]):
        x, w = x + 1, w - 1
    while w > 8 and blank(gray[y : y + h, x + w - 1]):
        w -= 1
    return x, y, w, h


def is_photographic(gray: np.ndarray) -> bool:
    """Photographs (microscopy, blots, gels) versus charts, diagrams and text.

    Charts sit on paper white and fill shapes with a few flat tones, so nearly all
    their pixels fall at one or two light levels (allowing for JPEG wobble).
    Photographs either spread over many levels or, like sparse fluorescence, sit on
    a dark background."""
    if gray.size == 0:
        return False
    small = cv2.resize(gray, (min(256, gray.shape[1]), min(256, gray.shape[0])), interpolation=cv2.INTER_AREA)
    if float((small >= 245).mean()) > 0.35:
        return False
    # a flat fill (a bar, a shaded box) is not a photograph; a nearly empty dark
    # fluorescence field is, so only light or mid-grey flat fills are excluded
    # (a bar slice with an error bar through it is still mostly one noise-free tone;
    # bright-field micrographs keep camera grain even in their background)
    med = float(np.median(small))
    if med > 40:
        flat = np.abs(small.astype(np.int16) - med) <= 6
        if flat.mean() > 0.6 and float(small[flat].std()) < 1.65:
            return False
    hist = np.bincount(small.ravel(), minlength=256).astype(np.float64)
    smooth = np.convolve(hist, np.ones(9), mode="same")  # +-4 levels of JPEG wobble
    rest = smooth.copy()
    peaks = []
    for _ in range(3):
        p = int(np.argmax(rest))
        peaks.append(p)
        rest[max(0, p - 8) : p + 9] = 0

    def mass(ps) -> float:
        near = np.zeros(256, bool)
        for p in ps:
            near[max(0, p - 4) : p + 5] = True
        return float(hist[near].sum() / hist.sum())

    # blots and gels also sit on a light background, but it is grey with grain, not
    # paper white; only flat fills on paper white count as graphics. Two tones (white
    # and a fill), or three (white, a fill and black outlines, as in a bar chart).
    if max(peaks[:2]) >= 245 and mass(peaks[:2]) > 0.8:
        return False
    if max(peaks) >= 245 and mass(peaks) > 0.88:
        return False
    return True


def _xy_cut(mask: np.ndarray, x0: int, y0: int, min_side: int, min_gap: int, out: list) -> None:
    """Recursively split a block along background gutters that run all the way
    across it (rows or columns with essentially no foreground)."""
    h, w = mask.shape
    for axis in (0, 1):  # 0: split by rows (horizontal gutters), 1: by columns
        fill = mask.mean(axis=1 - axis)  # per row (axis 0) or per column (axis 1)
        # a gutter can carry a panel letter, so "empty" means mostly empty
        empty = fill < 0.1
        n = len(empty)
        runs, start = [], None
        for k in range(n):
            if empty[k] and start is None:
                start = k
            elif not empty[k] and start is not None:
                runs.append((start, k))
                start = None
        if start is not None:
            runs.append((start, n))
        # interior gutters only, wide enough to be deliberate spacing
        cuts = [(a, b) for a, b in runs if a > 0 and b < n and b - a >= min_gap]
        if cuts:
            edges = [0] + [v for a, b in cuts for v in (a, b)] + [n]
            for s0, s1 in zip(edges[0::2], edges[1::2], strict=True):
                if s1 - s0 < min_side:
                    continue
                if axis == 0:
                    _xy_cut(mask[s0:s1], x0, y0 + s0, min_side, min_gap, out)
                else:
                    _xy_cut(mask[:, s0:s1], x0 + s0, y0, min_side, min_gap, out)
            return
    ys, xs = np.nonzero(mask)
    if xs.size:
        out.append((x0 + int(xs.min()), y0 + int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))


def _classifier():
    from .classifier import is_photo

    return is_photo


def extract_panels(
    image: np.ndarray,
    figure_id: str = "fig",
    min_side: int = 48,
    min_short_side: int = 24,
    min_area_frac: float = 0.002,
    tol: int = 10,
    classify: bool = True,
) -> list[Panel]:
    """Split a figure into panels in reading order. With `classify`, each panel is
    marked photographic or not by the learned classifier (classifier.py)."""
    photo = _classifier() if classify else (lambda g: True)
    gray = to_gray(image)
    h, w = gray.shape
    bg = _background(gray)
    if 40 < bg < 225:
        # a figure whose border is mid-grey is one photograph (an uncropped blot or gel
        # scan), not a layout on paper; cutting it into bands makes look-alikes
        return [Panel(figure_id, 0, 0, 0, w, h, photo(gray))]
    raw = (np.abs(gray.astype(np.int16) - bg) > tol).astype(np.uint8)
    # 1. drop text-sized blobs (panel letters, axis labels outside panels) so they
    #    cannot bridge the gutter between neighbouring panels
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    small = (stats[:, cv2.CC_STAT_WIDTH] < min_side) & (stats[:, cv2.CC_STAT_HEIGHT] < min_side)
    small[0] = False
    raw[np.isin(labels, np.nonzero(small)[0])] = 0
    # 2. close small holes inside panels (gaps in sparse fluorescence, white areas in
    #    brightfield), with a kernel smaller than any real gutter
    mask = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # 3. components, each split further along any gutter that crosses it entirely
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        if max(bw, bh) < min_side or min(bw, bh) < min_short_side:
            continue
        pieces: list = []
        _xy_cut(mask[y : y + bh, x : x + bw], int(x), int(y), min_short_side, max(4, int(min(h, w) * 0.006)), pieces)
        for px, py, pw, ph in pieces:
            # blots are often wide, short strips; text lines are shorter still
            if max(pw, ph) < min_side or min(pw, ph) < min_short_side or pw * ph < min_area_frac * h * w:
                continue
            px, py, pw, ph = _trim(gray, (px, py, pw, ph), bg, tol)
            if max(pw, ph) >= min_side and min(pw, ph) >= min_short_side:
                boxes.append((px, py, pw, ph))
    # reading order: rows top to bottom, then left to right
    boxes.sort(key=lambda b: (round(b[1] / max(1, h * 0.05)), b[0]))
    panels = []
    for i, (x, y, bw, bh) in enumerate(boxes):
        crop = gray[y : y + bh, x : x + bw]
        panels.append(Panel(figure_id, i, x, y, bw, bh, photo(crop)))
    return panels
