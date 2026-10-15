"""Dense copy-move detection inside one panel (translation clones).

Keypoints find large or textured copies, but a small pasted patch in a small panel
may hold only one or two keypoints. Here every block of the panel is compared with
every other (the shift-vector method of Fridrich et al., 2003): a pasted region
makes many neighbouring blocks agree on the same displacement, while look-alike
structures (two similar nuclei, two blot lanes) rarely line up on one exact shift
across a whole region, and if they do, the pixel check rejects them.
"""

from __future__ import annotations

import math
from collections import Counter

import cv2
import numpy as np

from .match import Match, PanelImage, along_edge, verify_pair


def _blocks(gray: np.ndarray, size: int, stride: int):
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
    h, w = g.shape
    feats, pos = [], []
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            b = g[y : y + size, x : x + size]
            sd = b.std()
            if sd < 6:  # flat background cannot show a copy
                continue
            d = cv2.resize(b, (8, 8), interpolation=cv2.INTER_AREA).ravel()
            d = (d - d.mean()) / (np.linalg.norm(d - d.mean()) + 1e-6)
            feats.append(d)
            pos.append((x, y))
    return np.array(feats, np.float32), np.array(pos, np.int32)


def find_dense_clones(
    p: PanelImage,
    min_blocks: int = 60,
    min_correlation: float = 0.8,
    max_clones: int = 3,
    max_side: int = 400,
) -> list[Match]:
    gray = p.gray
    h, w = gray.shape
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    gh, gw = gray.shape
    size = int(np.clip(round(min(gh, gw) * 0.09), 10, 24))
    # a copy lands at any offset, so a coarse grid splits it across misaligned blocks;
    # a 2-pixel stride keeps every copied block within a pixel of a grid block
    stride = 2
    feats, pos = _blocks(gray, size, stride)
    if len(feats) < 2 * min_blocks:
        return []
    min_shift = 1.5 * size
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 64})
    knn = flann.knnMatch(feats, feats, k=5)
    votes: Counter = Counter()
    members: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row in knn:
        for m in row[1:]:
            if m.distance > 0.35:  # descriptors are unit vectors; this is a close match
                break
            a, b = pos[m.queryIdx], pos[m.trainIdx]
            dx, dy = int(b[0] - a[0]), int(b[1] - a[1])
            if math.hypot(dx, dy) < min_shift:
                continue
            if (dx, dy) < (0, 0):  # canonical direction so a->b and b->a vote together
                dx, dy, a = -dx, -dy, b
            votes[(dx, dy)] += 1
            members.setdefault((dx, dy), []).append((int(a[0]), int(a[1])))
            break
    # Resampling and JPEG jitter a true shift by a pixel or two, splitting its votes;
    # sum each shift's votes over a 5x5 neighbourhood and keep local maxima.
    summed = {k: sum(votes.get((k[0] + i, k[1] + j), 0) for i in range(-2, 3) for j in range(-2, 3)) for k in votes}
    peaks = []
    for k, n in sorted(summed.items(), key=lambda kv: -kv[1]):
        if n < min_blocks:
            break
        if any(abs(k[0] - p[0]) <= 4 and abs(k[1] - p[1]) <= 4 for p, _ in peaks):
            continue
        peaks.append((k, n))
        if len(peaks) >= 8:
            break
    found: list[Match] = []
    for (dx, dy), n in peaks:
        region = np.zeros((gh, gw), np.uint8)
        for i in range(-2, 3):
            for j in range(-2, 3):
                for x, y in members.get((dx + i, dy + j), []):
                    region[y : y + size, x : x + size] = 1
        # back to the panel's full resolution
        if s < 1.0:
            region = cv2.resize(region, (w, h), interpolation=cv2.INTER_NEAREST)
        dx_full, dy_full = dx / s, dy / s
        M = np.array([[1, 0, dx_full], [0, 1, dy_full]], np.float64)
        if along_edge(region, dx, dy):
            continue
        corr, detail, texture, overlap = verify_pair(p.gray, p.gray, M, region)
        if corr < min_correlation:
            continue
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(c)
        qa = [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]]
        qb = [[x + dx_full, y + dy_full] for x, y in qa]
        found.append(
            Match(
                "clone",
                p.panel_id,
                p.panel_id,
                int(n),
                round(corr, 3),
                round(min(overlap, 1.0), 3),
                0.0,
                1.0,
                False,
                qa,
                qb,
                M.round(3).tolist(),
                detail=round(detail, 3),
                texture=round(texture, 3),
                extent=round(float(region.mean()), 4),
                method="dense",
            )
        )
    # strongest pixel evidence first, not most votes
    found.sort(key=lambda m: -(m.correlation + m.detail))
    found = found[:max_clones]
    return found
