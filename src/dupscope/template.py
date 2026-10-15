"""Fallback duplicate search for low-texture panels.

Sparse fluorescence fields (a few dim nuclei on black) give keypoint matching little
to work with. Here patches of one panel are searched directly in the other at low
resolution, over the 8 rotation/mirror orientations and a range of scales. The best
hypothesis is then checked at full resolution with the same pixel verification as
every other finding, so this widens recall without loosening the evidence bar.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .match import Match, PanelImage, _decompose, _quad, verify_pair

BASE = 96  # working size of the longer side of panel a
SCALES = (0.6, 0.7, 0.85, 1.0, 1.18, 1.4, 1.65)  # b relative to a


def _orient(img: np.ndarray, k: int, flip: bool) -> tuple[np.ndarray, np.ndarray]:
    """Rotate by k*90 degrees (counter-clockwise) then optionally mirror; also return
    the 3x3 matrix mapping original pixel coords to the oriented image."""
    h, w = img.shape
    out = np.rot90(img, k)
    # np.rot90 (k=1): (x, y) -> (y, w - 1 - x)
    R = np.eye(3)
    for _ in range(k % 4):
        R = np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], np.float64) @ R
        h, w = w, h
    if flip:
        out = out[:, ::-1]
        R = np.array([[-1, 0, out.shape[1] - 1], [0, 1, 0], [0, 0, 1]], np.float64) @ R
    return np.ascontiguousarray(out), R


def _patches(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = img.shape
    ph, pw = int(h * 0.45), int(w * 0.45)
    boxes = [(int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6))]
    for fy in (0.04, 0.51):
        for fx in (0.04, 0.51):
            boxes.append((int(w * fx), int(h * fy), pw, ph))
    return boxes


def _refine(ga: np.ndarray, gb: np.ndarray, M: np.ndarray, work: int = 256) -> np.ndarray:
    """Polish the coarse a->b transform with ECC (affine) at a moderate resolution.
    Falls back to the coarse transform when ECC does not converge."""
    s = min(1.0, work / max(ga.shape + gb.shape))
    a = cv2.GaussianBlur(cv2.resize(ga, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (0, 0), 1).astype(np.float32)
    b = cv2.GaussianBlur(cv2.resize(gb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), (0, 0), 1).astype(np.float32)
    S = np.diag([s, s, 1.0])
    M3 = np.vstack([M, [0, 0, 1]])
    # ECC estimates W with template(x) ~ input(W x); template = b, input = a, so W maps b->a
    W = (S @ np.linalg.inv(M3) @ np.linalg.inv(S))[:2].astype(np.float32)
    try:
        _, W = cv2.findTransformECC(
            b, a, W, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5), None, 3
        )
    except cv2.error:
        return M
    Wf = np.linalg.inv(S) @ np.vstack([W.astype(np.float64), [0, 0, 1]]) @ S
    out = np.linalg.inv(Wf)[:2]
    # keep the refinement only if it stays close to a similarity of sensible scale
    A = out[:, :2]
    sv = np.linalg.svd(A, compute_uv=False)
    if not (0.25 < sv[1] and sv[0] < 4 and sv[0] / sv[1] < 1.15):
        return M
    return out


def template_search(
    pa: PanelImage, pb: PanelImage, min_peak: float = 0.8, min_correlation: float = 0.7, min_overlap: float = 0.1
) -> Match | None:
    ga, gb = pa.gray, pb.gray
    fa = BASE / max(ga.shape)
    a_s = cv2.resize(ga, None, fx=fa, fy=fa, interpolation=cv2.INTER_AREA)
    best = None  # (peak, M_full)
    b_scaled = []
    for r in SCALES:
        fb = fa * r
        if max(gb.shape) * fb >= 24:
            b_scaled.append((fb, cv2.resize(gb, None, fx=fb, fy=fb, interpolation=cv2.INTER_AREA)))
    S_a = np.diag([fa, fa, 1.0])
    for k in range(4):
        for flip in (False, True):
            a_o, R = _orient(a_s, k, flip)
            patches = [(x, y, a_o[y : y + h, x : x + w]) for x, y, w, h in _patches(a_o)]
            patches = [(x, y, pt) for x, y, pt in patches if pt.std() >= 4]
            for fb, b_s in b_scaled:
                for x, y, patch in patches:
                    if patch.shape[0] > b_s.shape[0] or patch.shape[1] > b_s.shape[1]:
                        continue
                    res = cv2.matchTemplate(b_s, patch, cv2.TM_CCOEFF_NORMED)
                    _, peak, _, loc = cv2.minMaxLoc(res)
                    if peak >= min_peak and (best is None or peak > best[0]):
                        # a (full) -> a_s -> oriented -> b_s (shift by loc - patch origin) -> b (full)
                        T = np.array([[1, 0, loc[0] - x], [0, 1, loc[1] - y], [0, 0, 1]], np.float64)
                        S_b = np.diag([1 / fb, 1 / fb, 1.0])
                        best = (peak, (S_b @ T @ R @ S_a)[:2])
    if best is None:
        return None
    peak, M = best
    M = _refine(ga, gb, M)
    corr, detail, texture, overlap = verify_pair(ga, gb, M)
    if corr < min_correlation or overlap < min_overlap:
        return None
    rot, scale = _decompose(M)
    mirrored = bool(np.linalg.det(M[:, :2]) < 0)
    if mirrored:  # report rotation of the un-mirrored part
        rot = math.degrees(math.atan2(-M[1, 0], -M[0, 0]))
    quad_b = _quad(ga.shape[1], ga.shape[0], M)
    quad_b[:, 0] = quad_b[:, 0].clip(0, gb.shape[1] - 1)
    quad_b[:, 1] = quad_b[:, 1].clip(0, gb.shape[0] - 1)
    return Match(
        "duplicate",
        pa.panel_id,
        pb.panel_id,
        0,
        round(corr, 3),
        round(min(overlap, 1.0), 3),
        round(rot, 1),
        round(scale, 3),
        mirrored,
        _quad(ga.shape[1], ga.shape[0]).tolist(),
        quad_b.tolist(),
        M.round(4).tolist(),
        detail=round(detail, 3),
        texture=round(texture, 3),
        extent=round(min(overlap, 1.0), 3),
        method="template",
    )
