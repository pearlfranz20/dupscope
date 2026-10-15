"""Find duplicated image content between panels and within a panel.

Each candidate must pass two independent tests:

1. Geometry. SIFT keypoints are matched with Lowe's ratio test and a single
   similarity transform (rotation, uniform scale, translation, optionally a mirror)
   is fitted with RANSAC. Many microscopy images contain look-alike structures, so
   keypoints alone are not evidence.
2. Pixels. The transform is used to overlay one region on the other, and the
   overlapping pixels must correlate. Correlation is computed on ranks (Spearman),
   so brightness, contrast and gamma edits do not hide a copy, while two different
   fields of view of the same tissue do not correlate pixel-for-pixel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_SIDE = 1024
MAX_KEYPOINTS = 2000


@dataclass
class Features:
    keypoints: np.ndarray  # (n, 2) float32 xy
    descriptors: np.ndarray  # (n, 128) float32, RootSIFT
    shape: tuple[int, int]  # h, w of the (possibly resized) image
    scale: float  # resized / original


@dataclass
class PanelImage:
    """A panel's pixels plus cached features for itself and its mirror image."""

    panel_id: str
    gray: np.ndarray
    features: Features
    mirrored: Features


@dataclass
class Match:
    kind: str  # "duplicate" (between panels) or "clone" (within a panel)
    a: str
    b: str
    inliers: int
    correlation: float
    overlap_frac: float
    rotation_deg: float
    scale: float
    mirrored: bool
    # quadrilaterals (4x2, original pixel coords of each panel) for drawing
    region_a: list = field(default_factory=list)
    region_b: list = field(default_factory=list)
    # 2x3 transform from a's pixels to b's (to b mirrored left-right when `mirrored`)
    transform: list = field(default_factory=list)
    # correlation of fine texture (band-pass) only; copied pixels share their grain,
    # look-alike structures (two blot lanes, two nuclei) do not
    detail: float = 0.0
    # band-pass correlation after local contrast normalisation: flat background
    # counts as much as bright edges, so shared noise shows up even beside a blob
    texture: float = 0.0
    # size of the matched region as a share of the panel: a coincidental look-alike
    # is one small structure, a pasted patch carries several in the same arrangement
    extent: float = 0.0
    # how the pair was found: "keypoints", "template" (low-texture fallback) or "dense"
    method: str = "keypoints"

    @property
    def score(self) -> float:
        # pixel evidence only, so findings from different search methods rank on one
        # scale; fine detail separates copies from look-alikes better than raw intensity
        return round(0.5 * max(self.correlation, 0) + 0.5 * max(self.detail, 0), 3)


_sift = None


def _detector():
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=MAX_KEYPOINTS, contrastThreshold=0.01)
    return _sift


def _prep(gray: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = gray.shape
    # small panels are upsampled so fine structure yields keypoints; big ones capped
    s = min(MAX_SIDE / max(h, w), 2.0 if max(h, w) < 320 else 1.0)
    if s != 1.0:
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        gray = cv2.resize(gray, (int(round(w * s)), int(round(h * s))), interpolation=interp)
    # equalise so dim fluorescence and faint blots still yield keypoints
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray), s


def compute_features(gray: np.ndarray) -> Features:
    img, s = _prep(gray)
    kps, desc = _detector().detectAndCompute(img, None)
    if desc is None or len(kps) == 0:
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32), img.shape, s)
    # RootSIFT: L1-normalise then square-root; Hellinger distance matches better
    desc = desc / (np.abs(desc).sum(axis=1, keepdims=True) + 1e-7)
    desc = np.sqrt(desc).astype(np.float32)
    pts = np.array([k.pt for k in kps], dtype=np.float32)
    return Features(pts, desc, img.shape, s)


def load_panel(panel_id: str, gray: np.ndarray) -> PanelImage:
    return PanelImage(panel_id, gray, compute_features(gray), compute_features(np.ascontiguousarray(gray[:, ::-1])))


def _ratio_matches(da: np.ndarray, db: np.ndarray, ratio: float = 0.8) -> list[tuple[int, int]]:
    if len(da) < 2 or len(db) < 2:
        return []
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    return [(m.queryIdx, m.trainIdx) for m, n in knn if m.distance < ratio * n.distance]


def _rank(x: np.ndarray) -> np.ndarray:
    r = np.empty(x.size, dtype=np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(x.size)
    return r


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 200:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    d = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def _decompose(M: np.ndarray) -> tuple[float, float]:
    scale = math.hypot(M[0, 0], M[1, 0])
    return math.degrees(math.atan2(M[1, 0], M[0, 0])), scale


def _quad(w: int, h: int, M: np.ndarray | None = None) -> np.ndarray:
    q = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    if M is None:
        return q
    return cv2.transform(q[None], M)[0]


def _bandpass(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    # fine scale: at this band two different nuclei or blot lanes share only sensor
    # and film noise if the pixels themselves were copied
    return cv2.GaussianBlur(f, (0, 0), 0.7) - cv2.GaussianBlur(f, (0, 0), 1.6)


def _lcn(bp: np.ndarray) -> np.ndarray:
    local = np.sqrt(cv2.GaussianBlur(bp * bp, (0, 0), 4.0)) + 0.5
    return bp / local


def verify_pair(
    ga: np.ndarray, gb: np.ndarray, M: np.ndarray, region_mask_a: np.ndarray | None = None
) -> tuple[float, float, float, float]:
    """Warp `ga` into `gb`'s frame with M (2x3, original pixel coords) and compare the
    overlap. Returns (rank correlation of intensities, correlation of band-pass
    detail, correlation of contrast-normalised detail, overlap as a fraction of the
    smaller image or region)."""
    hb, wb = gb.shape
    src_mask = np.ones(ga.shape, np.uint8) if region_mask_a is None else region_mask_a
    # erode so interpolation at the borders does not dilute the correlation
    src_mask = cv2.erode(src_mask, np.ones((7, 7), np.uint8))
    warped = cv2.warpAffine(ga, M, (wb, hb), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(src_mask, M, (wb, hb), flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
    n = int(mask.sum())
    ref_area = min(ga.size if region_mask_a is None else int(region_mask_a.sum()), gb.size)
    overlap = n / max(1, ref_area)
    if n < 300:
        return 0.0, 0.0, 0.0, overlap
    # light smoothing removes JPEG block noise, which otherwise dominates the ranks of
    # dark fluorescence backgrounds
    sa = cv2.GaussianBlur(warped, (3, 3), 0)[mask].astype(np.float64)
    sb = cv2.GaussianBlur(gb, (3, 3), 0)[mask].astype(np.float64)
    # pixels that are background in both images say nothing about whether the
    # content is shared (polarity-agnostic: dark fluorescence and bright blots alike)
    da, db = np.abs(sa - np.median(sa)), np.abs(sb - np.median(sb))
    informative = (da > max(2.0, 0.3 * sa.std())) | (db > max(2.0, 0.3 * sb.std()))
    if informative.sum() < 200:
        return 0.0, 0.0, 0.0, overlap
    a, b = sa[informative], sb[informative]
    if a.std() < 2 or b.std() < 2:  # flat regions are not evidence
        return 0.0, 0.0, 0.0, overlap
    step = max(1, a.size // 60000)  # subsample big overlaps for speed
    corr = rank_correlation(a[::step], b[::step])
    bpa, bpb = _bandpass(warped), _bandpass(gb)
    pa, pb = bpa[mask], bpb[mask]
    detail = float(np.corrcoef(pa, pb)[0, 1]) if pa.std() > 0.3 and pb.std() > 0.3 else 0.0
    la, lb = _lcn(bpa)[mask], _lcn(bpb)[mask]
    texture = float(np.corrcoef(la, lb)[0, 1]) if la.std() > 0 and lb.std() > 0 else 0.0
    return corr, detail, texture, overlap


def _to_original(M_small: np.ndarray, sa: float, sb: float) -> np.ndarray:
    """Convert a transform between resized images to original pixel coords."""
    S_a = np.array([[sa, 0, 0], [0, sa, 0], [0, 0, 1]], np.float64)
    S_b_inv = np.array([[1 / sb, 0, 0], [0, 1 / sb, 0], [0, 0, 1]], np.float64)
    M3 = np.vstack([M_small, [0, 0, 1]])
    return (S_b_inv @ M3 @ S_a)[:2]


def _similarity_from(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity (rotation, uniform scale, translation), Umeyama-style."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    sxx = (a * a).sum()
    if sxx < 1e-9:
        return np.array([[1, 0, md[0] - ms[0]], [0, 1, md[1] - ms[1]]], np.float64)
    c = (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]).sum() / sxx
    d = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / sxx
    R = np.array([[c, -d], [d, c]])
    t = md - R @ ms
    return np.hstack([R, t[:, None]])


def ransac_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    thresh: float = 3.0,
    iters: int = 1500,
    scale_range: tuple[float, float] = (0.25, 4.0),
    seed: int = 0,
) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC for a similarity transform that rejects degenerate hypotheses.

    With many-to-one tentative matches (every nucleus resembles several others),
    OpenCV's estimator can settle on a collapsed model that maps everything to one
    point; here each 2-point hypothesis must have a plausible scale and the two
    sample points must be distinct in both images."""
    n = len(src)
    if n < 2:
        return None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, iters)
    j = rng.integers(0, n, iters)
    ps, pd = src[j] - src[i], dst[j] - dst[i]
    den = (ps**2).sum(1)
    ok = (i != j) & (den > 4.0) & ((pd**2).sum(1) > 4.0)
    c = np.where(ok, (ps[:, 0] * pd[:, 0] + ps[:, 1] * pd[:, 1]) / np.maximum(den, 1e-9), 0)
    d = np.where(ok, (ps[:, 0] * pd[:, 1] - ps[:, 1] * pd[:, 0]) / np.maximum(den, 1e-9), 0)
    scale = np.hypot(c, d)
    ok &= (scale >= scale_range[0]) & (scale <= scale_range[1])
    if not ok.any():
        return None, np.zeros(n, bool)
    c, d, i = c[ok], d[ok], i[ok]
    tx = dst[i, 0] - (c * src[i, 0] - d * src[i, 1])
    ty = dst[i, 1] - (d * src[i, 0] + c * src[i, 1])
    # project all points under every hypothesis (hypotheses x points)
    px = c[:, None] * src[None, :, 0] - d[:, None] * src[None, :, 1] + tx[:, None]
    py = d[:, None] * src[None, :, 0] + c[:, None] * src[None, :, 1] + ty[:, None]
    err = (px - dst[None, :, 0]) ** 2 + (py - dst[None, :, 1]) ** 2
    inl = err < thresh * thresh
    best = int(inl.sum(1).argmax())
    mask = inl[best]
    # refine on the inliers, then recount once
    for _ in range(2):
        if mask.sum() < 2:
            break
        M = _similarity_from(src[mask], dst[mask])
        proj = src @ M[:, :2].T + M[:, 2]
        mask = ((proj - dst) ** 2).sum(1) < thresh * thresh
    M = _similarity_from(src[mask], dst[mask]) if mask.sum() >= 2 else None
    if M is not None and not (scale_range[0] <= math.hypot(M[0, 0], M[1, 0]) <= scale_range[1]):
        return None, np.zeros(n, bool)
    # one keypoint in b may not explain several in a: count distinct targets only
    if M is not None:
        _, first = np.unique(np.round(dst[mask], 1), axis=0, return_index=True)
        keep = np.zeros(mask.sum(), bool)
        keep[first] = True
        idx = np.flatnonzero(mask)
        mask = np.zeros(n, bool)
        mask[idx[keep]] = True
    return M, mask


def _tentative_matches(da: np.ndarray, db: np.ndarray, want: int) -> list[tuple[int, int]]:
    """Ratio-test matches; when those are scarce (blob-like content such as nuclei,
    where every descriptor resembles several others), also the two nearest
    neighbours of every keypoint, leaving it to robust geometry to pick the
    consistent ones (RANSAC needs only two points per similarity hypothesis)."""
    pairs = _ratio_matches(da, db)
    if len(pairs) >= want or len(da) < 2 or len(db) < 2:
        return pairs
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    # closest first, capped: RANSAC cost grows with the number of candidates
    extra = sorted((m for row in knn for m in row), key=lambda m: m.distance)[:1000]
    return list(dict.fromkeys(pairs + [(m.queryIdx, m.trainIdx) for m in extra]))


def compare_panels(
    pa: PanelImage,
    pb: PanelImage,
    min_inliers: int = 10,
    min_correlation: float = 0.7,
    min_overlap: float = 0.1,
) -> Match | None:
    """Best duplicate relation between two different panels, or None."""
    best: Match | None = None
    fa = pa.features
    for mirrored, fb in ((False, pb.features), (True, pb.mirrored)):
        pairs = _tentative_matches(fa.descriptors, fb.descriptors, want=4 * min_inliers)
        if len(pairs) < min_inliers:
            continue
        src = fa.keypoints[[i for i, _ in pairs]]
        dst = fb.keypoints[[j for _, j in pairs]]
        M, inl = ransac_similarity(src, dst)
        if M is None or int(inl.sum()) < min_inliers:
            continue
        M = _to_original(M, fa.scale, fb.scale)
        target = pb.gray[:, ::-1] if mirrored else pb.gray
        corr, detail, texture, overlap = verify_pair(pa.gray, target, M)
        if corr < min_correlation or overlap < min_overlap:
            continue
        rot, scale = _decompose(M)
        quad_b = _quad(pa.gray.shape[1], pa.gray.shape[0], M)
        if mirrored:
            quad_b[:, 0] = pb.gray.shape[1] - 1 - quad_b[:, 0]
        quad_b[:, 0] = quad_b[:, 0].clip(0, pb.gray.shape[1] - 1)
        quad_b[:, 1] = quad_b[:, 1].clip(0, pb.gray.shape[0] - 1)
        m = Match(
            "duplicate",
            pa.panel_id,
            pb.panel_id,
            int(inl.sum()),
            round(corr, 3),
            round(min(overlap, 1.0), 3),
            round(rot, 1),
            round(scale, 3),
            mirrored,
            _quad(pa.gray.shape[1], pa.gray.shape[0]).tolist(),
            quad_b.tolist(),
            M.round(4).tolist(),
            detail=round(detail, 3),
            texture=round(texture, 3),
            extent=round(min(overlap, 1.0), 3),
        )
        if best is None or m.score > best.score:
            best = m
    return best


def along_edge(region: np.ndarray, dx: float, dy: float) -> bool:
    """True when the matched blocks form a thin strip parallel to the shift: a
    straight edge (panel border, scale bar, inset frame) looks the same at every
    shift along itself, which is not evidence of copying."""
    ys, xs = np.nonzero(region)
    if xs.size < 10:
        return True
    cov = np.cov(np.vstack([xs, ys]).astype(np.float64))
    evals, evecs = np.linalg.eigh(cov)
    if evals[0] <= 0:
        return True
    elongation = math.sqrt(evals[1] / evals[0])
    axis = evecs[:, 1]
    shift = np.array([dx, dy], np.float64)
    cos = abs(float(axis @ shift)) / (np.linalg.norm(shift) + 1e-9)
    return elongation > 2.0 and cos > math.cos(math.radians(20))


def find_clones(
    p: PanelImage,
    min_inliers: int = 8,
    min_correlation: float = 0.8,
    min_shift_frac: float = 0.06,
    max_clones: int = 3,
) -> list[Match]:
    """Regions copied to another place inside the same panel (copy-move)."""
    f = p.features
    h, w = p.gray.shape
    if len(f.descriptors) < 2 * min_inliers:
        return []
    min_shift = min_shift_frac * max(f.shape)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(f.descriptors, f.descriptors, k=3)
    pairs = []
    for row in knn:
        # row[0] is the keypoint itself; compare the 2nd and 3rd neighbours
        if len(row) < 3:
            continue
        _, m, n = row
        if m.distance < 0.6 * n.distance:
            i, j = m.queryIdx, m.trainIdx
            if np.linalg.norm(f.keypoints[i] - f.keypoints[j]) > min_shift:
                # orient each pair the same way so i->j and j->i agree
                a, b = (i, j) if tuple(f.keypoints[i]) < tuple(f.keypoints[j]) else (j, i)
                pairs.append((a, b))
    pairs = list(dict.fromkeys(pairs))
    found: list[Match] = []
    while len(pairs) >= min_inliers and len(found) < max_clones:
        src = f.keypoints[[a for a, _ in pairs]]
        dst = f.keypoints[[b for _, b in pairs]]
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None or inl is None or int(inl.sum()) < min_inliers:
            break
        inl = inl.ravel().astype(bool)
        # the cloned region: convex hull of inlier source points, padded
        hull = cv2.convexHull((src[inl] / f.scale).astype(np.float32))
        region = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(region, hull.astype(np.int32), 1)
        pad = max(3, int(0.02 * max(h, w)))
        region = cv2.dilate(region, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
        M = _to_original(M, f.scale, f.scale)
        shift = math.hypot(M[0, 2], M[1, 2])
        if along_edge(region, M[0, 2], M[1, 2]):
            pairs = [pr for pr, keep in zip(pairs, inl, strict=True) if not keep]
            continue
        corr, detail, texture, overlap = verify_pair(p.gray, p.gray, M, region)
        if corr >= min_correlation and shift > min_shift / f.scale and region.sum() > 400:
            rot, scale = _decompose(M)
            q = hull.reshape(-1, 2)
            found.append(
                Match(
                    "clone",
                    p.panel_id,
                    p.panel_id,
                    int(inl.sum()),
                    round(corr, 3),
                    round(min(overlap, 1.0), 3),
                    round(rot, 1),
                    round(scale, 3),
                    False,
                    q.tolist(),
                    cv2.transform(q[None].astype(np.float32), M)[0].tolist(),
                    M.round(4).tolist(),
                    detail=round(detail, 3),
                    texture=round(texture, 3),
                    extent=round(float(region.mean()), 4),
                )
            )
        pairs = [pr for pr, keep in zip(pairs, inl, strict=True) if not keep]
    return found
