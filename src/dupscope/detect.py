"""Run the duplication checks over all figures of one paper."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .dense import find_dense_clones
from .match import Match, compare_panels, find_clones, load_panel
from .panels import Panel, extract_panels, to_gray
from .template import template_search


@dataclass
class Thresholds:
    dup_min_inliers: int = 10
    dup_min_correlation: float = 0.7
    dup_min_overlap: float = 0.1
    clone_min_inliers: int = 8
    clone_min_correlation: float = 0.8
    dense_min_blocks: int = 60
    # Within-panel clone search is experimental: in compressed, resampled figures
    # the benchmark cannot separate real clones from natural repetition (similar
    # nuclei, regularly spaced blot lanes), so it is off unless asked for.
    clones: bool = False
    # when keypoint matching finds nothing, search one panel's patches directly in
    # the other (all 8 rotations/mirrorings, several scales); helps sparse, low-texture
    # fluorescence images, but on the benchmark it added more false alarms than finds,
    # so it is off by default
    template: bool = False
    min_detail: float = 0.0  # band-pass correlation floor, applied to every finding


@dataclass
class Finding:
    kind: str  # duplicate | clone
    category: str  # human-readable triage label
    panel_a: str
    panel_b: str
    score: float
    inliers: int
    correlation: float
    detail: float
    texture: float
    extent: float
    overlap_frac: float
    rotation_deg: float
    scale: float
    mirrored: bool
    region_a: list  # quad in figure pixel coords
    region_b: list
    method: str = "keypoints"


def categorise(m: Match) -> str:
    """A label that tells a reviewer what kind of reuse this looks like. These are
    leads for a human, not verdicts: some reuse (a zoomed inset, a shared control
    panel declared in the legend) is legitimate."""
    if m.kind == "clone":
        return "region repeated within a panel"
    rot = abs(((m.rotation_deg + 180) % 360) - 180)
    turned = rot > 5 or m.mirrored
    if m.scale > 1.6 or m.scale < 1 / 1.6:
        return "magnified region (often a legitimate inset)"
    if m.overlap_frac < 0.6:
        return "overlapping fields of view" + (" with rotation or mirroring" if turned else "")
    if turned:
        return "same image, rotated or mirrored"
    return "same image reused"


def _to_figure_coords(quad: list, p: Panel) -> list:
    return [[round(x + p.x, 1), round(y + p.y, 1)] for x, y in quad]


def load_figure(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return img


def analyse(figures: dict[str, np.ndarray], thresholds: Thresholds | None = None) -> tuple[list[Panel], list[Finding]]:
    """`figures` maps a figure id to its image. Returns all panels and the findings,
    strongest first."""
    t = thresholds or Thresholds()
    panels: list[Panel] = []
    loaded = {}
    for fig_id, img in figures.items():
        gray = to_gray(img)
        for p in extract_panels(img, fig_id):
            panels.append(p)
            if p.photographic:
                loaded[p.panel_id] = (p, load_panel(p.panel_id, p.crop(gray).copy()))

    findings: list[Finding] = []

    def add(m: Match, pa: Panel, pb: Panel):
        findings.append(
            Finding(
                m.kind,
                categorise(m),
                m.a,
                m.b,
                m.score,
                m.inliers,
                m.correlation,
                m.detail,
                m.texture,
                m.extent,
                m.overlap_frac,
                m.rotation_deg,
                m.scale,
                m.mirrored,
                _to_figure_coords(m.region_a, pa),
                _to_figure_coords(m.region_b, pb),
                m.method,
            )
        )

    for (pa, ia), (pb, ib) in itertools.combinations(loaded.values(), 2):
        m = compare_panels(ia, ib, t.dup_min_inliers, t.dup_min_correlation, t.dup_min_overlap)
        if m is None:
            # transforms are not symmetric under cropping and scale; try the other way round
            m2 = compare_panels(ib, ia, t.dup_min_inliers, t.dup_min_correlation, t.dup_min_overlap)
            if m2 is not None:
                add(m2, pb, pa)
            elif t.template:
                m3 = template_search(ia, ib, min_correlation=t.dup_min_correlation, min_overlap=t.dup_min_overlap)
                if m3 is not None:
                    add(m3, pa, pb)
        else:
            add(m, pa, pb)
    for p, img in loaded.values():
        if not t.clones:
            break
        clones = find_clones(img, t.clone_min_inliers, t.clone_min_correlation)
        clones += find_dense_clones(img, t.dense_min_blocks, t.clone_min_correlation)
        # the two clone finders often report the same copy; keep the stronger one
        kept: list[Match] = []
        for m in sorted(clones, key=lambda m: -m.score):
            tx, ty = m.transform[0][2], m.transform[1][2]
            if not any(abs(tx - k.transform[0][2]) < 8 and abs(ty - k.transform[1][2]) < 8 for k in kept):
                kept.append(m)
        for m in kept:
            add(m, p, p)
    findings = [f for f in findings if f.detail >= t.min_detail]
    findings.sort(key=lambda f: -f.score)
    return panels, findings


def findings_to_dicts(findings: list[Finding]) -> list[dict]:
    return [asdict(f) for f in findings]


COLOURS = [(0, 0, 230), (0, 140, 255), (200, 0, 160), (0, 160, 0), (200, 120, 0), (120, 60, 200)]


def draw_overlay(figures: dict[str, np.ndarray], panels: list[Panel], findings: list[Finding]) -> dict[str, np.ndarray]:
    """Copies of each figure with panel boxes and numbered finding regions drawn on."""
    out = {}
    for fig_id, img in figures.items():
        canvas = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if canvas.shape[2] == 4:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
        for p in panels:
            if p.figure_id == fig_id:
                colour = (170, 170, 170) if p.photographic else (220, 220, 220)
                cv2.rectangle(canvas, (p.x, p.y), (p.x + p.w - 1, p.y + p.h - 1), colour, 1)
        out[fig_id] = canvas
    for n, f in enumerate(findings, start=1):
        colour = COLOURS[(n - 1) % len(COLOURS)]
        for pid, quad in ((f.panel_a, f.region_a), (f.panel_b, f.region_b)):
            fig_id = pid.rsplit("#", 1)[0]
            canvas = out[fig_id]
            pts = np.array(quad, np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], True, colour, 2, cv2.LINE_AA)
            x, y = pts[:, 0, 0].min(), pts[:, 0, 1].min()
            cv2.putText(
                canvas, str(n), (int(x) + 3, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA
            )
    return out
