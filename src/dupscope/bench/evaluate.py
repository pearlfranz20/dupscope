"""Score the detector on synthetic papers.

The detector runs once with permissive thresholds and every candidate is kept with
its evidence (inliers, correlation, overlap), so thresholds can be chosen
afterwards on a development split and then applied unchanged to a held-out test
split.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..detect import Thresholds, analyse
from .compose import PAIR_KINDS, Paper

PERMISSIVE = Thresholds(
    dup_min_inliers=6,
    dup_min_correlation=0.3,
    dup_min_overlap=0.05,
    clone_min_inliers=6,
    clone_min_correlation=0.4,
    dense_min_blocks=24,
    min_detail=-1.0,
    clones=True,
)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter) if inter else 0.0


@dataclass
class Candidate:
    paper_id: str
    kind: str  # duplicate | clone
    gt_a: tuple | None
    gt_b: tuple | None
    inliers: int
    correlation: float
    detail: float
    texture: float
    extent: float
    overlap_frac: float
    scale: float
    label: str  # planted kind it matches, "hard_negative", or "none"
    method: str = "keypoints"


def run_paper(paper: Paper) -> dict:
    panels, findings = analyse(paper.figures, PERMISSIVE)
    # map each detected panel to the ground-truth panel it covers best
    gt = {(p.figure, p.index): (p.x, p.y, p.w, p.h) for p in paper.panels}
    det_to_gt = {}
    for p in panels:
        best, best_iou = None, 0.0
        for key, box in gt.items():
            if key[0] != p.figure_id:
                continue
            v = _iou((p.x, p.y, p.w, p.h), box)
            if v > best_iou:
                best, best_iou = key, v
        det_to_gt[p.panel_id] = best if best_iou >= 0.5 else None

    planted_pairs = {}
    planted_single = {}
    for pl in paper.planted:
        keys = [tuple(k) for k in pl.panels]
        if pl.kind in PAIR_KINDS:
            planted_pairs[frozenset(keys)] = pl.kind
        else:
            planted_single[keys[0]] = pl.kind
    hard = {frozenset(tuple(k) for k in pair) for pair in paper.hard_negatives}
    # Ground truth by content: two panels are a true duplicate pair whenever they show
    # overlapping pixels of the same source (planted or not, e.g. when two planted
    # reuses share an original), or the same blot.
    by_key = {(p.figure, p.index): p for p in paper.panels}

    def shares_pixels(a, b) -> bool:
        pa, pb = by_key[a], by_key[b]
        if pa.content == "chart" or pa.source != pb.source:
            return False
        if pa.content == "blot":
            return True
        ax, ay, aw, ah = pa.crop
        bx, by, bw, bh = pb.crop
        ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0, min(ay + ah, by + bh) - max(ay, by))
        return ix * iy > 0.05 * min(aw * ah, bw * bh)

    cands = []
    for f in findings:
        a, b = det_to_gt.get(f.panel_a), det_to_gt.get(f.panel_b)
        label = "none"
        if f.kind == "duplicate" and a and b and frozenset((a, b)) in planted_pairs:
            label = planted_pairs[frozenset((a, b))]
        elif f.kind == "duplicate" and a and b and a != b and shares_pixels(a, b):
            label = "incidental_duplicate"  # really shares pixels, though not a planted pair
        elif f.kind == "clone" and a and a in planted_single:
            label = planted_single[a]
        elif f.kind == "duplicate" and a and b and frozenset((a, b)) in hard:
            label = "hard_negative"
        cands.append(
            Candidate(
                paper.paper_id,
                f.kind,
                a,
                b,
                f.inliers,
                f.correlation,
                f.detail,
                f.texture,
                f.extent,
                f.overlap_frac,
                f.scale,
                label,
                f.method,
            ).__dict__
        )
    detected_panels = sum(1 for v in det_to_gt.values() if v is not None)
    return {
        "paper_id": paper.paper_id,
        "manipulated": bool(paper.planted),
        "planted": [p.kind for p in paper.planted],
        "n_hard_negatives": len(paper.hard_negatives),
        "n_gt_panels": len(paper.panels),
        "n_gt_photo_panels": sum(p.content != "chart" for p in paper.panels),
        "n_detected_panels": len(panels),
        "n_matched_panels": detected_panels,
        "candidates": cands,
    }


def passes(c: dict, t: Thresholds) -> bool:
    if c["detail"] < t.min_detail:
        return False
    if c["kind"] == "duplicate":
        if c.get("method") == "template":
            if not t.template:
                return False
        elif c["inliers"] < t.dup_min_inliers:
            return False
        return c["correlation"] >= t.dup_min_correlation and c["overlap_frac"] >= t.dup_min_overlap
    if not t.clones:
        return False
    need = t.dense_min_blocks if c.get("method") == "dense" else t.clone_min_inliers
    return c["inliers"] >= need and c["correlation"] >= t.clone_min_correlation


def score(results: list[dict], t: Thresholds) -> dict:
    """Recall per planted kind, false positives, and paper-level rates.

    With within-panel clone search off, planted clones are outside what is searched:
    they are left out of recall, and papers whose only manipulation is a clone count
    as neither clean nor manipulated."""
    per_kind: dict[str, list[int]] = {}
    fp = hard_fp = flagged_clean = flagged_manip = n_clean = n_manip = hard_total = 0
    tp_findings = 0
    for r in results:
        kept = [c for c in r["candidates"] if passes(c, t)]
        hits = {c["label"] for c in kept}
        searched = [k for k in r["planted"] if t.clones or k in PAIR_KINDS]
        for kind in searched:
            per_kind.setdefault(kind, [0, 0])
            per_kind[kind][1] += 1
        # a kind planted twice in one paper is rare; count it found if any hit has that label
        for kind in set(searched):
            if kind in hits:
                per_kind[kind][0] += searched.count(kind)
        fp += sum(c["label"] == "none" for c in kept)
        # duplicates that genuinely share pixels are neither hits nor false alarms
        hard_fp += sum(c["label"] == "hard_negative" for c in kept)
        tp_findings += sum(c["label"] not in ("none", "hard_negative", "incidental_duplicate") for c in kept)
        hard_total += r["n_hard_negatives"]
        if searched:
            n_manip += 1
            flagged_manip += any(c["label"] not in ("none", "hard_negative") for c in kept)
        elif not r["manipulated"]:
            n_clean += 1
            flagged_clean += any(c["label"] != "incidental_duplicate" for c in kept)
    planted_total = sum(v[1] for v in per_kind.values())
    planted_found = sum(v[0] for v in per_kind.values())
    total_kept = tp_findings + fp + hard_fp
    return {
        "recall": round(planted_found / max(1, planted_total), 3),
        "recall_by_kind": {
            k: {"found": v[0], "planted": v[1], "recall": round(v[0] / max(1, v[1]), 3)}
            for k, v in sorted(per_kind.items())
        },
        "precision": round(tp_findings / max(1, total_kept), 3),
        "false_positives": fp,
        "hard_negative_hits": hard_fp,
        "hard_negatives_total": hard_total,
        "clean_papers": n_clean,
        "clean_papers_flagged": flagged_clean,
        "manipulated_papers": n_manip,
        "manipulated_papers_caught": flagged_manip,
    }
