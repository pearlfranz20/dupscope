"""Score the detector on synthetic papers.

The detector runs once with permissive thresholds and every candidate is kept with
its evidence (inliers, correlation, overlap), so thresholds can be chosen
afterwards on a development split and then applied unchanged to a held-out test
split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

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


def _shares_pixels(pa: dict, pb: dict) -> bool:
    """Two ground-truth panels show overlapping pixels of the same original."""
    if pa["content"] == "chart" or pa["source"] != pb["source"]:
        return False
    if pa["content"] == "blot":
        return True
    ax, ay, aw, ah = pa["crop"]
    bx, by, bw, bh = pb["crop"]
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy > 0.05 * min(aw * ah, bw * bh)


def annotate(result: dict, truth: dict) -> dict:
    """Add ground truth that scoring needs, judged from the pixels' provenance rather
    than from what the generator meant to do:

    - planted_pairs: each planted between-panel reuse, scored on its own (a kind can
      be planted twice in one paper)
    - valid_hard_negatives: look-alike pairs that really are two regions of one
      original sharing no pixels (a planted reuse can overwrite one side of a pair)
    - incidental_pairs: panel pairs that share pixels without a plant (two random
      crops of one original can overlap); such a paper is not clean
    """
    panels = {(p["figure"], p["index"]): p for p in truth["panels"]}
    planted = [(pl["kind"], sorted(tuple(k) for k in pl["panels"])) for pl in truth["planted"]]
    pair_sets = {frozenset(ks) for kind, ks in planted if kind in PAIR_KINDS}
    hard = []
    for a, b in truth["hard_negatives"]:
        pa, pb = panels[tuple(a)], panels[tuple(b)]
        if pa["source"] == pb["source"] and not _shares_pixels(pa, pb):
            hard.append([list(a), list(b)])
    keys = sorted(panels)
    incidental = [
        [list(a), list(b)]
        for n, a in enumerate(keys)
        for b in keys[n + 1 :]
        if frozenset((a, b)) not in pair_sets and _shares_pixels(panels[a], panels[b])
    ]
    result = dict(result)
    result["planted_pairs"] = [[kind, [list(k) for k in ks]] for kind, ks in planted if kind in PAIR_KINDS]
    result["valid_hard_negatives"] = hard
    result["incidental_pairs"] = incidental
    return result


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
    by_key = {(p.figure, p.index): asdict(p) for p in paper.panels}

    def shares_pixels(a, b) -> bool:
        return _shares_pixels(by_key[a], by_key[b])

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
    return annotate(
        {
            "paper_id": paper.paper_id,
            "manipulated": bool(paper.planted),
            "planted": [p.kind for p in paper.planted],
            "n_hard_negatives": len(paper.hard_negatives),
            "n_gt_panels": len(paper.panels),
            "n_gt_photo_panels": sum(p.content != "chart" for p in paper.panels),
            "n_detected_panels": len(panels),
            "n_matched_panels": detected_panels,
            "candidates": cands,
        },
        paper.truth(),
    )


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


REAL = ("none", "hard_negative")


def score(results: list[dict], t: Thresholds) -> dict:
    """Recall per planted reuse, false alarms and paper-level rates. Results must be
    annotated with ground truth (see `annotate`).

    A planted between-panel reuse counts as found when a kept finding links exactly
    its two panels. With within-panel clone search off, planted clones are outside
    what is searched, so papers whose only manipulation is a clone are left out.
    Papers that contain an unplanted real duplicate are reported separately, not as
    clean. Findings on such duplicates count as real in precision."""
    per_kind: dict[str, list[int]] = {}
    fp = hard_fp = tp = 0
    clean = clean_flagged = manip = manip_caught = hard_total = 0
    incidental_papers = incidental_found = 0
    for r in results:
        kept = [c for c in r["candidates"] if passes(c, t)]
        kept_pairs = {frozenset((tuple(c["gt_a"]), tuple(c["gt_b"]))) for c in kept if c["gt_a"] and c["gt_b"]}
        for kind, ks in r["planted_pairs"]:
            per_kind.setdefault(kind, [0, 0])
            per_kind[kind][1] += 1
            per_kind[kind][0] += frozenset(tuple(k) for k in ks) in kept_pairs
        singles = [k for k in r["planted"] if k not in PAIR_KINDS]
        if t.clones:
            hits = {c["label"] for c in kept}
            for kind in singles:
                per_kind.setdefault(kind, [0, 0])
                per_kind[kind][1] += 1
            for kind in set(singles):
                per_kind[kind][0] += singles.count(kind) * (kind in hits)
        fp += sum(c["label"] == "none" for c in kept)
        hard_fp += sum(c["label"] == "hard_negative" for c in kept)
        tp += sum(c["label"] not in REAL for c in kept)
        hard_total += len(r["valid_hard_negatives"])
        searched = bool(r["planted_pairs"]) or (t.clones and bool(singles))
        if searched:
            manip += 1
            manip_caught += any(c["label"] not in REAL and c["label"] != "incidental_duplicate" for c in kept)
        elif r["incidental_pairs"]:
            incidental_papers += 1
            incidental_found += any(c["label"] == "incidental_duplicate" for c in kept)
        elif not singles:
            clean += 1
            clean_flagged += bool(kept)
    planted_total = sum(v[1] for v in per_kind.values())
    planted_found = sum(v[0] for v in per_kind.values())
    return {
        "found": planted_found,
        "planted": planted_total,
        "recall": round(planted_found / max(1, planted_total), 3),
        "recall_by_kind": {
            k: {"found": v[0], "planted": v[1], "recall": round(v[0] / max(1, v[1]), 3)}
            for k, v in sorted(per_kind.items())
        },
        "findings": tp + fp + hard_fp,
        "real_findings": tp,
        "precision": round(tp / max(1, tp + fp + hard_fp), 3),
        "false_positives": fp,
        "hard_negative_hits": hard_fp,
        "hard_negatives_total": hard_total,
        "clean_papers": clean,
        "clean_papers_flagged": clean_flagged,
        "manipulated_papers": manip,
        "manipulated_papers_caught": manip_caught,
        "papers_with_unplanted_duplicates": incidental_papers,
        "unplanted_duplicate_papers_flagged": incidental_found,
        "papers": len(results),
    }
