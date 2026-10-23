"""Train the photograph-versus-graphic panel classifier and export it to
src/dupscope/classifier.json (gradient-boosted trees, run with numpy only).

    python bench/train_panel_classifier.py DEVDATA_DIR DSB_DIR DEDUPE_JSON

Training data:
- results/panel_classifier/labels.json: panels from figures of retracted papers
  (the devdata branch, fetched by .github/workflows/devdata.yml), labelled by eye as
  photograph (micrograph, blot, gel, animal photo) or graphic (bar, fill pattern,
  plot, text, diagram). A random sample of panels ("random_sample"), plus every
  panel that took part in a finding on those papers in a first real run
  ("finding_panels");
- results/panel_classifier/comparison_labels.json: feature vectors of panels from
  comparison papers (never retracted), labelled the same way, under anonymous paper
  groups (bench/collect_comparison_panels.py);
- panels from synthetic benchmark papers with seeds 1000 to 1059, which the
  benchmark's development and test splits never use.

Writes results/panel_classifier/cv.json: photo recall and graphics kept, with whole
papers held out (5-fold, grouped by paper).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from dupscope.bench.compose import make_paper
from dupscope.bench.evaluate import _iou
from dupscope.bench.sources import load_pool
from dupscope.panelfeatures import NAMES, features
from dupscope.panels import extract_panels, to_gray

ROOT = Path(__file__).resolve().parent.parent
THRESHOLD = 0.3
PARAMS = {"n_estimators": 150, "max_depth": 3, "random_state": 0}


def real_set(devdata: Path):
    labels = json.loads((ROOT / "results" / "panel_classifier" / "labels.json").read_text())
    X, y, groups, random_sample, cache = [], [], [], [], {}
    for m in labels:
        key = (m["paper"], m["fig"])
        if key not in cache:
            cache[key] = cv2.imread(str(devdata / m["paper"] / f"{m['fig']}.png"), cv2.IMREAD_GRAYSCALE)
        x, yy, w, h = (int(m[k]) for k in ("x", "y", "w", "h"))
        X.append(features(cache[key][yy : yy + h, x : x + w]))
        y.append(m["label"])
        groups.append(m["paper"])
        random_sample.append(m.get("source", "random_sample") == "random_sample")
    return np.array(X), np.array(y), np.array(groups), np.array(random_sample)


def synthetic_set(dsb: str, dedupe: str):
    keep = set(json.loads(Path(dedupe).read_text())["kept"])
    pool = [(n, a) for n, a in load_pool(dsb) if n in keep]
    X, y = [], []
    for seed in range(1000, 1060):
        paper = make_paper(seed, pool, manipulated=seed % 2 == 1)
        for fig, img in paper.figures.items():
            g = to_gray(img)
            for d in extract_panels(img, fig, classify=False):
                gt = max(
                    (p for p in paper.panels if p.figure == fig),
                    key=lambda p: _iou((d.x, d.y, d.w, d.h), (p.x, p.y, p.w, p.h)),
                )
                ok = _iou((d.x, d.y, d.w, d.h), (gt.x, gt.y, gt.w, gt.h)) >= 0.5
                X.append(features(d.crop(g)))
                y.append(int(ok and gt.content != "chart"))
    return np.array(X), np.array(y)


def export(clf: GradientBoostingClassifier) -> dict:
    prior = float(clf.init_.class_prior_[1])
    trees = []
    for est in clf.estimators_[:, 0]:
        t = est.tree_
        trees.append(
            {
                "feature": t.feature.tolist(),
                "threshold": [round(float(v), 6) for v in t.threshold],
                "left": t.children_left.tolist(),
                "right": t.children_right.tolist(),
                "value": [round(float(v), 6) for v in t.value[:, 0, 0]],
            }
        )
    return {
        "features": NAMES,
        "init": float(np.log(prior / (1 - prior))),
        "learning_rate": clf.learning_rate,
        "threshold": THRESHOLD,
        "trees": trees,
    }


def comparison_set():
    """Feature vectors and labels of panels from comparison papers (never retracted),
    under anonymous paper groups. No images or identifiers are stored."""
    path = ROOT / "results" / "panel_classifier" / "comparison_labels.json"
    if not path.exists():
        return np.zeros((0, len(NAMES))), np.zeros(0, int), np.zeros(0, str), np.zeros(0, bool)
    rows = json.loads(path.read_text())
    return (
        np.array([r["features"] for r in rows]),
        np.array([r["label"] for r in rows]),
        np.array([r["group"] for r in rows]),
        np.array([r["current_photo"] for r in rows]),
    )


def _rates(pred, y) -> dict:
    return {
        "photo_recall": round(float((pred & (y == 1)).sum() / max(1, (y == 1).sum())), 3),
        "graphics_kept": int((pred & (y == 0)).sum()),
        "graphics": int((y == 0).sum()),
        "photographs": int((y == 1).sum()),
    }


def main(devdata, dsb, dedupe):
    X, y, groups, rs = real_set(Path(devdata))
    SX, Sy = synthetic_set(dsb, dedupe)
    CX, Cy, Cg, Ccur = comparison_set()
    AX, Ay, Ag = np.vstack([X, CX]), np.concatenate([y, Cy]), np.concatenate([groups, Cg])
    prob = np.zeros(len(Ay))
    for tr, te in GroupKFold(5).split(AX, Ay, Ag):
        clf = GradientBoostingClassifier(**PARAMS).fit(np.vstack([AX[tr], SX]), np.concatenate([Ay[tr], Sy]))
        prob[te] = clf.predict_proba(AX[te])[:, 1]
    real_prob, cmp_prob = prob[: len(y)], prob[len(y) :]
    # scored on the random sample of panels only: panels added because they took part in
    # a finding are a biased sample
    pred, yr = real_prob[rs] >= THRESHOLD, y[rs]
    labels = json.loads((ROOT / "results" / "panel_classifier" / "labels.json").read_text())
    sample = [m for m in labels if m.get("source", "random_sample") == "random_sample"]
    cv = {
        # graphics in the random sample that the earlier hand-written rules called
        # photographs (their decision is stored with each label as "photo")
        "old_rules_graphics_kept": sum(1 for m in sample if str(m["photo"]) == "1" and m["label"] == 0),
        "labelled_panels": int(len(y) + len(Cy)),
        "random_sample_panels": int(rs.sum()),
        "finding_panels": int((~rs).sum()),
        "comparison_panels": int(len(Cy)),
        "papers": int(len(set(groups))),
        "comparison_papers": int(len(set(Cg))),
        **_rates(pred, yr),
        "threshold": THRESHOLD,
    }
    if len(Cy):
        # comparison panels, held out by paper, with this classifier and with the one
        # before them (trained on retracted papers only), whose decision was recorded
        # when the panels were collected
        cv["comparison_held_out"] = _rates(cmp_prob >= THRESHOLD, Cy)
        cv["comparison_previous_classifier"] = _rates(Ccur.astype(bool), Cy)
    (ROOT / "results" / "panel_classifier" / "cv.json").write_text(json.dumps(cv, indent=1))
    clf = GradientBoostingClassifier(**PARAMS).fit(np.vstack([AX, SX]), np.concatenate([Ay, Sy]))
    (ROOT / "src" / "dupscope" / "classifier.json").write_text(json.dumps(export(clf)))
    print(json.dumps(cv, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
