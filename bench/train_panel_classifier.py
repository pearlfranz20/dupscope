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


def main(devdata, dsb, dedupe):
    X, y, groups, rs = real_set(Path(devdata))
    SX, Sy = synthetic_set(dsb, dedupe)
    prob = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = GradientBoostingClassifier(**PARAMS).fit(np.vstack([X[tr], SX]), np.concatenate([y[tr], Sy]))
        prob[te] = clf.predict_proba(X[te])[:, 1]
    # scored on the random sample of panels only: panels added because they took part in
    # a finding are a biased sample
    pred, yr = prob[rs] >= THRESHOLD, y[rs]
    cv = {
        "labelled_panels": int(len(y)),
        "random_sample_panels": int(rs.sum()),
        "finding_panels": int((~rs).sum()),
        "photographs": int(y.sum()),
        "papers": int(len(set(groups))),
        "photo_recall": round(float((pred & (yr == 1)).sum() / yr.sum()), 3),
        "graphics_kept": int((pred & (yr == 0)).sum()),
        "graphics": int((yr == 0).sum()),
        "threshold": THRESHOLD,
    }
    (ROOT / "results" / "panel_classifier" / "cv.json").write_text(json.dumps(cv, indent=1))
    clf = GradientBoostingClassifier(**PARAMS).fit(np.vstack([X, SX]), np.concatenate([y, Sy]))
    (ROOT / "src" / "dupscope" / "classifier.json").write_text(json.dumps(export(clf)))
    np.savez("/tmp/clf_check.npz", X=X, p=clf.predict_proba(X)[:, 1])
    print(cv)


if __name__ == "__main__":
    main(*sys.argv[1:4])
