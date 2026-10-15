"""Render the example images on the project page from benchmark papers.

    python bench/make_examples.py DSB_DIR DEDUPE_JSON docs/img

Needs the StarDist DSB2018 folder used by the benchmark. Paper seeds are from the
held-out test split; their findings use the shipped thresholds.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from dupscope.bench.compose import make_paper
from dupscope.bench.sources import load_pool
from dupscope.cli import default_thresholds
from dupscope.detect import analyse, draw_overlay
from dupscope.panels import to_gray
from dupscope.report import _crop_quad

SHOWCASE = 387  # a manipulated test paper: one contrast-adjusted reuse, one rotated reuse
FALSE_ALARM = 354  # the clean test paper the detector wrongly flags
MISSED = 269  # a test paper whose reuse is missed: a mostly dark field


def pair(fig_a, quad_a, fig_b, quad_b, h=220):
    a, b = _crop_quad(fig_a, quad_a), _crop_quad(fig_b, quad_b)
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    gap = np.full((h, 16), 255, np.uint8)
    return np.hstack([a, gap, b])


def main(dsb, dedupe, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    keep = set(json.loads(Path(dedupe).read_text())["kept"])
    pool = [(n, a) for n, a in load_pool(dsb) if n in keep]
    t = default_thresholds()
    meta = {}
    for name, seed in [("showcase", SHOWCASE), ("false_alarm", FALSE_ALARM)]:
        paper = make_paper(seed, pool, manipulated=seed % 2 == 1)
        panels, findings = analyse(paper.figures, t)
        gray = {k: to_gray(v) for k, v in paper.figures.items()}
        for n, f in enumerate(findings):
            fa, fb = f.panel_a.split("#")[0], f.panel_b.split("#")[0]
            cv2.imwrite(str(out / f"{name}_{n}.png"), pair(gray[fa], f.region_a, gray[fb], f.region_b))
        if name == "showcase":
            for fig_id, img in draw_overlay(paper.figures, panels, findings).items():
                img = cv2.resize(img, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out / f"{name}_{fig_id}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        meta[name] = {
            "seed": seed,
            "planted": [p.kind for p in paper.planted],
            "findings": [
                {
                    "category": f.category,
                    "panel_a": f.panel_a,
                    "panel_b": f.panel_b,
                    "correlation": f.correlation,
                    "detail": f.detail,
                    "rotation_deg": f.rotation_deg,
                    "mirrored": f.mirrored,
                    "scale": f.scale,
                }
                for f in findings
            ],
        }
    paper = make_paper(MISSED, pool, manipulated=True)
    pl = next(p for p in paper.planted if p.kind == "reuse_exact")
    gt = {(g.figure, g.index): g for g in paper.panels}
    crops = []
    for f, i in pl.panels:
        g = gt[(f, i)]
        crops.append(to_gray(paper.figures[f])[g.y : g.y + g.h, g.x : g.x + g.w])
    h = 220
    crops = [cv2.resize(c, (int(c.shape[1] * h / c.shape[0]), h)) for c in crops]
    crops = [np.clip(c.astype(np.float32) * 6, 0, 255).astype(np.uint8) for c in crops]  # brightened to be visible
    cv2.imwrite(str(out / "missed.png"), np.hstack([crops[0], np.full((h, 16), 255, np.uint8), crops[1]]))
    meta["missed"] = {"seed": MISSED, "kind": pl.kind, "detail": pl.detail}
    (out / "examples.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
