"""Render the real-paper example on the project page: a panel of a retracted paper
next to the matching region of another panel, aligned by the detector.

    python bench/make_real_example.py DEVDATA_DIR docs/img

Paper: Naunyn-Schmiedeberg's Archives of Pharmacology 396, 1931-1942 (2023),
doi:10.1007/s00210-023-02449-x, CC BY, retracted (Retraction Watch reasons include
duplication and manipulation of images). Figure 5, panels c and e.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from dupscope.cli import default_thresholds
from dupscope.detect import analyse

PMCID = "PMC10409850"


def main(devdata, out):
    out = Path(out)
    gray = cv2.imread(str(Path(devdata) / PMCID / "fig5.png"), cv2.IMREAD_GRAYSCALE)
    panels, findings = analyse({"fig5": gray}, default_thresholds())
    f = findings[0]
    by_id = {p.panel_id: p for p in panels}
    pa, pb = by_id[f.panel_a], by_id[f.panel_b]
    a, b = pa.crop(gray), pb.crop(gray)
    from dupscope.match import compare_panels, load_panel

    t = default_thresholds()
    m = compare_panels(
        load_panel("a", a), load_panel("b", b), t.dup_min_inliers, t.dup_min_correlation, t.dup_min_overlap
    )
    back = cv2.warpAffine(
        b, cv2.invertAffineTransform(np.array(m.transform)), (a.shape[1], a.shape[0]), borderValue=255
    )
    if m.mirrored:
        back = back[:, ::-1]
    gap = np.full((a.shape[0], 16), 255, np.uint8)
    pair = np.hstack([a, gap, back])
    cv2.imwrite(str(out / "real_reuse.png"), cv2.resize(pair, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
    fig = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for p, colour in ((pa, (40, 70, 200)), (pb, (40, 70, 200))):
        cv2.rectangle(fig, (p.x, p.y), (p.x + p.w, p.y + p.h), colour, 6)
    cv2.imwrite(str(out / "real_figure.png"), cv2.resize(fig, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA))
    (out / "real_example.json").write_text(
        json.dumps(
            {
                "pmcid": PMCID,
                "doi": "10.1007/s00210-023-02449-x",
                "panel_a": f.panel_a,
                "panel_b": f.panel_b,
                "category": f.category,
                "correlation": f.correlation,
                "detail": f.detail,
                "inliers": f.inliers,
                "rotation_deg": round(m.rotation_deg),
                "overlap": f.overlap_frac,
                "findings_in_paper": len(findings),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main(*sys.argv[1:3])
