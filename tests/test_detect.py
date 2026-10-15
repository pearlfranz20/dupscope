import numpy as np
from conftest import grid_figure, jpeg, tissue

from dupscope.detect import Thresholds, analyse, draw_overlay


def test_reuse_across_figures_reported():
    shared = tissue(20)
    f1 = jpeg(grid_figure([tissue(21), shared, tissue(22), tissue(23)]))
    f2 = jpeg(grid_figure([tissue(24), tissue(25), np.ascontiguousarray(np.fliplr(shared))]))
    panels, findings = analyse({"fig1": f1, "fig2": f2})
    assert len(panels) == 7
    assert len(findings) == 1
    f = findings[0]
    assert {f.panel_a, f.panel_b} == {"fig1#1", "fig2#2"}
    assert f.mirrored and f.category == "same image, rotated or mirrored"
    out = draw_overlay({"fig1": f1, "fig2": f2}, panels, findings)
    assert out["fig1"].shape[:2] == f1.shape[:2]


def test_clean_paper_has_no_findings():
    f1 = jpeg(grid_figure([tissue(i) for i in range(30, 34)]))
    f2 = jpeg(grid_figure([tissue(i) for i in range(34, 38)]))
    _, findings = analyse({"fig1": f1, "fig2": f2})
    assert findings == []


def test_clone_search_is_opt_in():
    img = tissue(50, 300, 300).copy()
    img[180:250, 170:250] = img[30:100, 40:120]
    fig = jpeg(grid_figure([img, tissue(51)]), 90)
    assert analyse({"fig1": fig})[1] == []
    _, findings = analyse({"fig1": fig}, Thresholds(clones=True))
    assert findings and findings[0].kind == "clone" and findings[0].panel_a == "fig1#0"
