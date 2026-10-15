import cv2
import numpy as np
from conftest import grid_figure, tissue

from dupscope.panels import extract_panels, is_photographic


def test_grid_panels_found_in_reading_order():
    fig = grid_figure([tissue(i) for i in range(4)])
    panels = extract_panels(fig, "fig1")
    assert len(panels) == 4
    assert all(p.photographic for p in panels)
    xs_ys = [(p.y // 100, p.x // 100) for p in panels]
    assert xs_ys == sorted(xs_ys)
    for p in panels:
        assert 250 <= p.w <= 270 and 250 <= p.h <= 270


def test_labels_do_not_merge_rows():
    fig = grid_figure([tissue(i) for i in range(4)])
    for x, y in [(4, 18), (288, 18), (4, 302), (288, 302)]:
        cv2.putText(fig, "A", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    assert len(extract_panels(fig, "f")) == 4


def test_chart_is_not_photographic():
    chart = np.full((200, 260), 255, np.uint8)
    for i, hgt in enumerate([60, 120, 90, 150]):
        cv2.rectangle(chart, (20 + i * 55, 190 - hgt), (55 + i * 55, 190), 90, -1)
    cv2.line(chart, (15, 190), (250, 190), 0, 2)
    assert not is_photographic(chart)
    assert is_photographic(tissue(1))


def test_flat_fill_is_not_photographic():
    assert not is_photographic(np.full((120, 120), 180, np.uint8))
