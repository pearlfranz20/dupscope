import cv2
import numpy as np
import pytest

from dupscope.match import load_panel
from dupscope.template import template_search


def sparse(seed):
    """A few dim nuclei on black: the case keypoint matching struggles with."""
    rng = np.random.default_rng(seed)
    img = np.zeros((300, 300), np.uint8)
    for _ in range(9):
        c = (int(rng.integers(20, 280)), int(rng.integers(20, 280)))
        cv2.circle(img, c, int(rng.integers(6, 14)), int(rng.integers(60, 200)), -1)
    return cv2.GaussianBlur(img, (0, 0), 2) + rng.integers(0, 6, img.shape).astype(np.uint8)


@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("flip", [False, True])
def test_every_orientation(k, flip):
    img = sparse(0)
    b = np.rot90(img, k)
    b = b[:, ::-1] if flip else b
    b = cv2.resize(np.ascontiguousarray(b), None, fx=0.8, fy=0.8)
    m = template_search(load_panel("a", img), load_panel("b", b))
    assert m is not None
    assert m.correlation > 0.95 and m.mirrored == flip and abs(m.scale - 0.8) < 0.03
    assert m.method == "template"


def test_different_fields_rejected():
    assert template_search(load_panel("a", sparse(0)), load_panel("b", sparse(1))) is None
