import pytest
from conftest import jpeg, tissue

from dupscope.dense import find_dense_clones
from dupscope.match import load_panel


def test_pasted_patch_found():
    img = tissue(11, 300, 300).copy()
    img[180:250, 170:250] = img[30:100, 40:120]
    clones = find_dense_clones(load_panel("p", jpeg(img, 90)))
    top = max(clones, key=lambda m: m.inliers)
    assert (top.transform[0][2], top.transform[1][2]) == (130, 150)
    assert top.inliers > 1000 and top.correlation > 0.95 and top.method == "dense"


@pytest.mark.parametrize("seed", range(12, 24))
def test_untouched_texture_gets_no_strong_clone(seed):
    # look-alike structures can produce weak candidates (this is why clone search is
    # experimental), but nothing near the vote count of a real pasted region
    clones = find_dense_clones(load_panel("p", jpeg(tissue(seed, 300, 300), 90)))
    assert all(c.inliers < 300 for c in clones)
