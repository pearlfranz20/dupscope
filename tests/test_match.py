import cv2
import numpy as np
from conftest import jpeg, tissue

from dupscope.match import compare_panels, load_panel, ransac_similarity, verify_pair


def _warp(img, angle, scale, mirror=False):
    if mirror:
        img = img[:, ::-1]
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    return cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), borderMode=cv2.BORDER_REFLECT)


def test_ransac_recovers_similarity_with_outliers():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 300, (80, 2))
    a = np.deg2rad(30)
    R = 1.5 * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = src @ R.T + [10, -5]
    dst[:30] = rng.uniform(0, 300, (30, 2))  # outliers
    M, inl = ransac_similarity(src.astype(np.float32), dst.astype(np.float32))
    assert M is not None
    assert inl.sum() >= 48
    np.testing.assert_allclose(M[:, :2], R, atol=0.02)


def test_exact_reuse_found():
    img = tissue(3)
    m = compare_panels(load_panel("a", img), load_panel("b", jpeg(img, 80)))
    assert m is not None and m.correlation > 0.9 and m.overlap_frac > 0.9
    assert not m.mirrored and abs(m.scale - 1) < 0.05


def test_rotated_mirrored_scaled_reuse_found():
    img = tissue(4, 300, 300)
    b = cv2.resize(np.ascontiguousarray(np.rot90(img)[:, ::-1]), None, fx=0.8, fy=0.8)
    m = compare_panels(load_panel("a", img), load_panel("b", jpeg(b)))
    assert m is not None and m.mirrored
    assert abs(m.scale - 0.8) < 0.05
    assert m.correlation > 0.85


def test_partial_overlap_found():
    big = tissue(5, 300, 450)
    a, b = big[:, :280], big[:, 170:]
    m = compare_panels(load_panel("a", a), load_panel("b", b))
    assert m is not None and 0.2 < m.overlap_frac < 0.7


def test_unrelated_panels_not_matched():
    assert compare_panels(load_panel("a", tissue(1)), load_panel("b", tissue(2))) is None


def test_disjoint_crops_of_one_image_not_matched():
    big = tissue(6, 260, 600)
    assert compare_panels(load_panel("a", big[:, :260]), load_panel("b", big[:, 330:590])) is None


def test_verify_pair_identity_vs_noise():
    img = tissue(7)
    M = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    corr, detail, _, overlap = verify_pair(img, img, M)
    assert corr > 0.99 and detail > 0.99 and overlap > 0.95
    corr2, _, _, _ = verify_pair(img, tissue(8), M)
    assert corr2 < 0.3
