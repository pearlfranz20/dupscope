import cv2
import numpy as np
import pytest


def tissue(seed: int, h: int = 260, w: int = 260) -> np.ndarray:
    """A textured grayscale image that looks enough like a micrograph for keypoints:
    blurred noise at two scales plus a few bright blobs."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.normal(0, 1, (h, w)), (0, 0), 6) * 400
    img += cv2.GaussianBlur(rng.normal(0, 1, (h, w)), (0, 0), 1.5) * 60
    for _ in range(12):
        c = (int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10)))
        cv2.circle(img, c, int(rng.integers(5, 14)), float(rng.uniform(40, 90)), -1)
    return np.clip(img + 110, 0, 255).astype(np.uint8)


def grid_figure(panels: list[np.ndarray], cols: int = 2, gutter: int = 24, cell: int = 260) -> np.ndarray:
    rows = (len(panels) + cols - 1) // cols
    fig = np.full((rows * (cell + gutter) + gutter, cols * (cell + gutter) + gutter), 255, np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        y, x = gutter + r * (cell + gutter), gutter + c * (cell + gutter)
        p = cv2.resize(p, (cell, cell), interpolation=cv2.INTER_AREA)
        fig[y : y + cell, x : x + cell] = p
    return fig


def jpeg(img: np.ndarray, q: int = 85) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


@pytest.fixture
def make_tissue():
    return tissue


@pytest.fixture
def make_grid():
    return grid_figure
