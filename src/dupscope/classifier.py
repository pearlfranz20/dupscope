"""Photograph-versus-graphic decision for a panel, from a small gradient-boosted
tree model stored in classifier.json (trained by bench/train_panel_classifier.py)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .panelfeatures import features

MODEL = Path(__file__).with_name("classifier.json")


@lru_cache(maxsize=1)
def _model() -> dict:
    return json.loads(MODEL.read_text())


def photo_probability(gray: np.ndarray) -> float:
    m = _model()
    x = features(gray)
    raw = m["init"]
    for t in m["trees"]:
        node = 0
        while t["left"][node] != -1:
            node = t["left"][node] if x[t["feature"][node]] <= t["threshold"][node] else t["right"][node]
        raw += m["learning_rate"] * t["value"][node]
    return float(1 / (1 + np.exp(-raw)))


def is_photo(gray: np.ndarray) -> bool:
    if gray.size == 0 or min(gray.shape) < 8:
        return False
    return photo_probability(gray) >= _model()["threshold"]
