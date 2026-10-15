"""Synthetic papers with known, planted image manipulations.

A paper is 2-3 figures. Each figure is a grid of panels (microscopy crops, western
blots, charts) with letter labels, saved as JPEG like a real PDF figure export.
Manipulated papers get one or two planted manipulations; every paper also gets
look-alike negatives (two non-overlapping fields from the same sample; blots that
share a layout), because those are what make real screening hard.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .blots import lane_bounds, make_blot

KINDS = (
    "reuse_exact",  # same image in two panels
    "reuse_rotflip",  # same image, rotated by 90/180/270 and/or mirrored
    "reuse_contrast",  # same image with brightness/contrast/gamma changed
    "overlap_crop",  # two crops of one image that partly overlap, shown as different
    "clone",  # region copied elsewhere inside one panel
    "blot_reuse",  # same blot strip in two panels
    "blot_lane",  # one blot lane copied onto another
)
PAIR_KINDS = {"reuse_exact", "reuse_rotflip", "reuse_contrast", "overlap_crop", "blot_reuse"}


@dataclass
class GTPanel:
    figure: str
    index: int
    x: int
    y: int
    w: int
    h: int
    content: str  # micro | blot | chart
    source: str = ""
    crop: list = field(default_factory=list)  # [x, y, w, h] in source pixels (micro)


@dataclass
class Planted:
    kind: str
    panels: list  # [(figure, index), ...] one entry for within-panel kinds
    detail: str = ""


@dataclass
class Paper:
    paper_id: str
    figures: dict = field(default_factory=dict)  # figure id -> uint8 image (not serialised)
    panels: list = field(default_factory=list)
    planted: list = field(default_factory=list)
    hard_negatives: list = field(default_factory=list)  # pairs of (figure, index)

    def truth(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "panels": [asdict(p) for p in self.panels],
            "planted": [asdict(p) for p in self.planted],
            "hard_negatives": self.hard_negatives,
        }


_CHARTS: list[np.ndarray] = []


def _charts(rng: np.random.Generator, n: int = 30) -> list[np.ndarray]:
    """Rendered bar/line/scatter charts, used as non-photographic distractors."""
    if _CHARTS:
        return _CHARTS
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for i in range(n):
        fig, ax = plt.subplots(figsize=(2.6, 2.4), dpi=90)
        kind = i % 3
        if kind == 0:
            k = int(rng.integers(2, 5))
            ax.bar(range(k), rng.uniform(1, 5, k), yerr=rng.uniform(0.1, 0.6, k), color="0.4", capsize=3)
            ax.set_ylabel("Response (a.u.)")
        elif kind == 1:
            ax.plot(np.cumsum(rng.standard_normal(40)), color="k")
            ax.set_xlabel("Time (s)")
        else:
            ax.scatter(rng.random(40), rng.random(40), s=10, color="0.2")
            ax.set_xlabel("Dose")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        _CHARTS.append(cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_GRAYSCALE))
    return _CHARTS


def _random_crop(rng, src: np.ndarray, aspect: float, min_side: int = 120) -> tuple[int, int, int, int]:
    h, w = src.shape
    max_w = min(w, int(h * aspect))
    cw = int(rng.integers(min(max_w, max(min_side, int(max_w * 0.45))), max_w + 1))
    ch = max(1, int(round(cw / aspect)))
    ch = min(ch, h)
    x0 = int(rng.integers(0, w - cw + 1))
    y0 = int(rng.integers(0, h - ch + 1))
    return x0, y0, cw, ch


def _overlapping_crop(rng, src: np.ndarray, crop: tuple, aspect: float) -> tuple[int, int, int, int] | None:
    """A crop of the same size that overlaps `crop` by roughly 30-75% of its area."""
    h, w = src.shape
    x0, y0, cw, ch = crop
    for _ in range(50):
        fx, fy = rng.uniform(0.3, 0.85), rng.uniform(0.3, 0.85)
        dx = int((1 - fx) * cw) * (1 if rng.random() < 0.5 else -1)
        dy = int((1 - fy) * ch) * (1 if rng.random() < 0.5 else -1)
        nx, ny = x0 + dx, y0 + dy
        if 0 <= nx and nx + cw <= w and 0 <= ny and ny + ch <= h and 0.3 <= fx * fy <= 0.75:
            return nx, ny, cw, ch
    return None


def _disjoint_crop(rng, src: np.ndarray, crop: tuple) -> tuple[int, int, int, int] | None:
    """A crop of the same size that does not overlap `crop`: a different field of view."""
    h, w = src.shape
    x0, y0, cw, ch = crop
    for _ in range(80):
        nx, ny = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
        if nx + cw <= x0 or nx >= x0 + cw or ny + ch <= y0 or ny >= y0 + ch:
            return nx, ny, cw, ch
    return None


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    interp = cv2.INTER_AREA if img.shape[1] > w else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def _cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale uniformly to cover w x h, then centre-crop (no stretching)."""
    s = max(w / img.shape[1], h / img.shape[0])
    big = cv2.resize(
        img, (max(w, round(img.shape[1] * s)), max(h, round(img.shape[0] * s))), interpolation=cv2.INTER_CUBIC
    )
    y0, x0 = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
    return np.ascontiguousarray(big[y0 : y0 + h, x0 : x0 + w])


def _tone(rng, img: np.ndarray) -> np.ndarray:
    g = rng.choice([rng.uniform(0.55, 0.8), rng.uniform(1.3, 1.8)])
    gain, bias = rng.uniform(0.75, 1.25), rng.uniform(-20, 20)
    out = 255.0 * (img / 255.0) ** g * gain + bias
    return np.clip(out, 0, 255).astype(np.uint8)


def _rotflip(rng, img: np.ndarray) -> tuple[np.ndarray, str]:
    k = int(rng.integers(0, 4))
    flip = bool(rng.random() < 0.5) or k == 0
    out = np.rot90(img, k)
    if flip:
        out = out[:, ::-1]
    return np.ascontiguousarray(out), f"rot{90 * k}" + ("+mirror" if flip else "")


def _clone(rng, img: np.ndarray) -> tuple[np.ndarray, str]:
    """Copy a patch to a non-overlapping place in the same panel."""
    h, w = img.shape
    out = img.copy()
    ps = int(min(h, w) * rng.uniform(0.18, 0.32))
    for _ in range(60):
        sx, sy = int(rng.integers(0, w - ps)), int(rng.integers(0, h - ps))
        tx, ty = int(rng.integers(0, w - ps)), int(rng.integers(0, h - ps))
        if abs(sx - tx) > ps or abs(sy - ty) > ps:
            patch = img[sy : sy + ps, sx : sx + ps]
            if patch.std() < 12:  # cloning empty background is invisible and pointless
                continue
            # feathered edge, as a careful forger (or a clone-stamp tool) would leave
            alpha = np.ones((ps, ps), np.float32)
            alpha = cv2.GaussianBlur(
                cv2.copyMakeBorder(alpha[2:-2, 2:-2], 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0), (5, 5), 0
            )
            dst = out[ty : ty + ps, tx : tx + ps].astype(np.float32)
            out[ty : ty + ps, tx : tx + ps] = (alpha * patch + (1 - alpha) * dst).astype(np.uint8)
            return out, f"{ps}px patch ({sx},{sy})->({tx},{ty})"
    return out, ""


def _blot_lane_copy(rng, blot: np.ndarray) -> tuple[np.ndarray, str]:
    lanes = lane_bounds(blot)
    if len(lanes) < 3:
        return blot, ""
    i, j = rng.choice(len(lanes), 2, replace=False)
    (a0, a1), (b0, b1) = lanes[i], lanes[j]
    wdt = min(a1 - a0, b1 - b0)
    out = blot.copy()
    out[:, b0 : b0 + wdt] = blot[:, a0 : a0 + wdt]
    return out, f"lane {i} -> lane {j}"


def make_paper(seed: int, pool: list[tuple[str, np.ndarray]], manipulated: bool) -> Paper:
    rng = np.random.default_rng(seed)
    charts = _charts(np.random.default_rng(12345))
    paper = Paper(f"paper{seed:04d}")
    n_figs = int(rng.integers(2, 4))

    # 1. lay out figures as grids of cells and decide each cell's content
    cells = []  # (fig, row, col, cw, ch)
    layouts = {}
    for f in range(n_figs):
        fig = f"fig{f + 1}"
        rows, cols = int(rng.integers(2, 4)), int(rng.integers(2, 5))
        cw, ch = int(rng.integers(170, 260)), int(rng.integers(150, 240))
        gutter = int(rng.integers(14, 26))
        layouts[fig] = (rows, cols, cw, ch, gutter)
        for r in range(rows):
            for c in range(cols):
                cells.append((fig, r, c, cw, ch))
    content = []
    for _ in cells:
        u = rng.random()
        content.append("chart" if u < 0.18 else "blot" if u < 0.33 else "micro")

    # 2. a spec per cell: what goes in it
    specs: list[dict] = []
    for (_fig, _r, _c, cw, ch), kind in zip(cells, content, strict=True):
        if kind == "micro":
            name, src = pool[int(rng.integers(len(pool)))]
            specs.append({"kind": "micro", "source": name, "src": src, "crop": _random_crop(rng, src, cw / ch)})
        elif kind == "blot":
            specs.append({"kind": "blot", "blot": make_blot(rng), "source": f"blot-{seed}-{len(specs)}"})
        else:
            specs.append({"kind": "chart", "chart": charts[int(rng.integers(len(charts)))], "source": "chart"})

    micro = [i for i, s in enumerate(specs) if s["kind"] == "micro"]
    blots = [i for i, s in enumerate(specs) if s["kind"] == "blot"]

    def pick_pair(ids):
        if len(ids) < 2:
            return None
        a, b = rng.choice(ids, 2, replace=False)
        return int(a), int(b)

    # 3. hard negatives in every paper: two different fields of the same sample
    pair = pick_pair(micro)
    if pair:
        a, b = pair
        src = specs[a]["src"]
        # smaller crops leave room for a second, separate field of view
        ch_, cw_ = src.shape
        aspect = specs[a]["crop"][2] / max(1, specs[a]["crop"][3])
        side = int(min(cw_ / 2, ch_) * rng.uniform(0.6, 0.95))
        w_ = max(64, min(int(side * min(aspect, 1.0) if aspect < 1 else side), cw_))
        h_ = max(64, min(int(w_ / aspect), ch_))
        first = (int(rng.integers(0, cw_ - w_ + 1)), int(rng.integers(0, ch_ - h_ + 1)), w_, h_)
        other = _disjoint_crop(rng, src, first)
        if other:
            specs[a] = dict(specs[a], crop=first)
            specs[b] = dict(specs[a], crop=other)
            specs[b]["hard_negative_of"] = a

    # 4. planted manipulations
    planted: list[tuple[str, list[int], str]] = []
    if manipulated:
        n = 1 if rng.random() < 0.6 else 2
        weights = np.array([1.0, 1.2, 1.0, 1.2, 1.2, 0.7, 0.7])
        used: set[int] = set()
        for _ in range(n):
            kind = str(rng.choice(KINDS, p=weights / weights.sum()))
            free_micro = [i for i in micro if i not in used and "hard_negative_of" not in specs[i]]
            free_blots = [i for i in blots if i not in used]
            if kind in ("reuse_exact", "reuse_rotflip", "reuse_contrast", "overlap_crop"):
                pr = pick_pair(free_micro)
                if not pr:
                    continue
                a, b = pr
                s = dict(specs[a])
                s.pop("hard_negative_of", None)
                detail = ""
                if kind == "overlap_crop":
                    oc = _overlapping_crop(rng, s["src"], s["crop"], 1.0)
                    if oc is None:
                        continue
                    s["crop"] = oc
                    detail = "partial overlap"
                elif kind == "reuse_rotflip":
                    s["post"] = "rotflip"
                elif kind == "reuse_contrast":
                    s["post"] = "tone"
                specs[b] = s
                planted.append((kind, [a, b], detail))
                used |= {a, b}
            elif kind == "clone":
                cand = [i for i in free_micro if specs[i]["src"].std() > 15]
                if not cand:
                    continue
                a = int(rng.choice(cand))
                specs[a] = dict(specs[a], post="clone")
                planted.append((kind, [a], ""))
                used.add(a)
            elif kind == "blot_reuse":
                pr = pick_pair(free_blots)
                if not pr:
                    continue
                a, b = pr
                specs[b] = dict(specs[a], post="mirror" if rng.random() < 0.3 else "")
                planted.append((kind, [a, b], ""))
                used |= {a, b}
            elif kind == "blot_lane":
                if not free_blots:
                    continue
                a = int(rng.choice(free_blots))
                specs[a] = dict(specs[a], post="lane")
                planted.append((kind, [a], ""))
                used.add(a)

    # 5. render
    canvases = {}
    for fig, (rows, cols, cw, ch, gutter) in layouts.items():
        top = 24
        canvases[fig] = np.full((top + rows * (ch + gutter) + gutter, cols * (cw + gutter) + gutter), 255, np.uint8)
    details: dict[int, str] = {}
    gt_index: dict[int, tuple[str, int]] = {}
    per_fig_count: dict[str, int] = {}
    for idx, ((fig, r, c, cw, ch), s) in enumerate(zip(cells, specs, strict=True)):
        canvas = canvases[fig]
        gutter = layouts[fig][4]
        x, y = gutter + c * (cw + gutter), 24 + gutter + r * (ch + gutter)
        if s["kind"] == "micro":
            x0, y0, w0, h0 = s["crop"]
            # uniform scale and centre crop: a reused image placed in a cell of another
            # shape is cropped, never stretched (stretching is not how figures are made)
            img = _cover(s["src"][y0 : y0 + h0, x0 : x0 + w0], cw, ch)
            post = s.get("post", "")
            if post == "rotflip":
                img, d = _rotflip(rng, img)
                img = _cover(img, cw, ch)
                details[idx] = d
            elif post == "tone":
                img = _tone(rng, img)
            elif post == "clone":
                img, details[idx] = _clone(rng, img)
            pw, ph = cw, ch
        elif s["kind"] == "blot":
            img = s["blot"]
            if s.get("post") == "lane":
                img, details[idx] = _blot_lane_copy(rng, img)
            elif s.get("post") == "mirror":
                img = np.ascontiguousarray(img[:, ::-1])
            scale = min(cw / img.shape[1], ch / img.shape[0], 1.6)
            img = _fit(img, max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale)))
            ph, pw = img.shape
            x, y = x + (cw - pw) // 2, y + (ch - ph) // 2
        else:
            img = s["chart"]
            scale = min(cw / img.shape[1], ch / img.shape[0])
            img = _fit(img, int(img.shape[1] * scale), int(img.shape[0] * scale))
            ph, pw = img.shape
            x, y = x + (cw - pw) // 2, y + (ch - ph) // 2
        canvas[y : y + ph, x : x + pw] = img
        k = per_fig_count.get(fig, 0)
        per_fig_count[fig] = k + 1
        label = chr(ord("A") + k) if k < 26 else str(k)
        cv2.putText(canvas, label, (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
        paper.panels.append(
            GTPanel(fig, k, x, y, pw, ph, s["kind"], s.get("source", ""), list(s["crop"]) if "crop" in s else [])
        )
        gt_index[idx] = (fig, k)

    # 6. export like a journal PDF: optional downscale, JPEG
    factor = float(rng.choice([1.0, 1.0, 0.85, 0.75]))
    quality = int(rng.integers(72, 93))
    for fig, canvas in canvases.items():
        if factor != 1.0:
            canvas = cv2.resize(canvas, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        _, enc = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
        paper.figures[fig] = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    if factor != 1.0:
        for p in paper.panels:
            p.x, p.y, p.w, p.h = (int(round(v * factor)) for v in (p.x, p.y, p.w, p.h))

    for kind, ids, detail in planted:
        extra = "; ".join(details[i] for i in ids if details.get(i))
        paper.planted.append(Planted(kind, [list(gt_index[i]) for i in ids], "; ".join(filter(None, [detail, extra]))))
    for i, s in enumerate(specs):
        if "hard_negative_of" in s and not any(i in ids for _, ids, _ in planted):
            paper.hard_negatives.append([list(gt_index[s["hard_negative_of"]]), list(gt_index[i])])
    return paper
