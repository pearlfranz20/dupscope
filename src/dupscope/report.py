"""A self-contained HTML page for a human reviewer: every finding with the two
regions side by side, the evidence numbers, and the annotated figures."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import cv2
import numpy as np

from .detect import Finding
from .panels import Panel

CSS = """
:root { --bg:#f7f8fa; --surface:#fff; --fg:#151b23; --muted:#5a6475; --line:#dfe3ea; --accent:#2b5fa8; --flag:#b4462a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#0f1318; --surface:#171c24; --fg:#e4e8ee; --muted:#a0aab9; --line:#2b333f; --accent:#86afe6; --flag:#f08a6c; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#0f1318; --surface:#171c24; --fg:#e4e8ee; --muted:#a0aab9; --line:#2b333f; --accent:#86afe6; --flag:#f08a6c; color-scheme:dark; }
body { background:var(--bg); color:var(--fg); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; margin:0; }
main { max-width:980px; margin:0 auto; padding-block:28px 60px;
  padding-left:max(16px, env(safe-area-inset-left,0px)); padding-right:max(16px, env(safe-area-inset-right,0px)); display:grid; gap:26px; }
h1 { font-size:1.6rem; margin:0; } h2 { font-size:1.15rem; margin:0; }
.note { background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:6px; padding:12px 14px; max-width:75ch; }
.finding { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:14px; display:grid; gap:10px; }
.pair { display:flex; flex-wrap:wrap; gap:12px; align-items:flex-start; }
.pair figure { margin:0; display:grid; gap:4px; } .pair img { max-width:100%; height:auto; border:1px solid var(--line); image-rendering:auto; }
figcaption, .muted { color:var(--muted); font-size:.85rem; }
table { border-collapse:collapse; font-size:.85rem; font-variant-numeric:tabular-nums; }
td, th { padding:3px 10px 3px 0; text-align:left; }
.cat { color:var(--flag); font-weight:600; }
.figs img { max-width:100%; height:auto; border:1px solid var(--line); }
"""


def _png(img: np.ndarray, max_side: int = 900) -> str:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(enc.tobytes()).decode()


def _crop_quad(fig: np.ndarray, quad: list, pad: int = 6) -> np.ndarray:
    q = np.array(quad)
    x0, y0 = np.floor(q.min(0)).astype(int) - pad
    x1, y1 = np.ceil(q.max(0)).astype(int) + pad
    h, w = fig.shape[:2]
    return fig[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]


def write_report(
    out_dir: str | Path,
    figures: dict[str, np.ndarray],
    panels: list[Panel],
    findings: list[Finding],
    overlays: dict[str, np.ndarray],
    title: str = "dupscope report",
) -> Path:
    out = Path(out_dir)
    parts = [
        f"<title>{html.escape(title)}</title>",
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        f"<style>{CSS}</style><main>",
        f"<h1>{html.escape(title)}</h1>",
        '<p class="note">Each finding is a lead for a person to check, not a verdict. Reuse can be '
        "legitimate (a magnified inset, a shared control image that the legend declares). Look at the two "
        "regions, read the figure legends, and judge.</p>",
        f'<p class="muted">{len(figures)} figures, {sum(p.photographic for p in panels)} image panels checked, '
        f"{len(findings)} findings.</p>",
    ]
    gray = {k: (v if v.ndim == 2 else cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)) for k, v in figures.items()}
    for n, f in enumerate(findings, start=1):
        fa, fb = f.panel_a.rsplit("#", 1)[0], f.panel_b.rsplit("#", 1)[0]
        a = _crop_quad(gray[fa], f.region_a)
        b = _crop_quad(gray[fb], f.region_b)
        transform = []
        if abs(f.rotation_deg) > 1:
            transform.append(f"rotated {f.rotation_deg:.0f}°")
        if f.mirrored:
            transform.append("mirrored")
        if abs(f.scale - 1) > 0.05:
            transform.append(f"scaled ×{f.scale:.2f}")
        parts += [
            '<section class="finding">',
            f'<h2>{n}. <span class="cat">{html.escape(f.category)}</span></h2>',
            '<div class="pair">',
            f'<figure><img alt="region in {html.escape(f.panel_a)}" src="{_png(a, 420)}">'
            f"<figcaption>{html.escape(f.panel_a)}</figcaption></figure>",
            f'<figure><img alt="region in {html.escape(f.panel_b)}" src="{_png(b, 420)}">'
            f"<figcaption>{html.escape(f.panel_b)}</figcaption></figure>",
            "</div>",
            "<table><tr><th>score</th><th>matched points</th><th>pixel correlation</th><th>fine-detail correlation</th>"
            "<th>overlap</th><th>transform</th></tr>",
            f"<tr><td>{f.score}</td><td>{f.inliers}</td><td>{f.correlation}</td><td>{f.detail}</td>"
            f"<td>{f.overlap_frac:.0%}</td><td>{html.escape(', '.join(transform) or 'none')}</td></tr></table>",
            "</section>",
        ]
    parts.append('<h2>Annotated figures</h2><div class="figs">')
    for fig_id, img in overlays.items():
        parts.append(
            f'<figure><img alt="{html.escape(fig_id)} annotated" src="{_png(img, 1400)}">'
            f"<figcaption>{html.escape(fig_id)}</figcaption></figure>"
        )
    parts.append("</div></main>")
    path = out / "report.html"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
