"""Build docs/index.html from the result files, so every number on the page comes
from results/benchmark/summary.json and (when present) results/retractions/.

    python docs/build.py            # page for GitHub Pages, images as files
    python docs/build.py --inline   # single self-contained file (images embedded)
"""

from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

from dupscope.bench.run import GRID

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
KIND_NAMES = {
    "reuse_exact": "Same image in two panels",
    "reuse_rotflip": "Same image, rotated or mirrored",
    "reuse_contrast": "Same image, brightness or contrast changed",
    "overlap_crop": "Two crops of one image that partly overlap",
    "blot_reuse": "Same western blot strip twice",
}


def pct(a: int, b: int) -> str:
    return f"{100 * a / max(1, b):.0f}%"


def img(name: str, alt: str, inline: bool) -> str:
    path = DOCS / "img" / name
    src = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode() if inline else f"img/{name}"
    return f'<img src="{src}" alt="{html.escape(alt)}" loading="lazy">'


def retraction_section(inline: bool) -> str:
    rdir = ROOT / "results" / "retractions"
    if not (rdir / "summary.json").exists():
        return """
<section id="retracted">
  <h2>Papers retracted for image problems</h2>
  <p>The second test uses real papers. The <a href="https://gitlab.com/crossref/retraction-watch-data">Retraction
  Watch database</a> lists neuroscience papers retracted because of duplicated or manipulated images. The check
  downloads their figures from PubMed Central's open-access collection and runs the detector on them. As a
  comparison, it also runs on papers from the same journal and year that were never retracted. Those are reported
  only as counts and are never named.</p>
  <p class="callout">This run has not happened yet. It needs internet access that the build machine lacked, so it
  runs as a GitHub Actions job (<code>.github/workflows/validate.yml</code>) and the results will appear here.</p>
</section>"""
    s = json.loads((rdir / "summary.json").read_text())
    r, c = s["retracted"], s["controls"]
    cases = json.loads((rdir / "retracted.json").read_text())
    flagged = [x for x in cases if x["result"]["n_findings"] > 0]
    flagged.sort(key=lambda x: -x["result"]["max_score"])
    rows = []
    for x in flagged:
        ret, res = x["retraction"], x["result"]
        doi = ret["original_doi"]
        link = (
            f'<a href="https://doi.org/{html.escape(doi)}">{html.escape(ret["title"][:90])}</a>'
            if doi
            else html.escape(ret["title"][:90])
        )
        rows.append(
            f'<tr><td>{link}<div class="sub">{html.escape(ret["journal"])}, {html.escape(ret["original_year"])}</div></td>'
            f'<td>{html.escape(", ".join(res["categories"]))}</td><td class="num">{res["n_findings"]}</td>'
            f'<td class="num">{res["max_score"]:.2f}</td></tr>'
        )
    table = (
        '<div class="table-wrap"><table><thead><tr><th>Retracted paper</th><th>What was found</th>'
        '<th class="num">Findings</th><th class="num">Top score</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
        if rows
        else ""
    )
    return f"""
<section id="retracted">
  <h2>Papers retracted for image problems</h2>
  <p>The second test uses real papers. The <a href="https://gitlab.com/crossref/retraction-watch-data">Retraction
  Watch database</a> lists neuroscience papers retracted because of duplicated or manipulated images. The check
  downloaded their figures from PubMed Central's open-access collection and ran the detector with the same
  thresholds as above. As a comparison, it also ran on papers from the same journal and year that were never
  retracted. Those are reported only as counts and are never named.</p>
  <div class="stats">
    <div class="stat"><div class="big">{r["flagged"]} of {r["analysed"]}</div><div>retracted papers flagged ({pct(r["flagged"], r["analysed"])})</div></div>
    <div class="stat"><div class="big">{c["flagged"]} of {c["analysed"]}</div><div>comparison papers flagged ({pct(c["flagged"], c["analysed"])})</div></div>
  </div>
  <p>A retraction notice says that a paper had an image problem, not which figure or what kind. Some problems are
  outside what this tool looks for, for example a spliced blot or a duplicate of an image from another paper. So
  the retracted share flagged is a floor on what a person could find, not a recall figure. A flag on a comparison
  paper is also not an accusation. Reuse is sometimes legitimate (a magnified inset, a shared control that the
  legend declares), and the detector also makes mistakes.</p>
  {table}
  <p class="sub">{r["skipped"]} retracted and {c["skipped"]} comparison papers were skipped because their figures
  were not in the open-access collection or had no image panels.</p>
</section>"""


def build(inline: bool = False) -> str:
    s = json.loads((ROOT / "results" / "benchmark" / "summary.json").read_text())
    ex = json.loads((DOCS / "img" / "examples.json").read_text())
    ms = json.loads((ROOT / "results" / "benchmark" / "misses.json").read_text())
    wg = json.loads((ROOT / "results" / "benchmark" / "summary_wider_grid.json").read_text())
    test, dev = s["test"], s["dev"]
    found = sum(v["found"] for v in test["recall_by_kind"].values())
    planted = sum(v["planted"] for v in test["recall_by_kind"].values())
    kind_rows = "".join(
        f'<tr><td>{KIND_NAMES[k]}</td><td class="num">{v["found"]} of {v["planted"]}</td>'
        f'<td class="bar-cell"><span class="bar" style="width:{100 * v["recall"]:.0f}%"></span></td></tr>'
        for k, v in sorted(test["recall_by_kind"].items(), key=lambda kv: -kv[1]["recall"])
    )
    clones = s["clone_search_experimental_test"]
    cl_strict = min(clones, key=lambda r: (r["clean_flag_rate"], -r["clone_recall"]))
    cl_loose = max(clones, key=lambda r: (r["clone_recall"], -r["clean_flag_rate"]))
    t = s["thresholds"]
    sc = ex["showcase"]["findings"]
    fa = ex["false_alarm"]["findings"][0]

    def finding_caption(f):
        bits = []
        if abs(f["rotation_deg"]) > 1:
            bits.append(f"rotated {abs(f['rotation_deg']):.0f}°")
        if f["mirrored"]:
            bits.append("mirrored")
        if abs(f["scale"] - 1) > 0.05:
            bits.append(f"resized ×{f['scale']:.2f}")
        how = ", ".join(bits) if bits else "same orientation"
        return f"{f['panel_a']} and {f['panel_b']}: {how}. Pixel correlation {f['correlation']:.2f}, fine-detail correlation {f['detail']:.2f}."

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>dupscope</title>
<meta name="description" content="Finds reused image panels in scientific figures for human review, with an honest benchmark.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>
:root {{
  --bg: #f7f8fa; --surface: #ffffff; --fg: #141a22; --muted: #586274; --faint: #8a93a3;
  --line: #e1e5eb; --accent: #2a5ea8; --accent-soft: #e6eef9; --series: #2a78d6; --flag: #b4462a;
  --code-bg: #eef1f5;
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --font-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0f1318; --surface: #161b22; --fg: #e4e8ee; --muted: #a2acbb; --faint: #7c8696;
  --line: #2a323d; --accent: #86afe6; --accent-soft: #1b2738; --series: #3987e5; --flag: #f08a6c;
  --code-bg: #1d242d; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #0f1318; --surface: #161b22; --fg: #e4e8ee; --muted: #a2acbb; --faint: #7c8696;
  --line: #2a323d; --accent: #86afe6; --accent-soft: #1b2738; --series: #3987e5; --flag: #f08a6c;
  --code-bg: #1d242d; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 16px; line-height: 1.6; margin: 0; }}
main {{ max-width: 760px; margin: 0 auto; padding-block: 40px 72px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 44px; }}
section, header {{ display: grid; gap: 14px; min-width: 0; }}
h1, h2 {{ font-family: var(--font-display); font-weight: 600; text-wrap: balance; margin: 0; letter-spacing: -0.01em; }}
h1 {{ font-size: clamp(1.9rem, 5vw, 2.6rem); line-height: 1.12; }}
h2 {{ font-size: 1.4rem; line-height: 1.25; }}
p, ul, ol {{ margin: 0; max-width: 68ch; }}
ul, ol {{ padding-left: 1.2em; display: grid; gap: 6px; }}
.eyebrow {{ font-family: var(--font-mono); font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); }}
.lede {{ font-size: 1.12rem; color: var(--muted); }}
.sub {{ color: var(--muted); font-size: 0.86rem; }}
code {{ font-family: var(--font-mono); font-size: 0.86em; background: var(--code-bg); padding: 1px 5px; border-radius: 4px; overflow-wrap: anywhere; }}
pre {{ background: var(--code-bg); padding: 12px 14px; border-radius: 6px; overflow-x: auto; margin: 0; font-size: 0.86rem; }}
pre code {{ background: none; padding: 0; }}
a {{ color: var(--accent); }}
.callout {{ background: var(--accent-soft); border-left: 3px solid var(--accent); padding: 10px 14px; border-radius: 4px; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
.stat {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; font-size: 0.9rem; color: var(--muted); }}
.stat .big {{ font-family: var(--font-display); font-size: 1.7rem; color: var(--fg); line-height: 1.2; font-variant-numeric: tabular-nums; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
th, td {{ text-align: left; padding: 7px 10px 7px 0; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font-weight: 600; color: var(--muted); font-size: 0.82rem; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.bar-cell {{ width: 35%; vertical-align: middle; }}
.bar {{ display: block; height: 10px; background: var(--series); border-radius: 2px; }}
figure {{ margin: 0; display: grid; gap: 6px; }}
figure img {{ max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 4px; background: #fff; }}
figcaption {{ color: var(--muted); font-size: 0.86rem; max-width: 68ch; }}
.figs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
.tag {{ color: var(--flag); font-weight: 600; }}
</style>
</head>
<body>
<main>
<header>
  <div class="eyebrow">dupscope</div>
  <h1>Finding reused images in scientific figures</h1>
  <p class="lede">A tool that checks a paper's figures for image panels that appear more than once, even when
  rotated, mirrored, resized, cropped or re-toned, and writes a report for a person to review. It was measured on
  synthetic papers with known answers, and is set up to run on real papers retracted for image problems.</p>
</header>

<section>
  <h2>Why this matters</h2>
  <p>Duplicated images are a common reason papers in biology and neuroscience are corrected or retracted. Some reuse is an honest mistake, some is deliberate, and a little is legitimate. Spotting it by eye
  means comparing every image panel with every other one, which nobody does reliably across a paper with dozens of
  panels. Software can do the comparing, and a person can do the judging.</p>
</section>

<section>
  <h2>How it works</h2>
  <ol>
    <li><strong>Split figures into panels.</strong> Labels are removed, then each figure is cut along its white
    gutters. Each panel is classed as an image (microscopy, blots) or a graphic (charts, diagrams). Only images are
    compared.</li>
    <li><strong>Compare every pair of image panels.</strong> Distinctive points are matched between the two panels
    (SIFT features). A geometric check keeps only matches that agree on one rotation, mirroring and resizing.</li>
    <li><strong>Check the pixels.</strong> The second panel is warped onto the first, and the overlapping pixels are
    compared. Copies agree on fine grain and noise, not just on shapes. Two different cells that look alike do not.</li>
    <li><strong>Report.</strong> Each finding shows the two regions side by side, labelled with what kind of reuse it
    looks like, for a person to judge.</li>
  </ol>
</section>

<section id="example">
  <h2>What a report shows</h2>
  <p>A synthetic test paper with two planted reuses. The detector found both and nothing else.</p>
  <div class="figs">
    <figure>{img("showcase_fig1.png", "Figure 1 with two panels outlined", inline)}<figcaption>Figure 1, with the flagged panels outlined.</figcaption></figure>
    <figure>{img("showcase_fig2.png", "Figure 2 with two panels outlined", inline)}<figcaption>Figure 2.</figcaption></figure>
  </div>
  <figure>{img("showcase_1.png", "A panel and its rotated, mirrored copy", inline)}<figcaption><span class="tag">{html.escape(sc[1]["category"])}.</span> {html.escape(finding_caption(sc[1]))}</figcaption></figure>
  <figure>{img("showcase_0.png", "A panel and a copy with changed brightness", inline)}<figcaption><span class="tag">{html.escape(sc[0]["category"])}.</span> {html.escape(finding_caption(sc[0]))} The planted copy had its brightness and contrast changed.</figcaption></figure>
</section>

<section id="benchmark">
  <h2>How well it works on papers with known answers</h2>
  <p>Real manipulated papers do not come with an answer key, so the first test uses synthetic papers. Each one has
  2 to 3 figures built from real microscopy images (the
  <a href="https://github.com/stardist/stardist/releases">Data Science Bowl 2018 nuclei set</a> and scikit-image
  samples), generated western blots and charts. Figures are saved as compressed JPEG, and half are shrunk, as
  happens in a real PDF. Some papers get planted reuse. Most also get look-alikes: two different regions of the
  same sample, which look similar but share no pixels. Those are the cases a careless detector flags.</p>
  <p>Thresholds were chosen on {s["n_dev"]} development papers, with a rule that at most
  {s["max_clean_flag_rate"]:.0%} of clean papers may be flagged. Wrongly flagging an honest paper is the costly
  mistake. The numbers below are from {s["n_test"]} separate test papers that played no part in tuning. Of
  those, {test["manipulated_papers"]} have planted reuse between panels and {test["clean_papers"]} have no reuse
  at all. Another {test["papers_with_unplanted_duplicates"]} ended up with a real duplicate the generator did not
  mean to make (two random crops of one image that overlap). They are scored apart from both groups, and the
  detector flagged {test["unplanted_duplicate_papers_flagged"]} of them. The remaining
  {test["papers"] - test["manipulated_papers"] - test["clean_papers"] - test["papers_with_unplanted_duplicates"]}
  only have a region pasted within one panel, which the default search does not look for.</p>
  <div class="stats">
    <div class="stat"><div class="big">{found} of {planted}</div><div>planted reuses found ({pct(found, planted)})</div></div>
    <div class="stat"><div class="big">{test["clean_papers_flagged"]} of {test["clean_papers"]}</div><div>clean papers wrongly flagged ({pct(test["clean_papers_flagged"], test["clean_papers"])})</div></div>
    <div class="stat"><div class="big">{test["hard_negative_hits"]} of {test["hard_negatives_total"]}</div><div>look-alike pairs wrongly flagged</div></div>
    <div class="stat"><div class="big">{test["real_findings"]} of {test["findings"]}</div><div>findings were real reuse</div></div>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Kind of reuse</th><th class="num">Found</th><th></th></tr></thead>
    <tbody>{kind_rows}</tbody>
  </table></div>
  <p class="sub">Counts are small, so a few cases move each rate a lot. On the development papers the same
  thresholds found {dev["found"]} of {dev["planted"]} and flagged {dev["clean_papers_flagged"]} of
  {dev["clean_papers"]} clean papers. The threshold search was widened once after an independent review, and its
  correlation floor is {min(GRID["dup_min_correlation"])} because below that a reviewer cannot see the match. With a
  floor of 0.3 the search would pick {wg["thresholds"]["dup_min_correlation"]}, which finds
  {wg["dev"]["found"]} of {wg["dev"]["planted"]} on the development papers and, on the test papers,
  {wg["test"]["found"]} of {wg["test"]["planted"]} while flagging {wg["test"]["clean_papers_flagged"]} of
  {wg["test"]["clean_papers"]} clean papers.</p>
</section>

<section id="limits">
  <h2>Where it falls short</h2>
  <ul>
    <li><strong>Mostly dark images.</strong> {"All" if ms["missed_microscopy"] == ms["missed"] else f"Of the {ms['missed']} reuses the test missed,"}
    {ms["missed_microscopy"]} {"missed reuses" if ms["missed_microscopy"] == ms["missed"] else ""} were microscopy panels in which at most
    {ms["max_bright_fraction_microscopy"]:.1%} of pixels rose above the dark background. A field with a few nuclei
    on black has little to match, and after JPEG compression the background is mostly noise.
    <figure>{img("missed.png", "Two mostly dark panels that are the same image, missed", inline)}<figcaption>A missed copy: the same image in two panels. Shown six times brighter than in the figure.</figcaption></figure></li>
    <li><strong>Small overlaps.</strong> Two crops that share a strip of their area are the hardest case
    ({test["recall_by_kind"].get("overlap_crop", {}).get("found", 0)} of {test["recall_by_kind"].get("overlap_crop", {}).get("planted", 0)} found).</li>
    <li><strong>Blots with a similar layout.</strong> The one false alarm in the test was two different western
    blots with a similar two-band layout, matched after turning one upside down and mirroring it
    (pixel correlation {fa["correlation"]:.2f}). Blots are simple patterns, so a reviewer should look closely at
    any blot finding with a modest score.
    <figure>{img("false_alarm_0.png", "Two different blots, wrongly flagged", inline)}<figcaption>Two different blots, wrongly flagged.</figcaption></figure></li>
    <li><strong>Copies within one panel are not searched by default.</strong> The tool can look for a region pasted
    elsewhere in the same image, but on the benchmark it could not tell real pastes from natural repetition (similar
    nuclei, evenly spaced blot lanes). At its strictest it found {cl_strict["clone_found"]} of
    {cl_strict["clone_planted"]} pastes and still flagged {cl_strict["clean_flag_rate"]:.1%} of clean papers. At
    its loosest it found {cl_loose["clone_found"]} of {cl_loose["clone_planted"]} and flagged
    {cl_loose["clean_flag_rate"]:.0%}. It is available as an experimental option.</li>
    <li><strong>Synthetic papers are not real papers.</strong> Real figures have insets, arrows, scale bars,
    composite colour images and stranger layouts. The retracted-paper check below is the test on real figures.</li>
  </ul>
</section>
{retraction_section(inline)}
<section id="use">
  <h2>Using it</h2>
<pre><code>pip install git+https://github.com/pearlfranz20/dupscope
dupscope scan fig1.png fig2.png fig3.png --out report/</code></pre>
  <p>Open <code>report/report.html</code>. Every finding is a lead for a person to check, not a verdict.
  Benchmark thresholds: at least {t["dup_min_inliers"]} geometrically consistent matched points, pixel
  correlation of at least {t["dup_min_correlation"]}, and at least {t["dup_min_overlap"]:.0%} of a panel
  overlapping. Code, benchmark generator and tests are on
  <a href="https://github.com/pearlfranz20/dupscope">GitHub</a>.</p>
</section>
</main>
</body>
</html>
"""
    return page


if __name__ == "__main__":
    inline = "--inline" in sys.argv
    out = DOCS / ("dupscope-inline.html" if inline else "index.html")
    out.write_text(build(inline), encoding="utf-8")
    print(out)
