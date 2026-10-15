"""Describe the planted reuses the shipped thresholds miss on the test split.

    python bench/analyse_misses.py DSB_DIR DEDUPE_JSON BENCH_OUT results/benchmark/misses.json

Regenerates each test paper with a miss (papers are deterministic in their seed) and
records how much of the missed panel is brighter than background.
"""

import json
import sys
from pathlib import Path

from dupscope.bench.compose import PAIR_KINDS, make_paper
from dupscope.bench.evaluate import passes
from dupscope.bench.sources import load_pool
from dupscope.cli import default_thresholds
from dupscope.panels import to_gray

BRIGHT = 40  # grey level separating signal from a dark background


def main(dsb, dedupe, bench_out, out):
    t = default_thresholds()
    keep = set(json.loads(Path(dedupe).read_text())["kept"])
    pool = [(n, a) for n, a in load_pool(dsb) if n in keep]
    rows = []
    for line in open(Path(bench_out) / "test" / "results.jsonl"):
        r = json.loads(line)
        hits = {c["label"] for c in r["candidates"] if passes(c, t)}
        missed = {k for k in r["planted"] if k in PAIR_KINDS and k not in hits}
        if not missed:
            continue
        paper = make_paper(int(r["paper_id"][5:]), pool, manipulated=True)
        gt = {(g.figure, g.index): g for g in paper.panels}
        for pl in paper.planted:
            if pl.kind not in missed:
                continue
            g = gt[tuple(pl.panels[0])]
            panel = to_gray(paper.figures[g.figure])[g.y : g.y + g.h, g.x : g.x + g.w]
            rows.append(
                {
                    "paper_id": r["paper_id"],
                    "kind": pl.kind,
                    "content": g.content,
                    "bright_fraction": round(float((panel > BRIGHT).mean()), 3),
                }
            )
    micro = [x for x in rows if x["content"] == "micro"]
    summary = {
        "missed": len(rows),
        "missed_microscopy": len(micro),
        "max_bright_fraction_microscopy": max((x["bright_fraction"] for x in micro), default=None),
        "bright_threshold": BRIGHT,
        "rows": rows,
    }
    Path(out).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
