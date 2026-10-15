"""Describe the planted reuses the shipped thresholds miss on the test split.

    python bench/analyse_misses.py DSB_DIR DEDUPE_JSON BENCH_OUT results/benchmark/misses.json

Regenerates each test paper with a miss (papers are deterministic in their seed) and
records how much of the missed panel is brighter than background.
"""

import json
import sys
from pathlib import Path

from dupscope.bench.compose import make_paper
from dupscope.bench.evaluate import annotate, passes
from dupscope.bench.sources import load_pool
from dupscope.cli import default_thresholds
from dupscope.panels import to_gray

BRIGHT = 40  # grey level separating signal from a dark background


def main(dsb, dedupe, bench_out, out):
    t = default_thresholds()
    keep = set(json.loads(Path(dedupe).read_text())["kept"])
    pool = [(n, a) for n, a in load_pool(dsb) if n in keep]
    rows = []
    split = Path(bench_out) / "test"
    truths = {json.loads(line)["paper_id"]: json.loads(line) for line in open(split / "truth.jsonl")}
    for line in open(split / "results.jsonl"):
        r = annotate(json.loads(line), truths[json.loads(line)["paper_id"]])
        kept = {
            frozenset((tuple(c["gt_a"]), tuple(c["gt_b"])))
            for c in r["candidates"]
            if passes(c, t) and c["gt_a"] and c["gt_b"]
        }
        missed = [(kind, ks) for kind, ks in r["planted_pairs"] if frozenset(tuple(k) for k in ks) not in kept]
        if not missed:
            continue
        paper = make_paper(int(r["paper_id"][5:]), pool, manipulated=True)
        gt = {(g.figure, g.index): g for g in paper.panels}
        for kind, ks in missed:
            g = gt[tuple(ks[0])]
            panel = to_gray(paper.figures[g.figure])[g.y : g.y + g.h, g.x : g.x + g.w]
            rows.append(
                {
                    "paper_id": r["paper_id"],
                    "kind": kind,
                    "content": g.content,
                    "panels": ks,
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
