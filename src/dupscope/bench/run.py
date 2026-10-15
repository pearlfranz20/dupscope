"""Generate the benchmark, run the detector, choose thresholds, report.

Papers with even seeds are clean, odd seeds are manipulated. Seeds [0, n_dev) form
the development split used to choose thresholds; [n_dev, n_dev + n_test) is the
held-out test split, scored once with the chosen thresholds.

Thresholds are chosen to keep false alarms rare: on the development split, at most
`max_clean_flag_rate` of clean papers may be flagged. Wrongly flagging a clean
paper is the expensive error for a tool like this, so recall is maximised only
inside that budget.
"""

from __future__ import annotations

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from ..detect import Thresholds
from .compose import PAIR_KINDS, make_paper
from .evaluate import annotate, run_paper, score

_POOL = None


def _init(pool_names: list[str] | None, dsb_dir: str):
    global _POOL
    from .sources import load_pool

    pool = load_pool(dsb_dir)
    if pool_names is not None:
        keep = set(pool_names)
        pool = [(n, a) for n, a in pool if n in keep]
    _POOL = pool


def _one(seed: int) -> tuple[dict, dict]:
    paper = make_paper(seed, _POOL, manipulated=seed % 2 == 1)
    return run_paper(paper), paper.truth()


def generate_and_run(seeds, dsb_dir: str, pool_names: list[str] | None, workers: int = 2, out: Path | None = None):
    results, truths = [], []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(pool_names, dsb_dir)) as ex:
        for r, t in ex.map(_one, seeds, chunksize=4):
            results.append(r)
            truths.append(t)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "results.jsonl", "w") as fh:
            for r in results:
                fh.write(json.dumps(r) + "\n")
        with open(out / "truth.jsonl", "w") as fh:
            for t in truths:
                fh.write(json.dumps(t) + "\n")
    return results


GRID = {
    "dup_min_inliers": [6, 8, 10, 12, 16],
    "dup_min_correlation": [0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
    "min_detail": [-1.0, 0.0, 0.3, 0.5, 0.6, 0.7],
    "template": [False, True],
}

CLONE_GRID = {
    "clone_min_inliers": [8, 12, 20, 30],
    "clone_min_correlation": [0.8, 0.85, 0.9, 0.95],
    "dense_min_blocks": [40, 80, 160, 320],
}


def choose_thresholds(dev: list[dict], max_clean_flag_rate: float = 0.05) -> tuple[Thresholds, dict]:
    """Duplicate-search thresholds with the most recall inside the false-alarm budget.
    Within-panel clone search stays off (see `clone_tradeoff`)."""
    best, best_key = None, None
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        params = dict(zip(keys, values, strict=True))
        t = Thresholds(dup_min_overlap=0.1, clones=False, **params)
        s = score(dev, t)
        rate = s["clean_papers_flagged"] / max(1, s["clean_papers"])
        if rate > max_clean_flag_rate:
            continue
        # recall first, then fewer false positives, then stricter (safer) settings
        strict = (params["dup_min_inliers"], params["dup_min_correlation"], params["min_detail"], -params["template"])
        key = (s["recall"], -s["false_positives"] - s["hard_negative_hits"], strict)
        if best_key is None or key > best_key:
            best, best_key = (t, s), key
    if best is None:
        raise RuntimeError("no threshold setting meets the false-alarm budget")
    return best


def clone_tradeoff(results: list[dict], base: Thresholds) -> list[dict]:
    """What turning on within-panel clone search costs: for each clone setting, the
    clone recall and the share of clean papers flagged (by anything)."""
    rows = []
    for values in itertools.product(*CLONE_GRID.values()):
        params = dict(zip(CLONE_GRID, values, strict=True))
        t = replace(base, clones=True, **params)
        s = score(results, t)
        single = {k: v for k, v in s["recall_by_kind"].items() if k not in PAIR_KINDS}
        found = sum(v["found"] for v in single.values())
        planted = sum(v["planted"] for v in single.values())
        rows.append(
            {
                **params,
                "clone_recall": round(found / max(1, planted), 3),
                "clone_found": found,
                "clone_planted": planted,
                "clean_flag_rate": round(s["clean_papers_flagged"] / max(1, s["clean_papers"]), 3),
            }
        )
    return rows


def summarise(dev: list[dict], test: list[dict], max_clean_flag_rate: float, pool_size: int | None) -> dict:
    t, dev_score = choose_thresholds(dev, max_clean_flag_rate)
    return {
        "thresholds": asdict(t),
        "max_clean_flag_rate": max_clean_flag_rate,
        "dev": dev_score,
        "test": score(test, t),
        "clone_search_experimental_dev": clone_tradeoff(dev, t),
        "clone_search_experimental_test": clone_tradeoff(test, t),
        "n_dev": len(dev),
        "n_test": len(test),
        "pool_size": pool_size,
    }


def _load(split_dir: Path) -> list[dict]:
    results = [json.loads(line) for line in open(split_dir / "results.jsonl")]
    truths = {json.loads(line)["paper_id"]: json.loads(line) for line in open(split_dir / "truth.jsonl")}
    return [annotate(r, truths[r["paper_id"]]) for r in results]


def rescore(out_dir: str, max_clean_flag_rate: float = 0.05, pool_size: int | None = None) -> dict:
    """Re-score saved detector output (results.jsonl, truth.jsonl) without rerunning."""
    out = Path(out_dir)
    summary = summarise(_load(out / "dev"), _load(out / "test"), max_clean_flag_rate, pool_size)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(
    dsb_dir: str,
    out_dir: str = "bench_out",
    n_dev: int = 200,
    n_test: int = 200,
    workers: int = 2,
    dedupe_file: str | None = None,
    max_clean_flag_rate: float = 0.05,
) -> dict:
    out = Path(out_dir)
    pool_names = json.loads(Path(dedupe_file).read_text())["kept"] if dedupe_file else None
    generate_and_run(range(0, n_dev), dsb_dir, pool_names, workers, out / "dev")
    generate_and_run(range(n_dev, n_dev + n_test), dsb_dir, pool_names, workers, out / "test")
    return rescore(out_dir, max_clean_flag_rate, len(pool_names) if pool_names else None)
