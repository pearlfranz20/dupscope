"""Collect panels from comparison papers (never retracted) to train the panel
classifier on, without naming any paper.

    DUPSCOPE_REVIEW_DIR=... python bench/collect_comparison_panels.py \
        --retracted results/retractions/retracted.json --mailto you@x.org --out DIR

The classifier was first trained only on panels from retracted papers, so it let more
graphics through in comparison papers (results/retractions/blinded_review.json). This
picks a second comparison paper for each analysed retracted paper, the same journal
and year, distinct from the one the evaluation uses (whose selection it replays), and
samples panels from it: up to 4 the current classifier calls photographs and 2 it
calls graphics. For each panel it writes a small thumbnail (for labelling by eye) and
the classifier's feature vector, under an anonymous id. The workflow encrypts the
folder before it leaves the runner. Only feature vectors and labels are ever
published, never images or identifiers.
"""

import argparse
import json
import random
from pathlib import Path

import cv2

from dupscope.panelfeatures import features
from dupscope.panels import extract_panels
from dupscope.validate.retractions import Client, control_pmcids, fetch_figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retracted", required=True)
    ap.add_argument("--mailto", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "dupscope"))
    ap.add_argument("--eval-seed", type=int, default=7)
    ap.add_argument("--train-seed", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    client = Client(Path(a.cache_dir), a.mailto)
    entries = json.loads(Path(a.retracted).read_text())
    seen = {e["retraction"]["pmcid"] for e in entries}
    analysed = [
        e
        for e in entries
        if not e["result"]["error"]
        and e["result"]["n_photo_panels"] > 0
        and e["retraction"]["journal"]
        and e["retraction"]["original_year"]
    ]
    # replay the evaluation's control choice so those papers are excluded
    for e in analysed:
        r = e["retraction"]
        seen.update(control_pmcids(client, r["journal"], r["original_year"], seen, 1, a.eval_seed))
    n_papers = n_panels = 0
    for e in analysed:
        r = e["retraction"]
        picks = control_pmcids(client, r["journal"], r["original_year"], seen, 1, a.train_seed)
        seen.update(picks)
        for pmcid in picks:
            try:
                figs = fetch_figures(client, pmcid)
            except Exception:  # noqa: BLE001 - one unreachable paper must not stop the run
                continue
            if not figs:
                continue
            group = f"cmp{n_papers:03d}"
            n_papers += 1
            panels = [(f, p, img) for f, img in figs.items() for p in extract_panels(img, f)]
            rng = random.Random(n_papers)
            photo = [x for x in panels if x[1].photographic]
            graphic = [x for x in panels if not x[1].photographic]
            chosen = rng.sample(photo, min(4, len(photo))) + rng.sample(graphic, min(2, len(graphic)))
            for k, (_, p, img) in enumerate(chosen):
                gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                crop = p.crop(gray)
                s = 160 / max(crop.shape)
                thumb = cv2.resize(
                    crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA
                )
                item = f"{group}_{k}"
                cv2.imwrite(str(out / f"{item}.png"), thumb)
                (out / f"{item}.json").write_text(
                    json.dumps(
                        {
                            "id": item,
                            "group": group,
                            "current_photo": bool(p.photographic),
                            "features": [round(float(v), 6) for v in features(crop)],
                        }
                    )
                )
                n_panels += 1
    print(n_papers, "papers,", n_panels, "panels")


if __name__ == "__main__":
    main()
