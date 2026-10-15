"""Download full-resolution figures of papers retracted for image problems, to
develop panel splitting and classification against real figures.

    python bench/fetch_dev_figures.py --mailto you@x.org --out devdata --limit 40

Only retracted papers (a public record) are fetched. For each one the folder holds
the PMC metadata JSON and the figure images, and manifest.json lists DOI, title,
licence and what was found. Used by .github/workflows/devdata.yml.
"""

import argparse
import json
import random
import re
from pathlib import Path

import httpx

from dupscope.validate.retractions import (
    FIGURE_EXT,
    PMC_S3,
    Client,
    download_retraction_watch,
    latest_version,
    map_pmcids,
    parse_retractions,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mailto", required=True)
    ap.add_argument("--out", default="devdata")
    ap.add_argument("--subject", default="Neuroscience,Neurology,Psychiatry")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--csv-path")
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "dupscope"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    client = Client(Path(a.cache_dir), a.mailto)
    csv_bytes = Path(a.csv_path).read_bytes() if a.csv_path else download_retraction_watch(client)
    cases = parse_retractions(csv_bytes, a.subject)
    pmc = map_pmcids(client, [c.original_pmid for c in cases])
    cases = [c for c in cases if c.original_pmid in pmc]
    for c in cases:
        c.pmcid = pmc[c.original_pmid]
    random.Random(7).shuffle(cases)
    manifest = {
        "subject": a.subject,
        "retractions_matching": len(parse_retractions(csv_bytes, a.subject)),
        "with_pmcid": len(cases),
        "papers": [],
    }
    for c in cases[: a.limit]:
        entry = {
            "pmcid": c.pmcid,
            "doi": c.original_doi,
            "title": c.title,
            "journal": c.journal,
            "year": c.original_year,
            "reasons": c.reasons,
        }
        try:
            version = latest_version(client, c.pmcid)
            entry["version"] = version
            if version:
                meta = json.loads(client.get(f"{PMC_S3}/{version}/{version}.json"))
                d = out / c.pmcid
                d.mkdir(exist_ok=True)
                (d / "metadata.json").write_text(json.dumps(meta, indent=1))
                entry["license"] = meta.get("license_code")
                media = meta.get("media_urls") or []
                # what the bucket actually holds under this version, listed or not
                listing = client.get(PMC_S3 + "/", {"list-type": "2", "prefix": f"{version}/"}).decode()
                entry["bucket_keys"] = re.findall(r"<Key>([^<]+)</Key>", listing)
                if not media:
                    media = [f"s3://pmc-oa-opendata/{k}" for k in entry["bucket_keys"]]
                entry["media"] = [u.split("?")[0].rsplit("/", 1)[-1] for u in media]
                for u in media:
                    name = u.split("?")[0].rsplit("/", 1)[-1]
                    if name.lower().endswith(FIGURE_EXT):
                        url = u.split("?")[0].replace("s3://pmc-oa-opendata", PMC_S3)
                        data = client.get(url)
                        if len(data) < 6_000_000:
                            (d / name).write_bytes(data)
        except (httpx.HTTPError, ValueError) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        manifest["papers"].append(entry)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
