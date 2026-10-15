"""Validate dupscope on papers already retracted for image problems.

Pipeline (network access to Crossref, NCBI and Europe PMC is needed, so this runs
on your own machine, not in a sandbox):

1. Retraction Watch database, published openly by Crossref as a CSV.
2. Keep retractions whose stated reasons include image duplication or manipulation,
   optionally only in a subject area (default: neuroscience).
3. Map each original paper's PubMed ID to PubMed Central and fetch its figure
   images from the PMC open-access dataset on AWS (pmc-oa-opendata, the
   replacement for the OA web service and FTP packages that NCBI retired in
   August 2026).
4. Run the detector on each paper's figures, and write a review report for every
   retracted paper with findings.
5. For comparison, run the same on matched control papers (same journal and year,
   not retracted). Controls are reported only as counts: their identifiers are never
   written to any output, so no unretracted paper can be named by this tool.

Every network call is cached on disk, so a run can be resumed or repeated offline.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import random
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import httpx
import numpy as np

from ..detect import Thresholds, analyse

log = logging.getLogger(__name__)

# Crossref publishes the Retraction Watch database in a git repository, updated daily
RW_CSV_URLS = (
    "https://gitlab.com/crossref/retraction-watch-data/-/raw/main/retraction_watch.csv",
    "https://api.labs.crossref.org/data/retractionwatch",
)
IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
PMC_S3 = "https://pmc-oa-opendata.s3.amazonaws.com"
EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

IMAGE_REASONS = (
    "Duplication of Image",
    "Manipulation of Images",
    "Concerns/Issues About Image",
)
RATE_LIMITED = ("ncbi.nlm.nih.gov", "ebi.ac.uk")
FIGURE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif")


@dataclass
class Retraction:
    record_id: str
    title: str
    journal: str
    subject: str
    original_pmid: str
    original_doi: str
    original_year: str
    reasons: list[str]
    pmcid: str = ""


@dataclass
class PaperResult:
    pmcid: str
    n_figures: int
    n_photo_panels: int
    n_findings: int
    max_score: float
    categories: list[str] = field(default_factory=list)
    error: str = ""


class Client:
    """HTTP with a disk cache and a polite rate limit (NCBI allows 3 requests/s)."""

    def __init__(self, cache: Path, mailto: str, min_interval: float = 0.4, client: httpx.Client | None = None):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.mailto = mailto
        self.min_interval = min_interval
        self._last = 0.0
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(60.0),
            follow_redirects=True,
            headers={"User-Agent": f"dupscope/0.1 (mailto:{mailto})"},
        )

    def get(self, url: str, params: dict | None = None) -> bytes:
        key = hashlib.sha256((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
        path = self.cache / key[:2] / key
        if path.exists():
            return path.read_bytes()
        delay = 1.0
        # NCBI and EBI ask for a few requests per second at most; the S3 bucket has no limit
        polite = httpx.URL(url).host.endswith(RATE_LIMITED)
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if polite and wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self._client.get(url, params=params)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(delay)
                    delay *= 2
                    continue
                r.raise_for_status()
            except httpx.TransportError:
                if attempt == 4:
                    raise
                time.sleep(delay)
                delay *= 2
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            return r.content
        raise RuntimeError("unreachable")


def download_retraction_watch(client: Client) -> bytes:
    errors = []
    for url in RW_CSV_URLS:
        try:
            return client.get(url, {"mailto": client.mailto} if "crossref.org" in url else None)
        except httpx.HTTPError as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        "could not download the Retraction Watch CSV ("
        + "; ".join(errors)
        + "). Clone https://gitlab.com/crossref/retraction-watch-data and pass its CSV with --csv-path."
    )


def parse_retractions(csv_bytes: bytes, subject: str | None = "Neuroscience") -> list[Retraction]:
    text = csv_bytes.decode("utf-8-sig", errors="replace")
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        if row.get("RetractionNature", "").strip() != "Retraction":
            continue
        reasons = [r.strip("+ ").strip() for r in row.get("Reason", "").split(";") if r.strip("+ ").strip()]
        if not any(any(key in r for key in IMAGE_REASONS) for r in reasons):
            continue
        if subject and subject.lower() not in row.get("Subject", "").lower():
            continue
        pmid = (row.get("OriginalPaperPubMedID") or "").strip()
        if not pmid or pmid == "0":
            continue
        date = row.get("OriginalPaperDate", "")
        year = re.search(r"(19|20)\d\d", date)
        out.append(
            Retraction(
                record_id=row.get("Record ID", ""),
                title=row.get("Title", ""),
                journal=row.get("Journal", ""),
                subject=row.get("Subject", ""),
                original_pmid=pmid,
                original_doi=row.get("OriginalPaperDOI", ""),
                original_year=year.group(0) if year else "",
                reasons=reasons,
            )
        )
    return out


def map_pmcids(client: Client, pmids: list[str]) -> dict[str, str]:
    """PubMed ID -> PMC ID for papers that are in PMC."""
    found = {}
    for i in range(0, len(pmids), 200):
        chunk = pmids[i : i + 200]
        data = json.loads(
            client.get(
                IDCONV_URL,
                {
                    "ids": ",".join(chunk),
                    "idtype": "pmid",
                    "format": "json",
                    "tool": "dupscope",
                    "email": client.mailto,
                },
            )
        )
        for rec in data.get("records", []):
            if rec.get("pmcid") and rec.get("pmid"):
                found[str(rec["pmid"])] = rec["pmcid"]
    return found


def latest_version(client: Client, pmcid: str) -> str | None:
    """The newest version prefix of an article in the PMC open-access bucket, such as
    "PMC1234567.2", or None if the article is not in the open-access dataset."""
    xml = client.get(PMC_S3 + "/", {"list-type": "2", "prefix": f"{pmcid}.", "delimiter": "/"}).decode(
        "utf-8", errors="replace"
    )
    versions = re.findall(rf"<Prefix>({re.escape(pmcid)}\.(\d+))/</Prefix>", xml)
    if not versions:
        return None
    return max(versions, key=lambda v: int(v[1]))[0]


def _https(url: str) -> str:
    """Media links in the metadata are S3 URLs; fetch them over HTTPS."""
    url = url.split("?", 1)[0]
    if url.startswith("s3://pmc-oa-opendata/"):
        return PMC_S3 + "/" + url[len("s3://pmc-oa-opendata/") :]
    return url


def figure_urls(client: Client, version: str) -> list[str]:
    meta = json.loads(client.get(f"{PMC_S3}/{version}/{version}.json"))
    urls = [_https(u) for u in meta.get("media_urls") or []]
    # supplementary files are named *-s001.*; figures *-g001.*, inline graphics *-i001.*
    return [u for u in urls if u.lower().endswith(FIGURE_EXT) and not re.search(r"-s\d+\.\w+$", u)]


def fetch_figures(client: Client, pmcid: str, max_figures: int = 40) -> dict[str, np.ndarray] | None:
    """Figure images of an open-access article, or None if it is not in the dataset.
    Thumbnails and small inline graphics (formulas, icons) are skipped by size."""
    version = latest_version(client, pmcid)
    if version is None:
        return None
    figs = {}
    for url in figure_urls(client, version):
        img = cv2.imdecode(np.frombuffer(client.get(url), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None or min(img.shape) < 150:
            continue
        figs[Path(url).stem] = img
        if len(figs) >= max_figures:
            break
    return figs


def run_on_pmcid(client: Client, pmcid: str, thresholds: Thresholds, report_dir: Path | None = None) -> PaperResult:
    """Analyse one article. With `report_dir`, a review report is written when there
    are findings (only ever used for retracted papers)."""
    try:
        figures = fetch_figures(client, pmcid)
        if figures is None:
            return PaperResult(pmcid, 0, 0, 0, 0.0, error="not in the PMC open-access dataset")
        if not figures:
            return PaperResult(pmcid, 0, 0, 0, 0.0, error="no figure images")
        panels, findings = analyse(figures, thresholds)
        dup = [f for f in findings if f.kind == "duplicate"]
        if report_dir is not None and dup:
            from ..detect import draw_overlay, findings_to_dicts
            from ..report import write_report

            report_dir.mkdir(parents=True, exist_ok=True)
            write_report(
                report_dir, figures, panels, dup, draw_overlay(figures, panels, dup), title=f"dupscope: {pmcid}"
            )
            (report_dir / "findings.json").write_text(json.dumps(findings_to_dicts(dup), indent=1, default=int))
        return PaperResult(
            pmcid,
            len(figures),
            sum(p.photographic for p in panels),
            len(dup),
            max((f.score for f in dup), default=0.0),
            sorted({f.category for f in dup}),
        )
    except Exception as exc:  # one broken article must not stop the run
        return PaperResult(pmcid, 0, 0, 0, 0.0, error=f"{type(exc).__name__}: {exc}")


def control_pmcids(client: Client, journal: str, year: str, exclude: set[str], k: int, seed: int) -> list[str]:
    """Open-access papers from the same journal and year that are not retracted."""
    query = (
        f'JOURNAL:"{journal}" AND PUB_YEAR:{year} AND OPEN_ACCESS:y AND IN_PMC:y'
        ' AND NOT PUB_TYPE:"Retracted Publication"'
    )
    data = json.loads(client.get(EPMC_SEARCH_URL, {"query": query, "format": "json", "pageSize": 100}))
    ids = [r["pmcid"] for r in data.get("resultList", {}).get("result", []) if r.get("pmcid")]
    ids = [i for i in ids if i not in exclude]
    random.Random(seed).shuffle(ids)
    return ids[:k]


_WORKER: tuple | None = None


def _init_worker(cache: str, mailto: str, min_interval: float, t: Thresholds):
    global _WORKER
    _WORKER = (Client(Path(cache), mailto, min_interval), t)


def _job(job: tuple[str, str | None]) -> PaperResult:
    client, t = _WORKER
    pmcid, report_dir = job
    return run_on_pmcid(client, pmcid, t, Path(report_dir) if report_dir else None)


def validate(
    out_dir: str | Path,
    mailto: str,
    subject: str | None = "Neuroscience",
    limit: int | None = None,
    controls_per_paper: int = 1,
    thresholds: Thresholds | None = None,
    csv_path: str | None = None,
    seed: int = 7,
    http: httpx.Client | None = None,
    min_interval: float = 0.4,
    cache_dir: str | Path | None = None,
    workers: int = 1,
) -> dict:
    """Writes summary.json, retracted.json and reports/<pmcid>/ to `out_dir`. The
    HTTP cache lives elsewhere (default ~/.cache/dupscope) because it holds the
    control papers' downloads, and the output folder must be safe to publish."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = Client(Path(cache_dir) if cache_dir else Path.home() / ".cache" / "dupscope", mailto, min_interval, http)
    t = thresholds or Thresholds()
    csv_bytes = Path(csv_path).read_bytes() if csv_path else download_retraction_watch(client)
    cases = parse_retractions(csv_bytes, subject)
    pmc = map_pmcids(client, [c.original_pmid for c in cases])
    cases = [c for c in cases if c.original_pmid in pmc]
    for c in cases:
        c.pmcid = pmc[c.original_pmid]
    random.Random(seed).shuffle(cases)
    if limit:
        cases = cases[:limit]
    log.info("%d retracted papers with PMC records", len(cases))

    # controls: same journal and year, never retracted; reported as counts only
    seen = {c.pmcid for c in cases}
    controls = []
    for c in cases:
        if not (c.journal and c.original_year):
            continue
        for pmcid in control_pmcids(client, c.journal, c.original_year, seen, controls_per_paper, seed):
            seen.add(pmcid)
            controls.append(pmcid)
    jobs = [(c.pmcid, str(out / "reports" / c.pmcid)) for c in cases] + [(pmcid, None) for pmcid in controls]
    if workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            workers,
            mp_context=multiprocessing.get_context("spawn"),  # forking a threaded parent can deadlock
            initializer=_init_worker,
            initargs=(str(client.cache), mailto, min_interval, t),
        ) as ex:
            results = list(ex.map(_job, jobs))
    else:
        results = [run_on_pmcid(client, pmcid, t, Path(rd) if rd else None) for pmcid, rd in jobs]
    retracted = [{"retraction": asdict(c), "result": asdict(r)} for c, r in zip(cases, results, strict=False)]
    control_results = [{k: v for k, v in asdict(r).items() if k != "pmcid"} for r in results[len(cases) :]]

    def summarise(results: list[dict]) -> dict:
        ok = [r for r in results if not r["error"] and r["n_photo_panels"] > 0]
        flagged = [r for r in ok if r["n_findings"] > 0]
        return {
            "analysed": len(ok),
            "flagged": len(flagged),
            "skipped": len(results) - len(ok),
            "flag_rate": round(len(flagged) / max(1, len(ok)), 3),
        }

    summary = {
        "subject": subject,
        "thresholds": asdict(t),
        "retracted": summarise([x["result"] for x in retracted]),
        "controls": summarise(control_results),
    }
    (out / "retracted.json").write_text(json.dumps(retracted, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def iter_retracted(path: str | Path) -> Iterator[dict]:
    yield from json.loads(Path(path).read_text())
