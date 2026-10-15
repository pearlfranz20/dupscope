"""The retraction check end to end, with every web service replaced by a fake."""

import csv
import io
import json

import cv2
import httpx
from conftest import grid_figure, jpeg, tissue

from dupscope.validate import retractions as rv

COLUMNS = [
    "Record ID",
    "Title",
    "Subject",
    "Journal",
    "OriginalPaperDate",
    "OriginalPaperDOI",
    "OriginalPaperPubMedID",
    "RetractionNature",
    "Reason",
]


def _csv(rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS)
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode()


ROWS = [
    # retracted for image duplication, neuroscience: in scope
    {
        "Record ID": "1",
        "Title": "Paper one",
        "Subject": "(BLS) Biology - Neuroscience;",
        "Journal": "J Neuro",
        "OriginalPaperDate": "3/4/2015 0:00",
        "OriginalPaperDOI": "10.1/one",
        "OriginalPaperPubMedID": "111",
        "RetractionNature": "Retraction",
        "Reason": "+Duplication of Image;+Investigation by Journal/Publisher;",
    },
    # image reasons but another field
    {
        "Record ID": "2",
        "Title": "Paper two",
        "Subject": "(BLS) Biology - Cancer;",
        "Journal": "J Onc",
        "OriginalPaperDate": "2016",
        "OriginalPaperDOI": "10.1/two",
        "OriginalPaperPubMedID": "222",
        "RetractionNature": "Retraction",
        "Reason": "+Manipulation of Images;",
    },
    # neuroscience, but retracted for something else
    {
        "Record ID": "3",
        "Title": "Paper three",
        "Subject": "(BLS) Biology - Neuroscience;",
        "Journal": "J Neuro",
        "OriginalPaperDate": "2015",
        "OriginalPaperDOI": "10.1/three",
        "OriginalPaperPubMedID": "333",
        "RetractionNature": "Retraction",
        "Reason": "+Plagiarism of Text;",
    },
    # a correction, not a retraction
    {
        "Record ID": "4",
        "Title": "Paper four",
        "Subject": "(BLS) Biology - Neuroscience;",
        "Journal": "J Neuro",
        "OriginalPaperDate": "2015",
        "OriginalPaperDOI": "10.1/four",
        "OriginalPaperPubMedID": "444",
        "RetractionNature": "Correction",
        "Reason": "+Duplication of Image;",
    },
]


def _png(img):
    return cv2.imencode(".png", img)[1].tobytes()


def fake_web():
    shared = tissue(60)
    figures = {
        # retracted paper: one image reused across figures
        "PMC100": [jpeg(grid_figure([tissue(61), shared])), jpeg(grid_figure([shared, tissue(62)]))],
        # control paper: all different
        "PMC900": [jpeg(grid_figure([tissue(63), tissue(64)])), jpeg(grid_figure([tissue(65), tissue(66)]))],
    }
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        url, q = str(req.url), dict(req.url.params)
        calls.append(url)
        if "retraction" in url:
            return httpx.Response(200, content=_csv(ROWS))
        if "idconv" in url:
            ids = q["ids"].split(",")
            recs = [{"pmid": 111, "pmcid": "PMC100", "requested-id": "111"}] if "111" in ids else []
            return httpx.Response(200, json={"status": "ok", "records": recs})
        if "europepmc" in url:
            res = [{"pmcid": "PMC100"}, {"pmcid": "PMC900"}]
            return httpx.Response(200, json={"resultList": {"result": res}})
        if req.url.host == "pmc-oa-opendata.s3.amazonaws.com":
            if req.url.path == "/" and "prefix" in q:
                pmcid = q["prefix"].rstrip(".")
                if pmcid not in figures:
                    return httpx.Response(200, text="<ListBucketResult></ListBucketResult>")
                body = (
                    f"<ListBucketResult><CommonPrefixes><Prefix>{pmcid}.1/</Prefix></CommonPrefixes>"
                    f"<CommonPrefixes><Prefix>{pmcid}.2/</Prefix></CommonPrefixes></ListBucketResult>"
                )
                return httpx.Response(200, text=body)
            parts = req.url.path.strip("/").split("/")
            pmcid = parts[0].split(".")[0]
            if parts[1].endswith(".json"):
                assert parts[0].endswith(".2")  # newest version chosen
                media = [
                    f"s3://pmc-oa-opendata/{parts[0]}/fig-g00{i + 1}.png?md5=abc" for i in range(len(figures[pmcid]))
                ]
                media += [
                    f"s3://pmc-oa-opendata/{parts[0]}/data-s001.xlsx?md5=x",
                    f"s3://pmc-oa-opendata/{parts[0]}/fig-s002.jpg?md5=y",
                ]
                return httpx.Response(200, json={"pmcid": pmcid, "media_urls": media})
            if parts[1].startswith("fig-g"):
                return httpx.Response(200, content=_png(figures[pmcid][int(parts[1][5:8]) - 1]))
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_parse_keeps_image_retractions_in_subject():
    cases = rv.parse_retractions(_csv(ROWS), "Neuroscience")
    assert [c.original_pmid for c in cases] == ["111"]
    assert cases[0].original_year == "2015" and "Duplication of Image" in cases[0].reasons
    assert {c.original_pmid for c in rv.parse_retractions(_csv(ROWS), None)} == {"111", "222"}


def test_validate_end_to_end(tmp_path):
    http, calls = fake_web()
    summary = rv.validate(tmp_path / "out", "me@example.org", http=http, min_interval=0, cache_dir=tmp_path / "cache")
    assert summary["retracted"] == {"analysed": 1, "flagged": 1, "skipped": 0, "flag_rate": 1.0}
    assert summary["controls"] == {"analysed": 1, "flagged": 0, "skipped": 0, "flag_rate": 0.0}
    detail = json.loads((tmp_path / "out" / "retracted.json").read_text())
    assert detail[0]["retraction"]["pmcid"] == "PMC100"
    assert detail[0]["result"]["n_findings"] == 1
    assert (tmp_path / "out" / "reports" / "PMC100" / "report.html").exists()
    # supplementary files are never downloaded
    assert not any("-s00" in c for c in calls)
    # the control's identifier appears in no output file and gets no report
    for f in (tmp_path / "out").rglob("*"):
        if f.is_file():
            assert b"PMC900" not in f.read_bytes(), f
    # second run is served from the cache
    n = len(calls)
    again = rv.validate(tmp_path / "out", "me@example.org", http=http, min_interval=0, cache_dir=tmp_path / "cache")
    assert len(calls) == n and again == summary
    # parallel workers make their own clients; with a warm cache they need no network
    par = rv.validate(tmp_path / "out", "me@example.org", min_interval=0, cache_dir=tmp_path / "cache", workers=2)
    assert par == summary
    found = json.loads((tmp_path / "out" / "reports" / "PMC100" / "findings.json").read_text())
    assert found[0]["category"] == "same image reused"


def test_missing_article_is_skipped_not_fatal(tmp_path):
    http, _ = fake_web()
    client = rv.Client(tmp_path, "me@example.org", 0, http)
    r = rv.run_on_pmcid(client, "PMC555", rv.Thresholds())
    assert r.error == "not in the PMC open-access dataset"


def test_publisher_figure_urls():
    xml = (
        '<article><fig id="f1"><graphic xlink:href="pone.0123456.g001"/></fig>'
        '<disp-formula><graphic xlink:href="pone.0123456.e001"/></disp-formula>'
        '<fig id="f2"><graphic xlink:href="pone.0123456.g002"/></fig></article>'
    )
    urls = rv.publisher_figure_urls("10.1371/journal.pone.0123456", xml)
    assert urls == [
        ["https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0123456.g001"],
        ["https://journals.plos.org/plosone/article/figure/image?size=large&id=10.1371/journal.pone.0123456.g002"],
    ]
    sn = rv.publisher_figure_urls(
        "10.1186/s12951-020-00626-1", '<fig><graphic xlink:href="12951_2020_626_Fig1_HTML.jpg"/></fig>'
    )
    base = "https://media.springernature.com/full/springer-static/image/art%3A10.1186%2Fs12951-020-00626-1"
    assert sn == [
        [f"{base}/MediaObjects/12951_2020_626_Fig1_HTML.png", f"{base}/MediaObjects/12951_2020_626_Fig1_HTML.jpg"]
    ]
    assert rv.publisher_figure_urls("10.1016/j.x.2020.1", xml) == []
