import json

import cv2
from conftest import grid_figure, jpeg, tissue
from typer.testing import CliRunner

from dupscope.cli import app


def test_scan_writes_report(tmp_path):
    shared = tissue(40)
    cv2.imwrite(str(tmp_path / "fig1.jpg"), jpeg(grid_figure([shared, tissue(41)])))
    cv2.imwrite(str(tmp_path / "fig2.jpg"), jpeg(grid_figure([tissue(42), shared])))
    out = tmp_path / "report"
    r = CliRunner().invoke(app, ["scan", str(tmp_path / "fig1.jpg"), str(tmp_path / "fig2.jpg"), "--out", str(out)])
    assert r.exit_code == 0, r.output
    data = json.loads((out / "findings.json").read_text())
    assert len(data["findings"]) == 1
    page = (out / "report.html").read_text()
    assert "same image reused" in page and "not a verdict" in page
    assert (out / "fig1.annotated.png").exists()
