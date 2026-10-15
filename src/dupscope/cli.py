"""dupscope command line.

dupscope scan fig1.jpg fig2.jpg --out report/        one paper's figures
dupscope bench --dsb data/dsb2018 --out bench_out     synthetic benchmark
dupscope validate-retractions --mailto you@x.org      retracted-paper check
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import typer

from .detect import Thresholds, analyse, draw_overlay, findings_to_dicts, load_figure
from .report import write_report

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Find duplicated image regions in figures.")

DEFAULTS = Path(__file__).with_name("thresholds.json")


def default_thresholds() -> Thresholds:
    """Thresholds chosen on the benchmark's development split (see bench/run.py)."""
    if DEFAULTS.exists():
        data = json.loads(DEFAULTS.read_text())
        names = {f.name for f in fields(Thresholds)}
        return Thresholds(**{k: v for k, v in data.items() if k in names})
    return Thresholds()


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s %(message)s")


@app.command()
def scan(
    images: list[Path] = typer.Argument(..., exists=True, dir_okay=False, help="All figures of one paper."),
    out: Path = typer.Option(Path("dupscope_report"), help="Output folder."),
    clones: bool = typer.Option(False, help="Also search for regions copied within a panel (experimental)."),
):
    """Check one paper's figures against each other and write a review report."""
    figures = {p.stem: load_figure(p) for p in images}
    t = default_thresholds()
    t.clones = clones
    panels, findings = analyse(figures, t)
    out.mkdir(parents=True, exist_ok=True)
    overlays = draw_overlay(figures, panels, findings)
    for fig_id, img in overlays.items():
        cv2.imwrite(str(out / f"{fig_id}.annotated.png"), img)
    (out / "findings.json").write_text(
        json.dumps(
            {"thresholds": asdict(t), "panels": [asdict(p) for p in panels], "findings": findings_to_dicts(findings)},
            indent=1,
            default=int,
        )
    )
    page = write_report(out, figures, panels, findings, overlays, title=f"dupscope: {images[0].parent.name or 'paper'}")
    typer.echo(f"{len(panels)} panels, {sum(p.photographic for p in panels)} images, {len(findings)} findings")
    typer.echo(f"report: {page}")


@app.command()
def bench(
    dsb: Path = typer.Option(..., exists=True, file_okay=False, help="Unpacked StarDist dsb2018 folder."),
    out: Path = typer.Option(Path("bench_out")),
    n_dev: int = typer.Option(200),
    n_test: int = typer.Option(200),
    workers: int = typer.Option(2),
    dedupe_file: Path | None = typer.Option(None, help="JSON from the pool de-duplication step."),
):
    """Generate synthetic papers, choose thresholds on the dev split, score the test split."""
    from .bench.run import main as run_bench

    summary = run_bench(str(dsb), str(out), n_dev, n_test, workers, str(dedupe_file) if dedupe_file else None)
    typer.echo(json.dumps(summary["test"], indent=1))


@app.command("validate-retractions")
def validate_retractions(
    mailto: str = typer.Option(..., help="Your email; Crossref and NCBI ask API users for one."),
    out: Path = typer.Option(Path("retraction_check")),
    subject: str = typer.Option("Neuroscience", help="Retraction Watch subject filter; empty for all."),
    limit: int = typer.Option(0, help="Check at most this many retracted papers (0 = all)."),
    controls: int = typer.Option(1, help="Matched non-retracted papers per retracted one."),
    csv_path: Path | None = typer.Option(None, exists=True, help="Use a downloaded Retraction Watch CSV."),
):
    """Run on papers retracted for image problems and on matched controls (counts only)."""
    from .validate.retractions import validate

    summary = validate(
        out, mailto, subject or None, limit or None, controls, default_thresholds(), str(csv_path) if csv_path else None
    )
    typer.echo(json.dumps(summary, indent=1))


if __name__ == "__main__":
    app()
