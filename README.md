# dupscope

Finds image panels that appear more than once in a paper's figures, even when rotated, mirrored, resized,
cropped or re-toned, and writes a report for a person to review. Every finding is a lead to check, not a verdict.

Project page with examples and results: https://pearlfranz20.github.io/dupscope/

## Use

```
pip install git+https://github.com/pearlfranz20/dupscope
dupscope scan fig1.png fig2.png fig3.png --out report/
```

This writes `report/report.html` (each finding with the two regions side by side), annotated figures and
`findings.json`.

## How it works

1. Split each figure into panels along its white gutters, after removing labels. Class each panel as an image
   (microscopy, blots) or a graphic (charts). Only images are compared.
2. For every pair of image panels, match SIFT keypoints and keep matches that agree on one similarity transform
   (rotation, mirroring, scale), using RANSAC.
3. Warp one panel onto the other and correlate the overlapping pixels, both raw and band-passed. Copies share fine
   grain and noise. Look-alike cells do not.
4. Label each finding (same image reused, rotated or mirrored, overlapping fields of view, magnified region) and
   write the report.

## Benchmark

Real manipulated papers have no answer key, so the benchmark generates synthetic papers: 2 to 3 figures of real
microscopy crops (DSB2018 nuclei and scikit-image samples), generated western blots and charts, saved as JPEG and
often shrunk. Half the papers get planted reuse. All get look-alikes (different regions of the same sample).
Thresholds are chosen on 200 development papers so that at most 5% of clean
papers are flagged. Results on 200 held-out test papers:

| | |
|---|---|
| Planted reuses found | 59 of 80 |
| Clean papers wrongly flagged | 3 of 111 |
| Look-alike pairs wrongly flagged | 0 of 181 |
| Findings that were real reuse | 94% |

| Kind of reuse | Found |
|---|---|
| Same image, rotated or mirrored | 17 of 17 |
| Same western blot strip twice | 8 of 9 |
| Same image in two panels | 17 of 20 |
| Same image, brightness or contrast changed | 15 of 26 |
| Two crops of one image that partly overlap | 2 of 8 |

Full numbers: `results/benchmark/summary.json`.

### Limits

- 20 of the 21 missed reuses were mostly dark microscopy panels (at most
  10% of pixels above background), where there is little to match.
- Partly overlapping crops are the hardest case.
- Bar charts can be cut into bars that look like reused images. This caused the worst false alarm.
- Searching for a region pasted within one panel (`--clones`) is experimental and off by default. On the benchmark
  it could not separate real pastes from natural repetition such as similar nuclei or evenly spaced blot lanes.
  See `clone_search_experimental_test` in the summary.
- A template-matching fallback for low-texture panels exists (`Thresholds.template`) but is off, because on the
  benchmark it added more false alarms than finds.
- Synthetic papers are simpler than real ones.

Reproduce: download the StarDist `dsb2018.zip` release asset, then
`pip install -e ".[bench]"` and
`dupscope bench --dsb path/to/dsb2018 --out bench_out --dedupe-file results/benchmark/pool_dedupe.json`.

## Retracted papers

`dupscope validate-retractions` runs on neuroscience papers that Retraction Watch lists as retracted for image
duplication or manipulation (a public record), with figures from the PubMed Central open-access dataset. It also
runs on matched papers from the same journal and year that were not retracted, and reports those only as counts:
their identifiers are never written to the output. The HTTP cache, which does hold them, lives outside the output
folder (`~/.cache/dupscope` by default) and should not be published.

It runs as a GitHub Actions job (Actions tab, "Retraction check"), which commits results to
`results/retractions/`. A retraction notice says a paper had an image problem, not where, and some problems are
outside what this tool checks, so the share of retracted papers flagged is not a recall figure.

## Development

```
pip install -e ".[dev]"
pytest
ruff check src tests
```

MIT licence.
