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
often shrunk. Some papers get planted reuse, and most get look-alikes (different regions of one sample that share
no pixels). Thresholds are chosen on 200 development papers so that at most
5% of clean papers are flagged, then applied once to 200 held-out test
papers. Of those, 70 have planted reuse between panels and 99 have no
reuse. 14 more contain an accidental real duplicate from the generator and
are scored separately (the detector flagged 6). The rest only have a paste
within one panel, which the default search does not cover.

| Test split | |
|---|---|
| Planted reuses found | 58 of 80 |
| Clean papers wrongly flagged | 1 of 99 |
| Look-alike pairs wrongly flagged | 0 of 178 |
| Findings that were real reuse | 72 of 73 |

| Kind of reuse | Found |
|---|---|
| Same western blot strip twice | 9 of 9 |
| Same image, rotated or mirrored | 17 of 17 |
| Same image in two panels | 16 of 20 |
| Same image, brightness or contrast changed | 13 of 26 |
| Two crops of one image that partly overlap | 3 of 8 |

Full numbers, including the development split: `results/benchmark/summary.json`.
The threshold search was widened once after an independent review. Its correlation floor is 0.5, because below
that a reviewer cannot see the match. With a floor of 0.3 it would pick 0.4,
finding 63 of 80 test reuses but flagging
4 of 99 clean papers
(`results/benchmark/summary_wider_grid.json`).

### Limits

- All 22 missed reuses were mostly dark microscopy panels (at most
  11.1% of pixels above background), where there is little to match.
- Partly overlapping crops are the hardest case.
- The one test false alarm was two different western blots with the same band layout, matched after flipping one.
- Searching for a region pasted within one panel (`--clones`) is experimental and off by default. On the benchmark
  it could not separate real pastes from natural repetition such as similar nuclei or evenly spaced blot lanes.
  See `clone_search_experimental_dev` and `clone_search_experimental_test` in the summary.
- A template-matching fallback for low-texture panels exists (`Thresholds.template`) but is off, because on the
  benchmark it added more false alarms than finds.
- Synthetic papers are simpler than real ones, and the development and test papers draw on the same image pool.

Reproduce: download the StarDist `dsb2018.zip` release asset, then
`pip install -e ".[bench]"` and
`dupscope bench --dsb path/to/dsb2018 --out bench_out --dedupe-file results/benchmark/pool_dedupe.json`.
`dupscope bench --out bench_out --rescore` re-scores saved output without rerunning the detector.

## Real figures

Real figures broke the first version: hatched bar fills, text and plots were read as photographs. A small
gradient-boosted classifier, trained on 1106 hand-labelled panels from 60 retracted
papers plus synthetic panels, now decides photograph or graphic. On a random sample of 768
labelled panels, with whole papers held out, it keeps 96.9% of photographs and lets through
33 of 416 graphics (the old rules let through 234).
Retrain with `bench/train_panel_classifier.py`.

## Retracted papers

`dupscope validate-retractions` runs on papers in neuroscience, neurology and psychiatry that Retraction Watch lists
as retracted with image problems among the reasons (a public record). PubMed Central withholds the images of
retracted articles, so figures come from the publisher where possible (PLOS and Springer Nature). It also runs on
one matched paper per analysed retracted paper (same journal and year, not retracted) and reports those only as
counts: their identifiers are never written to the output. The HTTP and result cache, which does hold them, lives
outside the output folder (`~/.cache/dupscope` by default) and should not be published.

Latest run (`results/retractions/summary.json`):

| | Retracted | Comparison |
|---|---|---|
| Analysed | 79 | 64 |
| Any finding | 59 | 28 |
| Strongest finding 0.5 or more | 52 | 24 |
| Strongest finding 0.7 or more | 34 | 15 |
| Strongest finding 0.8 or more | 20 | 11 |
| Strongest finding 0.9 or more | 12 | 5 |

Many papers not marked as retracted have findings too (insets, merged fluorescence channels, declared reused
controls), so a single finding says little. At score cut-offs of 0.8 or 0.9 the gap between the groups is small
enough to be chance at this sample size. The panel classifier was trained partly on panels from these retracted
papers and on none of the comparison papers, which probably flatters the gap. The strongest finding of each of the 20 retracted papers scoring 0.8 or more was checked
by eye: 14 look like the same image used twice and 6 are false alarms (`results/retractions/reviewed.json`).
A retraction notice says a paper had an image problem, not where, so the share of retracted papers flagged is not a
recall figure.

A blinded review then showed the strongest finding of every flagged paper, both groups shuffled and unlabelled,
to a second person. The same pixels were found in 36 of 59 retracted papers' findings and
9 of 28 comparison papers' (Fisher's exact p = 0.021). 16 of the comparison
findings were graphics, the main remaining weakness (`results/retractions/blinded_review.json`). The review
measures pixel reuse, not misconduct.

It runs as a GitHub Actions job (Actions tab, "Retraction check"), which commits results to
`results/retractions/`. Reruns resume from a cache.

## Development

```
pip install -e ".[dev]"
pytest
ruff check src tests bench docs
ruff format --check src tests bench docs
```

MIT licence.
