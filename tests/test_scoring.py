from dupscope.bench.evaluate import annotate, score
from dupscope.detect import Thresholds


def panel(fig, i, source, crop, content="micro"):
    return {
        "figure": fig,
        "index": i,
        "content": content,
        "source": source,
        "crop": crop,
        "x": 0,
        "y": 0,
        "w": 10,
        "h": 10,
    }


def cand(a, b, label, corr=0.95):
    return {
        "kind": "duplicate",
        "gt_a": a,
        "gt_b": b,
        "inliers": 40,
        "correlation": corr,
        "detail": 0.8,
        "texture": 0.8,
        "extent": 1.0,
        "overlap_frac": 1.0,
        "scale": 1.0,
        "label": label,
        "method": "keypoints",
    }


def paper(pid, panels, planted, hard, cands):
    truth = {"paper_id": pid, "panels": panels, "planted": planted, "hard_negatives": hard}
    result = {
        "paper_id": pid,
        "planted": [p["kind"] for p in planted],
        "candidates": cands,
        "n_hard_negatives": len(hard),
    }
    return annotate(result, truth)


def test_kind_planted_twice_counts_each_pair():
    panels = [panel("f1", i, f"s{i // 2}", [0, 0, 50, 50]) for i in range(4)]
    planted = [
        {"kind": "reuse_contrast", "panels": [["f1", 0], ["f1", 1]], "detail": ""},
        {"kind": "reuse_contrast", "panels": [["f1", 2], ["f1", 3]], "detail": ""},
    ]
    r = paper("p1", panels, planted, [], [cand(["f1", 0], ["f1", 1], "reuse_contrast")])
    s = score([r], Thresholds())
    assert (s["found"], s["planted"]) == (1, 2)


def test_unplanted_duplicate_paper_is_not_clean():
    panels = [
        panel("f1", 0, "a", [0, 0, 50, 50]),
        panel("f1", 1, "a", [20, 20, 50, 50]),
        panel("f1", 2, "b", [0, 0, 9, 9]),
    ]
    r = paper("p2", panels, [], [], [cand(["f1", 0], ["f1", 1], "incidental_duplicate")])
    s = score([r], Thresholds())
    assert s["clean_papers"] == 0 and s["papers_with_unplanted_duplicates"] == 1
    assert s["unplanted_duplicate_papers_flagged"] == 1 and s["real_findings"] == 1


def test_broken_look_alike_pairs_are_not_counted():
    panels = [
        panel("f1", 0, "a", [0, 0, 50, 50]),
        panel("f1", 1, "a", [100, 100, 50, 50]),
        panel("f1", 2, "b", [0, 0, 50, 50]),
        panel("f1", 3, "c", [0, 0, 50, 50]),
    ]
    hard = [[["f1", 0], ["f1", 1]], [["f1", 2], ["f1", 3]]]  # the second no longer shares a source
    r = paper("p3", panels, [], hard, [cand(["f1", 0], ["f1", 1], "hard_negative")])
    s = score([r], Thresholds())
    assert s["hard_negatives_total"] == 1 and s["hard_negative_hits"] == 1
    assert s["clean_papers"] == 1 and s["clean_papers_flagged"] == 1
