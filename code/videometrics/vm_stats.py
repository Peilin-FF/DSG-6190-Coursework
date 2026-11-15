"""Per-benchmark summaries of our measurements on the released videos.

Inputs : results/noref_per_video.csv (vm_noref.py), results/physiq_per_case.csv (vm_physiq.py, optional),
         results/physiq_still_reference.csv (vm_still_reference.py, optional)
Outputs: results/noref_means.csv          mean, median, std and n per benchmark x method x metric
         results/noref_paired.csv         CorrFlow minus each method on the same prompts: mean difference,
                                          bootstrap 95% CI (10,000 resamples of prompts, seed 0),
                                          prompts where CorrFlow is higher, n
         results/noref_paired_flow_log2.csv  flow as a per-prompt log2 ratio CorrFlow / method (scale free):
                                          mean, bootstrap 95% CI (same resampling, seed 0), geometric-mean
                                          ratio 2^mean, prompts where CorrFlow moves more, n
         results/physiq_means.csv         mean, std and n per method x agreement metric
         results/physiq_paired.csv        CorrFlow minus each method per case (same statistics; for MSE,
                                          "corrflow_better" counts cases where CorrFlow is LOWER)
         results/tables/noref_means_rows.tex   rows per benchmark: mean flicker, MEDIAN flow, mean subject
                                          consistency, mean CLIP x100, n; a dagger marks 9.8 fps videos. The pooled
                                          "T2V (all prompts)" means are not written: they mix n = 22, 8 and 2.
         results/tables/physiq_means_rows.tex  mean agreement per method
         results/tables/physiq_still_row.tex   the same columns for the still-video reference (optional row)
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vm_common as C  # noqa: E402

RES = os.path.join(C.VM, "results")
NOREF_METRICS = ["flicker", "flow_px_s", "subject", "clip"]
PHYSIQ_METRICS = ["spatial_iou", "spatiotemporal_iou", "weighted_spatial_iou", "mse"]
LOWER_IS_BETTER = {"mse"}
B = 10000


def boot_ci(d, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", path, len(rows))


def order_key(m):
    return C.METHOD_ORDER.index(m) if m in C.METHOD_ORDER else 99


def summarise(rows, metrics, group_of, unit_of):
    """rows: dicts with case_id, method and metric values. group_of(row) -> group label."""
    vals = defaultdict(dict)  # (group, method) -> {case: {metric: v}}
    for r in rows:
        vals[(group_of(r), r["method"])][unit_of(r)] = {m: float(r[m]) for m in metrics}
    means, paired = [], []
    groups = []
    for g, _ in vals:
        if g not in groups:
            groups.append(g)
    for g in groups:
        methods = sorted({m for (gg, m) in vals if gg == g}, key=order_key)
        for m in methods:
            d = vals[(g, m)]
            for k in metrics:
                x = np.array([d[c][k] for c in d])
                means.append({"group": g, "method": m, "metric": k, "mean": float(x.mean()),
                              "median": float(np.median(x)),
                              "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "n": len(x)})
        if "CorrFlow" not in methods:
            continue
        ours = vals[(g, "CorrFlow")]
        for m in methods:
            if m in ("CorrFlow", "Ground truth"):
                continue
            other = vals[(g, m)]
            common = sorted(set(ours) & set(other))
            for k in metrics:
                dd = np.array([ours[c][k] - other[c][k] for c in common])
                lo, hi = boot_ci(dd)
                better = int((dd < 0).sum()) if k in LOWER_IS_BETTER else int((dd > 0).sum())
                paired.append({"group": g, "method": m, "metric": k, "mean_diff": float(dd.mean()),
                               "ci_lo": lo, "ci_hi": hi, "corrflow_higher": int((dd > 0).sum()),
                               "corrflow_better": better, "ties": int((dd == 0).sum()), "n": len(common)})
    return means, paired


def low_fps_pairs(rows):
    """(benchmark display name, method) pairs with videos at a native 9.8 fps (temporal steps of 1/9.8 s)."""
    return {(C.BENCH_NAME[r["benchmark"]], r["method"]) for r in rows if abs(float(r["src_fps"]) - 9.8) < 1e-6}


def tex_rows_noref(means, path, low_fps):
    """One row per benchmark x method: mean flicker, median flow (px/s), mean subject, mean CLIP (x100), n.

    Flow uses the median because its per-prompt values are heavy tailed (one VideoPhy2 prompt of CorrFlow at
    422.6 px/s lifts its mean to 81.2 against a median of 14.1). The pooled "T2V (all prompts)" means are left
    out: the methods there cover 22, 8 or 2 prompts, so their means are not comparable.
    """
    lut = {(r["group"], r["method"], r["metric"]): r for r in means}
    lines = []
    for g in [C.BENCH_NAME[b] for b in C.BENCH_ORDER]:
        ms = sorted({r["method"] for r in means if r["group"] == g}, key=order_key)
        for i, m in enumerate(ms):
            cells = [g if i == 0 else "", m + (r"\textsuperscript{\dag}" if (g, m) in low_fps else "")]
            for k in NOREF_METRICS:
                if k == "flow_px_s":
                    cells.append("%.1f" % lut[(g, m, k)]["median"])
                else:
                    v = lut[(g, m, k)]["mean"]
                    cells.append("%.2f" % (100 * v) if k == "clip" else "%.4f" % v)
            cells.append(str(lut[(g, m, NOREF_METRICS[0])]["n"]))
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    if lines and lines[-1] == r"\midrule":
        lines = lines[:-1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)


def flow_log2(rows, group_of):
    """Paired flow comparison on a log scale: d = log2(flow_CorrFlow / flow_method) per common prompt."""
    vals = defaultdict(dict)
    for r in rows:
        vals[(group_of(r), r["method"])][r["case_id"]] = float(r["flow_px_s"])
    out = []
    groups = []
    for g, _ in vals:
        if g not in groups:
            groups.append(g)
    for g in groups:
        if (g, "CorrFlow") not in vals:
            continue
        ours = vals[(g, "CorrFlow")]
        for m in sorted({m for (gg, m) in vals if gg == g}, key=order_key):
            if m in ("CorrFlow", "Ground truth"):
                continue
            common = sorted(set(ours) & set(vals[(g, m)]))
            d = np.log2(np.array([ours[c] for c in common]) / np.array([vals[(g, m)][c] for c in common]))
            lo, hi = boot_ci(d)
            out.append({"group": g, "method": m, "metric": "flow_log2_ratio", "mean_log2": float(d.mean()),
                        "ci_lo": lo, "ci_hi": hi, "geo_ratio": float(2.0 ** d.mean()),
                        "corrflow_higher": int((d > 0).sum()), "n": len(common)})
    return out


def main():
    rows = read(os.path.join(RES, "noref_per_video.csv"))
    for r in rows:
        r["bench"] = C.BENCH_NAME[r["benchmark"]]
    means, paired = summarise(rows, NOREF_METRICS, lambda r: r["bench"], lambda r: r["case_id"])
    t2v = [dict(r) for r in rows if r["task"] == "t2v"]
    m2, p2 = summarise(t2v, NOREF_METRICS, lambda r: "T2V (all prompts)", lambda r: r["case_id"])
    means += m2
    paired += p2
    write(os.path.join(RES, "noref_means.csv"), means)
    write(os.path.join(RES, "noref_paired.csv"), paired)
    write(os.path.join(RES, "noref_paired_flow_log2.csv"),
          flow_log2(rows, lambda r: r["bench"]) + flow_log2(t2v, lambda r: "T2V (all prompts)"))
    tex_rows_noref(means, os.path.join(RES, "tables", "noref_means_rows.tex"), low_fps_pairs(rows))
    pq = os.path.join(RES, "physiq_per_case.csv")
    if os.path.exists(pq):
        prow = read(pq)
        pm, pp = summarise(prow, PHYSIQ_METRICS, lambda r: "Physics-IQ", lambda r: r["case_id"])
        write(os.path.join(RES, "physiq_means.csv"), pm)
        write(os.path.join(RES, "physiq_paired.csv"), pp)
        lut = {(r["method"], r["metric"]): r for r in pm}
        ms = sorted({r["method"] for r in pm}, key=order_key)
        lines = []
        for m in ms:
            cells = [m] + ["%.3f" % lut[(m, k)]["mean"] if k != "mse" else "%.4f" % lut[(m, k)]["mean"]
                           for k in PHYSIQ_METRICS]
            lines.append(" & ".join(cells) + r" \\")
        open(os.path.join(RES, "tables", "physiq_means_rows.tex"), "w").write("\n".join(lines) + "\n")
        print("wrote physiq_means_rows.tex")
    sq = os.path.join(RES, "physiq_still_reference.csv")
    if os.path.exists(sq):
        srow = read(sq)
        cells = ["Still video"] + [("%.4f" if k == "mse" else "%.3f") % np.mean([float(r[k]) for r in srow])
                                   for k in PHYSIQ_METRICS]
        open(os.path.join(RES, "tables", "physiq_still_row.tex"), "w").write(" & ".join(cells) + r" \\" + "\n")
        print("wrote physiq_still_row.tex")


if __name__ == "__main__":
    main()
