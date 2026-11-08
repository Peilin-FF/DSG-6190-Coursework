"""Data figures of the report (vector PDF + PNG preview), from results/predictions.parquet.

The style follows the figures of Hsu et al. (2025): their Office colour palette, sans-serif text, no grid,
one message per figure and the values printed next to the bars. Accuracies are averaged over the four VLMs
and over the three schema samples of every schema-based method.

  python scripts/figures.py --results results --out report/figures --data data
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.4, "lines.markersize": 4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

# Colours of the framework figure of Hsu et al. (2025), used with one meaning each in all figures.
BLUE, LBLUE = "#4E95D9", "#B4C7E7"      # DSG, steps of the DSG ablation
GREEN, ORANGE = "#4EA72E", "#E97132"    # our two changes: exclusive grounding, question components
LORANGE = "#F8CBAD"                     # other prompting methods (CoT)
LYELLOW, GOLD = "#FFF2CC", "#DEA817"    # schema controls (fill, edge)
GRAY, DGRAY = "#D0CECE", "#7F7F7F"      # base VLM, ground truth and zero lines

MODELS = [("qwen3-vl-4b", "Qwen3-VL-4B"), ("qwen3-vl-8b", "Qwen3-VL-8B"),
          ("internvl3.5-8b", "InternVL3.5-8B"), ("llava-ov-7b", "LLaVA-OV-7B")]
SETS = [("vad", "VAB"), ("maze", "Maze-Nav"), ("maze_nolegend", "Maze-Nav\nno legend"), ("ai2d", "AI2D")]
COL_W, FULL_W = 3.25, 6.875
RUNS = [0, 1, 2]
FULL_N = {"vad": 540, "maze": 600, "maze_nolegend": 600, "ai2d": 556}
ONE_RUN = ("blind", "direct", "cot", "ccot", "dsg_sc3", "dsg_sc5", "dsg_gate")  # deterministic or one vote per item


def acc(df, model, dataset, variant, runs=None, qtype=None):
    """Mean accuracy (%) over the complete runs of one model (a run must cover every question of the test set)."""
    if runs is None:
        runs = [0] if variant.split("@")[0] in ONE_RUN else RUNS
    d = df[(df.model == model) & (df.dataset == dataset) & (df.variant == variant) & df.run.isin(runs)]
    sizes = d.groupby("run")["id"].nunique()
    d = d[d.run.isin(sizes[sizes >= FULL_N[dataset]].index)]
    if qtype is not None:
        d = d[d.qtype == qtype]
    if d.empty:
        return np.nan
    return d.groupby("run")["correct"].mean().mean() * 100


def mean_acc(df, dataset, variant, qtype=None, models=MODELS):
    """Mean over the models; nan unless every model has a complete run."""
    vals = [acc(df, mk, dataset, variant, qtype=qtype) for mk, _ in models]
    # rounded like the table entries, so that a plotted difference equals the difference of two table values
    return round(float(np.mean(vals)), 1) if all(np.isfinite(vals)) else np.nan


def label_bars(ax, xs, vals, horizontal=False, plus=True, pad=0.25, size=6):
    for x, v in zip(xs, vals):
        if not np.isfinite(v):
            continue
        s = (f"{v:+.1f}" if plus and abs(v) >= 0.05 else f"{v:.1f}").replace("-", "\u2212")
        if horizontal:
            ax.text(v + (pad if v >= 0 else -pad), x, s, va="center", ha="left" if v >= 0 else "right", fontsize=size)
        else:
            ax.text(x, v + (pad if v >= 0 else -pad), s, ha="center", va="bottom" if v >= 0 else "top", fontsize=size)


def save(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.pdf"))
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=220)
    plt.close(fig)


def fig_qtype(df, out):
    """VAB: change over the base VLM per question type, single-round prompts against multi-round DSG."""
    qts = [("counting", "Counting"), ("binary", "Binary-choice"), ("open", "Open-ended")]
    vs = [("cot", "CoT (single round)", LORANGE), ("dsg_schema", "+ schema (single round)", LBLUE),
          ("dsg_full", "DSG (multi-round)", BLUE)]
    fig, ax = plt.subplots(figsize=(COL_W, 1.55))
    w = 0.26
    for k, (v, name, c) in enumerate(vs):
        ds = [mean_acc(df, "vad", v, qt) - mean_acc(df, "vad", "direct", qt) for qt, _ in qts]
        xs = np.arange(len(qts)) + (k - 1) * w
        ax.bar(xs, ds, w * 0.92, color=c, label=name)
        label_bars(ax, xs, ds)
    ax.axhline(0, color=DGRAY, lw=0.6)
    ax.set_xticks(range(len(qts)), [q for _, q in qts])
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Change (points)")
    ax.margins(y=0.18)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.9)
    save(fig, out, "fig_qtype")


def fig_ablation(df, out):
    """VAB: change over the base VLM for the ablation of Hsu et al. (2025) and for our schema controls."""
    rows = [("dsg_schema", "+ schema", LBLUE), ("dsg_seq", "+ grounding", LBLUE), ("dsg_hier", "+ hierarchical", LBLUE),
            ("dsg_full", "DSG", BLUE), None, ("dsg_generic", "Generic schema", LYELLOW),
            ("dsg_wrong", "Schema of another concept", LYELLOW), ("dsg_shuffle", "Groundings of another image", LYELLOW)]
    base = mean_acc(df, "vad", "direct")
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    ys, names, y = [], [], 0.0
    for r in rows:
        if r is None:
            y += 0.6
            continue
        v, name, c = r
        d = mean_acc(df, "vad", v) - base
        ax.barh(-y, d, 0.66, color=c, edgecolor=GOLD if c == LYELLOW else "none", linewidth=0.6)
        label_bars(ax, [-y], [d], horizontal=True, pad=0.08)
        ys.append(-y)
        names.append(name)
        y += 1
    ax.axvline(0, color=DGRAY, lw=0.6)
    ax.set_yticks(ys, names)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Change over base VLM on VAB (points; base VLM {base:.1f}%)")
    ax.margins(x=0.18)
    save(fig, out, "fig_ablation")


def fig_counts(df, out, data_root, model="qwen3-vl-8b"):
    """Maze-Nav turn-counting questions: distribution of true counts and of the counts that the VLM answers."""
    items = {}
    with open(os.path.join(data_root, "manifests", "maze.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            items[r["id"]] = r
    bins = np.arange(6)
    d = df[(df.model == model) & (df.dataset == "maze") & df.qtype.isin(["right_turns", "total_turns"])]

    def hist(counts):
        h = np.bincount(np.minimum(np.asarray(counts, dtype=int), 5), minlength=6)[:6]
        return 100 * h / max(h.sum(), 1)

    truth = hist([int(items[i]["answer"]) for i in d[(d.variant == "direct") & (d.run == 0)]["id"]])
    preds = {}
    for v, runs in (("direct", [0]), ("dsg_full", RUNS)):
        x = d[(d.variant == v) & d.run.isin(runs)].dropna(subset=["pred_idx"])
        preds[v] = hist([int(items[r.id]["choices"][int(r.pred_idx)]) for r in x.itertuples()])
    fig, ax = plt.subplots(figsize=(COL_W, 1.45))
    w = 0.27
    for k, (h, name, c) in enumerate([(truth, "Ground truth", DGRAY), (preds["direct"], "Base VLM", GRAY),
                                      (preds["dsg_full"], "DSG", BLUE)]):
        ax.bar(bins + (k - 1) * w, h, w * 0.92, color=c, label=name)
    ax.set_xticks(bins, ["0", "1", "2", "3", "4", "5+"])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Number of turns")
    ax.set_ylabel("Share of answers (%)")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.9)
    save(fig, out, "fig_counts")


def fig_hparam(df, out, model="qwen3-vl-8b"):
    """VAB, one setting changed at a time for the model with three schema samples per setting:
    (a) schema size, (b) image resolution, (c) other settings."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.45), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    panels = [(axes[0], [(3, "dsg_full", "direct"), (5, "dsg_full@c5", "direct"), (7, "dsg_full@c7", "direct"),
                         (9, "dsg_full@c9", "direct")], "Components in the example schema", "(a) Schema size"),
              (axes[1], [(448, "dsg_full@res448", "direct@res448"), (672, "dsg_full@res672", "direct@res672"),
                         (1024, "dsg_full", "direct"), (1344, "dsg_full@res1344", "direct@res1344")],
               "Longer image side (pixels)", "(b) Image resolution")]
    for ax, pts, xlabel, title in panels:
        xs = [x for x, _, _ in pts]
        ax.plot(xs, [acc(df, model, "vad", b) for _, _, b in pts], color=DGRAY, ls="--", marker="o", mfc="white",
                label="Base VLM")
        ax.plot(xs, [acc(df, model, "vad", v) for _, v, _ in pts], color=BLUE, marker="o", label="DSG")
        ax.set_xticks(xs)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.margins(y=0.2)
    axes[0].set_ylabel("Accuracy (%)")
    a0 = axes[0]  # direct line labels instead of a legend box
    a0.text(9.4, acc(df, model, "vad", "dsg_full@c9"), "DSG", color=BLUE, va="center", fontsize=6.5)
    a0.text(9.4, acc(df, model, "vad", "direct"), "Base VLM", color=DGRAY, va="center", fontsize=6.5)
    a0.set_xlim(2.4, 11.6)
    ax = axes[2]
    ref = round(acc(df, model, "vad", "dsg_full"), 1)
    cfgs = [("dsg_full@sentence", "Sentence groundings"), ("dsg_full@self8b", "Schema written by the VLM"),
            ("dsg_sc3", "Vote over 3 schemas"), ("dsg_sc5", "Vote over 5 schemas")]
    ds = [round(acc(df, model, "vad", v), 1) - ref for v, _ in cfgs]
    ys = -np.arange(len(cfgs))
    ax.barh(ys, ds, 0.62, color=[BLUE if d >= 0 else LBLUE for d in ds])
    label_bars(ax, ys, ds, horizontal=True, pad=0.06)
    ax.axvline(0, color=DGRAY, lw=0.6)
    ax.set_yticks(ys, [n for _, n in cfgs])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Change over DSG (points; DSG {ref:.1f}%)")
    ax.set_title("(c) Other settings", loc="left")
    ax.margins(x=0.22)
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "fig_hparam")


def fig_ext(df, out):
    """Change over the base VLM on each test set for DSG and for our two changes to DSG."""
    vs = [("dsg_full", "DSG", BLUE), ("dsg_x", "+ exclusive grounding", GREEN), ("dsg_q", "+ question components", ORANGE)]
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    w = 0.28
    for k, (v, name, c) in enumerate(vs):
        ds = [mean_acc(df, s, v) - mean_acc(df, s, "direct") for s, _ in SETS]
        xs = np.arange(len(SETS)) + (k - 1) * w
        ax.bar(xs, ds, w * 0.92, color=c, label=name)
        label_bars(ax, xs, ds, plus=False, pad=0.3, size=5.5)
    ax.axhline(0, color=DGRAY, lw=0.6)
    ax.set_xticks(range(len(SETS)), [s for _, s in SETS])
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel("Change (points)")
    ax.margins(y=0.15)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.9)
    save(fig, out, "fig_ext")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--data", default=os.environ.get("DSG_DATA", "data"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_parquet(os.path.join(a.results, "predictions.parquet"))
    for f in (fig_qtype, fig_ablation, fig_counts, fig_hparam, fig_ext):
        try:
            f(df, a.out, a.data) if f is fig_counts else f(df, a.out)
            print("ok", f.__name__)
        except Exception as e:  # keep the other figures when one input is still missing
            print("FAILED", f.__name__, repr(e))


if __name__ == "__main__":
    main()
