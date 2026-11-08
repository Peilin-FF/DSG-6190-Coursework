"""Data figures of the report (vector PDF + PNG preview), from results/predictions.parquet.

  python scripts/figures.py --results results --out report/figures
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.18,
    "grid.linestyle": "-", "lines.linewidth": 1.4, "lines.markersize": 4, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

MODELS = [("qwen3-vl-4b", "Qwen3-VL-4B"), ("qwen3-vl-8b", "Qwen3-VL-8B"),
          ("internvl3.5-8b", "InternVL3.5-8B"), ("llava-ov-7b", "LLaVA-OV-7B")]
OKABE = {"qwen3-vl-4b": "#56B4E9", "qwen3-vl-8b": "#0072B2", "internvl3.5-8b": "#009E73", "llava-ov-7b": "#E69F00"}
MARK = {"qwen3-vl-4b": "o", "qwen3-vl-8b": "s", "internvl3.5-8b": "^", "llava-ov-7b": "D"}
SETS = [("vad", "VAD"), ("maze", "Maze"), ("maze_nolegend", "Maze\nno legend"), ("ai2d", "AI2D")]
COL_W, FULL_W = 3.25, 6.875
RUNS = [0, 1, 2]


FULL_N = {"vad": 540, "maze": 600, "maze_nolegend": 600, "ai2d": 556}


def acc(df, model, dataset, variant, runs=RUNS, qtype=None):
    """Mean and std over complete runs only (a run must cover every question of the test set)."""
    d = df[(df.model == model) & (df.dataset == dataset) & (df.variant == variant) & df.run.isin(runs)]
    sizes = d.groupby("run")["id"].nunique()
    complete = sizes[sizes >= FULL_N[dataset]].index
    d = d[d.run.isin(complete)]
    if qtype is not None:
        d = d[d.qtype == qtype]
    if d.empty:
        return np.nan, np.nan
    per = d.groupby("run")["correct"].mean() * 100
    return per.mean(), (per.std(ddof=1) if len(per) > 1 else 0.0)


def save(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.pdf"))
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=220)
    plt.close(fig)


def fig_gains(df, out):
    """Accuracy changes (percentage points): DSG over Direct, and each extension over DSG."""
    panels = [("dsg_full", "direct", "DSG $-$ Direct"), ("dsg_x", "dsg_full", "DSG-X $-$ DSG"),
              ("dsg_q", "dsg_full", "DSG-Q $-$ DSG")]
    mats = []
    for v, ref, _ in panels:
        m = np.full((len(MODELS), len(SETS)), np.nan)
        for i, (mk, _) in enumerate(MODELS):
            for j, (ds, _) in enumerate(SETS):
                a, _ = acc(df, mk, ds, v)
                b, _ = acc(df, mk, ds, ref, runs=[0] if ref == "direct" else RUNS)
                m[i, j] = a - b
        mats.append(m)
    vmax = max(np.nanmax(np.abs(m)) for m in mats if np.isfinite(m).any())
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.6), sharey=True)
    for ax, m, (_, _, title) in zip(axes, mats, panels):
        im = ax.imshow(m, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(m[i, j]) > 0.55 * vmax else "black")
        ax.set_xticks(range(len(SETS)), [s for _, s in SETS], fontsize=6.5)
        ax.set_yticks(range(len(MODELS)), [n for _, n in MODELS])
        ax.set_title(title)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("pp", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    save(fig, out, "fig_gains")


def fig_ablation(df, out):
    steps = [("direct", "Direct"), ("dsg_schema", "+Schema"), ("dsg_seq", "+Ground."), ("dsg_hier", "+Hier."),
             ("dsg_full", "Full DSG")]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    x = np.arange(len(steps))
    for mk, name in MODELS:
        ys, es = zip(*[acc(df, mk, "vad", v, runs=[0] if v == "direct" else RUNS) for v, _ in steps])
        ax.errorbar(x, ys, yerr=es, color=OKABE[mk], marker=MARK[mk], capsize=2, label=name, lw=1.2)
    ax.set_xticks(x, [s for _, s in steps])
    ax.set_ylabel("VAD accuracy (%)")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.8, handlelength=1.2, columnspacing=0.8)
    save(fig, out, "fig_ablation")


def fig_hparam(df, out, models=("qwen3-vl-8b", "internvl3.5-8b")):
    """One setting at a time on VAD. A model enters a panel only when every point of that panel is complete."""
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.75), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    names = dict(MODELS)
    # (a) schema size
    ax = axes[0]
    sizes = [(3, "dsg_full"), (5, "dsg_full@c5"), (7, "dsg_full@c7"), (9, "dsg_full@c9")]
    for mk in models:
        pts = [acc(df, mk, "vad", v) for _, v in sizes]
        if not all(np.isfinite(y) for y, _ in pts):
            continue
        ys, es = zip(*pts)
        ax.errorbar([k for k, _ in sizes], ys, yerr=es, color=OKABE[mk], marker=MARK[mk], capsize=2, label=f"{names[mk]}: DSG")
        ax.axhline(acc(df, mk, "vad", "direct", runs=[0])[0], color=OKABE[mk], ls="--", lw=0.9, label=f"{names[mk]}: Direct")
    ax.set_xticks([3, 5, 7, 9])
    ax.set_xlabel("components in the example schema")
    ax.set_ylabel("VAD accuracy (%)")
    ax.set_title("(a) schema size")
    ax.legend(fontsize=5.8, loc="upper right")
    # (b) resolution
    ax = axes[1]
    res = [(448, "@res448"), (672, "@res672"), (1024, ""), (1344, "@res1344")]
    for mk in models:
        yd = [acc(df, mk, "vad", "direct" + t, runs=[0])[0] for _, t in res]
        pg = [acc(df, mk, "vad", "dsg_full" + t, runs=RUNS if mk == "qwen3-vl-8b" else [0]) for _, t in res]
        if not all(np.isfinite(y) for y in yd) or not all(np.isfinite(y) for y, _ in pg):
            continue
        ax.plot([r for r, _ in res], yd, color=OKABE[mk], ls="--", marker=MARK[mk], mfc="white", label=f"{names[mk]}: Direct")
        yg, eg = zip(*pg)
        ax.errorbar([r for r, _ in res], yg, yerr=eg, color=OKABE[mk], marker=MARK[mk], capsize=2, label=f"{names[mk]}: DSG")
    ax.set_xticks([448, 672, 1024, 1344])
    ax.set_xlabel("longer image side (px)")
    ax.set_title("(b) image resolution")
    ax.legend(fontsize=5.8, loc="lower right")
    # (c) other settings, Qwen3-VL-8B
    ax = axes[2]
    mk = models[0]
    cfgs = [("dsg_full", "default (phrase)"), ("dsg_full@sentence", "sentence grounding"),
            ("dsg_full@self8b", "schema by the 8B VLM"), ("dsg_sc3", "vote over 3 schemas"), ("dsg_sc5", "vote over 5 schemas")]
    pts = [acc(df, mk, "vad", v, runs=[0] if v.startswith("dsg_sc") else RUNS) for v, _ in cfgs]
    y = np.arange(len(cfgs))[::-1]
    ax.barh(y, [p[0] for p in pts], xerr=[p[1] for p in pts], color=OKABE[mk], height=0.6, capsize=1.5,
            error_kw={"lw": 0.7})
    for yy, (v, e) in zip(y, pts):
        if np.isfinite(v):
            ax.text(v + (e if np.isfinite(e) else 0) + 0.2, yy, f"{v:.1f}", va="center", fontsize=6)
    d0 = acc(df, mk, "vad", "direct", runs=[0])[0]
    ax.axvline(d0, color="black", ls="--", lw=0.9)
    ax.text(d0 - 0.15, -0.75, "Direct", fontsize=6, va="center", ha="right")
    ax.set_yticks(y, [c for _, c in cfgs], fontsize=6.3)
    ax.set_xlim(d0 - 3, max(p[0] for p in pts if np.isfinite(p[0])) + 2.5)
    ax.set_xlabel("VAD accuracy (%)")
    ax.set_title(f"(c) other settings, {names[mk]}")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig_hparam")


def fig_qtype(df, out):
    qts = [("counting", "Counting"), ("binary", "Binary"), ("open", "Open")]
    vs = [("direct", "Direct", "#B0B0B0"), ("cot", "CoT", "#CC79A7"), ("dsg_full", "DSG", "#0072B2"),
          ("dsg_q", "DSG-Q (ours)", "#D55E00")]
    fig, ax = plt.subplots(figsize=(COL_W, 1.85))
    w = 0.8 / len(vs)
    for k, (v, name, c) in enumerate(vs):
        means, per_model = [], []
        for qt, _ in qts:
            vals = [acc(df, mk, "vad", v, runs=[0] if v in ("direct", "cot") else RUNS, qtype=qt)[0] for mk, _ in MODELS]
            means.append(np.nanmean(vals))
            per_model.append(vals)
        xs = np.arange(len(qts)) + (k - (len(vs) - 1) / 2) * w
        ax.bar(xs, means, w * 0.9, color=c, label=name)
        for x0, vals in zip(xs, per_model):
            ax.scatter([x0] * len(vals), vals, s=5, color="black", zorder=3, lw=0)
    ax.set_xticks(range(len(qts)), [q for _, q in qts])
    ax.set_ylabel("VAD accuracy (%)")
    ax.set_ylim(30, None)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, out, "fig_qtype")


def fig_counts(df, out, data_root, model="qwen3-vl-8b"):
    """P(predicted count | true count) on Maze-Nav turn questions, Direct vs DSG (run 0)."""
    import json
    items = {}
    with open(os.path.join(data_root, "manifests", "maze.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            items[r["id"]] = r
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.55), sharey=True)
    for ax, (v, title) in zip(axes, [("direct", "Direct"), ("dsg_full", "DSG")]):
        d = df[(df.model == model) & (df.dataset == "maze") & (df.variant == v) & (df.run == 0)
               & df.qtype.isin(["right_turns", "total_turns"])]
        m = np.zeros((5, 6))
        for _, r in d.iterrows():
            it = items[r["id"]]
            if pd.isna(r["pred_idx"]):
                continue
            t = min(int(it["answer"]), 4)
            p_ = min(int(it["choices"][int(r["pred_idx"])]), 5)
            m[t, p_] += 1
        m = m / np.maximum(m.sum(1, keepdims=True), 1)
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(5):
            for j in range(6):
                if m[i, j] >= 0.1:
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                            color="white" if m[i, j] > 0.5 else "black")
        ax.set_xticks(range(6), ["0", "1", "2", "3", "4", "5+"])
        ax.set_yticks(range(5), ["0", "1", "2", "3", "4+"])
        ax.set_xlabel("predicted turns")
        ax.set_title(title)
        ax.grid(False)
    axes[0].set_ylabel("true turns")
    save(fig, out, "fig_counts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--data", default=os.environ.get("DSG_DATA", "data"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_parquet(os.path.join(a.results, "predictions.parquet"))
    for f in (fig_gains, fig_ablation, fig_hparam, fig_qtype, fig_counts):
        try:
            f(df, a.out) if f is not fig_counts else f(df, a.out, a.data)
            print("ok", f.__name__)
        except Exception as e:  # keep the other figures when one input is still missing
            print("FAILED", f.__name__, repr(e))


if __name__ == "__main__":
    main()
