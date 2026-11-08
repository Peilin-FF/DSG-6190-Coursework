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
SETS = [("vad", "VAD"), ("maze", "Maze-Nav"), ("maze_nolegend", "Maze-Nav\n(no legend)"), ("ai2d", "AI2D-\nConcepts")]
COL_W, FULL_W = 3.25, 6.875
RUNS = [0, 1, 2]


def acc(df, model, dataset, variant, runs=RUNS, qtype=None):
    d = df[(df.model == model) & (df.dataset == dataset) & (df.variant == variant) & df.run.isin(runs)]
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
    """Accuracy change over Direct (percentage points) per model and test set."""
    variants = [("dsg_full", "DSG $-$ Direct"), ("dsg_q", "DSG-Q (ours) $-$ Direct")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 1.55), sharey=True)
    vmax = 0
    mats = []
    for v, _ in variants:
        m = np.full((len(MODELS), len(SETS)), np.nan)
        for i, (mk, _) in enumerate(MODELS):
            for j, (ds, _) in enumerate(SETS):
                a, _ = acc(df, mk, ds, v)
                b, _ = acc(df, mk, ds, "direct", runs=[0])
                m[i, j] = a - b
        mats.append(m)
        vmax = max(vmax, np.nanmax(np.abs(m)) if np.isfinite(m).any() else 1)
    for ax, m, (_, title) in zip(axes, mats, variants):
        im = ax.imshow(m, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(m[i, j]) > 0.55 * vmax else "black")
        ax.set_xticks(range(len(SETS)), [s for _, s in SETS])
        ax.set_yticks(range(len(MODELS)), [n for _, n in MODELS])
        ax.set_title(title)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
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
    ax.legend(ncol=2, loc="lower right")
    save(fig, out, "fig_ablation")


def fig_hparam(df, out, models=("qwen3-vl-8b", "internvl3.5-8b")):
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.75))
    names = dict(MODELS)
    # (a) schema size
    ax = axes[0]
    sizes = [(3, "dsg_full"), (5, "dsg_full@c5"), (7, "dsg_full@c7"), (9, "dsg_full@c9")]
    for mk in models:
        ys, es = zip(*[acc(df, mk, "vad", v) for _, v in sizes])
        ax.errorbar([s for s, _ in sizes], ys, yerr=es, color=OKABE[mk], marker=MARK[mk], capsize=2,
                    label=f"{names[mk]} DSG")
        d0, _ = acc(df, mk, "vad", "direct", runs=[0])
        ax.axhline(d0, color=OKABE[mk], ls="--", lw=0.9, label=f"{names[mk]} Direct")
    ax.set_xticks([3, 5, 7, 9])
    ax.set_xlabel("components in the example schema")
    ax.set_ylabel("VAD accuracy (%)")
    ax.set_title("(a) schema size")
    # (b) resolution
    ax = axes[1]
    res = [(448, "@res448"), (672, "@res672"), (1024, ""), (1344, "@res1344")]
    for mk in models:
        yd = [acc(df, mk, "vad", "direct" + t, runs=[0])[0] for _, t in res]
        yg, eg = zip(*[acc(df, mk, "vad", "dsg_full" + t) for _, t in res])
        ax.plot([r for r, _ in res], yd, color=OKABE[mk], ls="--", marker=MARK[mk], mfc="white")
        ax.errorbar([r for r, _ in res], yg, yerr=eg, color=OKABE[mk], marker=MARK[mk], capsize=2)
    ax.set_xticks([448, 672, 1024, 1344])
    ax.set_xlabel("longer image side (px)")
    ax.set_title("(b) image resolution")
    # (c) other settings
    ax = axes[2]
    cfgs = [("dsg_full", "default"), ("dsg_full@sentence", "sentence\ngrounding"), ("dsg_full@self8b", "self-\nschema"),
            ("dsg_sc3", "vote\n$K{=}3$"), ("dsg_sc5", "vote\n$K{=}5$")]
    w = 0.38
    for k, mk in enumerate(models):
        ys, es = zip(*[acc(df, mk, "vad", v, runs=[0] if v.startswith("dsg_sc") else RUNS) for v, _ in cfgs])
        ax.bar(np.arange(len(cfgs)) + (k - 0.5) * w, ys, w * 0.92, yerr=es, color=OKABE[mk], capsize=1.5,
               error_kw={"lw": 0.7}, label=names[mk])
        d0, _ = acc(df, mk, "vad", "direct", runs=[0])
        ax.axhline(d0, color=OKABE[mk], ls="--", lw=0.9)
    ax.set_xticks(range(len(cfgs)), [c for _, c in cfgs], fontsize=6.3)
    lo = np.nanmin([acc(df, mk, "vad", "direct", runs=[0])[0] for mk in models]) - 8
    ax.set_ylim(lo, None)
    ax.set_title("(c) grounding, schema source, voting")
    axes[0].legend(fontsize=5.8, loc="lower left")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_parquet(os.path.join(a.results, "predictions.parquet"))
    for f in (fig_gains, fig_ablation, fig_hparam, fig_qtype):
        try:
            f(df, a.out)
            print("ok", f.__name__)
        except Exception as e:  # keep the other figures when one input is still missing
            print("FAILED", f.__name__, repr(e))


if __name__ == "__main__":
    main()
