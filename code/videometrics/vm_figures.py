"""Figures of our measurements on the released CorrFlow videos (house plot style, print size for a CVPR page).

Inputs : results/noref_means.csv, results/noref_paired.csv, results/noref_paired_flow_log2.csv,
         results/noref_per_video.csv (native frame rates), results/physiq_means.csv, results/physiq_per_case.csv,
         results/physiq_still_reference.csv
Outputs: figures/*.png (600 dpi) + *.pdf
  noref_means_<metric>          one panel per metric, one marker per method and benchmark: the mean over the
                                prompts (flicker, subject consistency, CLIP) or the median (optical flow) (3.3 in)
  noref_means_grid              the four panels in one full-width figure (6.9 in)
  noref_means_legend            legend strip for the single panels
  noref_paired_grid             CorrFlow minus each method on the same prompts: mean and bootstrap 95% CI, flow as a
                                log2 ratio; no interval for FinePhyEval (2 prompts) (6.9 in)
  physiq_agreement_<metric>     one panel per agreement metric: mean over the 10 cases per method, the still-video
                                reference dashed where it is above 0 (1.65 in)
  physiq_agreement_row          the four agreement panels in one full-width row: spatial IoU and weighted spatial IoU
                                first, then spatiotemporal IoU and MSE (6.9 in)
  physiq_agreement_legend       legend strip for the single agreement panels
  physiq_agreement_cases        per-case values, cases ordered by CorrFlow's spatial-IoU gain over its base Wan2.2
                                (6.9 in, appendix)
Font: bold TeX Gyre Termes (a Times clone, TrueType conversion registered from ~/.local/share/fonts/tex-gyre-termes,
see fonts/README.txt); DejaVu Serif if it is missing. Text is 7-8 pt at the printed width.
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vm_common as C  # noqa: E402
import plot_style as ps  # noqa: E402

RES = os.path.join(C.VM, "results")
FIG = os.path.join(C.VM, "figures")
TIMES_CLONE = "TeX Gyre Termes"
FONT_DIRS = [os.path.expanduser("~/.local/share/fonts/tex-gyre-termes")]

COLOR = {"CorrFlow": ps.PALETTE["blue"], "Wan2.1": ps.PALETTE["blue_light"], "Wan2.2": ps.PALETTE["blue_light"],
         "PhysRAG": ps.PALETTE["pink"], "VideoREPA": ps.PALETTE["pink_light"], "DreamWorld": ps.PALETTE["base_gray"],
         "VPT": "#D6D6D6", "VLiPP": ps.PALETTE["pink_light"], "WMReward": "#696969", "Ground truth": "#1F1F1F"}
HATCH = {"VLiPP": "////"}
EDGE = {"VideoREPA": ps.PALETTE["pink"], "VLiPP": ps.PALETTE["pink_text"], "Wan2.1": ps.PALETTE["blue"],
        "Wan2.2": ps.PALETTE["blue"], "VPT": "#999999"}
MARKER = {"VLiPP": "D"}            # VLiPP shares VideoREPA's tint; the diamond keeps the two apart
STILL_COLOR = ps.PALETTE["spine"]  # still-video reference line (the real clip is the black dashed line elsewhere)
TEXT_PT = 7                        # value and count labels (the brief asks for 7-9 pt)
BENCH = [C.BENCH_NAME[b] for b in C.BENCH_ORDER]
BENCH_TICK = {"VideoPhy2": "VideoPhy2\n(T2V, 6)", "Physics-IQ": "Physics-IQ\n(I2V, 10)",
              "FinePhyEval": "FinePhyEval\n(T2V, 2)", "VBench": "VBench\n(T2V, 14)"}
NOREF_LABEL = {"flicker": "Mean flicker score", "flow_px_s": "Median optical flow (px/s)",
               "subject": "Mean subject consistency", "clip": "Mean CLIP score (×100)"}
NOREF_STAT = {"flicker": "mean", "flow_px_s": "median", "subject": "mean", "clip": "mean"}
PHYSIQ_ORDER = ["spatial_iou", "weighted_spatial_iou", "spatiotemporal_iou", "mse"]
PHYSIQ_LABEL = {"spatial_iou": "Spatial IoU ↑", "spatiotemporal_iou": "Spatiotemporal IoU ↑",
                "weighted_spatial_iou": "Weighted spatial IoU ↑", "mse": "MSE (×1000) ↓"}


def register_times_clone():
    """Make the TrueType TeX Gyre Termes faces known to matplotlib without rebuilding the shared font cache."""
    added = []
    for d in FONT_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*.ttf"))):
            mpl.font_manager.fontManager.addfont(f)
            added.append(f)
    return added


def print_rc():
    """House rcParams scaled to print size (text 7-8 pt at 3.3 / 6.9 in) with a bold Times clone."""
    added = register_times_clone()
    ps.apply()
    have_times = any(e.name == TIMES_CLONE for e in mpl.font_manager.fontManager.ttflist)
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.8, "grid.linewidth": 0.5, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "hatch.linewidth": 0.5, "legend.borderpad": 0.3, "legend.handlelength": 1.4, "legend.columnspacing": 1.0,
        "legend.handletextpad": 0.4, "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })
    if have_times:
        mpl.rcParams["font.serif"] = [TIMES_CLONE] + [f for f in mpl.rcParams["font.serif"] if f != TIMES_CLONE]
        mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": TIMES_CLONE + ":bold",
                             "mathtext.it": TIMES_CLONE + ":italic:bold", "mathtext.bf": TIMES_CLONE + ":bold"})
    else:
        mpl.rcParams["mathtext.fontset"] = "dejavuserif"
    prop = mpl.font_manager.FontProperties(family=mpl.rcParams["font.family"], weight="bold")
    used = "%s (%s)" % (prop.get_name(), mpl.font_manager.findfont(prop))
    print("[figures] registered %d font files; font used: %s" % (len(added), used))
    return used


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def order_key(m):
    return C.METHOD_ORDER.index(m)


def bar_kw(m, lw=0.4):
    return dict(color=COLOR[m], hatch=HATCH.get(m), edgecolor=EDGE.get(m, "white" if m not in HATCH else "#595959"),
                linewidth=lw)


def marker_kw(m, size):
    return dict(marker=MARKER.get(m, "o"), s=size, color=COLOR[m], edgecolors=EDGE.get(m, "#595959"),
                linewidths=0.6)


def plain_log_ticks(axis):
    """Log axis with plain tick labels (0.3, 1, 3, 10, ...) instead of mathtext powers."""
    axis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=(1.0, 3.0)))
    axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: "%g" % v))
    axis.set_minor_locator(mpl.ticker.NullLocator())


def base_label(methods):
    bases = [m for m in methods if m in ("Wan2.1", "Wan2.2")]
    if len(bases) > 1:
        return "Base (%s)" % " / ".join(bases)
    return "%s (base)" % bases[0] if bases else ""


def legend_handles(methods, kind="bar", still=False, with_gt=False):
    """One entry per method; the two base models share a colour and one entry."""
    blabel = base_label(methods)
    hs, done_base = [], False
    for m in methods:
        if m in ("Wan2.1", "Wan2.2"):
            if done_base:
                continue
            done_base = True
            label = blabel
        else:
            label = m
        if kind == "bar":
            hs.append(Patch(facecolor=COLOR[m], hatch=HATCH.get(m), edgecolor=EDGE.get(m, COLOR[m]), linewidth=0.5,
                            label=label))
        else:
            hs.append(Line2D([0], [0], ls="", marker=MARKER.get(m, "o"), markersize=4.5, markerfacecolor=COLOR[m],
                             markeredgecolor=EDGE.get(m, "#595959"), markeredgewidth=0.6, label=label))
    if with_gt:
        hs.append(Line2D([0], [0], color=COLOR["Ground truth"], ls="--", lw=1.0, label="Ground truth"))
    if still:
        hs.append(Line2D([0], [0], color=STILL_COLOR, ls="--", lw=1.0, label="Still video"))
    return hs


def scale(metric, v):
    return 100 * v if metric == "clip" else v


# ----------------------------------------------------------------------------- no-reference summaries (markers)
def draw_noref_panel(ax, metric, means):
    stat = NOREF_STAT[metric]
    lut = {(r["group"], r["method"]): r for r in means if r["metric"] == metric}
    step = 0.13
    allv = []
    for gi, b in enumerate(BENCH):
        ms = sorted([m for (g, m) in lut if g == b and m != "Ground truth"], key=order_key)
        xs = gi + (np.arange(len(ms)) - (len(ms) - 1) / 2) * step
        for x, m in zip(xs, ms):
            v = scale(metric, float(lut[(b, m)][stat]))
            allv.append(v)
            ax.scatter([x], [v], zorder=4 if m == "CorrFlow" else 3, **marker_kw(m, 22 if m == "CorrFlow" else 17))
        if (b, "Ground truth") in lut:
            v = scale(metric, float(lut[(b, "Ground truth")][stat]))
            allv.append(v)
            ax.plot([xs[0] - step / 2, xs[-1] + step / 2], [v, v], ls="--", lw=1.0, color=COLOR["Ground truth"],
                    zorder=2)
    if metric == "flow_px_s":
        ax.set_yscale("log")
        ax.set_ylim(0.6 * min(allv), 1.7 * max(allv))
        plain_log_ticks(ax.yaxis)
        ax.grid(True, axis="y", which="major")
    else:
        lo, hi = min(allv), max(allv)
        pad = 0.1 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xticks(range(len(BENCH)))
    ax.set_xticklabels([BENCH_TICK[b] for b in BENCH])
    ax.set_xlim(-0.5, len(BENCH) - 0.5)
    ax.set_ylabel(NOREF_LABEL[metric])
    ax.tick_params(axis="x", length=0)


def fig_noref_means(means):
    methods = [m for m in C.METHOD_ORDER if m != "Ground truth" and any(r["method"] == m for r in means)]
    metrics = ["flicker", "flow_px_s", "subject", "clip"]
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(3.3, 1.75))
        draw_noref_panel(ax, metric, means)
        ps.save(fig, os.path.join(FIG, "noref_means_%s" % metric))
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.75))
    for ax, metric in zip(axes.flat, metrics):
        draw_noref_panel(ax, metric, means)
    hs = legend_handles(methods, kind="marker", with_gt=True)
    fig.legend(handles=hs, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), frameon=True,
               prop={"weight": "bold", "size": 7})
    fig.tight_layout(h_pad=0.6, w_pad=1.0, rect=(0, 0, 1, 0.9))
    ps.save(fig, os.path.join(FIG, "noref_means_grid"))
    fig = plt.figure(figsize=(6.9, 0.4))
    fig.legend(handles=hs, loc="center", ncol=5, frameon=True, prop={"weight": "bold", "size": 7})
    ps.save(fig, os.path.join(FIG, "noref_means_legend"))


# ----------------------------------------------------------------------------- paired differences
def fig_noref_paired(paired, flow_log2, low_fps):
    metrics = ["flicker", "flow_px_s", "subject", "clip"]
    rows = []
    for b in BENCH:
        ms = sorted({r["method"] for r in paired if r["group"] == b}, key=order_key)
        rows += [(b, m) for m in ms]
    lut = {(r["group"], r["method"], r["metric"]): r for r in paired}
    flut = {(r["group"], r["method"]): r for r in flow_log2}
    y, heads, pos = [], [], 0.0
    nb = {b: max(int(lut[(b, m, "clip")]["n"]) for (bb, m) in rows if bb == b) for b in BENCH}
    for b in BENCH:
        heads.append((-pos, "%s (%d prompts)" % (b, nb[b])))
        pos += 1.0
        for bb, m in rows:
            if bb == b:
                y.append(-pos)
                pos += 1.0
        pos += 0.3
    y = np.array(y)
    xlabel = {"flicker": "Flicker score diff.\n(×100)", "flow_px_s": "Optical flow ratio\n(log2)",
              "subject": "Subject consistency\ndiff. (×100)", "clip": "CLIP score diff.\n(×100)"}
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 3.6), sharey=True)
    for ax, metric in zip(axes, metrics):
        for yi, (b, m) in zip(y, rows):
            if metric == "flow_px_s":
                r = flut[(b, m)]
                mean, lo, hi = float(r["mean_log2"]), float(r["ci_lo"]), float(r["ci_hi"])
            else:
                r = lut[(b, m, metric)]
                mean, lo, hi = [100 * float(r[k]) for k in ("mean_diff", "ci_lo", "ci_hi")]
            k, n = int(r["corrflow_higher"]), int(r["n"])
            ax.barh(yi, mean, 0.72, **bar_kw(m), zorder=2)
            if n > 2:  # with two prompts the "interval" is only the range of two values: no whisker
                ax.plot([lo, hi], [yi, yi], color="#1F1F1F", lw=0.8, zorder=3)
            ax.text(1.03, yi, "%d/%d" % (k, n), transform=ax.get_yaxis_transform(), va="center", ha="left",
                    fontsize=TEXT_PT, fontweight="bold")
        for hy, _ in heads[1:]:
            ax.axhline(hy + 0.65, color=ps.PALETTE["spine"], lw=0.5, ls="-", zorder=1)
        ax.axvline(0, color=ps.PALETTE["spine"], lw=0.8, zorder=1)
        ax.set_xlabel(xlabel[metric])
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        lim = max(abs(v) for v in ax.get_xlim())
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, symmetric=True, steps=[1, 2, 5, 10]))
        ax.set_ylim(min(y) - 0.7, 0.7)
        ax.tick_params(axis="y", length=0)
    ticks = list(y) + [h for h, _ in heads]
    labels = [m + ("†" if (b, m) in low_fps else "") for b, m in rows] + [t for _, t in heads]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(labels)
    for lab, t in zip(axes[0].get_yticklabels(), ticks):
        if t in [h for h, _ in heads]:
            lab.set_fontstyle("italic")
            lab.set_color(ps.PALETTE["spine"])
    axes[0].tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=2.4)
    ps.save(fig, os.path.join(FIG, "noref_paired_grid"))


# ----------------------------------------------------------------------------- Physics-IQ agreement
def fig_physiq(pmeans, pcases, still_rows):
    methods = sorted({r["method"] for r in pmeans}, key=order_key)
    lut = {(r["method"], r["metric"]): float(r["mean"]) for r in pmeans}
    still = {k: float(np.mean([float(r[k]) for r in still_rows])) for k in PHYSIQ_ORDER}

    def unit(metric, v):
        return v * 1000 if metric == "mse" else v

    def panel(ax, metric):
        vals = [unit(metric, lut[(m, metric)]) for m in methods]
        ref = unit(metric, still[metric])
        xs = np.arange(len(methods))
        for x, m, v in zip(xs, methods, vals):
            ax.bar(x, v, 0.72, **bar_kw(m), zorder=2)
            ax.annotate(("%.1f" if metric == "mse" else "%.2f") % v, (x, v), xytext=(0, 1.0),
                        textcoords="offset points", ha="center", va="bottom", fontsize=TEXT_PT, fontweight="bold",
                        zorder=4)
        top = max(vals + [ref])
        if ref > 0:
            ax.axhline(ref, color=STILL_COLOR, ls="--", lw=1.0, zorder=3)
        ax.set_ylim(0, top * 1.22)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.set_xticks(xs)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel(PHYSIQ_LABEL[metric])
        ax.set_xlim(-0.85, len(methods) - 0.15)

    for metric in PHYSIQ_ORDER:
        fig, ax = plt.subplots(figsize=(1.65, 1.6))
        panel(ax, metric)
        ps.save(fig, os.path.join(FIG, "physiq_agreement_%s" % metric))
    hs = legend_handles(methods, kind="bar", still=True)
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.75))
    for ax, metric in zip(axes, PHYSIQ_ORDER):
        panel(ax, metric)
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), bbox_to_anchor=(0.5, 1.06), frameon=True,
               prop={"weight": "bold", "size": 7})
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.9))
    ps.save(fig, os.path.join(FIG, "physiq_agreement_row"))
    fig = plt.figure(figsize=(6.9, 0.25))
    fig.legend(handles=hs, loc="center", ncol=len(hs), frameon=True, prop={"weight": "bold", "size": 7})
    ps.save(fig, os.path.join(FIG, "physiq_agreement_legend"))

    # per case (appendix): cases ordered by CorrFlow's spatial-IoU gain over its base (largest gain at the top)
    v = {(r["case_id"], r["method"], k): float(r[k]) for r in pcases for k in PHYSIQ_ORDER}
    cases = sorted({r["case_id"] for r in pcases},
                   key=lambda c: -(v[(c, "CorrFlow", "spatial_iou")] - v[(c, "Wan2.2", "spatial_iou")]))
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.6), sharey=True)
    yy = -np.arange(len(cases))
    off = (np.arange(len(methods)) - (len(methods) - 1) / 2) * 0.14
    for ax, metric in zip(axes, PHYSIQ_ORDER):
        for j, m in enumerate(methods):
            xs = [unit(metric, v[(c, m, metric)]) for c in cases]
            ax.scatter(xs, yy + off[j], zorder=4 if m == "CorrFlow" else 3,
                       **marker_kw(m, 14 if m == "CorrFlow" else 10))
        ax.set_xlabel({"spatial_iou": "Spatial IoU ↑", "spatiotemporal_iou": "Spatiotemporal\nIoU ↑",
                       "weighted_spatial_iou": "Weighted spatial\nIoU ↑",
                       "mse": "MSE (×1000) ↓\n(log scale)"}[metric])
        ax.tick_params(axis="y", length=0)
        if metric == "mse":
            ax.set_xscale("log")
            plain_log_ticks(ax.xaxis)
        else:
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        for yv in yy[:-1]:
            ax.axhline(yv - 0.5, color=ps.PALETTE["grid"], lw=0.4, ls="--", zorder=0)
        ax.set_ylim(yy[-1] - 0.5, 0.5)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(["%s %s" % (C.case_short(c), C.PHYSIQ_CASE_LABEL[c]) for c in cases])
    axes[0].tick_params(axis="y", length=0)
    hs = [Line2D([0], [0], marker=MARKER.get(m, "o"), ls="", markersize=4, markerfacecolor=COLOR[m],
                 markeredgecolor=EDGE.get(m, "#595959"), markeredgewidth=0.5,
                 label=m + (" (base)" if m in ("Wan2.1", "Wan2.2") else "")) for m in methods]
    fig.legend(handles=hs, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.04), frameon=True,
               prop={"weight": "bold", "size": 7})
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.93))
    ps.save(fig, os.path.join(FIG, "physiq_agreement_cases"))
    return cases


def low_fps_pairs(per_video):
    return {(C.BENCH_NAME[r["benchmark"]], r["method"]) for r in per_video if abs(float(r["src_fps"]) - 9.8) < 1e-6}


def main():
    used = print_rc()
    os.makedirs(FIG, exist_ok=True)
    means = read(os.path.join(RES, "noref_means.csv"))
    paired = read(os.path.join(RES, "noref_paired.csv"))
    flow_log2 = read(os.path.join(RES, "noref_paired_flow_log2.csv"))
    per_video = read(os.path.join(RES, "noref_per_video.csv"))
    low = low_fps_pairs(per_video)
    print("[figures] 9.8 fps rows:", sorted(low))
    fig_noref_means([r for r in means if r["group"] in BENCH])
    fig_noref_paired([r for r in paired if r["group"] in BENCH], [r for r in flow_log2 if r["group"] in BENCH], low)
    order = fig_physiq(read(os.path.join(RES, "physiq_means.csv")), read(os.path.join(RES, "physiq_per_case.csv")),
                       read(os.path.join(RES, "physiq_still_reference.csv")))
    print("[figures] per-case order (largest spatial-IoU gain first):", [C.case_short(c) for c in order])
    with open(os.path.join(FIG, "FONT_USED.txt"), "w") as f:
        f.write(used + "\n")
    print("done; files:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
