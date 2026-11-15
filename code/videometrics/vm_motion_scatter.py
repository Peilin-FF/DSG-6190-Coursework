"""Motion and temporal scores of the released text-to-video clips (our measurement): flicker score and
subject consistency against optical flow, one point per clip, with Spearman correlations."""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import vm_figures as F  # noqa: E402
import plot_style as ps  # noqa: E402


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r


def spearman(x, y):
    return float(np.corrcoef(ranks(np.asarray(x)), ranks(np.asarray(y)))[0, 1])


def main():
    F.print_rc()
    rows = [r for r in csv.DictReader(open(os.path.join(F.RES, "noref_per_video.csv"))) if r["task"] == "t2v"]
    methods = sorted({r["method"] for r in rows}, key=F.order_key)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.35))
    for ax, metric, label in zip(axes, ["flicker", "subject"], ["Flicker score", "Subject consistency"]):
        xs = [float(r["flow_px_s"]) for r in rows]
        ys = [float(r[metric]) for r in rows]
        rho = spearman([math.log(x) for x in xs], ys)
        for m in methods[::-1]:
            pts = [(float(r["flow_px_s"]), float(r[metric])) for r in rows if r["method"] == m]
            if not pts:
                continue
            px, py = zip(*pts)
            ax.scatter(px, py, s=16 if m == "CorrFlow" else 11, c=F.COLOR.get(m, "#A5A5A5"),
                       edgecolors=F.EDGE.get(m, "#595959"), linewidths=0.5, zorder=4 if m == "CorrFlow" else 3)
        ax.set_xscale("log")
        F.plain_log_ticks(ax.xaxis)
        ax.set_xlabel("Optical flow (px/s, log scale)")
        ax.set_ylabel(label)
        ax.text(0.97, 0.06, f"Spearman $\\rho$ = {rho:.2f}, {len(rows)} clips".replace("-", "\u2212"), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, fontweight="bold")
        print(metric, "rho", round(rho, 3), "n", len(rows))
    hs = [Line2D([0], [0], marker="o", ls="", markersize=4.5 if m == "CorrFlow" else 3.8,
                 markerfacecolor=F.COLOR.get(m, "#A5A5A5"), markeredgecolor=F.EDGE.get(m, "#595959"),
                 markeredgewidth=0.5, label=m + (" (base)" if m == "Wan2.1" else "")) for m in methods]
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), bbox_to_anchor=(0.5, 1.02), frameon=True,
               prop={"weight": "bold", "size": 7})
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    ps.save(fig, os.path.join(F.FIG, "motion_scatter"))
    print("wrote motion_scatter")


if __name__ == "__main__":
    main()
