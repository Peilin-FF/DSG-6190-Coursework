"""Schematic figures for the CE6190 Project 2 report: our view of CorrFlow, in the colour style of the
authors' figures (blue #0072b2, green #008b6a, orange #ce641d, ink #26333d; pastel boxes: base #d9e5f4,
clean prediction #fdf1dc, refiner #e5d9f7, condition #e8f5f2, sampler #f4f5f7). Output: PDF + PNG."""
import os
import sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

R = "/mnt/data_3T/peilin/ce6190_p2"
OUT = f"{R}/schematics/figures"
SP = f"{R}/media/sampling-pair"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.2})
INK, GREY, LIGHT = "#26333d", "#5b6670", "#8a9099"
BLUE, GREEN, ORANGE, PURPLE = "#0072b2", "#008b6a", "#ce641d", "#7d5fb0"
COL = {"base": ("#d9e5f4", "#83a1ba"), "clean": ("#fdf1dc", "#d9a45a"), "refiner": ("#e5d9f7", "#9a78c4"),
       "cond": ("#e8f5f2", "#86aaa5"), "sampler": ("#f4f5f7", "#8a9099"), "extra": ("#fdf1dc", "#d9a45a")}


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, kind, fs=7.2, bold=False, r=0.05, lw=0.8, z=3, color=INK):
    fc, ec = COL[kind]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=color, zorder=z + 1,
                fontweight="bold" if bold else "normal", linespacing=1.2)


def arrow(ax, p1, p2, color=INK, lw=0.9, ls="-", rad=0.0, z=2, ms=7):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=z))


def poly_arrow(ax, pts, color=INK, lw=0.9, ls="-", z=2, ms=7):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
    ax.plot([xs[-2], xs[-1]], [ys[-2], ys[-1]], color=color, lw=lw, ls=ls, zorder=z)
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, z=z, ms=ms)


def plus(ax, x, y, r=0.065, color=INK):
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=0.9, zorder=4))
    ax.plot([x - r * 0.6, x + r * 0.6], [y, y], color=color, lw=0.9, zorder=5)
    ax.plot([x, x], [y - r * 0.6, y + r * 0.6], color=color, lw=0.9, zorder=5)


def panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc="white",
                                ec="#c5c9cf", lw=0.7, zorder=0))
    ax.text(x + 0.10, y + h - 0.09, title, ha="left", va="top", fontsize=7.8, fontweight="bold", color=INK, zorder=5)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), dpi=600)   # dpi sets the resolution of the embedded thumbnails
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", name)


def teaser():
    W, H = 6.875, 1.86
    fig, ax = canvas(W, H)
    gap = 0.08
    pw = (W - 2 * gap) / 3
    titles = ["(a) Training-time adaptation", "(b) Inference-time steering", "(c) CorrFlow"]
    notes = ["Pays during training and can\nlose adherence to the prompt",
             "Pays in every sampling run with\nextra model calls or memory",
             "One small refiner pass per step;\nthe base prediction stays the anchor"]
    bases = ["Base model\n(adapted)", "Base model\n(fixed)", "Base model\n(trained jointly)"]
    for i in range(3):
        x0 = i * (pw + gap)
        panel(ax, x0 + 0.01, 0.02, pw - 0.02, H - 0.04, titles[i])
        bx, sx, y = x0 + 0.58, x0 + pw - 0.46, 0.92
        box(ax, bx, y, 0.86, 0.38, bases[i], "base")
        box(ax, sx, y, 0.56, 0.38, "Sampler", "sampler")
        poly_arrow(ax, [(sx, y - 0.19), (sx, 0.58), (bx, 0.58), (bx, y - 0.19)], color=LIGHT, lw=0.8)
        ax.text((bx + sx) / 2, 0.62, r"next state $\mathbf{x}_\sigma$", ha="center", va="bottom", fontsize=6.4, color=GREY)
        ax.text(x0 + pw / 2, 0.27, notes[i], ha="center", va="center", fontsize=6.5, color=GREY, style="italic",
                linespacing=1.25)
        if i < 2:
            arrow(ax, (bx + 0.43, y), (sx - 0.28, y), color=BLUE)
            ax.text((bx + 0.43 + sx - 0.28) / 2, y + 0.05, r"$\mathbf{v}$", ha="center", va="bottom", fontsize=7.4, color=BLUE)
        if i == 0:
            box(ax, bx, 1.45, 1.02, 0.32, "Extra objective\nor physics data", "extra", fs=6.6)
            arrow(ax, (bx, 1.29), (bx, y + 0.19), color=ORANGE, ls=(0, (3, 2)))
            ax.text(bx + 0.06, 1.20, "updates weights", ha="left", va="center", fontsize=6.3, color=ORANGE)
        if i == 1:
            box(ax, sx - 0.20, 1.45, 1.14, 0.32, "Reward model, retrieval\nor motion planner", "cond", fs=6.6)
            arrow(ax, (sx, 1.29), (sx, y + 0.19), color=GREEN)
            ax.text(sx - 0.06, 1.20, "guides each run", ha="right", va="center", fontsize=6.3, color=GREEN)
        if i == 2:
            px = (bx + 0.43 + sx - 0.28) / 2 + 0.02
            box(ax, px + 0.02, 1.45, 0.62, 0.30, "Refiner", "refiner", bold=True)
            arrow(ax, (bx + 0.43, y), (px - 0.065, y), color=BLUE)
            ax.text(bx + 0.43 + 0.07, y + 0.05, r"$\mathbf{v}_\theta$", ha="left", va="bottom", fontsize=7.2, color=BLUE)
            plus(ax, px, y)
            arrow(ax, (px + 0.065, y), (sx - 0.28, y), color=GREEN)
            ax.text((px + sx - 0.28) / 2 + 0.05, y + 0.03, r"$\mathbf{v}^{\kappa}$", ha="center", va="bottom", fontsize=7.2, color=GREEN)
            poly_arrow(ax, [(bx, y + 0.19), (bx, 1.45), (px - 0.29, 1.45)], color=INK, lw=0.8)
            ax.text(bx + 0.05, 1.22, r"$\hat{\mathbf{x}}_0,\ \mathbf{h}$", ha="left", va="center", fontsize=6.9, color=INK)
            arrow(ax, (px, 1.30), (px, y + 0.065), color=GREEN)
            ax.text(px + 0.05, 1.19, r"$\kappa\hat{\mathbf{r}}$", ha="left", va="center", fontsize=7.0, color=GREEN)
    save(fig, "fig_intro_placements")


def read_mid(arm, step, frame=40):
    cap = cv2.VideoCapture(f"{SP}/{arm}/step_{step:03d}.mp4")
    i, img = 0, None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i == frame:
            img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            break
        i += 1
    cap.release()
    return img


def method():
    W, H = 6.875, 2.30
    fig, ax = canvas(W, H)
    wa = 3.86
    panel(ax, 0.01, 0.02, wa - 0.02, H - 0.04, "(a) One corrected sampling step")
    # lanes
    bx, bw = 0.86, 0.60
    box(ax, bx, 1.27, bw, 1.06, "Base\nmodel", "base", bold=True)
    ax.text(0.24, 1.27, r"$\mathbf{x}_\sigma$", ha="center", va="center", fontsize=8.4, color=INK)
    arrow(ax, (0.36, 1.27), (bx - bw / 2, 1.27))
    yv, yc, yh = 1.66, 1.24, 0.90
    cx, cw = 1.74, 0.86
    rx, rw = 2.56, 0.64
    px = 2.92
    ex, ew = 3.50, 0.60
    box(ax, cx, yc, cw, 0.40, "Clean prediction\n" + r"$\hat{\mathbf{x}}_0=\mathbf{x}_\sigma-\sigma\mathbf{v}_\theta$", "clean", fs=6.6)
    box(ax, rx, yc, rw, 0.40, "Refiner\n" + r"$\hat{\mathbf{r}}\approx\mathbf{v}-\mathbf{v}_\theta$", "refiner", fs=6.6)
    arrow(ax, (bx + bw / 2, yc), (cx - cw / 2, yc))
    arrow(ax, (cx + cw / 2, yc), (rx - rw / 2, yc))
    arrow(ax, (bx + bw / 2, yv), (px - 0.065, yv), color=BLUE)
    ax.text(1.95, yv + 0.04, r"velocity $\mathbf{v}_\theta$", ha="center", va="bottom", fontsize=6.8, color=BLUE)
    poly_arrow(ax, [(bx + bw / 2, yh), (rx - 0.12, yh), (rx - 0.12, yc - 0.20)], color=INK, lw=0.8, ls=(0, (1.2, 1.4)))
    ax.text(1.74, yh - 0.05, r"hidden states $\mathbf{h}_\ell$", ha="center", va="top", fontsize=6.6, color=INK)
    box(ax, 1.72, 0.47, 2.36, 0.24, r"Condition $\mathbf{c}$ and noise level $\sigma$", "cond", fs=6.6)
    arrow(ax, (bx, 0.59), (bx, 1.27 - 0.53), color=GREY, lw=0.8)
    arrow(ax, (rx + 0.14, 0.59), (rx + 0.14, yc - 0.20), color=GREY, lw=0.8)
    plus(ax, px, yv)
    poly_arrow(ax, [(rx + rw / 2, yc + 0.08), (px, yc + 0.08), (px, yv - 0.065)], color=GREEN)
    ax.text(px + 0.05, 1.47, r"$\kappa\hat{\mathbf{r}}$", ha="left", va="center", fontsize=7.0, color=GREEN)
    box(ax, ex, yv, ew, 0.32, "Euler step", "sampler")
    arrow(ax, (px + 0.065, yv), (ex - ew / 2, yv), color=GREEN)
    ax.text((px + 0.065 + ex - ew / 2) / 2, yv + 0.05, r"$\mathbf{v}^{\kappa}$", ha="center", va="bottom", fontsize=7.0, color=GREEN)
    poly_arrow(ax, [(ex, yv - 0.16), (ex, 0.20), (0.24, 0.20), (0.24, 1.13)], color=LIGHT, lw=0.8)
    ax.text(2.95, 0.235, "next state, 50 steps", ha="center", va="bottom", fontsize=6.3, color=GREY)
    # (b) correction along a released trajectory
    x0 = wa + 0.07
    wb = W - x0 - 0.01
    panel(ax, x0, 0.02, wb, H - 0.04, "(b) Where the correction acts")
    ax.text(x0 + 0.10, H - 0.25, "clean prediction of the middle frame, released trajectory",
            ha="left", va="top", fontsize=6.2, color=GREY, style="italic")
    steps = [1, 10, 20, 30, 40, 50]
    sig = {1: 1.000, 10: 0.958, 17: 0.914, 20: 0.891, 30: 0.784, 40: 0.585, 49: 0.172, 50: 0.093}
    axl, axr, ya = x0 + 0.48, x0 + wb - 0.24, 0.78
    pos = lambda s: axl + (s - 1) / 49 * (axr - axl)
    band = (pos(17), pos(49))
    tw = 0.38
    # background box over the shown steps that lie inside the correction interval (20, 30, 40),
    # aligned with the edges of their thumbnails; the label gives the exact steps 17 to 49
    bl, br = pos(20) - tw / 2 - 0.06, pos(40) + tw / 2 + 0.06
    ax.add_patch(FancyBboxPatch((bl, 0.46), br - bl, 1.14, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=COL["refiner"][0], ec="none", alpha=0.8, zorder=0.5))
    ax.text((bl + br) / 2, 1.66, r"correction interval $\sigma\in[0.15,0.92]$, steps 17$-$49", ha="center", va="bottom",
            fontsize=6.2, color=PURPLE)
    th = tw * 480 / 832
    cxs = [pos(s) for s in steps]   # each thumbnail sits above its own step
    for s, c in zip(steps, cxs):
        img = read_mid("with_refiner", s)
        if img is None:
            sys.exit(f"no frame for step {s}")
        y0 = 1.25
        ax.imshow(img, extent=(c - tw / 2, c + tw / 2, y0, y0 + th), zorder=4, interpolation="lanczos")
        inside = 17 <= s <= 49
        ax.add_patch(Rectangle((c - tw / 2, y0), tw, th, fc="none", ec=COL["refiner"][1] if inside else "#8a9099",
                               lw=1.0 if inside else 0.4, zorder=5))
        ax.plot([pos(s), pos(s)], [ya - 0.03, ya + 0.03], color=INK, lw=0.7, zorder=3)
        ax.text(pos(s), ya - 0.07, f"{s}", ha="center", va="top", fontsize=6.4, color=INK)
        ax.text(pos(s), ya - 0.20, f"{sig[s]:.2f}", ha="center", va="top", fontsize=5.9, color=GREY)
    ax.plot([axl, axr], [ya, ya], color=INK, lw=0.8, zorder=3)
    ax.text(axl - 0.20, ya - 0.07, "step", ha="right", va="top", fontsize=6.0, color=GREY)
    ax.text(axl - 0.20, ya - 0.20, r"$\sigma$", ha="right", va="top", fontsize=6.4, color=GREY)
    ax.text((cxs[0] + cxs[1]) / 2, 1.12, "layout forms", ha="center", va="center", fontsize=6.4, color=INK, style="italic")
    ax.text((bl + br) / 2, 1.12, "detail is refined", ha="center", va="center", fontsize=6.4, color=PURPLE, style="italic")
    ax.text(x0 + wb / 2, 0.17, r"$\kappa=0.2$ before guidance with scale 5: residual weight $5\kappa=1$",
            ha="center", va="center", fontsize=6.2, color=GREY)
    save(fig, "fig_method_step")


if __name__ == "__main__":
    teaser()
    method()
