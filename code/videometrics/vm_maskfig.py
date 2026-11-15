"""Case figures of the Physics-IQ motion masks (where the motion happens over the 5 s).

Each tile: ground-truth frame 0 (grey) with a method's motion area (mask OR-ed over the 80 frames) filled in
blue and the ground-truth motion area outlined in pink. Tiles: Ground truth | CorrFlow | Wan2.2 (base).
Inputs : results/physiq_masks.npz, results/physiq_per_case.csv, the released ground-truth clips.
Outputs: figures/masks/<case>.png/.pdf (one row of three tiles per case, 6.9 in wide)
         figures/physiq_mask_cases.png/.pdf (two cases stacked: the largest and the smallest spatial-IoU gain of
         CorrFlow over its base Wan2.2; rows labelled with the case id and a short name from the prompt)
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import cv2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vm_common as C  # noqa: E402
import vm_physiq as P  # noqa: E402
import vm_figures as VF  # noqa: E402
import plot_style as ps  # noqa: E402

RES = os.path.join(C.VM, "results")
FIG = os.path.join(C.VM, "figures")
COLS = ["Ground truth", "CorrFlow", "Wan2.2"]
BLUE = np.array([0x64, 0x9C, 0xD8], dtype=np.float32)
PINK = (0xFF, 0x85, 0xA1)


def tile(frame0, mask_any, gt_any):
    g = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY).astype(np.float32)
    base = np.repeat((0.35 * g + 0.65 * 235)[..., None], 3, axis=2)  # light grey background
    out = base.copy()
    out[mask_any] = 0.35 * base[mask_any] + 0.65 * BLUE
    out = out.clip(0, 255).astype(np.uint8)
    cnts, _ = cv2.findContours(gt_any.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, PINK, 3)
    return out


def main():
    VF.print_rc()
    z = np.load(os.path.join(RES, "physiq_masks.npz"))
    rows = list(csv.DictReader(open(os.path.join(RES, "physiq_per_case.csv"))))
    siou = {(r["case_id"], r["method"]): float(r["spatial_iou"]) for r in rows}
    cases = {c["id"]: c for c in C.load_cases() if c["task"] == "i2v"}
    os.makedirs(os.path.join(FIG, "masks"), exist_ok=True)
    tiles = {}
    for cid, c in cases.items():
        gt = [v for v in c["videos"] if v["method"] == "Ground truth"][0]
        f0 = P.prepare(gt)[0][0]
        unpack = lambda m: np.unpackbits(z[cid + "__" + m], axis=-1)[:, :, :P.SIZE[0]].astype(bool).any(0)
        gt_any = unpack("Ground truth")
        tiles[cid] = [(m, tile(f0, unpack(m), gt_any)) for m in COLS]
        fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.42))
        for ax, (m, t) in zip(axes, tiles[cid]):
            ax.imshow(t)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            lab = m if m == "Ground truth" else "%s: spatial IoU %.2f" % ("Wan2.2 (base)" if m == "Wan2.2" else m,
                                                                         siou[(cid, m)])
            ax.set_xlabel(lab, fontsize=7)
        fig.tight_layout(w_pad=0.4)
        ps.save(fig, os.path.join(FIG, "masks", cid.replace("physics-iq-veried-", "")))
    ranked = sorted(cases, key=lambda cid: siou[(cid, "CorrFlow")] - siou[(cid, "Wan2.2")])
    pick = [ranked[-1], ranked[0]]  # largest and smallest gain of CorrFlow over its base in spatial IoU
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 2.95))
    for r, cid in enumerate(pick):
        name = "%s\n%s" % (C.case_short(cid), C.PHYSIQ_CASE_LABEL[cid])
        for k, (m, t) in enumerate(tiles[cid]):
            ax = axes[r, k]
            ax.imshow(t)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            lab = m if m == "Ground truth" else "%s: spatial IoU %.2f" % ("Wan2.2 (base)" if m == "Wan2.2" else m,
                                                                         siou[(cid, m)])
            ax.set_xlabel(lab, fontsize=7)
            if k == 0:
                ax.set_ylabel(name, fontsize=7)
    fig.tight_layout(w_pad=0.4, h_pad=1.3)
    ps.save(fig, os.path.join(FIG, "physiq_mask_cases"))
    print("picked", pick, [(siou[(c, "CorrFlow")], siou[(c, "Wan2.2")]) for c in pick])


if __name__ == "__main__":
    main()
