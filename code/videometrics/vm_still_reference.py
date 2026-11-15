"""Still-video reference for the Physics-IQ agreement metrics (CPU only).

For each of the 10 image-to-video cases, a video that never moves (the conditioning switch frame repeated for
the 80 frames of the 5 s window) is scored against the released ground-truth clip with the same ported steps as
vm_physiq.py (16 fps, 960x540, official motion masks and metric definitions). The reference shows how the four
raw metrics treat an output without motion: it has no motion mask, so its spatial IoU and weighted spatial IoU
are 0, while its spatiotemporal IoU equals the share of ground-truth frames without detected motion and its MSE
is the pixel change of the real clip.
Inputs : the switch-frame images and ground-truth clips of the project page, results/physiq_per_case.csv
Outputs: results/physiq_still_reference.csv   per case: the four metrics, the share of empty ground-truth frames
         results/physiq_still_vs_methods.csv  per method and metric: mean of the method and of the still video,
                                              cases where the still video scores better than the method
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vm_common as C  # noqa: E402
import vm_physiq as P  # noqa: E402

RES = os.path.join(C.VM, "results")
METRICS = ["spatial_iou", "spatiotemporal_iou", "weighted_spatial_iou", "mse"]
LOWER_IS_BETTER = {"mse"}
SWITCH = os.path.join(C.MEDIA, "public", "media", "video", "physics-iq-veried", "01_input", "%s_switch-frame.jpg")


def write(path, rows):
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", path, len(rows))


def main():
    rows = []
    for c in [c for c in C.load_cases() if c["task"] == "i2v"]:
        gt = [v for v in c["videos"] if v["method"] == "Ground truth"][0]
        gfr, _, _ = P.prepare(gt)
        gm = P.official_masks(gfr)
        sw = cv2.cvtColor(cv2.imread(SWITCH % c["stem"]), cv2.COLOR_BGR2RGB)
        sw = cv2.resize(sw, P.SIZE)  # bilinear, as every generated frame in vm_physiq.prepare
        still = np.repeat(sw[None], P.N, axis=0)
        m = P.official_masks(still)
        st = P.iou_per_frame(gm, m)
        sp = float(P.iou_per_frame(gm.any(0)[None], m.any(0)[None])[0])
        wsp = P.weighted_iou(gm.sum(0, dtype=np.uint16) / len(gm), m.sum(0, dtype=np.uint16) / len(m))
        mse = float(((gfr.astype(np.float32) / 255.0 - still.astype(np.float32) / 255.0) ** 2).mean())
        rows.append({"case_id": c["id"], "reference": "still_switch_frame", "spatial_iou": sp,
                     "spatiotemporal_iou": float(st.mean()), "weighted_spatial_iou": wsp, "mse": mse,
                     "motion_share": float(m.mean()), "gt_empty_frame_share": float((gm.sum(axis=(1, 2)) == 0).mean())})
        r = rows[-1]
        print("%-26s still: sIoU %.3f stIoU %.4f wsIoU %.3f MSE %.5f motion %.1e | GT empty frames %.4f" % (
            c["id"], sp, r["spatiotemporal_iou"], wsp, mse, r["motion_share"], r["gt_empty_frame_share"]), flush=True)
    write(os.path.join(RES, "physiq_still_reference.csv"), rows)

    still = {r["case_id"]: r for r in rows}
    per_case = list(csv.DictReader(open(os.path.join(RES, "physiq_per_case.csv"))))
    methods = []
    for r in per_case:
        if r["method"] not in methods:
            methods.append(r["method"])
    out = []
    for m in methods:
        mine = {r["case_id"]: r for r in per_case if r["method"] == m}
        cases = sorted(mine)
        for k in METRICS:
            a = np.array([float(mine[c][k]) for c in cases])
            s = np.array([float(still[c][k]) for c in cases])
            better = int((s < a).sum()) if k in LOWER_IS_BETTER else int((s > a).sum())
            out.append({"method": m, "metric": k, "method_mean": float(a.mean()), "still_mean": float(s.mean()),
                        "still_median": float(np.median(s)), "still_better": better, "ties": int((s == a).sum()),
                        "n": len(cases)})
    write(os.path.join(RES, "physiq_still_vs_methods.csv"), out)


if __name__ == "__main__":
    main()
