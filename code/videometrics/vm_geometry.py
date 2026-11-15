"""Check how each image-to-video output maps onto the 16:9 ground-truth frame (one-off check before measuring).

For every Physics-IQ case: compare frame 0 of each method with the ground-truth frame 0 and with the
switch-frame image under three candidate mappings (full resize, centre crop in width, letterbox).
Also find which ground-truth frame matches the switch frame best. Writes results/geometry_check.csv.
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vm_common as C  # noqa: E402

W, H = 480, 270  # comparison size (16:9)


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def candidates(fr, ref):
    """fr: method frame (h, w, 3); ref: 16:9 reference (H, W, 3) at comparison size. Returns {mapping: ncc}."""
    h, w = fr.shape[:2]
    aspect_m = w / h
    aspect_r = 16 / 9
    out = {}
    # 1) full-frame resize: the method frame shows the whole 16:9 scene
    out["resize"] = ncc(gray(cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)), gray(ref))
    # 2) centre crop in width: the method frame shows the central part of the scene, full height
    frac = aspect_m / aspect_r
    if frac < 1:
        cw = int(round(W * frac))
        x0 = (W - cw) // 2
        out["crop_w"] = ncc(gray(cv2.resize(fr, (cw, H), interpolation=cv2.INTER_AREA)), gray(ref[:, x0:x0 + cw]))
        # 3) letterbox: the whole scene squeezed into a band of the method frame
        bh = int(round(h * frac))
        y0 = (h - bh) // 2
        out["letterbox"] = ncc(gray(cv2.resize(fr[y0:y0 + bh], (W, H), interpolation=cv2.INTER_AREA)), gray(ref))
    return out


def main():
    cases = [c for c in C.load_cases() if c["task"] == "i2v"]
    rows = []
    for c in cases:
        sw_path = os.path.join(C.MEDIA, "public", "media", "video", "physics-iq-veried", "01_input",
                               c["stem"] + "_switch-frame.jpg")
        sw = cv2.cvtColor(cv2.imread(sw_path), cv2.COLOR_BGR2RGB)
        sw_s = cv2.resize(sw, (W, H), interpolation=cv2.INTER_AREA)
        gt = [v for v in c["videos"] if v["method"] == "Ground truth"][0]
        gtf = C.read_all_frames(gt["path"])
        gt0 = cv2.resize(gtf[0], (W, H), interpolation=cv2.INTER_AREA)
        best = [(ncc(gray(cv2.resize(gtf[k], (W, H), interpolation=cv2.INTER_AREA)), gray(sw_s)), k) for k in range(6)]
        rows.append({"case": c["id"], "method": "Ground truth", "w": gtf.shape[2], "h": gtf.shape[1],
                     "ref": "switch", "mapping": "gt_frame_vs_switch",
                     "ncc": max(best)[0], "note": "best gt frame %d; ncc per frame 0-5: %s" % (
                         max(best)[1], " ".join("%.4f" % b[0] for b in best))})
        for v in c["videos"]:
            if v["method"] == "Ground truth":
                continue
            f0 = C.read_all_frames(v["path"])[0]
            for refname, ref in (("gt0", gt0), ("switch", sw_s)):
                for mp, val in candidates(f0, ref).items():
                    rows.append({"case": c["id"], "method": v["method"], "w": f0.shape[1], "h": f0.shape[0],
                                 "ref": refname, "mapping": mp, "ncc": val, "note": ""})
        print(c["id"], "done", flush=True)
    out = os.path.join(C.VM, "results", "geometry_check.csv")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    # summary: best mapping per method (mean ncc over cases)
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["method"], r["ref"], r["mapping"])].append(r["ncc"])
    for k in sorted(agg):
        print("%-14s %-7s %-20s mean ncc %.4f  min %.4f  (n=%d)" % (k[0], k[1], k[2], np.mean(agg[k]),
                                                                  np.min(agg[k]), len(agg[k])))
    for r in rows:
        if r["mapping"] == "gt_frame_vs_switch":
            print(r["case"], r["note"])


if __name__ == "__main__":
    main()
