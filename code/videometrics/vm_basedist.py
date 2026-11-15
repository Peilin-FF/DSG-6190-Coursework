"""How far each released output is from the base model's output on the same case (descriptive, CPU only).

I2V (Physics-IQ): frames prepared exactly as in vm_physiq.py (16 fps, 80 frames, 960x540); per-frame MSE
(frames in [0, 1]) between each method and the Wan2.2 output, and between each method and CorrFlow.
T2V: frames on the 16 fps grid at 704x480 (vm_common); frame-0 NCC (grey) and mean MSE between CorrFlow and
each other method, to see which pairs start from the same noise (a near-identical first frame).
Writes results/base_distance.csv and results/base_distance_per_frame.json.
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vm_common as C  # noqa: E402
import vm_physiq as P  # noqa: E402


def ncc(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def main():
    rows, curves = [], {}
    for c in C.load_cases():
        if c["task"] == "i2v":
            fr = {v["method"]: P.prepare(v)[0].astype(np.float32) / 255.0 for v in c["videos"]}
            for ref in ("Wan2.2", "CorrFlow"):
                for m, x in fr.items():
                    if m == ref:
                        continue
                    mse_t = ((x - fr[ref]) ** 2).mean(axis=(1, 2, 3))
                    rows.append({"case_id": c["id"], "benchmark": c["benchmark"], "ref": ref, "method": m,
                                 "frame0_ncc": ncc((x[0] * 255).astype(np.uint8), (fr[ref][0] * 255).astype(np.uint8)),
                                 "mse_mean": float(mse_t.mean()), "mse_f20": float(mse_t[20]),
                                 "mse_f40": float(mse_t[40]), "mse_f79": float(mse_t[79])})
                    curves["%s|%s|%s" % (c["id"], ref, m)] = mse_t.tolist()
        else:
            fr = {v["method"]: C.load_grid(v, "t2v")["frames"] for v in c["videos"]}
            ref = "CorrFlow"
            for m, x in fr.items():
                if m == ref:
                    continue
                d = (x.astype(np.float32) / 255.0 - fr[ref].astype(np.float32) / 255.0) ** 2
                mse_t = d.mean(axis=(1, 2, 3))
                rows.append({"case_id": c["id"], "benchmark": c["benchmark"], "ref": ref, "method": m,
                             "frame0_ncc": ncc(x[0], fr[ref][0]), "mse_mean": float(mse_t.mean()),
                             "mse_f20": float(mse_t[20]), "mse_f40": float(mse_t[40]), "mse_f79": float(mse_t[79])})
        print(c["id"], "done", flush=True)
    res = os.path.join(C.VM, "results")
    with open(os.path.join(res, "base_distance.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(res, "base_distance_per_frame.json"), "w") as f:
        json.dump(curves, f)
    # short summary
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["benchmark"], r["ref"], r["method"])].append((r["frame0_ncc"], r["mse_mean"]))
    for k in sorted(agg):
        a = np.array(agg[k])
        print("%-18s ref %-8s %-12s n=%2d frame0 NCC mean %.4f min %.4f | MSE mean %.5f" % (
            k[0], k[1], k[2], len(a), a[:, 0].mean(), a[:, 0].min(), a[:, 1].mean()))


if __name__ == "__main__":
    main()
