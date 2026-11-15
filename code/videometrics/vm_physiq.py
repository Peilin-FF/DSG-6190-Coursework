"""Agreement with the ground truth for the 10 image-to-video (Physics-IQ) cases, Physics-IQ style.

The steps follow the official code (google-deepmind/physics-IQ-benchmark, physiq/fps_changer.py,
physiq/binary_mask_generator.py, physiq/calculate_and_write_metrics_to_csv.py), ported here:
  1. frame rate: a video that is not 16 fps (the 30 fps ground truth, the 9.8 fps VLiPP output) is changed to
     16 fps with the benchmark's own resampler (first 5 s, linear blend of the two nearest frames);
     16 fps outputs are used as they are. The first 80 frames (5 s) are kept.
  2. size: every frame is resized to 960x540 (cv2.resize, bilinear, as in the official loader; the official
     loader also evaluates at width/4 x height/4 of the 3840x2160 real videos, i.e. 960x540).
  3. motion masks (official generator): grey, Gaussian blur 5x5, running-average background
     (accumulateWeighted, alpha 0.3, started from frame 0), |frame - background| > 10, morphological open
     then close (5x5); the mask of frame 0 is empty.
  4. metrics against the single released ground-truth take (official definitions):
     spatial IoU (masks OR-ed over time), spatiotemporal IoU (per-frame IoU, 1 when both masks are empty,
     mean over frames), weighted spatial IoU (per-pixel fraction of frames with motion; sum min / sum max),
     MSE (frames in [0, 1], mean over pixels and frames).
Differences from the official pipeline: one view and one real take only (no physical-variance
normalisation, raw values), the ground truth is the released 1280x720 30 fps clip (not the 4K original),
masks are computed at 960x540 for every video and kept in memory (no lossy mask-video round trip).
Writes results/physiq_per_case.csv, results/physiq_per_frame.json, results/physiq_masks.npz.
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vm_common as C  # noqa: E402

FPS = 16
N = 80
SIZE = (960, 540)


def official_fps_change(frames, fps_orig, fps_new=FPS):
    """Port of physiq/fps_changer.py: first 5 s, linear blend between neighbouring frames, uint8 cast."""
    duration = len(frames) / fps_orig
    sub = min(5, duration)
    fr = frames[:int(sub * fps_orig)]
    n_o = len(fr)
    n_new = int(sub * fps_new)
    out = []
    for j in range(n_new):
        a = j * (n_o - 1) / (n_new - 1)
        i = int(a)
        a -= i
        f1 = fr[i].astype(np.float32)
        f2 = fr[min(i + 1, n_o - 1)].astype(np.float32)
        out.append(((1 - a) * f1 + a * f2).astype(np.uint8))
    return np.stack(out)


def official_masks(frames, thr=10):
    """Port of physiq/binary_mask_generator.py (masks kept as booleans)."""
    T, H, W = frames.shape[:3]
    masks = np.zeros((T, H, W), dtype=bool)
    g = cv2.GaussianBlur(cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY), (5, 5), 0)
    avg = g.astype("float")
    kernel = np.ones((5, 5), np.uint8)
    for t in range(1, T):
        g = cv2.GaussianBlur(cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY), (5, 5), 0)
        cv2.accumulateWeighted(g, avg, 0.3)
        avg_g = cv2.convertScaleAbs(avg)
        diff = cv2.absdiff(g, avg_g)
        _, b = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
        b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
        masks[t] = b > 127
    return masks


def iou_per_frame(m1, m2):
    inter = np.logical_and(m1, m2).sum(axis=(1, 2))
    union = np.logical_or(m1, m2).sum(axis=(1, 2))
    return np.where(union == 0, 1.0, inter / np.maximum(union, 1))


def weighted_iou(w1, w2):
    inter = np.minimum(w1, w2)
    union = np.maximum(w1, w2)
    valid = union > 0
    if valid.sum() == 0:
        return 1.0
    return float(inter[valid].sum() / union[valid].sum())


def prepare(video):
    allf = C.read_all_frames(video["path"])
    fps = float(video["fps"])
    if abs(fps - FPS) > 1e-6:
        fr = official_fps_change(allf, fps)
        how = "official fps change %.4g->16" % fps
    else:
        fr = allf
        how = "native 16 fps"
    if len(fr) < N:
        raise ValueError("%s: only %d frames" % (video["path"], len(fr)))
    fr = fr[:N]
    fr = np.stack([cv2.resize(f, SIZE) for f in fr])
    return fr, how, len(allf)


def main():
    cases = [c for c in C.load_cases() if c["task"] == "i2v"]
    rows, per_frame, masks_out = [], {}, {}
    for c in cases:
        gt = [v for v in c["videos"] if v["method"] == "Ground truth"][0]
        gfr, ghow, gn = prepare(gt)
        gm = official_masks(gfr)
        g_sp = gm.any(0)
        g_w = gm.sum(0, dtype=np.uint16) / len(gm)
        gf = gfr.astype(np.float32) / 255.0
        masks_out[c["id"] + "|Ground truth"] = np.packbits(gm, axis=-1)
        print(c["id"], "GT", ghow, "decoded", gn, "motion px share %.4f" % gm.mean(), flush=True)
        per_frame[c["id"] + "|Ground truth"] = {"mask_share": gm.mean(axis=(1, 2)).tolist()}
        for v in c["videos"]:
            if v["method"] == "Ground truth":
                continue
            fr, how, n = prepare(v)
            m = official_masks(fr)
            st = iou_per_frame(gm, m)
            sp = float(iou_per_frame(g_sp[None], m.any(0)[None])[0])
            wsp = weighted_iou(g_w, m.sum(0, dtype=np.uint16) / len(m))
            mse_t = ((gf - fr.astype(np.float32) / 255.0) ** 2).mean(axis=(1, 2, 3))
            rows.append({"case_id": c["id"], "method": v["method"], "folder": v["folder"], "resample": how,
                         "n_decoded": n, "spatial_iou": sp, "spatiotemporal_iou": float(st.mean()),
                         "weighted_spatial_iou": wsp, "mse": float(mse_t.mean()),
                         "motion_share_gen": float(m.mean()), "motion_share_gt": float(gm.mean()),
                         "spatial_area_gen": float(m.any(0).mean()), "spatial_area_gt": float(g_sp.mean())})
            per_frame[c["id"] + "|" + v["method"]] = {"st_iou": st.tolist(), "mse": mse_t.tolist(),
                                                      "mask_share": m.mean(axis=(1, 2)).tolist()}
            masks_out[c["id"] + "|" + v["method"]] = np.packbits(m, axis=-1)
            r = rows[-1]
            print("  %-10s %-28s sIoU %.3f stIoU %.3f wsIoU %.3f MSE %.4f motion %.4f" % (
                v["method"], how, r["spatial_iou"], r["spatiotemporal_iou"], r["weighted_spatial_iou"],
                r["mse"], r["motion_share_gen"]), flush=True)
    res = os.path.join(C.VM, "results")
    with open(os.path.join(res, "physiq_per_case.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(res, "physiq_per_frame.json"), "w") as f:
        json.dump(per_frame, f)
    np.savez_compressed(os.path.join(res, "physiq_masks.npz"), **{k.replace("|", "__"): v for k, v in masks_out.items()})
    print("wrote physiq_per_case.csv rows", len(rows))


if __name__ == "__main__":
    main()
