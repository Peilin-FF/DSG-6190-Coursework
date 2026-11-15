"""Check our port of the Physics-IQ metric code against the official functions (one-off check).

1) metric functions on random masks/frames: ours vs official (numpy only).
2) mask generator on one real case: our in-memory masks vs the official generate_mask (which writes the mask
   as an mp4v video that is read back and thresholded at 127). Input frames go through a lossless FFV1 file so
   both see identical pixels. Reports the pixel agreement and the effect on the four metrics.
Writes results/physiq_equivalence.txt.
"""
import os
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vm_common as C  # noqa: E402
import vm_physiq as P  # noqa: E402

OFFICIAL = os.path.join(C.VM, "ext", "physiq_zip", "physics-IQ-benchmark-main")
sys.path.insert(1, OFFICIAL)
from physiq import calculate_and_write_metrics_to_csv as OM  # noqa: E402
from physiq import binary_mask_generator as OG  # noqa: E402

lines = []


def say(s):
    print(s, flush=True)
    lines.append(s)


def part1():
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(20):
        p1, p2 = rng.uniform(0.0, 0.3, 2)
        m1 = rng.random((80, 54, 96)) < p1
        m2 = rng.random((80, 54, 96)) < p2
        if trial == 0:
            m1[:] = False
            m2[:] = False
        a = P.iou_per_frame(m1, m2)
        b = np.array(OM.spatiotemporal_iou_per_frame(m1, m2))
        s_ours = float(P.iou_per_frame(m1.any(0)[None], m2.any(0)[None])[0])
        s_off = OM.spatiotemporal_iou_per_frame(OM.spatial_binary_masks(m1)[None], OM.spatial_binary_masks(m2)[None])[0]
        w_ours = P.weighted_iou(m1.sum(0, dtype=np.uint16) / len(m1), m2.sum(0, dtype=np.uint16) / len(m2))
        w_off = OM.compute_weighted_spatial_iou(OM.weighted_spatial_mask(m1), OM.weighted_spatial_mask(m2))
        f1 = rng.random((80, 54, 96, 3))
        f2 = rng.random((80, 54, 96, 3))
        mse_ours = ((f1.astype(np.float32) - f2.astype(np.float32)) ** 2).mean(axis=(1, 2, 3))
        mse_off = np.array(OM.mse_per_frame(f1, f2))
        worst = max(worst, np.abs(a - b).max(), abs(s_ours - s_off), abs(w_ours - w_off), np.abs(mse_ours - mse_off).max())
    say("part 1: metric functions, max |ours - official| over 20 random trials = %.3g" % worst)


def part2(case_id="physics-iq-veried-p0019", method="CorrFlow"):
    c = [c for c in C.load_cases() if c["id"] == case_id][0]
    out = {}
    tmp = tempfile.mkdtemp(dir=os.path.join(C.VM, "logs"))
    for who in ("Ground truth", method):
        v = [x for x in c["videos"] if x["method"] == who][0]
        fr, how, _ = P.prepare(v)
        src = os.path.join(tmp, "0001_perspective-center_%s.avi" % who.replace(" ", ""))
        wr = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"FFV1"), 16, P.SIZE)
        for f in fr:
            wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        wr.release()
        back = C.read_all_frames(src)
        same = bool(np.array_equal(back, fr))
        ours = P.official_masks(back)
        dst = os.path.join(tmp, "0001_perspective-center_%s.mp4" % who.replace(" ", ""))
        OG.generate_mask(src, dst, is_real=False)
        produced = [f for f in os.listdir(tmp) if "video-masks" in f and who.replace(" ", "") in f]
        mv = C.read_all_frames(os.path.join(tmp, produced[0]))
        off = mv[:, :, :, 0] > 127
        n = min(len(off), len(ours))
        agree = float((off[:n] == ours[:n]).mean())
        say("part 2: %s %s: lossless input %s, frames ours %d official %d, pixel agreement %.5f, "
            "motion share ours %.4f official %.4f" % (case_id, who, same, len(ours), len(off), agree,
                                                     ours.mean(), off[:n].mean()))
        out[who] = (back, ours, off[:n])
    (g, gm_o, gm_f), (x, m_o, m_f) = out["Ground truth"], out[method]
    for tag, gm, m in (("our masks", gm_o, m_o), ("official mask videos", gm_f, m_f)):
        st = P.iou_per_frame(gm, m).mean()
        sp = P.iou_per_frame(gm.any(0)[None], m.any(0)[None])[0]
        w = P.weighted_iou(gm.sum(0, dtype=np.uint16) / len(gm), m.sum(0, dtype=np.uint16) / len(m))
        say("part 2: %s vs GT with %s: spatial %.4f spatiotemporal %.4f weighted %.4f" % (method, tag, sp, st, w))


if __name__ == "__main__":
    part1()
    part2()
    part2("physics-iq-veried-p0115", "Wan2.2")
    with open(os.path.join(C.VM, "results", "physiq_equivalence.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
