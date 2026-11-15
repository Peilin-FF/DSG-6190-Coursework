"""Shared helpers for the videometrics track (our light measurements on the released CorrFlow videos).

Case list, method names, decoding, the common 16 fps grid and the common frame size.
All paths are absolute; nothing here writes files.
"""
from __future__ import annotations

import csv
import json
import os
from fractions import Fraction

import cv2
import numpy as np

ROOT = "/mnt/data_3T/peilin/ce6190_p2"
MEDIA = os.path.join(ROOT, "media", "corrflow-project")
INV_CSV = os.path.join(ROOT, "inventory", "videos_inventory.csv")
CASES_JSON = os.path.join(ROOT, "inventory", "cases.json")
VM = os.path.join(ROOT, "videometrics")

GRID_FPS = 16           # common frame rate
GRID_N = 80             # first 5 s at 16 fps
T2V_SIZE = (704, 480)   # (W, H) common T2V size: centre crop, no rescaling (704 is the narrowest T2V width)
I2V_SIZE = (960, 540)   # (W, H) common I2V size (Physics-IQ style evaluation size)

# Display names and benchmark order used in every table and figure.
BENCH_ORDER = ["videophy2", "physics-iq-veried", "finephyeval", "vbench"]
BENCH_NAME = {"videophy2": "VideoPhy2", "physics-iq-veried": "Physics-IQ", "finephyeval": "FinePhyEval",
              "vbench": "VBench"}
METHOD_NAME = {"Ours": "CorrFlow", "Ground truth": "Ground truth", "Wan2.1": "Wan2.1", "Wan2.2": "Wan2.2",
               "PhysRAG": "PhysRAG", "VideoREPA": "VideoREPA", "DreamWorld": "DreamWorld", "VPT": "VPT",
               "VLiPP": "VLiPP", "WMReward": "WMReward"}
METHOD_ORDER = ["CorrFlow", "Wan2.1", "Wan2.2", "PhysRAG", "VideoREPA", "DreamWorld", "VPT", "VLiPP", "WMReward",
                "Ground truth"]

# Short names of the 10 Physics-IQ cases for figure labels, written from each case's prompt (what moves).
# The project-page titles stay in cases.json; one of them ("Mirror ball fall", p0115) does not describe the motion:
# the prompt says the tennis ball on a cord in front of a mirror "is slowly lowered".
PHYSIQ_CASE_LABEL = {
    "physics-iq-veried-p0019": "ball rolls onto a ramp",
    "physics-iq-veried-p0027": "ball rolls over glass",
    "physics-iq-veried-p0040": "knife cuts a tangerine",
    "physics-iq-veried-p0046": "domino dropped in a mug",
    "physics-iq-veried-p0084": "rotating statue and shadow",
    "physics-iq-veried-p0110": "match lowered into water",
    "physics-iq-veried-p0115": "ball lowered on a cord",
    "physics-iq-veried-p0133": "paper dropped into water",
    "physics-iq-veried-p0140": "burning paper with smoke",
    "physics-iq-veried-p0155": "ball rolls to a kettlebell",
}


def case_short(case_id):
    """'physics-iq-veried-p0115' -> 'p0115' (Physics-IQ ids only)."""
    return case_id.replace("physics-iq-veried-", "")


PHYSIQ_PREFIX = ("Static locked-off single-shot with fixed frame throughout filmed with constant framerate in "
                 "real-time. The scene shows a realistic scientific demonstration. The scene only contains the "
                 "described setup and actions.")


def load_cases():
    """Return a list of cases; each case has id, benchmark, task, prompt (short), stem and videos.

    videos: list of dicts with method (display name), path (absolute), fps (Fraction), nb_frames, width, height.
    """
    inv = {}
    with open(INV_CSV) as f:
        for r in csv.DictReader(f):
            inv[(r["case_id"], r["label"])] = r
    cases = json.load(open(CASES_JSON))
    out = []
    for c in cases:
        prompt = c["prompt"].strip()
        if c["benchmark"] == "physics-iq-veried":
            assert prompt.startswith(PHYSIQ_PREFIX), c["id"]
            prompt = prompt[len(PHYSIQ_PREFIX):].strip()
        vids = []
        for m in c["methods"]:
            r = inv[(c["id"], m["label"])]
            vids.append({
                "method": METHOD_NAME[m["label"]],
                "label": m["label"],
                "folder": m["folder"],
                "path": os.path.join(MEDIA, m["src"]),
                "fps": Fraction(r["r_fps"]),
                "nb_frames": int(r["nb_frames"]),
                "width": int(r["width"]),
                "height": int(r["height"]),
            })
        out.append({"id": c["id"], "benchmark": c["benchmark"], "task": c["task"], "prompt": prompt,
                    "stem": c["stem"], "videos": vids})
    return out


def read_all_frames(path):
    """Decode every frame as RGB uint8 (T, H, W, 3)."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError("no frames decoded: " + path)
    return np.stack(frames)


def grid_indices(n_src, fps, grid_fps=GRID_FPS, grid_n=GRID_N):
    """Nearest source frame for each grid time t_k = k / grid_fps (like ffmpeg's fps filter, round=near)."""
    fps = float(fps)
    t = np.arange(grid_n) / grid_fps
    j = np.floor(t * fps + 0.5 + 1e-9).astype(int)
    if j.max() > n_src - 1:
        raise ValueError("video shorter than the grid: n_src=%d, need index %d" % (n_src, j.max()))
    return j


def centre_crop_to_aspect(frames, out_w, out_h):
    """Centre crop (no rescale) to out_w x out_h; frames must be at least that large."""
    h, w = frames.shape[1:3]
    if w < out_w or h < out_h:
        raise ValueError("frame %dx%d smaller than crop %dx%d" % (w, h, out_w, out_h))
    x0 = (w - out_w) // 2
    y0 = (h - out_h) // 2
    return frames[:, y0:y0 + out_h, x0:x0 + out_w]


def resize_frames(frames, out_w, out_h):
    """Resize every frame to out_w x out_h (area filter when shrinking, bilinear when enlarging)."""
    h, w = frames.shape[1:3]
    if (w, h) == (out_w, out_h):
        return frames
    interp = cv2.INTER_AREA if (out_w < w and out_h < h) else cv2.INTER_LINEAR
    return np.stack([cv2.resize(f, (out_w, out_h), interpolation=interp) for f in frames])


def to_common_size(frames, task):
    """T2V: centre crop to 704x480 (all T2V videos are 480 px high). I2V: resize the full frame to 960x540.

    I2V note: the 1248x704 and 1280x720 frames are 16:9 already; the 720x480 VLiPP frames are a full-frame
    resize of the 16:9 input (checked with vm_geometry.py), so a plain resize restores the geometry.
    """
    if task == "t2v":
        return centre_crop_to_aspect(frames, *T2V_SIZE)
    return resize_frames(frames, *I2V_SIZE)


def load_grid(video, task):
    """Decode a video, sample the 16 fps grid (80 frames) and bring it to the common size.

    Returns dict: frames (80, H, W, 3) uint8, src_idx (80,), n_decoded, fps (float).
    """
    allf = read_all_frames(video["path"])
    n = len(allf)
    idx = grid_indices(n, video["fps"])
    g = to_common_size(allf[idx], task)
    return {"frames": np.ascontiguousarray(g), "src_idx": idx, "n_decoded": n, "fps": float(video["fps"])}


def distinct_positions(src_idx):
    """Grid positions kept after dropping consecutive repeats of the same source frame."""
    keep = [0]
    for k in range(1, len(src_idx)):
        if src_idx[k] != src_idx[keep[-1]]:
            keep.append(k)
    return np.array(keep)
