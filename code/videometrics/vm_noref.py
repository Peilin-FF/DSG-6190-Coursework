"""No-reference scores for every released video (our light measurement; no generation).

Protocol (one rule for every video):
  * 16 fps grid over the first 5 s (80 frames, nearest source frame) at the common size
    (T2V: centre crop to 704x480, no rescaling; I2V: full-frame resize to 960x540).
  * Per-frame scores (CLIP) use all 80 grid frames.
  * Temporal scores (flicker, optical flow, subject consistency) use the grid with consecutive repeats of
    the same source frame removed. Only the 9.8 fps videos (PhysRAG on VideoPhy2/VBench, VLiPP) repeat
    frames; their temporal steps are 1/9.8 s instead of 1/16 s.
Scores:
  flicker      = 1 - mean |I_t - I_{t-1}| / 255   (RGB, all pixels, consecutive distinct grid frames; VBench form)
  flow_px_s    = mean RAFT-large flow magnitude between consecutive distinct grid frames, divided by the
                 source time gap (pixels per second at the common size)
  subject      = VBench subject consistency with DINO ViT-B/16 CLS features:
                 mean_t 0.5 * (max(0, cos(f_t, f_{t-1})) + max(0, cos(f_t, f_0)))
  clip         = mean over the 80 grid frames of cos(CLIP ViT-L/14 image, CLIP text of the short prompt)
Writes results/noref_per_video.csv and results/noref_per_frame.json.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vm_common as C  # noqa: E402

HF = os.path.join(C.VM, "weights", "hf")


def load_models(dev):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    from transformers import CLIPModel, CLIPTokenizer, ViTModel
    w = Raft_Large_Weights.C_T_SKHT_V2
    raft = raft_large(weights=w, progress=False).to(dev).eval()
    dino = ViTModel.from_pretrained(os.path.join(HF, "dino-vitb16"), add_pooling_layer=False).to(dev).eval()
    clip = CLIPModel.from_pretrained(os.path.join(HF, "clip-vit-large-patch14")).to(dev).eval()
    tok = CLIPTokenizer.from_pretrained(os.path.join(HF, "clip-vit-large-patch14"))
    return raft, dino, clip, tok


@torch.no_grad()
def raft_flow_mag(raft, frames, pairs, dev, bs):
    """Mean flow magnitude (px) for each (a, b) pair of grid positions."""
    h, w = frames.shape[1:3]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    out = []
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        a = torch.from_numpy(frames[[p[0] for p in chunk]]).to(dev).permute(0, 3, 1, 2).float()
        b = torch.from_numpy(frames[[p[1] for p in chunk]]).to(dev).permute(0, 3, 1, 2).float()
        a = a / 127.5 - 1.0  # the RAFT weights expect [-1, 1]
        b = b / 127.5 - 1.0
        if ph or pw:
            a = F.pad(a, (0, pw, 0, ph), mode="replicate")
            b = F.pad(b, (0, pw, 0, ph), mode="replicate")
        flows = raft(a, b)  # list over refinement iterations; last is the final estimate
        fl = flows[-1][:, :, :h, :w]
        mag = torch.sqrt((fl ** 2).sum(1))
        out.extend(mag.mean(dim=(1, 2)).cpu().tolist())
    return out


@torch.no_grad()
def dino_feats(dino, frames, dev, bs=40):
    """VBench preprocessing: shorter side to 224 (aspect kept, no crop), /255, ImageNet normalisation."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    h, w = frames.shape[1:3]
    if h <= w:
        nh, nw = 224, int(round(w * 224 / h))
    else:
        nw, nh = 224, int(round(h * 224 / w))
    feats = []
    for i in range(0, len(frames), bs):
        x = torch.from_numpy(frames[i:i + bs]).to(dev).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=False) / 255.0
        x = (x - mean) / std
        o = dino(pixel_values=x, interpolate_pos_encoding=True)
        feats.append(F.normalize(o.last_hidden_state[:, 0], dim=-1))
    return torch.cat(feats)


@torch.no_grad()
def clip_image_feats(clip, frames, dev, bs=40):
    """OpenAI CLIP preprocessing: shorter side to 224 (bicubic), centre crop 224, CLIP normalisation."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
    h, w = frames.shape[1:3]
    if h <= w:
        nh, nw = 224, int(round(w * 224 / h))
    else:
        nw, nh = 224, int(round(h * 224 / w))
    feats = []
    for i in range(0, len(frames), bs):
        x = torch.from_numpy(frames[i:i + bs]).to(dev).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False, antialias=True)
        y0 = (nh - 224) // 2
        x0 = (nw - 224) // 2
        x = x[:, :, y0:y0 + 224, x0:x0 + 224].clamp(0, 255) / 255.0
        x = (x - mean) / std
        vo = clip.vision_model(pixel_values=x)
        emb = clip.visual_projection(vo.pooler_output)
        feats.append(F.normalize(emb, dim=-1))
    return torch.cat(feats)


@torch.no_grad()
def clip_text_feat(clip, tok, text, dev):
    full = tok(text, truncation=False)["input_ids"]
    enc = tok(text, truncation=True, max_length=77, padding="max_length", return_tensors="pt")
    to = clip.text_model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev))
    emb = clip.text_projection(to.pooler_output)
    return F.normalize(emb, dim=-1)[0], len(full), len(full) > 77


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N videos (smoke test)")
    ap.add_argument("--cases", default="", help="comma-separated case ids (smoke test)")
    ap.add_argument("--out_tag", default="")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    raft, dino, clip, tok = load_models(dev)
    print("models loaded in %.1f s" % (time.time() - t0), flush=True)
    cases = C.load_cases()
    if args.cases:
        want = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in want]
    jobs = [(c, v) for c in cases for v in c["videos"]]
    if args.limit:
        jobs = jobs[:args.limit]
    rows, per_frame = [], {}
    text_cache = {}
    for n, (c, v) in enumerate(jobs):
        t1 = time.time()
        g = C.load_grid(v, c["task"])
        fr = g["frames"]
        keep = C.distinct_positions(g["src_idx"])
        pairs = [(int(keep[i - 1]), int(keep[i])) for i in range(1, len(keep))]
        gaps = np.array([(g["src_idx"][b] - g["src_idx"][a]) / g["fps"] for a, b in pairs])
        # flicker (VBench form) on consecutive distinct frames
        mad = np.array([np.abs(fr[b].astype(np.float32) - fr[a].astype(np.float32)).mean() for a, b in pairs])
        flicker = 1.0 - mad.mean() / 255.0
        # optical flow
        bs = 16 if c["task"] == "t2v" else 8
        mag = np.array(raft_flow_mag(raft, fr, pairs, dev, bs))
        flow_px_s = float(np.mean(mag / gaps))
        flow_px_step = float(np.mean(mag))
        # subject consistency (VBench) on the distinct sequence
        df = dino_feats(dino, fr[keep], dev)
        cos_prev = (df[1:] * df[:-1]).sum(-1).clamp(min=0).cpu().numpy()
        cos_first = (df[1:] * df[0:1]).sum(-1).clamp(min=0).cpu().numpy()
        subject = float(np.mean(0.5 * (cos_prev + cos_first)))
        # CLIP text alignment over all 80 grid frames
        if c["prompt"] not in text_cache:
            text_cache[c["prompt"]] = clip_text_feat(clip, tok, c["prompt"], dev)
        tf, ntok, trunc = text_cache[c["prompt"]]
        cf = clip_image_feats(clip, fr, dev)
        clip_per_frame = (cf @ tf).cpu().numpy()
        row = {
            "case_id": c["id"], "benchmark": c["benchmark"], "task": c["task"], "method": v["method"],
            "folder": v["folder"], "src_fps": "%.4g" % g["fps"], "src_w": v["width"], "src_h": v["height"],
            "n_decoded": g["n_decoded"], "n_grid": len(fr), "n_distinct": len(keep),
            "mean_step_s": float(gaps.mean()),
            "flicker": float(flicker), "mad": float(mad.mean()),
            "flow_px_s": flow_px_s, "flow_px_step": flow_px_step,
            "subject": subject, "subject_first": float(cos_first.mean()), "subject_prev": float(cos_prev.mean()),
            "clip": float(clip_per_frame.mean()), "clip_tokens": ntok, "clip_truncated": int(trunc),
        }
        rows.append(row)
        per_frame[c["id"] + "|" + v["method"]] = {
            "src_idx": g["src_idx"].tolist(), "keep": keep.tolist(), "mad": mad.tolist(),
            "flow_px": mag.tolist(), "gaps_s": gaps.tolist(), "dino_prev": cos_prev.tolist(),
            "dino_first": cos_first.tolist(), "clip": clip_per_frame.tolist(),
        }
        print("[%d/%d] %s %s fps=%s n=%d distinct=%d flicker=%.4f flow=%.1f px/s subject=%.4f clip=%.4f (%.1f s)" % (
            n + 1, len(jobs), c["id"], v["method"], row["src_fps"], g["n_decoded"], len(keep), flicker,
            flow_px_s, subject, row["clip"], time.time() - t1), flush=True)
    os.makedirs(os.path.join(C.VM, "results"), exist_ok=True)
    out_csv = os.path.join(C.VM, "results", "noref_per_video%s.csv" % args.out_tag)
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(C.VM, "results", "noref_per_frame%s.json" % args.out_tag), "w") as f:
        json.dump(per_frame, f)
    print("wrote", out_csv, "rows", len(rows), "total %.1f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
