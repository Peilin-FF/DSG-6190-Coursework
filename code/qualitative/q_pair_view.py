"""Viewing aid for the sampling pair: rows = arm x frame index, columns = steps. Writes a small JPEG.
Usage: python q_pair_view.py OUT.jpg 'steps' 'frames' [cellw]"""
import sys, json, os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

P2 = '/mnt/data_3T/peilin/ce6190_p2'
SP = f'{P2}/media/sampling-pair'
FONT = ImageFont.truetype(os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf', 'DejaVuSans-Bold.ttf'), 13)
out = sys.argv[1]
steps = [int(s) for s in sys.argv[2].split(',')]
fidx = [int(s) for s in sys.argv[3].split(',')]
cellw = int(sys.argv[4]) if len(sys.argv) > 4 else 190
man = json.load(open(f'{SP}/manifest.json'))
sig = {s['step']: s['sigma'] for s in man['steps']}


def read(path, want):
    cap = cv2.VideoCapture(path)
    got, i = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    return got

rows = []
for arm in ('without_refiner', 'with_refiner'):
    per_step = {s: read(f'{SP}/{arm}/step_{s:03d}.mp4', set(fidx)) for s in steps}
    for f in fidx:
        rows.append((f'{arm[:7]} f{f}', [per_step[s][f] for s in steps]))
ch = int(round(cellw * 480 / 832))
lw, gap, th = 120, 3, 34
W = lw + len(steps) * (cellw + gap)
H = th + len(rows) * (ch + gap)
sheet = Image.new('RGB', (W, H), (255, 255, 255))
d = ImageDraw.Draw(sheet)
for j, s in enumerate(steps):
    d.text((lw + j * (cellw + gap) + 3, 2), f'step {s}', fill=(0, 0, 0), font=FONT)
    d.text((lw + j * (cellw + gap) + 3, 17), f'sigma {sig[s]:.3f}', fill=(0, 0, 0), font=FONT)
for r, (lab, frs) in enumerate(rows):
    y = th + r * (ch + gap)
    d.text((3, y + ch // 2 - 7), lab, fill=(0, 0, 0), font=FONT)
    for j, fr in enumerate(frs):
        sheet.paste(Image.fromarray(cv2.resize(fr, (cellw, ch), interpolation=cv2.INTER_AREA)), (lw + j * (cellw + gap), y))
sheet.save(out, quality=78)
print('wrote', out, sheet.size)
