"""Viewing aid: ROI crops of selected methods at many times, written as a small JPEG.
Usage: python q_zoom.py CASE_ID 'x0,y0,x1,y1' 'times' 'label1,label2,...' OUT.jpg [cellw]
  ROI in relative coordinates of each frame; times in seconds; labels as in cases.json
  ('Ours', 'Ground truth', 'Wan2.2', ...)."""
import sys, json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib, os

P2 = '/mnt/data_3T/peilin/ce6190_p2'
MEDIA = f'{P2}/media/corrflow-project'
FONT = ImageFont.truetype(os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf', 'DejaVuSans-Bold.ttf'), 14)

cid, roi, times, labels, out = sys.argv[1:6]
cellw = int(sys.argv[6]) if len(sys.argv) > 6 else 220
x0, y0, x1, y1 = [float(v) for v in roi.split(',')]
times = [float(t) for t in times.split(',')]
labels = labels.split(',')
case = [c for c in json.load(open(f'{P2}/inventory/cases.json')) if c['id'] == cid][0]
rows = []
for lab in labels:
    m = [m for m in case['methods'] if m['label'] == lab][0]
    n = int(m['nb_frames'])
    p, q = (m['r_fps'].split('/') + ['1'])[:2]
    fps = float(p) / float(q)
    want = [min(n - 1, int(round(t * fps))) for t in times]
    cap = cv2.VideoCapture(f"{MEDIA}/{m['src']}")
    got, i = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    crops = []
    for k in want:
        fr = got[k]
        H, W = fr.shape[:2]
        c = fr[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
        ch = int(round(cellw * c.shape[0] / c.shape[1]))
        crops.append(cv2.resize(c, (cellw, ch), interpolation=cv2.INTER_AREA))
    rows.append((lab, want, crops))
ch = max(c.shape[0] for _, _, cs in rows for c in cs)
lw, gap, th = 110, 3, 20
Wt = lw + len(times) * (cellw + gap)
Ht = th + len(rows) * (ch + gap)
sheet = Image.new('RGB', (Wt, Ht), (255, 255, 255))
d = ImageDraw.Draw(sheet)
for j, t in enumerate(times):
    d.text((lw + j * (cellw + gap) + 4, 2), f'{t:.2f}s', fill=(0, 0, 0), font=FONT)
for r, (lab, want, crops) in enumerate(rows):
    y = th + r * (ch + gap)
    d.text((4, y + ch // 2 - 7), lab[:12], fill=(0, 0, 0), font=FONT)
    for j, c in enumerate(crops):
        sheet.paste(Image.fromarray(c), (lw + j * (cellw + gap), y))
sheet.save(out, quality=80)
print('wrote', out, sheet.size)
