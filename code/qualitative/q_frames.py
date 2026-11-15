"""Viewing aid: chosen frame indices of one method's clip, in a row, as a small JPEG.
Usage: python q_frames.py CASE_ID LABEL 'f1,f2,...' OUT.jpg [cellw]"""
import json, os, sys
import cv2
import matplotlib
from PIL import Image, ImageDraw, ImageFont

P2 = '/mnt/data_3T/peilin/ce6190_p2'
FONT = ImageFont.truetype(os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf', 'DejaVuSans-Bold.ttf'), 13)
cid, lab, fids, out = sys.argv[1:5]
cellw = int(sys.argv[5]) if len(sys.argv) > 5 else 200
fids = [int(f) for f in fids.split(',')]
m = [m for m in [c for c in json.load(open(f'{P2}/inventory/cases.json')) if c['id'] == cid][0]['methods'] if m['label'] == lab][0]
p, q = (m['r_fps'].split('/') + ['1'])[:2]
fps = float(p) / float(q)
cap = cv2.VideoCapture(f"{P2}/media/corrflow-project/{m['src']}")
got, i = {}, 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if i in fids:
        got[i] = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    i += 1
cells = [got[f].resize((cellw, round(cellw * got[f].height / got[f].width)), Image.LANCZOS) for f in fids]
sheet = Image.new('RGB', (len(cells) * (cellw + 3), cells[0].height + 18), 'white')
d = ImageDraw.Draw(sheet)
for j, (f, c) in enumerate(zip(fids, cells)):
    sheet.paste(c, (j * (cellw + 3), 18))
    d.text((j * (cellw + 3) + 3, 2), f'f{f} {f / fps:.3f}s', fill=(0, 0, 0), font=FONT)
sheet.save(out, quality=82)
print('wrote', out, sheet.size)
