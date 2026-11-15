"""Viewing aid: the exact strip frames (frames/<case>/<folder>__k<j>.png, the five figure columns)
of chosen rows, optionally cropped, at a larger size, written as one small JPEG.
Usage: python q_cols.py CASE_ID 'x0,y0,x1,y1' 'label1,label2,...' 'k0,k1,...' OUT.jpg [cellw]"""
import json, os, sys
import matplotlib
from PIL import Image, ImageDraw, ImageFont

Q = '/mnt/data_3T/peilin/ce6190_p2/qualitative'
FONT = ImageFont.truetype(os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf', 'DejaVuSans-Bold.ttf'), 14)
cid, roi, labels, ks, out = sys.argv[1:6]
cellw = int(sys.argv[6]) if len(sys.argv) > 6 else 300
x0, y0, x1, y1 = [float(v) for v in roi.split(',')]
ks = [int(k) for k in ks.split(',')]
meths = {m['label']: m for m in json.load(open(f'{Q}/frame_index.json'))[cid]['methods']}
rows = []
for lab in labels.split(','):
    m = meths[lab]
    cells = []
    for k in ks:
        im = Image.open(f"{Q}/frames/{cid}/{m['folder']}__k{k}.png").convert('RGB')
        W, H = im.size
        im = im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
        cells.append((m['times_s'][k], im.resize((cellw, round(cellw * im.height / im.width)), Image.LANCZOS)))
    rows.append((lab, cells))
ch = max(c.height for _, cs in rows for _, c in cs)
lw, gap, th = 110, 3, 20
sheet = Image.new('RGB', (lw + len(ks) * (cellw + gap), th + len(rows) * (ch + gap)), 'white')
d = ImageDraw.Draw(sheet)
for r, (lab, cells) in enumerate(rows):
    y = th + r * (ch + gap)
    d.text((4, y + ch // 2 - 7), lab[:12], fill=(0, 0, 0), font=FONT)
    for j, (t, c) in enumerate(cells):
        x = lw + j * (cellw + gap)
        sheet.paste(c, (x, y))
        if r == 0:
            d.text((x + 4, 2), f'k{ks[j]}', fill=(0, 0, 0), font=FONT)
        d.text((x + 4, y + 2), f'{t:.2f}s', fill=(255, 255, 0), font=FONT)
sheet.save(out, quality=82)
print('wrote', out, sheet.size)
