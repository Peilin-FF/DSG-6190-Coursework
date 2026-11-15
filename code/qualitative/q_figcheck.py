"""Check the figure PNGs: pixel size, size at 600 dpi, opaque, ink in the first and last pixel columns
and rows (must be none), and the CorrFlow outline (#649CD8) and label (#527FD4) colours.
Usage: python q_figcheck.py DIR"""
import glob, json, os, sys
import numpy as np
from PIL import Image
d = sys.argv[1]
idx = json.load(open(os.path.join(d, 'figures_index.json')))
for f in sorted(glob.glob(os.path.join(d, 'fig_*.png'))):
    if f.endswith('_old.png'):
        continue
    im = Image.open(f)
    a = np.asarray(im.convert('RGBA'))
    rgb = a[..., :3].astype(int)
    ink = (rgb.min(axis=2) < 200)
    name = os.path.basename(f)[:-4]
    def count(hexc):
        c = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)])
        return int((np.abs(rgb - c).sum(axis=2) == 0).sum())
    e = idx.get(name, {})
    print(f"{name:44s} px={im.size} in@600={tuple(round(v / 600, 3) for v in im.size)} index_in=({e.get('width_in')}, {e.get('height_in')}) "
          f"dpi={tuple(round(v) for v in im.info.get('dpi', (0, 0)))} alpha_min={a[..., 3].min()} "
          f"ink: col0={int(ink[:, 0].sum())} col1={int(ink[:, 1].sum())} lastcol={int(ink[:, -1].sum())} "
          f"row0={int(ink[0].sum())} lastrow={int(ink[-1].sum())} #649CD8={count('#649CD8')} #527FD4={count('#527FD4')}")
