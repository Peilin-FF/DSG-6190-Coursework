"""Make small JPEG copies of PNG images for viewing over a slow link.
Usage: python q_tojpg.py SRC_DIR DST_DIR [max_width] [quality] [glob]"""
import sys, os, glob
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
mw = int(sys.argv[3]) if len(sys.argv) > 3 else 1300
q = int(sys.argv[4]) if len(sys.argv) > 4 else 80
pat = sys.argv[5] if len(sys.argv) > 5 else '*.png'
os.makedirs(dst, exist_ok=True)
tot = 0
for f in sorted(glob.glob(os.path.join(src, pat))):
    if f.endswith('_old.png'):   # superseded versions are kept for the record, not previewed
        continue
    im = Image.open(f).convert('RGB')
    if im.width > mw:
        im = im.resize((mw, round(im.height * mw / im.width)), Image.LANCZOS)
    out = os.path.join(dst, os.path.splitext(os.path.basename(f))[0] + '.jpg')
    im.save(out, quality=q)
    tot += os.path.getsize(out)
print('wrote', dst, 'total bytes', tot)
