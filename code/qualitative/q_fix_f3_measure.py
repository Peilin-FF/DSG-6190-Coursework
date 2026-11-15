"""Measure title widths (in) at print size, and list the fonts the other tracks use."""
import glob, json, re, sys
sys.path.insert(0, '/mnt/data_3T/peilin/ce6190_p2/qualitative/scripts')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_style as ps
ps.apply()
plt.rcParams.update({'font.serif': ['DejaVu Serif'], 'font.weight': 'bold'})
cands = [
    '(a) VBench: a sports ball on the right of a baseball bat',
    '(b) VBench: a chair and a couch',
    '(c) Physics-IQ: paper smoke',
    '(d) Physics-IQ: domino in juice',
    '(a) VBench: a panda standing on a surfboard',
    '(b) VBench: a person is bending metal',
    'VideoPhy2: hair in a stream of air',
    '(a) VBench: ball on the right of a baseball bat',
]
f = plt.figure(figsize=(3, 3), dpi=600)
r = f.canvas.get_renderer()
for size in (7.5, 7.0):
    for s in cands:
        t = f.text(0, 0, s, fontsize=size, fontweight='bold')
        print(f'{size} pt  {t.get_window_extent(r).width / f.dpi:.3f} in  {s}')
print('panel width main: 5*0.583+4*0.025 =', 5 * 0.583 + 4 * 0.025, ' + G_CASE 0.08')
for p in glob.glob('/mnt/data_3T/peilin/ce6190_p2/*/figures/*index*.json') + glob.glob('/mnt/data_3T/peilin/ce6190_p2/*/*index*.json'):
    try:
        txt = open(p).read()
    except Exception as e:
        continue
    fonts = sorted(set(re.findall(r'"font[a-z_]*"\s*:\s*"([^"]+)"', txt)))
    print(p, fonts)
for p in glob.glob('/mnt/data_3T/peilin/ce6190_p2/*/scripts/*.py'):
    txt = open(p).read()
    hits = sorted(set(re.findall(r"(DejaVu Serif|STIXGeneral|Times New Roman|Nimbus Roman|TeX Gyre Termes)", txt)))
    if hits:
        print(p, hits)
