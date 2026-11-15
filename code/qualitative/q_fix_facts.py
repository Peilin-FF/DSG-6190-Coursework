"""Facts for the 16 Nov notes: (1) coarse PSNR lines of p0019, p0046, p0140 (inventory step);
(2) frame-index patterns per benchmark and method (frame_index.json); (3) all-frame change of the
CorrFlow clips of the 5 multi-object / relation prompts at 128x72 (written to zoom3/cf_multiobject_frame_change.json)."""
import json, re
import cv2
import numpy as np

P2 = '/mnt/data_3T/peilin/ce6190_p2'
Q = f'{P2}/qualitative'
print('== (1) PSNR lines')
for l in open(f'{P2}/inventory/pairmatrix_log.txt'):
    if re.match(r'physics-iq-veried-p0(019|046|140) ', l):
        print(l.rstrip())
print('== (2) frame-index patterns (benchmark, label) -> n, fps, size, idx, last time')
idx = json.load(open(f'{Q}/frame_index.json'))
pat = {}
for cid, c in idx.items():
    for m in c['methods']:
        k = (c['benchmark'], m['label'], m['n'], round(m['fps'], 3), tuple(m['size']), tuple(m['idx']), m['times_s'][-1])
        pat[k] = pat.get(k, 0) + 1
for k, v in sorted(pat.items()):
    print(f'{k[0]:18s} {k[1]:12s} n={k[2]:3d} fps={k[3]:5g} size={k[4]} idx={list(k[5])} last={k[6]} clips={v}')
print('== (3) CorrFlow all-frame change, 128x72 grey, mean absolute difference (0-255)')
cases = {c['id']: c for c in json.load(open(f'{P2}/inventory/cases.json'))}
res = {}
for cid in ['vbench-vb-03', 'vbench-vb-04', 'vbench-vb-05', 'vbench-vb-13', 'vbench-vb-14']:
    m = [m for m in cases[cid]['methods'] if m['label'] == 'Ours'][0]
    cap = cv2.VideoCapture(f"{P2}/media/corrflow-project/{m['src']}")
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (128, 72), interpolation=cv2.INTER_AREA).astype(np.float32))
    d = [float(np.abs(fr[i] - fr[i - 1]).mean()) for i in range(1, len(fr))]
    res[cid] = {'frames': len(fr), 'max_consecutive_mad': round(max(d), 2), 'at_frame': int(np.argmax(d)) + 1,
                'median_consecutive_mad': round(float(np.median(d)), 2),
                'mad_first_vs_last': round(float(np.abs(fr[-1] - fr[0]).mean()), 2)}
    print(cid, res[cid])
json.dump({'about': 'CorrFlow clips of the multi-object and relation prompts: mean absolute grey-level difference '
                    'between consecutive frames at 128x72 (0-255 scale); a sudden object loss would show as a spike.',
           'cases': res}, open(f'{Q}/zoom3/cf_multiobject_frame_change.json', 'w'), indent=1)
print('wrote zoom3/cf_multiobject_frame_change.json')
