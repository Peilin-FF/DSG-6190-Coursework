"""Track the yellow tennis ball in every frame of every row of one Physics-IQ case (default p0019).
Ball = largest connected blob of saturated yellow-green pixels (OpenCV HSV: H 25-50, S >= 110, V >= 110).
Writes zoom3/<case>_ball_track.json (per frame: time, area fraction, centroid x, y relative to the frame)
and prints a table at 0.25-s steps plus the intervals in which no ball is found.
Usage: python q_track_ball.py [CASE_ID]"""
import json, sys
import cv2
import numpy as np

P2 = '/mnt/data_3T/peilin/ce6190_p2'
MEDIA = f'{P2}/media/corrflow-project'
cid = sys.argv[1] if len(sys.argv) > 1 else 'physics-iq-veried-p0019'
MIN_FRAC = 2e-4          # smallest blob (fraction of frame area) accepted as the ball
case = [c for c in json.load(open(f'{P2}/inventory/cases.json')) if c['id'] == cid][0]
out = {'case': cid, 'rule': 'largest blob, HSV H 25-50, S>=110, V>=110, area >= 2e-4 of the frame', 'methods': {}}
for m in case['methods']:
    p, q = (m['r_fps'].split('/') + ['1'])[:2]
    fps = float(p) / float(q)
    cap = cv2.VideoCapture(f"{MEDIA}/{m['src']}")
    rows, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        H, W = fr.shape[:2]
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (25, 110, 110), (50, 255, 255))
        n, lab, st, cen = cv2.connectedComponentsWithStats(mask, 8)
        rec = {'f': i, 't': round(i / fps, 3), 'area': 0.0, 'x': None, 'y': None}
        if n > 1:
            k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            a = st[k, cv2.CC_STAT_AREA] / (H * W)
            if a >= MIN_FRAC:
                rec.update(area=round(float(a), 5), x=round(float(cen[k][0] / W), 3), y=round(float(cen[k][1] / H), 3))
        rows.append(rec)
        i += 1
    cap.release()
    gaps, start = [], None
    for r in rows:
        if r['x'] is None and start is None:
            start = r['t']
        if r['x'] is not None and start is not None:
            gaps.append([start, r['t']])
            start = None
    if start is not None:
        gaps.append([start, 'end'])
    seen = [r for r in rows if r['x'] is not None]
    top = min(seen, key=lambda r: r['y']) if seen else None
    out['methods'][m['label']] = {'fps': fps, 'n': len(rows), 'size': [W, H], 'no_ball_intervals_s': gaps,
                                  'topmost': top, 'frames': rows}
    print(f"== {m['label']}  fps {fps:g}  frames {len(rows)}  no-ball intervals (start, first frame seen again): {gaps}")
    print('   topmost point:', top)
    line = []
    for t in np.arange(0, 5.001, 0.25):
        r = rows[min(len(rows) - 1, int(round(t * fps)))]
        line.append(f"{t:4.2f}s " + ('  --  ' if r['x'] is None else f"({r['x']:.2f},{r['y']:.2f})"))
    for j in range(0, len(line), 7):
        print('   ' + ' | '.join(line[j:j + 7]))
json.dump(out, open(f"{P2}/qualitative/zoom3/{cid.split('-')[-1]}_ball_track.json", 'w'), indent=0)
print('wrote', f"zoom3/{cid.split('-')[-1]}_ball_track.json")
