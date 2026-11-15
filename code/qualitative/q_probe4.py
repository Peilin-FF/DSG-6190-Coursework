import json, os, subprocess
P2 = '/mnt/data_3T/peilin/ce6190_p2'
m = json.load(open(f'{P2}/media/sampling-pair/manifest.json'))
print('manifest keys:', list(m.keys()))
files = m.get('files', {})
ks = list(files.keys())
print('n files', len(ks))
for k in ks[:3] + ks[49:52]:
    print(k, json.dumps(files[k])[:800])
for k in m.keys():
    if k not in ('steps', 'files', 'prompt'):
        print(k, '=', json.dumps(m[k])[:600])
# probe one step video with ffprobe if available, else cv2
import shutil
print('ffprobe:', shutil.which('ffprobe'), 'ffmpeg:', shutil.which('ffmpeg'))
import cv2
for arm in ('without_refiner', 'with_refiner'):
    p = f'{P2}/media/sampling-pair/{arm}/step_001.mp4'
    cap = cv2.VideoCapture(p)
    print(arm, 'frames', cap.get(cv2.CAP_PROP_FRAME_COUNT), 'fps', cap.get(cv2.CAP_PROP_FPS), 'w', cap.get(cv2.CAP_PROP_FRAME_WIDTH), 'h', cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
    print(arm, 'decoded frames', n)
# fonts
import matplotlib.font_manager as fm
names = sorted(set(f.name for f in fm.fontManager.ttflist))
print('Times-like fonts:', [n for n in names if 'Times' in n or 'Nimbus Roman' in n or 'Liberation Serif' in n or 'TeX Gyre' in n])
print('DejaVu:', [n for n in names if 'DejaVu' in n])
