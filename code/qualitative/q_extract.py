"""Extract time-aligned frames from the released CorrFlow comparison videos and build one
contact sheet per case (rows = methods, CorrFlow last; GT first for image-to-video).

Times: t = 0, 1.25, 2.5, 3.75, 5.0 s (0/25/50/75/100 % of a common 5-s span).
Frame index for a video with n frames at f fps: min(n - 1, round(t * f)).
Outputs (under /mnt/data_3T/peilin/ce6190_p2/qualitative):
  frames/<case_id>/<method_folder>__k<j>.png   full-resolution frames
  frames/<case_id>/input.png                    conditioning image (image-to-video only)
  contact/<case_id>.png                         contact sheet for viewing
  frame_index.json                              per case/method: n, fps, indices, times
Usage: python q_extract.py [--times 0,1.25,2.5,3.75,5] [--tag ''] [--only id1,id2]
"""
import argparse, json, os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

P2 = '/mnt/data_3T/peilin/ce6190_p2'
MEDIA = f'{P2}/media/corrflow-project'
OUT = f'{P2}/qualitative'
FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')

LABEL = {  # display names
    'Ours': 'CorrFlow', 'Ground truth': 'Ground truth', 'Wan2.1': 'Wan2.1-1.3B', 'Wan2.2': 'Wan2.2-5B',
}


def order_methods(c):
    ms = list(c['methods'])
    gt = [m for m in ms if m['label'] == 'Ground truth']
    ours = [m for m in ms if m['ours']]
    base = [m for m in ms if m['label'] in ('Wan2.1', 'Wan2.2')]
    rest = [m for m in ms if m not in gt and m not in ours and m not in base]
    return gt + base + rest + ours


def read_frames(path, want):
    cap = cv2.VideoCapture(path)
    got, i = {}, 0
    want = set(want)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return got, i


def fit_cell(img, cw, ch, bg=255):
    h, w = img.shape[:2]
    s = min(cw / w, ch / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    cell = np.full((ch, cw, 3), bg, np.uint8)
    y0, x0 = (ch - nh) // 2, (cw - nw) // 2
    cell[y0:y0 + nh, x0:x0 + nw] = r
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--times', default='0,1.25,2.5,3.75,5')
    ap.add_argument('--tag', default='')
    ap.add_argument('--only', default='')
    ap.add_argument('--cellw', type=int, default=300)
    a = ap.parse_args()
    times = [float(x) for x in a.times.split(',')]
    cases = json.load(open(f'{P2}/inventory/cases.json'))
    only = set(x for x in a.only.split(',') if x)
    font = ImageFont.truetype(os.path.join(FONT_DIR, 'DejaVuSans-Bold.ttf'), 15)
    font_s = ImageFont.truetype(os.path.join(FONT_DIR, 'DejaVuSans.ttf'), 13)
    index = {}
    os.makedirs(f'{OUT}/contact', exist_ok=True)
    for c in cases:
        cid = c['id']
        if only and cid not in only:
            continue
        fdir = f'{OUT}/frames{a.tag}/{cid}'
        os.makedirs(fdir, exist_ok=True)
        rows, info = [], []
        if c.get('input'):
            inp = cv2.cvtColor(cv2.imread(f"{MEDIA}/public/{c['input']}"), cv2.COLOR_BGR2RGB)
            Image.fromarray(inp).save(f'{fdir}/input.png')
        for m in order_methods(c):
            path = f"{MEDIA}/{m['src']}"
            n, fps = int(m['nb_frames']), float(m['fps'])
            if m.get('r_fps') and '/' in m['r_fps']:
                p, q = m['r_fps'].split('/')
                fps = float(p) / float(q)
            idx = [min(n - 1, int(round(t * fps))) for t in times]
            got, ndec = read_frames(path, idx)
            assert ndec == n, (cid, m['folder'], ndec, n)
            frs = [got[i] for i in idx]
            for j, fr in enumerate(frs):
                Image.fromarray(fr).save(f"{fdir}/{m['folder']}__k{j}.png")
            rows.append((LABEL.get(m['label'], m['label']), frs))
            info.append({'label': m['label'], 'folder': m['folder'], 'n': n, 'fps': fps,
                         'idx': idx, 'times_s': [round(i / fps, 4) for i in idx],
                         'size': [int(frs[0].shape[1]), int(frs[0].shape[0])]})
        index[cid] = {'task': c['task'], 'benchmark': c['benchmark'], 'prompt': c['prompt'],
                      'stem': c['stem'], 'times_req_s': times, 'methods': info}
        # contact sheet
        cw = a.cellw
        ch = int(round(cw * 480 / 832))
        lw, th, gap = 150, 48, 4
        W = lw + len(times) * (cw + gap)
        H = th + len(rows) * (ch + gap)
        sheet = Image.new('RGB', (W, H), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        d.text((6, 4), f"{cid} [{c['task']}]", fill=(0, 0, 0), font=font)
        d.text((6, 24), (c['prompt'][:150] + ('...' if len(c['prompt']) > 150 else '')), fill=(60, 60, 60), font=font_s)
        for j, t in enumerate(times):
            pass
        for r, (lab, frs) in enumerate(rows):
            y = th + r * (ch + gap)
            d.text((6, y + ch // 2 - 8), lab, fill=(30, 90, 200) if lab == 'CorrFlow' else (0, 0, 0), font=font)
            for j, fr in enumerate(frs):
                x = lw + j * (cw + gap)
                sheet.paste(Image.fromarray(fit_cell(fr, cw, ch)), (x, y))
                ti = info[r]['times_s'][j]
                d.text((x + 3, y + 2), f'{ti:.2f}s', fill=(255, 255, 0), font=font_s)
        sheet.save(f'{OUT}/contact/{cid}{a.tag}.png')
        print(cid, len(rows), 'rows ->', f'{OUT}/contact/{cid}{a.tag}.png', flush=True)
    jpath = f'{OUT}/frame_index{a.tag}.json'
    old = json.load(open(jpath)) if (only and os.path.exists(jpath)) else {}
    old.update(index)
    json.dump(old, open(jpath, 'w'), indent=1)
    print('wrote', jpath, len(old))


if __name__ == '__main__':
    main()
