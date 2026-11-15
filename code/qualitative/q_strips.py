"""Print-ready qualitative strips and the sampling-pair figure for Part C (house plot style).

Strips: rows = methods (ground truth first for image-to-video, CorrFlow last), columns = time
(0, 1.25, 2.5, 3.75, 5 s; nearest frame of each video, indices in frame_index.json).
Frames keep their aspect ratio inside a common cell (white bars for narrower videos).
Two cases of one benchmark can share one label column.
Each figure is written twice from the same layout: PNG with images at 600 px/in, and PDF with
images at 300 px/in (embedded without resampling) to keep the PDF small.
16 Nov correction pass: panel titles are the release case names (none reuses the paper's figure
wording); titles 7 pt with a fit check; 0.02-in left margin; the WMReward row is left out of the
Physics-IQ strips; the hairdryer strip uses the full 6.9-in width; the sampling-pair bracket names
the active noise interval and its steps (read from the manifest).
Usage: python q_strips.py [figure_name ...]   (no argument = all figures)
"""
import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_style as ps  # noqa: E402

P2 = '/mnt/data_3T/peilin/ce6190_p2'
Q = f'{P2}/qualitative'
FIG = f'{Q}/figures'
SP = f'{P2}/media/sampling-pair'

ps.apply()
_names = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((n for n in ('Times New Roman', 'Times', 'Nimbus Roman', 'TeX Gyre Termes') if n in _names), 'DejaVu Serif')
plt.rcParams.update({
    'font.serif': [FONT], 'font.size': 7.5, 'font.weight': 'bold',
    'savefig.bbox': 'standard', 'savefig.pad_inches': 0.0,
    'mathtext.fontset': 'dejavuserif', 'pdf.fonttype': 42, 'ps.fonttype': 42,
})
print('[q_strips] font family used:', FONT)

LAB_PT, TIME_PT, TITLE_PT = 7.5, 7.0, 7.0
BLUE, BLUE_TEXT = ps.PALETTE['blue'], ps.PALETTE['blue_text']
NAMES = {'Ours': 'CorrFlow', 'Ground truth': 'Ground\ntruth', 'Wan2.1': 'Wan2.1-1.3B', 'Wan2.2': 'Wan2.2-5B', 'VLiPP': 'VLIPP'}
TIMES = ['0 s', '1.25 s', '2.5 s', '3.75 s', '5 s']
IDX = json.load(open(f'{Q}/frame_index.json'))

G_CELL = 0.025   # gap between cells (in)
G_CASE = 0.08    # gap between two cases of one block (in)
G_BLOCK = 0.10   # gap between blocks (in)
TITLE_H = 0.15   # height of a case title line (in)
TIME_H = 0.12    # height of the time header (in)
LAB_PAD = 0.08   # space between label column and first cell (in)
M_LEFT, M_TOP, M_RIGHT, M_BOTTOM = 0.02, 0.02, 0.02, 0.03   # outer margins (in): keep label serifs and the CorrFlow outline inside


def whole_px(H, dpi=600):
    """Round a figure height up to a whole pixel at the PNG resolution, so no edge row is cut."""
    return math.ceil(H * dpi - 1e-6) / dpi


def text_width_in(strings, size):
    """Widest rendered width (in) of the strings, measured at the output resolution."""
    f = plt.figure(figsize=(3, 3), dpi=600)
    r = f.canvas.get_renderer()
    w = 0.0
    for s in strings:
        t = f.text(0, 0, s, fontsize=size, fontweight='bold')
        w = max(w, t.get_window_extent(r).width / f.dpi)
    plt.close(f)
    return w


class Canvas:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.fig = plt.figure(figsize=(W, H))
        self.fig.patch.set_facecolor('white')

    def ax(self, x, y, w, h):
        a = self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H])
        a.set_axis_off()
        return a

    def text(self, x, y, s, **kw):
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)

    def rect(self, x, y, w, h, **kw):
        r = Rectangle((x / self.W, 1 - (y + h) / self.H), w / self.W, h / self.H,
                      transform=self.fig.transFigure, fill=False, **kw)
        self.fig.add_artist(r)

    def line(self, x0, y0, x1, y1, **kw):
        self.fig.add_artist(plt.Line2D([x0 / self.W, x1 / self.W], [1 - y0 / self.H, 1 - y1 / self.H],
                                       transform=self.fig.transFigure, **kw))


def fit_image(path_or_arr, w_in, h_in, ppi):
    im = Image.open(path_or_arr).convert('RGB') if isinstance(path_or_arr, str) else Image.fromarray(path_or_arr)
    W, H = im.size
    s = min(w_in * ppi / W, h_in * ppi / H)
    nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
    return np.asarray(im.resize((nw, nh), Image.LANCZOS))


def show(cv, x, y, w, h, arr):
    """Place arr centred in the cell (x, y, w, h); the axes box shrinks to the image aspect."""
    a = cv.ax(x, y, w, h)
    a.imshow(arr, interpolation='none', aspect='equal')
    a.set_anchor('C')
    return a


# ----------------------------------------------------------------------------------- strips
def block_rows(cases, drop=()):
    rows = [m['label'] for m in IDX[cases[0]]['methods'] if m['label'] not in drop]
    for c in cases[1:]:
        assert [m['label'] for m in IDX[c]['methods'] if m['label'] not in drop] == rows, (cases, rows)
    return rows


def cell_aspect(case):
    ours = [m for m in IDX[case]['methods'] if m['label'] == 'Ours'][0]
    w, h = ours['size']
    return h / w


def strip_figure(name, blocks, width, drop=(), ppi_list=((600, 'png'), (300, 'pdf'))):
    """blocks: list of dicts {cases: [cid, ...], titles: [str, ...]} (cases in one block share rows).
    drop: method labels left out of every block (e.g. 'WMReward')."""
    all_labels = []
    for b in blocks:
        all_labels += [NAMES.get(r, r) for r in block_rows(b['cases'], drop)]
    lab_w = M_LEFT + text_width_in(all_labels, LAB_PT) + LAB_PAD
    ncase = max(len(b['cases']) for b in blocks)
    usable = width - M_RIGHT - G_CELL / 2
    c = (usable - lab_w - (ncase - 1) * G_CASE - ncase * 4 * G_CELL) / (5 * ncase)
    panel_w = 5 * c + 4 * G_CELL
    title_w = {t: text_width_in([t], TITLE_PT) for b in blocks for t in b['titles']}
    for t, w in title_w.items():   # a title must not run into the next panel
        assert w <= panel_w, f'title wider than its panel ({w:.3f} > {panel_w:.3f} in): {t}'
    lay, H = [], M_TOP
    for bi, b in enumerate(blocks):
        rows = block_rows(b['cases'], drop)
        h = c * cell_aspect(b['cases'][0])
        bh = TITLE_H + TIME_H + len(rows) * h + (len(rows) - 1) * G_CELL
        lay.append((H, rows, h))
        H += bh + (G_BLOCK if bi < len(blocks) - 1 else 0.0)
    H = whole_px(H + M_BOTTOM)
    written = []
    for ppi, fmt in ppi_list:
        cv = Canvas(width, H)
        for (y0, rows, h), b in zip(lay, blocks):
            for k, (cid, title) in enumerate(zip(b['cases'], b['titles'])):
                xc = lab_w + k * (5 * c + 4 * G_CELL + G_CASE)
                cv.text(xc, y0, title, fontsize=TITLE_PT, fontweight='bold', ha='left', va='top')
                for j, t in enumerate(TIMES):
                    cv.text(xc + j * (c + G_CELL) + c / 2, y0 + TITLE_H + TIME_H - 0.015, t,
                            fontsize=TIME_PT, fontweight='bold', ha='center', va='baseline')
                meths = {m['label']: m for m in IDX[cid]['methods']}
                for r, lab in enumerate(rows):
                    yr = y0 + TITLE_H + TIME_H + r * (h + G_CELL)
                    folder = meths[lab]['folder']
                    for j in range(5):
                        arr = fit_image(f'{Q}/frames/{cid}/{folder}__k{j}.png', c, h, ppi)
                        show(cv, xc + j * (c + G_CELL), yr, c, h, arr)
                    if lab == 'Ours':
                        cv.rect(xc - G_CELL / 2, yr - G_CELL / 2, 5 * c + 5 * G_CELL, h + G_CELL,
                                edgecolor=BLUE, linewidth=0.9)
            for r, lab in enumerate(rows):
                yr = y0 + TITLE_H + TIME_H + r * (h + G_CELL)
                cv.text(M_LEFT, yr + h / 2, NAMES.get(lab, lab), fontsize=LAB_PT, fontweight='bold',
                        ha='left', va='center', color=BLUE_TEXT if lab == 'Ours' else 'black',
                        linespacing=1.0)
        out = f'{FIG}/{name}.{fmt}'
        cv.fig.savefig(out, format=fmt, dpi=600 if fmt == 'png' else 300)
        plt.close(cv.fig)
        written.append(out)
    print(name, f'{width:.2f} x {H:.2f} in', 'cell', f'{c:.3f} in', '->', written, flush=True)
    return {'name': name, 'width_in': round(width, 3), 'height_in': round(H, 3), 'cell_in': round(c, 3),
            'label_col_in': round(lab_w, 3), 'files': written,
            'cases': [cid for b in blocks for cid in b['cases']],
            'titles': [t for b in blocks for t in b['titles']],
            'title_width_in': {t: round(w, 3) for t, w in title_w.items()}, 'panel_width_in': round(panel_w, 3),
            'rows': [[NAMES.get(r, r).replace('\n', ' ') for r in block_rows(b['cases'], drop)] for b in blocks],
            'dropped_rows': list(drop), 'columns_s': TIMES,
            'text_pt': {'labels': LAB_PT, 'times': TIME_PT, 'titles': TITLE_PT},
            'margins_in': {'left': M_LEFT, 'top': M_TOP, 'right': M_RIGHT, 'bottom': M_BOTTOM}}


# ------------------------------------------------------------------------------ sampling pair
def read_frames(path, want):
    cap = cv2.VideoCapture(path)
    got, i = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return got, i


def pair_figure(name, steps, frame_ids, width, ppi_list=((600, 'png'), (300, 'pdf'))):
    man = json.load(open(f'{SP}/manifest.json'))
    sig = {s['step']: s['sigma'] for s in man['steps']}
    seeds, kappa, fps = man['seeds'], man['refinerScale'], float(man['sourceFPS'])
    arms = [('without_refiner', f"Without\nrefiner\n(seed {seeds['without_refiner']})"),
            ('with_refiner', f"With refiner\nκ = {kappa['with_refiner']}\n(seed {seeds['with_refiner']})")]
    data, nframes = {}, {}
    for arm, _ in arms:
        for s in steps:
            got, n = read_frames(f'{SP}/{arm}/step_{s:03d}.mp4', set(frame_ids))
            data[(arm, s)] = got
            nframes[(arm, s)] = n
    multi = len(frame_ids) > 1
    flabels = [f't = {f / fps:g} s' for f in frame_ids]
    arm_w = text_width_in([lab for _, lab in arms], LAB_PT)
    fr_w = text_width_in(flabels, TIME_PT) if multi else 0.0
    lab_w = M_LEFT + arm_w + LAB_PAD + ((fr_w + LAB_PAD) if multi else 0.0)
    n = len(steps)
    usable = width - M_RIGHT - G_CELL / 2
    c = (usable - lab_w - (n - 1) * G_CELL) / n
    h = c * 480 / 832
    bracket, head, g_arm = 0.17, 0.30, 0.07
    nrow = len(arms) * len(frame_ids)
    H = whole_px(M_TOP + bracket + head + nrow * h + (nrow - 1) * G_CELL + (g_arm if multi else 0) + M_BOTTOM)
    lo, hi = 0.15, 0.92   # active noise interval of the refiner in the paper's text-to-video setting
    active = [i for i, s in enumerate(steps) if lo <= sig[s] <= hi]
    in_iv = [s['step'] for s in man['steps'] if lo <= s['sigma'] <= hi]   # all 50 steps, not only the shown ones
    blabel = (f'correction interval in the paper’s setting: σ ∈ [{lo:g}, {hi:g}] '
              f'(steps {in_iv[0]}–{in_iv[-1]})')
    bl_w = text_width_in([blabel], TIME_PT)
    written = []
    for ppi, fmt in ppi_list:
        cv = Canvas(width, H)
        yb = M_TOP + bracket
        if active:
            xa = lab_w + active[0] * (c + G_CELL) + 0.03
            xb = lab_w + active[-1] * (c + G_CELL) + c - 0.03
            cv.line(xa, yb - 0.02, xb, yb - 0.02, color=BLUE, linewidth=0.9)
            for xx in (xa, xb):
                cv.line(xx, yb - 0.02, xx, yb + 0.02, color=BLUE, linewidth=0.9)
            xm = min(max((xa + xb) / 2, lab_w + bl_w / 2), width - M_RIGHT - bl_w / 2)
            cv.text(xm, yb - 0.04, blabel, fontsize=TIME_PT, fontweight='bold', ha='center', va='bottom',
                    color=BLUE_TEXT)
        for i, s in enumerate(steps):
            x = lab_w + i * (c + G_CELL)
            cv.text(x + c / 2, yb + head - 0.03, f'Step {s}\nσ = {sig[s]:.3f}', fontsize=TIME_PT,
                    fontweight='bold', ha='center', va='bottom', linespacing=1.15)
        y = yb + head
        for ai, (arm, lab) in enumerate(arms):
            if multi and ai > 0:
                y += g_arm
            y_start = y
            for fi, f in enumerate(frame_ids):
                for i, s in enumerate(steps):
                    arr = fit_image(data[(arm, s)][f], c, h, ppi)
                    show(cv, lab_w + i * (c + G_CELL), y, c, h, arr)
                if multi:
                    cv.text(M_LEFT + arm_w + LAB_PAD, y + h / 2, flabels[fi], fontsize=TIME_PT, fontweight='bold',
                            ha='left', va='center', color=BLUE_TEXT if arm == 'with_refiner' else 'black')
                y += h + G_CELL
            y_end = y - G_CELL
            if arm == 'with_refiner':
                cv.rect(lab_w - G_CELL / 2, y_start - G_CELL / 2, n * c + n * G_CELL, (y_end - y_start) + G_CELL,
                        edgecolor=BLUE, linewidth=0.9)
            cv.text(M_LEFT, (y_start + y_end) / 2, lab, fontsize=LAB_PT, fontweight='bold', ha='left', va='center',
                    color=BLUE_TEXT if arm == 'with_refiner' else 'black', linespacing=1.05)
        out = f'{FIG}/{name}.{fmt}'
        cv.fig.savefig(out, format=fmt, dpi=600 if fmt == 'png' else 300)
        plt.close(cv.fig)
        written.append(out)
    print(name, f'{width:.2f} x {H:.2f} in', 'cell', f'{c:.3f} in', '->', written, flush=True)
    return {'name': name, 'width_in': round(width, 3), 'height_in': round(H, 3), 'cell_in': round(c, 3),
            'files': written, 'steps': steps, 'sigma': [round(sig[s], 4) for s in steps],
            'frame_ids': frame_ids, 'frame_times_s': [f / fps for f in frame_ids],
            'frames_per_video': {f'{a}/{s}': nframes[(a, s)] for a, s in nframes},
            'seeds': seeds, 'refinerScale': kappa, 'active_steps_in_paper_interval': [steps[i] for i in active],
            'paper_interval_steps_of_50': [in_iv[0], in_iv[-1]], 'bracket_label': blabel,
            'bracket_label_width_in': round(bl_w, 3),
            'margins_in': {'left': M_LEFT, 'top': M_TOP, 'right': M_RIGHT, 'bottom': M_BOTTOM}}


# ------------------------------------------------------------------------------------ figures
# Panel titles = release case names (media/corrflow-project/src/data/videos.json "title", lower case,
# ", front view" and "in the ocean in sunset" cut to fit); none reuses the paper's figure wording.
T_VB13 = 'VBench: a sports ball on the right of a baseball bat'
T_VB03 = 'VBench: a chair and a couch'
T_P0140 = 'Physics-IQ: paper smoke'
T_P0046 = 'Physics-IQ: domino in juice'
PIQ_DROP = ('WMReward',)   # Physics-IQ strips show GT, Wan2.2-5B, VLiPP, PhysRAG, CorrFlow
FIGS = {
    'fig_qual_main_t2v_i2v': lambda: strip_figure('fig_qual_main_t2v_i2v', [
        {'cases': ['vbench-vb-13', 'vbench-vb-03'], 'titles': [f'(a) {T_VB13}', f'(b) {T_VB03}']},
        {'cases': ['physics-iq-veried-p0140', 'physics-iq-veried-p0046'], 'titles': [f'(c) {T_P0140}', f'(d) {T_P0046}']},
    ], 6.9, drop=PIQ_DROP),
    'fig_qual_t2v_vbench_good': lambda: strip_figure('fig_qual_t2v_vbench_good', [
        {'cases': ['vbench-vb-13', 'vbench-vb-03'], 'titles': [f'(a) {T_VB13}', f'(b) {T_VB03}']},
    ], 6.9),
    'fig_qual_i2v_physiq_good_fail': lambda: strip_figure('fig_qual_i2v_physiq_good_fail', [
        {'cases': ['physics-iq-veried-p0140', 'physics-iq-veried-p0046'], 'titles': [f'(a) {T_P0140}', f'(b) {T_P0046}']},
    ], 6.9, drop=PIQ_DROP),
    'fig_qual_app_videophy2_hairdryer': lambda: strip_figure('fig_qual_app_videophy2_hairdryer', [
        {'cases': ['videophy2-vp2_0019'], 'titles': ['VideoPhy2: hair in a stream of air']},
    ], 6.9),
    'fig_qual_app_vbench_mixed': lambda: strip_figure('fig_qual_app_vbench_mixed', [
        {'cases': ['vbench-vb-10', 'vbench-vb-01'],
         'titles': ['(a) VBench: a panda standing on a surfboard', '(b) VBench: a person is bending metal']},
    ], 6.9),
    'fig_sampling_pair_clean_pred': lambda: pair_figure('fig_sampling_pair_clean_pred', [1, 10, 20, 30, 40, 50], [40], 6.9),
    'fig_sampling_pair_clean_pred_first_last': lambda: pair_figure('fig_sampling_pair_clean_pred_first_last', [1, 10, 20, 30, 40, 50], [0, 80], 6.9),
}

if __name__ == '__main__':
    os.makedirs(FIG, exist_ok=True)
    names = sys.argv[1:] or list(FIGS)
    log_path = f'{FIG}/figures_index.json'
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    log = {k: v for k, v in log.items() if k in FIGS}
    for nm in names:
        log[nm] = FIGS[nm]()
        log[nm]['font'] = FONT
    json.dump(log, open(log_path, 'w'), indent=1)
    print('wrote', log_path)
