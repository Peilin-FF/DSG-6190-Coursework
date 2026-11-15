"""16 Nov correction pass of the per-case notes (CHECK.md P1-P9, S1, T1).
1. Keeps the first version as case_notes_old.json, applies the corrected notes and verdicts below,
   recomputes verdict_tally from the 32 verdict_class fields (overall and per benchmark only, scope rule S1)
   and writes case_notes.json.
2. With a template path: fills <!--TALLY-->, <!--CASES--> and <!--FIGTABLE--> from case_notes.json,
   figures_index.json and the release titles, and writes NOTES_FINAL.md.
Usage: python q_fix_notes.py [TEMPLATE.md]"""
import collections, json, os, shutil, sys

P2 = '/mnt/data_3T/peilin/ce6190_p2'
Q = f'{P2}/qualitative'
UPDATES = {
    'physics-iq-veried-p0019': dict(
        verdict='better (right path order, late return)', verdict_class='better',
        note=('Ground truth (ball track in zoom3/p0019_ball_track.json): the ball rolls up the ramp (highest point at 0.60 s), '
              'rolls back and leaves at the left edge at 1.77 s. Wan2.2 and WMReward: the ball first rolls toward the pipe '
              '(0.25-1.25 s), then up the ramp, and reaches the top edge at 4.7-5 s (reverse order). VLiPP: the ball drifts slowly '
              'toward the ramp until about 2.3 s and back, and stays in view. PhysRAG: the ball first rolls toward the pipe '
              '(0.25-0.75 s), then rolls back and forth between the pipe and the ramp, and goes up the ramp at 4.5-5 s. '
              'CorrFlow: the ball follows the ground-truth order (up the ramp, then back toward the pipe), but it rolls slower '
              'and further: it passes the top edge of the frame at about 2.1 s, is out of view from 2.19 to 4.06 s, rolls back '
              'down the ramp and is near the pipe at 5 s, whereas the ground-truth ball turns back at about 0.6 s and is gone '
              'after 1.77 s. The only row with the ground-truth order.')),
    'physics-iq-veried-p0046': dict(
        verdict='worse (wrong outcome)', verdict_class='worse',
        note=('Ground truth: the grabber releases the domino into the liquid; the domino is out of sight by 0.75 s '
              '(zoom/p0046.jpg, zoom3/p0046_all_rows.jpg) and the grabber stays at the top edge. Wan2.2 and WMReward: the grabber '
              'goes into the mug and pushes the domino under; a hand appears at 2.25-4 s. VLiPP: the domino turns into a small grey '
              'cube that floats on the surface from 0.75 s. PhysRAG: the grabber lets go and the domino floats flat on the surface '
              'from 1 s to 5 s. CorrFlow: the grabber lowers the domino into the mug; from 2.5 s the domino lies flat at the '
              'surface, brownish, until 5 s, and the grabber withdraws to the top edge: the ceramic domino floats instead of sinking.')),
    'physics-iq-veried-p0110': dict(
        verdict='mixed (flame at the water line)', verdict_class='mixed',
        note=('Prompt: "The lit match is being lowered into the glass of water". Ground truth: the match on the stick moves to the '
              'water at the right of the glass; the head is burnt and no flame is visible. Wan2.2 and WMReward: the stick vanishes '
              'at 4-5 s. VLiPP: the stick turns horizontal and a large white smoke cloud forms at 2.5-5 s. PhysRAG: little change. '
              'CorrFlow: the match follows the ground-truth path, but a flame flares at its head at the water line at 3-4 s, where '
              'the ground truth shows no flame; the flame is gone at 5 s.')),
    'physics-iq-veried-p0115': dict(
        verdict='better', verdict_class='better',
        note=('Ground truth: a tennis ball that hangs from an orange cord in front of a mirror is lowered to the table by about '
              '1.25 s, and its reflection follows. Wan2.2 and WMReward: the ball goes down, then rises again from about 3.1 s. '
              'VLiPP: the ball detaches and falls, and its reflection does not follow. PhysRAG: the ball rises to the top edge at '
              '3.75-5 s. CorrFlow: the ball on the cord and its reflection are lowered to the table by about 1.25 s and stay there, '
              'as in the ground truth (zoom3/p0115_all_rows.jpg).')),
    'physics-iq-veried-p0140': dict(
        verdict='better (closest smoke amount)', verdict_class='better',
        note=('Ground truth: a thin, steady vertical plume; the paper is unchanged. Wan2.2 and WMReward: the plume thickens into '
              'a bright curling column from about 2 s. VLiPP: a wide, faint haze at 1.25-1.9 s and near 4.4 s, and the paper chars '
              'black from about 1.9 s. PhysRAG: thick turbulent smoke from 2.25 s. CorrFlow: thin wisps as in the ground truth, '
              'but more diffuse, and drifting left from about 3 s (3.75 s and 5 s columns), whereas the ground-truth plume stays '
              'vertical; the paper is unchanged except a small dark spot (zoom3/p0140_all_rows.jpg, zoom3/cols_p0140.jpg).')),
    'videophy2-vp2_0019': dict(
        verdict='mixed (clean frames, action missing)', verdict_class='mixed',
        note=('Generation prompt (release prompt file, fed to all methods): a woman stands before a large mirror with her '
              'reflection in view, lifts her hair with her left hand, and "her right hand expertly wields a sleek, white hairdryer, '
              'its nozzle directed towards her hair". PhysRAG: one woman holds the dryer at chest height and tosses her hair at '
              '3.1-3.75 s; the dryer does not point at the hair. VideoREPA: a mirror scene with two dryers in view (one at the left '
              'edge, one at the reflected woman\'s ear) at 0 s and 3.1-5 s. DreamWorld: three women in view in every column (one in '
              'the foreground with her back to the camera and two more behind her), where the prompt describes one woman and her '
              'reflection; a dryer points at the hair in the 3.75 s and 5 s columns. VPT: a mirror scene; the reflection holds the '
              'dryer at chest height; the foreground hand and dryer are motion-blurred at 1.9-3.75 s. CorrFlow: one woman and a '
              'matching reflection, sharp frames; she lifts her hair with one hand at 0.6-1.25 s, but the dryer stays at waist '
              'height with its nozzle pointing away from her hair in every frame we read; nearly static after 2.5 s '
              '(zoom3/vp0019_all_rows.jpg, zoom3/cols_vp0019.jpg).')),
    'vbench-vb-01': dict(
        verdict='worse (action absent)', verdict_class='worse',
        note=('PhysRAG: a smiling man holds an orange object at 1.25-2.5 s; no metal. VideoREPA: static close-up of a metal '
              'bracket. VPT: hands turn a small metal ring (the closest to working metal). CorrFlow: a man rests his head on a '
              'wooden step beside a metal tube; the tube is diagonal at 0-1.9 s and lies horizontally at the bottom from 2.5 s; '
              'nothing bends (zoom3/vb01_all_rows.jpg, zoom3/cols_vb01_cf.jpg).')),
    'vbench-vb-10': dict(
        verdict='mixed (panda on the board, board covered by a wave for a while)', verdict_class='mixed',
        note=('PhysRAG: a bear-like animal stands in the surf; no board. VideoREPA: a cartoon panda sits on a board; its body '
              'stretches at 1.9-2.5 s. VPT: a panda swims; no board. CorrFlow: a photo-like panda sits on a board at 0-2.5 s; from '
              '3.1 s a wave covers the board and the panda sits in the water; the board is visible again at 5 s. No method shows a '
              'panda standing on a board (zoom3/vb10_all_rows.jpg, zoom3/cols_vb10_cf.jpg, zoom3/cols_vb10_others.jpg).')),
    'vbench-vb-13': dict(
        verdict='better (stable identity)', verdict_class='better',
        note=('PhysRAG: close-up of a baseball; a striped object enters the right edge from 2.5 s; no clear bat. VideoREPA: a blue '
              'ball to the right of a bat (relation met), but at 2.5-3.1 s a hand touches the ball and its pattern changes from blue '
              'to white-blue (blue again from 3.75 s). VPT: a baseball alone on green; no bat. CorrFlow: bat at the left, baseball at '
              'the right, unchanged for 5 s; relation met with stable object identity (VideoREPA also meets the relation, with an '
              'appearance change) (zoom3/vb13_all_rows.jpg).')),
    'vbench-vb-03': dict(
        verdict='better', verdict_class='better',
        note=('PhysRAG: one dark couch, no chair. VideoREPA: blurry beige shapes, neither object clear. VPT: dark close-up of a '
              'couch arm with a white pillow. CorrFlow: pink armchair and green couch side by side, sharp and fixed for 5 s; the '
              'only method with both objects clearly visible (zoom3/vb03_all_rows.jpg).')),
    'videophy2-vp2_0256': dict(
        verdict='more stable, no full rotation in any sampled frame', verdict_class='mixed', note=None),
}


def tally(cases):
    t = {'all': collections.Counter()}
    for c in cases:
        b = c['id'].split('-')[0]
        b = 'physics-iq' if b == 'physics' else b
        t.setdefault(b, collections.Counter())[c['verdict_class']] += 1
        t['all'][c['verdict_class']] += 1
    order = ['better', 'equal', 'mixed', 'worse']
    return {k: {o: v.get(o, 0) for o in order} for k, v in t.items()}


def main():
    path, old = f'{Q}/case_notes.json', f'{Q}/case_notes_old.json'
    if not os.path.exists(old):
        shutil.copy2(path, old)
    d = json.load(open(old))
    for c in d['cases']:
        u = UPDATES.get(c['id'])
        if u:
            c['verdict'], c['verdict_class'] = u['verdict'], u['verdict_class']
            if u['note']:
                c['note'] = u['note']
    assert len(d['cases']) == 32 and all(c['id'] in {x['id'] for x in d['cases']} for c in d['cases'])
    d['_about'] = ('Per-case visual notes on the 32 released CorrFlow comparison cases. Sources: contact sheets '
                   'qualitative/contact/<id>.png (nearest frames to 0, 1.25, 2.5, 3.75 and 5 s; indices in frame_index.json) and '
                   'denser views for 20 cases: zoom/ (8 cases and the sampling pair; 7 or 9 times, 2 to 5 rows), zoom2/ (9 cases; '
                   '9 times, 2 rows) and zoom3/ (16 Nov: every row of the 7 chosen cases and of p0115 at 9 times, every row of p0019 '
                   'at 0.25-s steps with a ball track, and larger views of the exact figure columns). One viewer, visual reading '
                   'only; all cases were selected by the authors for their project page. in_paper = the case appears in a figure '
                   'of the CorrFlow paper. verdict is about CorrFlow against the other rows of the same case. 16 Nov: notes and '
                   'verdicts corrected after CHECK.md (p0019 re-read from dense frames: mixed -> better); first version in '
                   'case_notes_old.json.')
    d['dense_views'] = {
        'zoom': ['physics-iq-veried-p0046', 'physics-iq-veried-p0110', 'physics-iq-veried-p0140', 'vbench-vb-01', 'vbench-vb-02',
                 'vbench-vb-04', 'vbench-vb-10', 'videophy2-vp2_0019', 'sampling pair'],
        'zoom2': ['finephyeval-p016', 'finephyeval-p019', 'physics-iq-veried-p0019', 'physics-iq-veried-p0084',
                  'physics-iq-veried-p0155', 'videophy2-vp2_0256', 'videophy2-vp2_0273', 'videophy2-vp2_0362', 'videophy2-vp2_0534'],
        'zoom3': ['physics-iq-veried-p0019', 'physics-iq-veried-p0115', 'physics-iq-veried-p0140', 'physics-iq-veried-p0046',
                  'videophy2-vp2_0019', 'vbench-vb-13', 'vbench-vb-03', 'vbench-vb-10', 'vbench-vb-01'],
    }
    ids = set(x for v in d['dense_views'].values() for x in v if x != 'sampling pair')
    d['dense_views']['cases_with_a_dense_view'] = len(ids)
    d['dense_views']['cases_with_contact_sheet_only'] = sorted(c['id'] for c in d['cases'] if c['id'] not in ids)
    tr = json.load(open(f'{Q}/zoom3/p0019_ball_track.json'))
    d['p0019_ball_track'] = {k: {'no_ball_intervals_s': v['no_ball_intervals_s'],
                                 'topmost': {kk: v['topmost'][kk] for kk in ('t', 'x', 'y')}}
                             for k, v in tr['methods'].items()}
    d['p0019_ball_track']['_about'] = tr['rule'] + '; x, y relative to the frame (y = 0 is the top edge); intervals = [first frame without the ball, first frame with it again]'
    d['sampling_pair'] = ('Clean-sample estimates (manifest field cleanPrediction: VAE decode of x_sigma - sigma * v_guided after '
                          'CFG and the optional refiner, before the ODE update) of two independently sampled trajectories: seed 3 '
                          'with refinerScale 0 and seed 12 with refinerScale 0.2 (correction strength). Step 1 (sigma 1.000): a red '
                          'blur in both arms. Step 10 (sigma 0.958): each arm already shows the layout that it keeps to step 50 '
                          '(without the refiner: a popsicle lying on a plate; with the refiner: a watermelon-slice dessert on a '
                          'green leaf). Steps 20-50: only details sharpen (seed dots, sauce edge). First and last frames (t = 0 and '
                          '5 s): in the without-refiner arm the object changes shape over time (a lying popsicle at 0 s, an upright '
                          'red cylinder at 5 s); in the with-refiner arm one slice shape stays. Both are visible at step 10, before '
                          'step 17 (sigma 0.914), the first step inside the active noise interval of the paper text-to-video '
                          'setting (sigma in [0.15, 0.92], steps 17-49; the manifest does not record the interval). The arms use '
                          'different seeds and initial noise (manifest: "not ... a matched-noise causal ablation"), so the pair '
                          'is not evidence of a refiner effect.')
    mo = d['multi_object_and_relation_prompts']
    fc = json.load(open(f'{Q}/zoom3/cf_multiobject_frame_change.json'))
    mo['corrflow_frame_change_128x72'] = fc['cases']
    mo['note'] = ('CorrFlow shows every named object in the sampled frames of all 5 prompts; the all-frame change '
                  '(zoom3/cf_multiobject_frame_change.json) has no spike in vb-03, vb-05, vb-13, vb-14 (largest consecutive '
                  'change <= 0.70 of 255), and the vb-04 maximum (9.58 at frame 26) is the arm that moves the knife '
                  '(zoom3/frames_vb04_cf.jpg), the knife stays in view.')
    d['verdict_tally'] = tally(d['cases'])
    d['_verdict_tally_scope'] = 'overall and per benchmark only; the split by in_paper is not reported (CHECK.md S1)'
    json.dump(d, open(path, 'w'), indent=1, ensure_ascii=False)
    print('wrote', path)
    print(json.dumps(d['verdict_tally']))
    if len(sys.argv) > 1:
        render(d, sys.argv[1])


def render(d, tpl):
    vt = d['verdict_tally']
    rows = [('All 32', 'all'), ('VBench (14)', 'vbench'), ('Physics-IQ (10)', 'physics-iq'),
            ('VideoPhy2 (6)', 'videophy2'), ('FinePhyEval (2)', 'finephyeval')]
    tab = ['| Set | Better | Equal | Mixed | Worse | Better or equal |', '|---|---|---|---|---|---|']
    for lab, k in rows:
        v = vt[k]
        n = sum(v.values())
        tab.append(f"| {lab} | {v['better']} | {v['equal']} | {v['mixed']} | {v['worse']} | {v['better'] + v['equal']} of {n} |")
    titles = {}
    vj = json.load(open(f'{P2}/media/corrflow-project/src/data/videos.json'))
    def walk(o):
        if isinstance(o, dict):
            if 'id' in o and 'title' in o:
                titles[o['id']] = o['title']
            for x in o.values():
                walk(x)
        elif isinstance(o, list):
            for x in o:
                walk(x)
    walk(vj)
    groups = [('FinePhyEval (text to video)', 'finephyeval'), ('VideoPhy2 (text to video)', 'videophy2'),
              ('VBench (text to video)', 'vbench'), ('Physics-IQ Verified (image to video, with ground truth)', 'physics-iq')]
    lines = []
    for gl, gk in groups:
        lines.append(f'**{gl}**\n')
        for c in d['cases']:
            if not c['id'].startswith(gk):
                continue
            role = f"; {c['role']}" if c['role'] else ''
            inp = 'in a paper figure' if c['in_paper'] else 'not in the paper figures'
            v, cls = c['verdict'], c['verdict_class']
            rest = v[len(cls):].strip() if v.startswith(cls) else ''
            label = rest[1:-1] if rest.startswith('(') and rest.endswith(')') else v
            vtxt = f"**{cls}**" if v == cls else f"**{cls}** ({label})"
            lines.append(f"- **{c['id']}** (\"{titles.get(c['id'], '?')}\"; {inp}{role}). Verdict: {vtxt}. {c['note']}")
        lines.append('')
    fi = json.load(open(f'{Q}/figures/figures_index.json'))
    ft = ['| File (PNG + PDF, same stem) | Size (in) | Cell (in) | Panels and rows |', '|---|---|---|---|']
    for k, v in fi.items():
        if 'titles' in v:
            pr = '; '.join(v['titles']) + '. Rows: ' + ' / '.join(', '.join(r) for r in v['rows'])
            if v['dropped_rows']:
                pr += f". Left out: {', '.join(v['dropped_rows'])}"
        else:
            pr = (f"steps {v['steps']}, frames {v['frame_ids']}; bracket: {v['bracket_label']}; "
                  f"seeds {v['seeds']}, refinerScale {v['refinerScale']}")
        ft.append(f"| `{k}` | {v['width_in']} x {v['height_in']} | {v['cell_in']} | {pr} |")
    s = open(tpl).read()
    s = s.replace('<!--TALLY-->', '\n'.join(tab)).replace('<!--CASES-->', '\n'.join(lines)).replace('<!--FIGTABLE-->', '\n'.join(ft))
    assert '<!--' not in s, 'unfilled placeholder'
    open(f'{Q}/NOTES_FINAL.md', 'w').write(s)
    print('wrote', f'{Q}/NOTES_FINAL.md', len(s), 'chars')


if __name__ == '__main__':
    main()
