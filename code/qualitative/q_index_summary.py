"""Print the distinct frame-index patterns used per (fps, n) from frame_index.json and the figure index."""
import json
Q = '/mnt/data_3T/peilin/ce6190_p2/qualitative'
idx = json.load(open(f'{Q}/frame_index.json'))
pat = {}
for cid, c in idx.items():
    for m in c['methods']:
        key = (m['label'] if m['label'] in ('Ground truth',) else '', round(m['fps'], 3), m['n'])
        pat.setdefault((round(m['fps'], 3), m['n']), (m['idx'], m['times_s'], set()))[2].add(m['label'])
for (fps, n), (ii, tt, labs) in sorted(pat.items()):
    print(f'fps={fps} n={n} idx={ii} times={tt} methods={sorted(labs)}')
fig = json.load(open(f'{Q}/figures/figures_index.json'))
for k, v in fig.items():
    print(k, v['width_in'], 'x', v['height_in'], 'cell', v['cell_in'], v.get('cases', v.get('steps')), v['font'])
