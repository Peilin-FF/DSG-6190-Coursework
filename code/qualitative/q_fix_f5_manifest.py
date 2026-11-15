"""Print the sampling-pair manifest (short values), the release README lines on alignment and compression,
and the coarse PSNR lines for Physics-IQ p0019, p0046, p0140 and p0115."""
import json, re
m = json.load(open('/mnt/data_3T/peilin/ce6190_p2/media/sampling-pair/manifest.json'))
for k, v in m.items():
    if k == 'steps':
        print('steps: n =', len(v), 'first:', v[0], 'last:', v[-1])
        for s in v:
            if s['step'] in (1, 10, 16, 17, 20, 30, 40, 49, 50):
                print('   ', {kk: s[kk] for kk in s if kk in ('step', 'sigma', 't', 'timestep')})
    else:
        print(k, ':', json.dumps(v)[:600])
print('--- README')
lines = open('/mnt/data_3T/peilin/ce6190_p2/media/corrflow-project/public/media/video/README.txt').read().splitlines()
for i, l in enumerate(lines[:60], 1):
    print(f'{i:3d} {l}')
print('--- PSNR log lines')
for l in open('/mnt/data_3T/peilin/ce6190_p2/inventory/pairmatrix_log.txt'):
    if re.search(r'p0019|p0046|p0140|p0115|mean|summary|Physics', l, re.I):
        print(l.rstrip()[:240])
