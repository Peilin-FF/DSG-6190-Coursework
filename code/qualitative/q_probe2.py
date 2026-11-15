import json, os, glob
P2 = '/mnt/data_3T/peilin/ce6190_p2'
d = json.load(open(f'{P2}/inventory/cases.json'))
print('type', type(d).__name__, 'len', len(d))
if isinstance(d, dict):
    print('keys', list(d.keys())[:50])
    items = list(d.items())
    print(json.dumps(items[0], indent=1)[:3500])
else:
    print(json.dumps(d[0], indent=1)[:3500])
    print('----- item 18')
    print(json.dumps(d[18], indent=1)[:2500])

root = f'{P2}/media/corrflow-project/public/media/video'
print('---- video root')
for b in sorted(os.listdir(root)):
    print(b, sorted(os.listdir(os.path.join(root, b))))
print('---- sampling pair')
sp = f'{P2}/media/sampling-pair'
for r, ds, fs in os.walk(sp):
    print(r, len(ds), len(fs), sorted(fs)[:6])
mf = glob.glob(f'{sp}/**/manifest.json', recursive=True)
print('manifests', mf)
for m in mf:
    print(open(m).read()[:4000])
