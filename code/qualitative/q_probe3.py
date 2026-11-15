import json, os, glob
P2 = '/mnt/data_3T/peilin/ce6190_p2'
d = json.load(open(f'{P2}/inventory/cases.json'))
print('top-level keys of a case:', list(d[0].keys()))
for c in d:
    ms = ', '.join(f"{m['label']}={m['folder']}({m['nb_frames']}f,{m['fps']}fps,{m['duration']}s,{m['width']}x{m['height']})" for m in c['methods'])
    print(c['id'], c['task'], 'input=', c.get('input'), '|', ms)
root = f'{P2}/media/corrflow-project/public/media/video'
print('---- video root')
for b in sorted(os.listdir(root)):
    p = os.path.join(root, b)
    if os.path.isdir(p):
        print(b, sorted(os.listdir(p)))
    else:
        print('FILE', b, os.path.getsize(p))
print('---- README.txt')
print(open(os.path.join(root, 'README.txt')).read()[:3000])
print('---- sampling pair')
sp = f'{P2}/media/sampling-pair'
for r, ds, fs in os.walk(sp):
    print(r, len(ds), len(fs), sorted(fs)[:8])
mf = glob.glob(f'{sp}/**/manifest.json', recursive=True)
print('manifests', mf)
for m in mf:
    print(open(m).read()[:6000])
