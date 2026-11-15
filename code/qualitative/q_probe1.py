import json, os, subprocess
P2 = '/mnt/data_3T/peilin/ce6190_p2'
d = json.load(open(f'{P2}/inventory/cases.json'))
print(type(d), len(d))
if isinstance(d, dict):
    print(list(d.keys())[:40])
    k0 = list(d.keys())[0]
    first = d[k0]
else:
    first = d[0]
print(json.dumps(first, indent=1)[:5000])
