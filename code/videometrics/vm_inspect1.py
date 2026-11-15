import json, csv, collections, sys
inv = '/mnt/data_3T/peilin/ce6190_p2/inventory/videos_inventory.csv'
cj = '/mnt/data_3T/peilin/ce6190_p2/inventory/cases.json'
rows = list(csv.DictReader(open(inv)))
print('rows', len(rows))
print('columns', list(rows[0].keys()))
cnt = collections.Counter((r['benchmark'], r['task']) for r in rows)
print(cnt)
cnt2 = collections.Counter((r['benchmark'], r['label'], r['folder'], r['width'], r['height'], r['fps'], r['r_fps'], r['nb_frames'], r['duration']) for r in rows)
for k, v in sorted(cnt2.items()):
    print(v, k)
data = json.load(open(cj))
print(type(data), len(data))
if isinstance(data, dict):
    print(list(data.keys())[:20])
    first = data[list(data.keys())[0]]
else:
    first = data[0]
def show(o, ind=0, maxlen=300):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, (dict, list)):
                print(' ' * ind + str(k) + ':')
                show(v, ind + 2)
            else:
                s = str(v)
                print(' ' * ind + str(k) + ': ' + s[:maxlen])
    elif isinstance(o, list):
        print(' ' * ind + 'list len %d' % len(o))
        if o:
            show(o[0], ind + 2)
    else:
        print(' ' * ind + str(o)[:maxlen])
show(first)
