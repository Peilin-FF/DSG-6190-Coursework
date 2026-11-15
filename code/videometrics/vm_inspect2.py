import json
cj = '/mnt/data_3T/peilin/ce6190_p2/inventory/cases.json'
data = json.load(open(cj))
for c in data:
    print('==', c['id'], c['task'], c['benchmark'], '| input:', c.get('input'))
    print('   prompt:', c['prompt'])
    if not c.get('prompt_file_equals_json'):
        print('   prompt_file_text:', (c.get('prompt_file_text') or '')[:600])
    print('   methods:', [(m['label'], m['folder']) for m in c['methods']])
    extra = [k for k in c.keys() if k not in ('id','task','benchmark','title','prompt','input','stem','prompt_file','prompt_file_text','prompt_file_equals_json','methods')]
    if extra:
        print('   extra keys:', {k: str(c[k])[:200] for k in extra})
