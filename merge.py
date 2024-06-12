import os
import json
import pdb
import numpy as np

res=[]
for i,item in (enumerate(os.listdir("result"))):
    if not item.endswith("json"):
        continue
    path=os.path.join("result", item)
    with open(path) as f:
        data=json.load(f)
        res.append(data)


result = {}

for key in res[0].keys():
    all_values = [d[key] for d in res]
    aggregated_values = zip(*all_values)
    averages = []
    for pair in aggregated_values:
        average=np.mean(pair)
        averages.append(average)
    result[key] = averages
with open("merge_submit.json", 'w', encoding='utf-8') as wf:
    json.dump(result, wf, indent=4, ensure_ascii=False)



