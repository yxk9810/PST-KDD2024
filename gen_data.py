import utils_tan as utils
from os.path import join
import os
from tqdm import tqdm
import json
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import numpy as np
import json
import re
import pdb
import random

# dblp = json.load(open("dblp_data.json","r",encoding="utf-8"))

def fielter_url(text):    
    text = re.sub(r'<.*?>',"",text)
    text = " ".join([x for x in text.strip().split(" ") if not ("<" in x or "=" in x or ">" in x)])
    return text

random.seed(47)
np.random.seed(47)


x_train = []
y_train = []
x_valid = []
y_valid = []
data = []
total_ref = 0
data_dir = "data"
papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")

# rule_data = json.load(open("data/paper_source_gen_by_rule.json","r",encoding="utf-8"))

n_papers = len(papers)
papers = sorted(papers, key=lambda x: x["_id"])
# pids_rule = set(rule_data.keys())
# pids_train.update(pids_rule)
# pids = pids_rule
pids = [p["_id"] for p in papers]

in_dir = join(data_dir, "paper-xml")
# files = []
# for f in os.listdir(in_dir):
#     if f.endswith(".xml"):
#         files.append(f)

paper2title = {}
pid_to_source_titles = dd(list)
for paper in tqdm(papers):
    pid = paper["_id"]
    paper2title[pid] = paper["title"]
    for ref in paper["refs_trace"]:
        pid_to_source_titles[pid].append(ref["title"].lower())


# tmp = {k:list(rule_data[k].values()) for k in rule_data}
# pid_to_source_titles.update(tmp)
# files = sorted(files)
# for file in tqdm(files):
for cur_pid in tqdm(pids):
    f = open(join(in_dir, cur_pid + ".xml"), encoding='utf-8')
    xml = f.read()
    bs = BeautifulSoup(xml, "xml")

    source_titles = pid_to_source_titles[cur_pid]
    if len(source_titles) == 0:
        continue

    references = bs.find_all("biblStruct")
    # total_ref+=len(references)
    bid_to_title = {}
    n_refs = 0
    for ref in references:
        if "xml:id" not in ref.attrs:
            continue
        bid = ref.attrs["xml:id"]
        if ref.analytic is None:
            continue
        if ref.analytic.title is None:
            continue
        bid_to_title[bid] = ref.analytic.title.text.lower()
        b_idx = int(bid[1:]) + 1
        if b_idx > n_refs:
            n_refs = b_idx
    # total_ref+=n_refs
    flag = False

    cur_pos_bib = []
    total_ref+=len(bid_to_title)
    for bid in bid_to_title:
        cur_ref_title = bid_to_title[bid]
        for label_title in source_titles:
            if fuzz.ratio(cur_ref_title, label_title) >= 80:
                flag = True
                cur_pos_bib.append(bid)
    
    # cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib
    cur_neg_bib = [x for x in bid_to_title.keys() if x not in cur_pos_bib]
    # total_ref+=len(bid_to_title.keys())

    # if not flag:
    #     continue

    # if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
    #     continue

    bib_to_contexts = utils.find_bib_context(xml)

    n_pos = len(cur_pos_bib)
    n_neg = n_pos * 10
    n_neg = min(n_neg,len(cur_neg_bib))
    # cur_neg_bib_sample = np.random.choice(list(cur_neg_bib), n_neg, replace=True)
    cur_neg_bib_sample = np.random.choice(cur_neg_bib, n_neg, replace=False)
    # cur_neg_bib_sample = list(cur_neg_bib)
    for bib in cur_pos_bib:
        length = len(bib_to_contexts[bib])
        cur_context = ",".join(bib_to_contexts[bib])
        cur_context = bid_to_title.get(bib," ") + " [SEP] " + cur_context
        # cur_context = paper2title.get(cur_pid," ") + " [SEP] " + cur_context
        cur_context = str(length) + " [SEP] " + cur_context
        # cur_context = str(dblp.get(paper2title.get(cur_pid," "),{"n_citation":" "})["n_citation"])+" [SEP] " + cur_context
        data.append((cur_context,1))
        data.append((cur_context,1))
        data.append((cur_context,1))

    for bib in cur_neg_bib_sample:
        length = len(bib_to_contexts[bib])
        cur_context = ",".join(bib_to_contexts[bib])
        cur_context = bid_to_title.get(bib," ") + " [SEP] " + cur_context
        # cur_context = paper2title.get(cur_pid," ") + " [SEP] " + cur_context
        cur_context = str(length) + " [SEP] " + cur_context
        # cur_context = str(dblp.get(paper2title.get(cur_pid," "),{"n_citation":" "})["n_citation"])+" [SEP] " + cur_context
        data.append((cur_context,0))

data = [[fielter_url(x),y] for x,y in data]
data = [[x,y] for x,y in data if x.strip()]


print(len(data))
json.dump(data,open("train_data.json","w",encoding="utf-8"),ensure_ascii=False,indent=4)
#13050