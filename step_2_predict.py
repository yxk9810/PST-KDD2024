import os
from os.path import join
from tqdm import tqdm
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import trange
from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score
import logging
import utils

from train_1 import *
from tqdm import tqdm
logging.getLogger("transformers").setLevel(logging.ERROR)

import re
def fielter_url(text):
    text = re.sub(r'<.*?>', "", text)
    text = " ".join([x for x in text.strip().split(" ") if not ("<" in x or "=" in x or ">" in x)])
    return text

opt = ini_args()
set_seed(opt.seed)
model = Class_model(opt)
model.load_state_dict(torch.load("model_out/{}".format(opt.model_name),map_location="cpu"))
model.to(device)
model.eval()
num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_non_learnable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Number of learnable parameters: {num_learnable_params}")
print(f"Number of non-learnable parameters: {num_non_learnable_params}")
print(num_learnable_params+num_non_learnable_params)
def gen_kddcup_valid_submission_bert():
    data_dir = "data"
    papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")
    sub_example_dict = utils.load_json(data_dir, "submission_example_test.json")
    xml_dir = join(data_dir, "paper-xml")
    sub_dict = {}

    for paper in tqdm(papers):
        cur_pid = paper["_id"]
        file = join(xml_dir, cur_pid + ".xml")
        f = open(file, encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

        references = bs.find_all("biblStruct")
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

        bib_to_contexts = utils.find_bib_context(xml,dist=150)
        # bib_sorted = sorted(bib_to_contexts.keys())
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]
        
        y_score = [0] * n_refs

        assert len(sub_example_dict[cur_pid]) == n_refs
        # continue
        bib_to_contexts_len=[len(bib_to_contexts[bib]) for bib in bib_to_contexts]
        max_len=max(bib_to_contexts_len)
        min_len=min(bib_to_contexts_len)
        contexts_sorted=[]
        kong_idx=[]
        for i,bib in enumerate(bib_sorted):
            if len(bid_to_title.get(bib,"" ))==0:
                kong_idx.append(i)
            temp=paper.get("title"," ").lower()+"[SEP]"+bid_to_title.get(bib,"" )+"[SEP]"+" ".join(bib_to_contexts[bib]) 
            if len(bib_to_contexts[bib])==max_len:
                pre="[unused0] "
            elif len(bib_to_contexts[bib])==min_len:
                pre="[unused1] "
            else:
                pre="[unused2] "
            cur_context=pre+" "+temp
            contexts_sorted.append(cur_context)
        # contexts_sorted = [" ".join(bib_to_contexts[bib]) for bib in bib_sorted]
        contexts_sorted = [fielter_url(x) for x in contexts_sorted]
        predicted_scores = model.predict(contexts_sorted)
        for ii in range(len(predicted_scores)):
            bib_idx = int(bib_sorted[ii][1:])
            # print("bib_idx", bib_idx)
            y_score[bib_idx] = float(predicted_scores[ii])
            if ii in kong_idx:
                    y_score[bib_idx] = float(predicted_scores[ii])*0.1
            #     y_score[bib_idx] = float(predicted_scores[ii])*0.0 #49.6
        sub_dict[cur_pid] = y_score

    utils.dump_json(sub_dict,"./result/",opt.test_file)
gen_kddcup_valid_submission_bert()