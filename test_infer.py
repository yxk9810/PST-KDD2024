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
import utils_tan as utils

from test_train import *
from tqdm import tqdm
logging.getLogger("transformers").setLevel(logging.ERROR)

import re
def fielter_url(text):    
    text = re.sub(r'<.*?>',"",text)
    text = " ".join([x for x in text.strip().split(" ") if not ("<" in x or "=" in x or ">" in x)])
    return text

# dblp = json.load(open("dblp_data.json","r",encoding="utf-8"))
opt = ini_args()
set_seed(opt.seed)
model = Class_model(opt)
model.load_state_dict(torch.load("model_out/{}".format(opt.model_name),map_location="cpu"))
model.to(device)
model.eval()


def gen_kddcup_valid_submission_bert():
    data_dir = "data"
    papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")
    sub_example_dict = utils.load_json(data_dir, "submission_example_test.json")
    xml_dir = join(data_dir, "paper-xml")
    sub_dict = {}

    for paper in tqdm(papers):
        s_title = paper["title"]
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
            

        bib_to_contexts = utils.find_bib_context(xml)
        # bib_sorted = sorted(bib_to_contexts.keys())
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]
        
        y_score = [0] * n_refs
        try:
            assert len(sub_example_dict[cur_pid]) == n_refs
        except:
            pdb.set_trace()
        # continue
        # contexts_sorted = [" ".join(bib_to_contexts[bib]) for bib in bib_sorted]
        lengths = [len(bib_to_contexts[bib]) for bib in bib_sorted]
        flag = [1 if len(bid_to_title.get(bib,""))>0 else 0 for bib in bib_sorted]
        # n_citations = [dblp.get(bid_to_title.get(bib," "),{"n_citation":" "})["n_citation"] for bib in bib_sorted]
        contexts_sorted = [bid_to_title.get(bib," ") + " [SEP] " + ",".join(bib_to_contexts[bib]) for bib in bib_sorted]
        # contexts_sorted = [s_title + " [SEP] " + x for x in contexts_sorted]
        # contexts_sorted = [fielter_url(x) for x in contexts_sorted]
        assert len(contexts_sorted)==len(lengths)
        contexts_sorted = [str(x)+" [SEP] "+y for x,y in zip(lengths,contexts_sorted)]
        # contexts_sorted = [str(x)+" [SEP] "+y for x,y in zip(n_citations,contexts_sorted)]
        contexts_sorted = [fielter_url(x) for x in contexts_sorted]
        predicted_scores = model.predict(contexts_sorted)
        assert len(predicted_scores) == len(flag)
        predicted_scores = [x if y==1 else x*0.1 for x,y in zip(predicted_scores,flag)]
        
        for ii in range(len(predicted_scores)):
            bib_idx = int(bib_sorted[ii][1:])
            # print("bib_idx", bib_idx)
            y_score[bib_idx] = float(predicted_scores[ii])
        
        sub_dict[cur_pid] = y_score
    
    utils.dump_json(sub_dict,"result",opt.test_file)

gen_kddcup_valid_submission_bert()