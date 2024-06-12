import random
import torch
import numpy as np
import os
import argparse
from random import shuffle
from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer,BertModel
from transformers import AutoModel, AutoTokenizer
from torch.cuda.amp import autocast as ac
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import precision_score,f1_score,recall_score
from torch.optim.swa_utils import AveragedModel
import json
import pdb
import re
from torch.optim.swa_utils import AveragedModel
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_class = 2


# FGM
class FGM:
    def __init__(self, model: nn.Module, eps=0.5):
        self.model = (
            model.module if hasattr(model, "module") else model
        )
        self.eps = eps
        self.backup = {}

    # only attack word embedding
    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}

def set_seed(seed):
    """
    设置随机种子
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def ini_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', default='./data',
                        help='the data dir of raw data')

    parser.add_argument('--output_dir', default='./model_out/',
                        help='the output dir for model checkpoints')

    parser.add_argument('--bert_dir', default='***',
                        help='bert dir for ernie / roberta-wwm / uer')

    parser.add_argument('--task_type', default='span',
                        help='crf / span / mrc')

    parser.add_argument('--loss_type', default='ls_ce',
                        help='loss type for span/mrc')
    # other args
    parser.add_argument('--seed', type=int, default=47, help='random seed')

    parser.add_argument('--gpu_ids', type=str, default=['0'],
                        help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')

    parser.add_argument('--mode', type=str, default='train',
                        help='train / stack')
    # train args
    parser.add_argument('--train_epochs', default=8, type=int,
                        help='Max training epoch')

    parser.add_argument('--dropout_prob', default=0.1, type=float,
                        help='drop ner_out probability')

    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate for the bert module')

    parser.add_argument('--other_lr', default=2e-3, type=float,
                        help='learning rate for the module except bert')

    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='max grad clip')

    parser.add_argument('--warmup_proportion', default=0.1, type=float)

    parser.add_argument('--weight_decay', default=0.01, type=float)

    parser.add_argument('--adam_epsilon', default=1e-8, type=float)

    parser.add_argument('--train_batch_size', default=128, type=int)

    parser.add_argument('--attack_train', default='', type=str,
                        help='fgm / pgd attack train when training')

    parser.add_argument('--test_file', default='',type=str)
    parser.add_argument('--model_name', default='model_1.pt',type=str)

    parser.add_argument('--fgm_param', default=0.3 ,type=float)

    return parser.parse_args()


def get_train_data():
    data = json.load(open("./train_data.json","r",encoding="utf-8"))
    shuffle(data)
    shuffle(data)
    split_index = int(len(data) * 0.8)

    # 按照索引切分数据和标签
    train = data[:split_index]
    dev= data[split_index:]
    train=data
    temp=[]
    for line in train:
        if line[1]==1:
            temp.append(line)
            temp.append(line)
        temp.append(line)
    shuffle(temp)
    train=temp
    return train,dev
    


class My_Dataset(Dataset):
    def __init__(self,train_feature,opt):
        self.data = train_feature
        self.nums = len(train_feature)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.bert_dir)
        self.opt = opt

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        sample = self.data[index]
        return sample

    def collate_fn(self,batch_data):
        input_ids,token_type_ids,attention_mask,labels = [],[],[],[]
        max_len = 510
        # max_len = 128
        for sample in batch_data:
            text,l = sample
            encode_dict = self.tokenizer.encode_plus(text,truncation = True,
                                                     add_special_tokens=True,
                                                     max_length = max_len,
                                                     padding = 'max_length',
                                                     return_token_type_ids=True)

            input_ids.append(encode_dict['input_ids'])
            token_type_ids.append(encode_dict['token_type_ids'])
            attention_mask.append(encode_dict['attention_mask'])
            labels.append(int(l))

        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        labels = torch.tensor(labels).long()


        result = ['input_ids', 'token_type_ids','attention_mask',"labels"]
        return dict(zip(result,[input_ids, token_type_ids, attention_mask,labels]))


def build_optimizer_and_scheduler(opt, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))


    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt.weight_decay, 'lr': opt.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt.other_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(opt.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_pred = torch.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()


        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_pred, target,
                                                                                   reduction=self.reduction,
                                                                                   ignore_index=self.ignore_index)


class Class_model(torch.nn.Module):
    def __init__(self,opt,dropout_prob=0.1):
        super(Class_model, self).__init__()
        logger.info("load weight dir from {}".format(opt.bert_dir))
        if "ernie" ==  opt.bert_dir:
            self.bert_module = ErnieModel.from_pretrained(opt.bert_dir)
        else:
            self.bert_module = AutoModel.from_pretrained(opt.bert_dir)
        self.opt = opt
        out_dims = self.bert_module.config.hidden_size
        mid_linear_dims = 128
        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        out_dims = mid_linear_dims
        self.out_fc = nn.Linear(out_dims,num_class)
        reduction = 'none'
        opt.loss_type ='ce'
        if opt.loss_type == 'ce':
            # weight = torch.tensor([0.5, 5.0])
            self.criterion = nn.CrossEntropyLoss()
            # self.criterion = nn.CrossEntropyLoss(weight=weight)
            # self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        elif opt.loss_type == 'ls_ce':
            self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        else:
            self.criterion = FocalLoss(reduction=reduction)

        init_blocks = [self.mid_linear,self.out_fc]

        self._init_weights(init_blocks)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.bert_dir)

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,labels=None):

        bert_outputs = self.bert_module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        if self.opt.bert_dir=="electra" or self.opt.bert_dir=="xlnet":
            seq_out  = bert_outputs.last_hidden_state[:,0,:]
        else:
            seq_out = bert_outputs[1]

        seq_out = self.mid_linear(seq_out)

        out_logits = self.out_fc(seq_out)

        out = (out_logits,)

        if labels is not None and self.training:
            loss = self.criterion(out_logits,labels).mean()
            out = (loss, ) + out

        return out

    @torch.no_grad()
    def predict(self,candidates):
        res = []
        inputs = self.batch_data(candidates)
        output = self.forward(**inputs)[0]
        prob = [x[1] for x in torch.sigmoid(output).cpu().detach().numpy()]
        # prob_1 = [x[1] for x in torch.sigmoid(output).cpu().detach().numpy()]
        # logits_softmax=F.softmax(output,dim=-1)
        # predict_logits=logits_softmax.cpu().detach().numpy()
        # predict_logits = np.vstack(predict_logits)
        # prob=predict_logits[:, -1]
        return prob


    def batch_data(self,candidates):
        input_ids,token_type_ids,attention_mask, = [],[],[]
        max_len = 510
        # max_len = 128
        for sample in candidates:
            encode_dict = self.tokenizer.encode_plus(sample,
                                       truncation = True,
                                       add_special_tokens = True,
                                       max_length = max_len,
                                       padding = 'max_length',
                                       return_token_type_ids=True)



            input_ids.append(encode_dict['input_ids'])
            token_type_ids.append(encode_dict['token_type_ids'])
            attention_mask.append(encode_dict['attention_mask'])

        input_ids = torch.tensor(input_ids).long().to(device)
        token_type_ids = torch.tensor(token_type_ids).long().to(device)
        attention_mask = torch.tensor(attention_mask).long().to(device)

        result = ['input_ids', 'token_type_ids','attention_mask']
        return dict(zip(result,[input_ids, token_type_ids, attention_mask]))


def save_model(opt, model,epoch,name="class_model.pt"):
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir, exist_ok=True)

    # take care of model distributed / parallel training
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    torch.save(model_to_save.state_dict(), os.path.join(opt.output_dir,name))

def model_evaluate(model,dev_load,opt,device,epoch):
    model.eval()
    true,predict = [],[]
    predict_logits=[]
    with torch.no_grad():
        for batch,batch_data in enumerate(dev_load):

            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)

            output = model(**batch_data)[0]
            logits=[x[1] for x in torch.sigmoid(output).cpu().detach().numpy()]

            logits_softmax=F.softmax(output,dim=-1)
            predict_logits.extend(logits_softmax.cpu().detach().numpy())
            
            output = list(torch.argmax(output,-1).cpu().numpy())
            label = list(batch_data["labels"].cpu().numpy())
            predict+=output
            true+=label


    # precision = precision_score(true,predict,average='macro')
    # recall = recall_score(true,predict,average='macro')
    # f1 = f1_score(true,predict,average='macro')
    precision = precision_score(true,predict)
    recall = recall_score(true,predict)
    f1 = f1_score(true,predict)
    # logger.info("epoch:{},precision:{},recall:{},f1:{}".format(epoch+1,precision, recall, f1))
    from sklearn.metrics import average_precision_score
    predict_logits = np.vstack(predict_logits)
    predict_logits=predict_logits[:, -1]
    map=average_precision_score(np.array(true),np.array(predict_logits))
    logger.info("epoch:{},precision:{},recall:{},f1:{},map:{}".format(epoch+1,precision, recall, f1,map))
    return map


def model_predict(model,opt,device):
    model.eval()
    results = {}
    data=json.load(open(file="./ent_recall.json",encoding="utf-8",mode="r"))
    for ent,candidates in tqdm(data.items()):
        res = model.predict(ent,candidates)
        js = {"recall":res,"candidates":candidates}
        results[ent] = js
    json.dump(results,open(file=os.path.join("./files",opt.test_file),encoding="utf-8",mode="w"),ensure_ascii=False,indent=4)

def train(opt):
    train,dev = get_train_data()
    logger.info("train:{},dev:{}".format(len(train),len(dev)))
    # 计算类别权重
    train_labels=[line[1]for line in train ]
    # class_weights = torch.tensor(len(train_labels) / (2 * np.bincount(train_labels)), dtype=torch.float)

    # # 定义损失函数并设置权重
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    train_data = My_Dataset(train,opt)
    dev_data = My_Dataset(dev,opt)
    fn = train_data.collate_fn
    train_loader = DataLoader(dataset=train_data,batch_size=opt.train_batch_size,shuffle=True,collate_fn=fn,num_workers=8)
    dev_loader = DataLoader(dataset=dev_data, batch_size=opt.train_batch_size,shuffle=False,collate_fn=fn,num_workers=8)

    model = Class_model(opt)
    model.to(device)

    t_total = len(train_loader) * opt.train_epochs
    optimizer, scheduler = build_optimizer_and_scheduler(opt, model, t_total)
    model.zero_grad()
    best_f1 = 0.0
    fgm = FGM(model=model,eps=opt.fgm_param)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(opt.train_epochs):
        torch.cuda.empty_cache()
        model.train()
        for batch_data in tqdm(train_loader):
            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(device)
            with ac():
                loss = model(**batch_data)[0]
            scaler.scale(loss).backward()

            fgm.attack()
            with ac():
                loss_adv = model(**batch_data)[0]
            scaler.scale(loss_adv).backward()
            fgm.restore()

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.zero_grad()

        # f1 = model_evaluate(model,dev_loader,opt,device,epoch)
        if epoch+1==4:
            swa_model = AveragedModel(model)
        if epoch+1>=5:
            swa_model.update_parameters(model)

        # if f1>best_f1:
        #     best_f1 = f1
        #     save_model(opt, model, epoch,opt.model_name)
        #     if epoch>=4:
        #         save_model(opt, swa_model.module, epoch,opt.model_name)
        #     else:
        #         save_model(opt, model, epoch,opt.model_name)

    # save_model(opt, model, epoch,opt.model_name)
    # save_model(opt, swa_model.module, epoch,"swa_model.pt")
    save_model(opt, swa_model.module, epoch,opt.model_name)


if __name__ == '__main__':
    opt = ini_args()
    logger.info("lr={},seed={},train_batch_size={}".format(opt.lr,opt.seed,opt.train_batch_size))
    set_seed(opt.seed)
    train(opt)


