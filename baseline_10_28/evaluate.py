#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/11 10:39
# @Author  : hit-itnlp-fengmq
# @FileName: evaluate.py
# @Software: PyCharm
'''
针对完全匹配和N/A问题的准确率计算

'''#
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AdamW, WEIGHTS_NAME, CONFIG_NAME
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import get_linear_schedule_with_warmup
from collections import namedtuple
from sentence_transformers import SentenceTransformer, util
# 随机数种子
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(2021)


# 加载模型
def get_premodel(path=r"D:\Anaconda\learn\_Bert\pre_train_model\albert-base-v2"):
    '''
    :param path: 预训练模型在本机的路径
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path)  # Embedding(30000, 128) [MASK]:4
    print("model load end!")
    return model, tokenizer
class myDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item = self.data[index]
        q = item['question']
        a = item['answer']
        text = item['text']
        # encode= self.tokenizer(q, text, add_special_tokens=True, return_tensors="pt")
        encode = self.tokenizer.encode_plus(q.strip().lower(), text.strip().lower(), add_special_tokens=True,
                                            max_length=512, padding='max_length',
                                            return_attention_mask=True, return_tensors='pt',
                                            truncation=True)
        input_ids, token_type_ids, attention_mask = encode['input_ids'], encode['token_type_ids'], encode[
            'attention_mask']
        # 获取起始位置
        start, end = self.start_end(a, q, text)
        return input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze(), torch.tensor(
            start), torch.tensor(end)

    def __len__(self):
        return len(self.data)

    def start_end(self, answer, q, text):
        # 有问题
        answer_encode = self.tokenizer(answer.strip().lower())['input_ids'][1:-1]
        text_encode = self.tokenizer(q.strip().lower(), text.strip().lower())['input_ids']
        start_end = ()
        for i in range(len(text_encode)):
            if text_encode[i] == answer_encode[0]:
                j = 0
                for j in range(len(answer_encode)):
                    if text_encode[i + j] != answer_encode[j]:
                        break
                if j == len(answer_encode) - 1:
                    if text_encode[i + j] != answer_encode[j]:
                        continue
                    start_end = (i, i + len(answer_encode) - 1)
            if len(start_end) != 0:
                return start_end
        if len(start_end) == 0:
            start_end = (0, 0)
        return start_end

class MyModel(nn.Module):
    def __init__(self, albert, QAhead):
        super().__init__()
        self.albert = albert

        self.qa_outputs = QAhead  # self.qa_outputs = nn.Linear(768, 2,bias=True)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None):
        albert_output = self.albert(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
        last_hidden_state = albert_output.last_hidden_state  # torch.Size([batchSize, 512, 768]) 512与max_length相关
        logits = self.qa_outputs(last_hidden_state)  # torch.Size([batchSize, 512, 2])
        start_logits, end_logits = logits.split(1, dim=-1)  # 分离出的start_logits/end_logits形状为([batchSize, 512, 1])
        start_logits = start_logits.squeeze(-1)  # ([batchSize, 512])
        end_logits = end_logits.squeeze(-1)  # ([batchSize, 512])

        Outputs = namedtuple('Outputs', 'start_logits, end_logits')
        outputs = Outputs(start_logits, end_logits)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            Outputs = namedtuple('Outputs', 'loss,start_logits, end_logits')
            outputs = Outputs(total_loss, start_logits, end_logits)
        return outputs
def evaluate_match(model,testdataloader,device):
    model.eval()
    test_acc2 = []
    for data in tqdm(testdataloader):
        with torch.no_grad():
            data = tuple(t.to(device) for t in data)
            input_ids, token_type_ids, attention_mask, start, end = data
            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            start_positions=start, end_positions=end)
            start_pred = torch.argmax(outputs.start_logits, dim=1)
            end_pred = torch.argmax(outputs.end_logits, dim=1)



            starts = (np.array((start_pred == start).cpu()))
            ends = (np.array((end_pred == end).cpu()))
            acc2 = np.all(np.array([starts, ends]).T, axis=-1).astype(int)
            test_acc2.extend(acc2)
    print(len(test_acc2))
    return test_acc2
def evaluate_na(Sbert,data):
    def scores(question: str, text: list, model, top_k: int = 2):
        text_embeddings = model.encode(text, convert_to_tensor=True)
        query_embedding = model.encode(question, convert_to_tensor=True)

        if torch.cuda.is_available():
            text_embeddings.to('cuda')
            query_embedding = torch.unsqueeze(query_embedding, dim=0)
            query_embedding.to('cuda')
            text_embeddings = util.normalize_embeddings(text_embeddings)
            query_embedding = util.normalize_embeddings(query_embedding)
        top_results = util.semantic_search(query_embedding, text_embeddings,
                                           top_k=top_k)  # [[{corpus_id:,score:},{},{}]]
        top_results = [item['score'] for item in top_results[0]]

        return np.mean(top_results)
    result=[]
    for i in tqdm(data):
        text=i['text']
        ques=i['question'].strip("")
        score = scores(ques, text, Sbert)
        if score<0.58:
            result.append(1)
        else:
            result.append(0)
    print("len na::", len(result))
    return result


_, tokenizer = get_premodel(r'/home/mqfeng/R2QA/pretrain_recipeQA/large/epoch2')
with open(r'QAhead_large.pickle', 'rb') as file:
    QAhead = pickle.load(file)


model = torch.load(r'/home/mqfeng/R2QA/Baseline/fold0_epoch0/model.pt')
with open(r"subdata/test_data.json",'r',encoding='utf-8') as f:
    match = json.load(f)
test_Dataset = myDataset(match, tokenizer)
test_Dataloader = DataLoader(test_Dataset, batch_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
match_result=evaluate_match(model,test_Dataloader,device)

Sbert = SentenceTransformer(r'/home/mqfeng/preModels/all-MiniLM-L6-v2')
with open(r"subdata/na.json",'r',encoding='utf-8') as f:
    match = json.load(f)

na_result = evaluate_na(Sbert,match)

all = match_result+na_result

print('acc:',all.count(1)/len(all))
print(match_result)
print(na_result)
