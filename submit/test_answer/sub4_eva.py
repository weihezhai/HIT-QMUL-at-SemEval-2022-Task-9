#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/14 19:20
# @Author  : hit-itnlp-fengmq
# @FileName: sub4_eva.py
# @Software: PyCharm

# 1080 机器上
# /home/mqfeng/R2QA/Baseline/save 路径下
# 保存的模型和测试集代码

import json
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


def scores(question: str, text: list, model, top_k: int = 4):
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
    text_ids = [item['corpus_id'] for item in top_results[0]]
    # 还原顺序
    text_ids.sort()
    result = []
    for id in text_ids:
        result.append(text[id].strip())
    return result


class MyModel(nn.Module):
    def __init__(self, albert):
        super().__init__()
        self.albert = albert
        self.linear1 = nn.Linear(1024, 1)
        self.attention = nn.Linear(1024, 1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        albert_output = self.albert(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
        last_hidden_state = albert_output.last_hidden_state  # torch.Size([batchSize, 512, 768]) 512与max_length相关
        cls = last_hidden_state[:, 0]

        all_tokens = last_hidden_state[:, 1:]
        temp = self.attention(all_tokens.transpose(1, 0)).transpose(0, 1)
        weight = torch.softmax(temp, 1)
        vec = torch.sum(weight * all_tokens, 1)
        y1 = self.linear1(torch.relu(cls + vec))
        return y1


def get_premodel(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path)  # Embedding(30000, 128) [MASK]:4
    print("model load end!")
    return model, tokenizer


def get_text_from_docid(docid: str):
    with open("test_all_text_tag.json") as f:
        texts = json.load(f)
    text = []
    for item in texts[docid]['text_tags']:
        text.append(item["text"])
    return text


model, tokenizer = get_premodel(path=r'/home/mqfeng/R2QA/pretrain_recipeQA/large/epoch2')
myModel = MyModel(model.albert)
myModel.load_state_dict(torch.load(r'/home/mqfeng/R2QA/Baseline/save/save3/model_2.pth'))
Sbert = SentenceTransformer(r'/home/mqfeng/preModels/all-MiniLM-L6-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
myModel.to(device)

with open("/home/mqfeng/R2QA/Baseline/save/start4.json") as f:
    data4 = json.load(f)
result = {}
with torch.no_grad():
    for item in tqdm(data4):
        q_id = item['q_id'].strip()
        question = item['question'].strip()
        doc_id = item['doc_id'].strip()
        texts = get_text_from_docid(doc_id)
        texts = " ".join(scores(question, texts, Sbert))

        encode = tokenizer.encode_plus(question, texts, add_special_tokens=True,
                                       max_length=512, padding='max_length',
                                       return_attention_mask=True, return_tensors='pt',
                                       truncation=True)
        input_ids, token_type_ids, attention_mask = encode['input_ids'], encode['token_type_ids'], encode[
            'attention_mask']

        pred = myModel(input_ids.to(device), token_type_ids=token_type_ids.to(device),
                       attention_mask=attention_mask.to(device))
        pred = torch.round(torch.sigmoid(pred.squeeze(1))).int()
        if pred == 0:
            ans = "the first event"
        else:
            ans = "the second event"
        if doc_id not in result.keys():
            result[doc_id] = {}
        result[doc_id][q_id] = ans
fw = open("sub4_ans.json", "w", encoding="utf-8")
json.dump(result, fw, ensure_ascii=False, indent=4)
fw.close()
