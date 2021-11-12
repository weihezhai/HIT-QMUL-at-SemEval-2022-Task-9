#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/11 13:47
# @Author  : hit-itnlp-fengmq
# @FileName: event.py
# @Software: PyCharm
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import get_linear_schedule_with_warmup

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
    text_ids = [item['corpus_id'] for item in top_results[0]]
    # 还原顺序
    text_ids.sort()
    result = []
    for id in text_ids:
        result.append(text[id].strip())
    return result


def split_data(path=r'text_data.json', fold=0):
    '''
    :param path: text_data.json的路径
    :param fold: 第几折，默认总共五折交叉
    :return: 返回划分好的训练集与验证集,是一个list，list的元素为一个字典，包含{qa：''，text：''}，其中qa以四个空格分割，q,a=qa.split("     ")
    '''

    Sbert = SentenceTransformer(r'/home/mqfeng/preModels/all-MiniLM-L6-v2')

    f = open(path, 'r', encoding='utf-8')
    json_data = json.load(f)
    f.close()

    datas = []
    for key, value in tqdm(json_data.items()):

        questions = value['question']
        text = value['text']
        for question in questions:
            ques, answer = question.split("     ")

            if answer.lower() == "the first event" or answer.lower() == "the second event":
                data = {}
                data['question'] = ques
                data['answer'] = answer
                result_text = scores(ques, text, Sbert, top_k=4)
                data['text'] = "".join(result_text)
                datas.append(data)

    num_data = len(datas)
    num_test = num_data // 5
    random.shuffle(datas)
    test = datas[fold * num_test:(fold + 1) * num_test]
    if fold == 0:
        train = datas[num_test:]
    else:
        train = datas[:num_test * fold]
        train.extend(datas[num_test * (fold + 1):])

    print("nums of train data:{}".format(len(train)))
    print("nums of val data:{}".format(len(test)))
    print("data split end")
    del Sbert
    return train, test


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
                                            max_length=450, padding='max_length',
                                            return_attention_mask=True, return_tensors='pt',
                                            truncation=True)
        input_ids, token_type_ids, attention_mask = encode['input_ids'], encode['token_type_ids'], encode[
            'attention_mask']
        if a == 'the first event':
            answer = torch.tensor([0])
        else:
            answer = torch.tensor([1])
        return input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze(), answer

    def __len__(self):
        return len(self.data)


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


def train(model, train_dataloader, testdataloader, device, epoch=3):
    model.to(device)
    optim = AdamW(model.parameters(), lr=1e-5, weight_decay=0.2)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=200, num_training_steps=len(train_dataloader) * epoch)

    for epoch in range(epoch):
        model.train()
        preds = []
        ys = []
        loss_train = []
        loop = tqdm(train_dataloader, leave=True)
        for data in tqdm(loop):
            optim.zero_grad()
            data = tuple(t.to(device) for t in data)
            input_ids, token_type_ids, attention_mask, answer = data
            pred = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(pred, answer.float())

            loss.backward()
            optim.step()
            scheduler.step()

            loss_train.append(loss.item())
            preds.extend(torch.round(torch.sigmoid(pred.squeeze(1))).int().tolist())
            ys.extend(answer.squeeze(1).tolist())

        acc = accuracy_score(ys, preds)
        prec = precision_score(ys, preds)
        recall = recall_score(ys, preds)
        f1 = f1_score(ys, preds)
        print("train loss:{}  acc:{} prec:{}  recall:{} f1:{}".format(np.mean(loss_train), acc, prec, recall, f1))

        model.eval()
        preds_t = []
        ys_t = []
        loss_t = []
        for data in tqdm(testdataloader):
            with torch.no_grad():
                data = tuple(t.to(device) for t in data)
                input_ids, token_type_ids, attention_mask, answer = data
                pred = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                loss = criterion(pred, answer.float())


                loss_t.append(loss.item())
                preds_t.extend(torch.round(torch.sigmoid(pred.squeeze(1))).int().tolist())
                ys_t.extend(answer.squeeze(1).tolist())
        acc_t = accuracy_score(ys_t, preds_t)
        prec_t = precision_score(ys_t, preds_t)
        recall_t = recall_score(ys_t, preds_t)
        f1_t = f1_score(ys_t, preds_t)
        print("test loss:{}  acc:{} prec:{}  recall:{} f1:{}".format(np.mean(loss_t), acc_t, prec_t, recall_t, f1_t))
        print("\n")


model, tokenizer = get_premodel(r'/home/mqfeng/R2QA/pretrain_recipeQA/large/epoch2')
myModel = MyModel(model.albert)
train_data, test_data = split_data(r'text_data.json')
train_Dataset = myDataset(train_data, tokenizer)
test_Dataset = myDataset(test_data, tokenizer)
train_Dataloader = DataLoader(train_Dataset, batch_size=4, shuffle=True)
test_Dataloader = DataLoader(test_Dataset, batch_size=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(myModel, train_Dataloader, test_Dataloader, device, epoch=5)
