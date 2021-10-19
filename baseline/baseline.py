#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/19 22:24
# @Author  : fengmq
# @FileName: baseline.py
# @Software: PyCharm

# 按照知乎代码微调  参考：https://zhuanlan.zhihu.com/p/357528657

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import numpy as np
import random
import json
from torch.utils.data import Dataset, DataLoader

# 随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(2021)

# 加载模型
def get_premodel(path=r"E:\RecipeQA\transformers_models\bert-large-uncased-whole-word-masking-finetuned-squad"):
    '''
    :param path: 预训练模型在本机的路径  下载链接（https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/tree/main）
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForQuestionAnswering.from_pretrained(path)
    print("model load end!")
    return model,tokenizer

# Dataset
def split_data(path=r'text_data.json',fold=0):
    '''
    总共1000个食谱，10202条问答对
    :param path: text_data.json的路径
    :param fold: 第几折，默认总共五折交叉
    :return: 返回划分好的训练集与验证集,是一个list，list的元素为一个字典，包含{qa：''，text：''}，其中qa以四个空格分割，q,a=qa.split("     ")
    '''
    datas=[]
    f = open(path, 'r', encoding='utf-8')
    json_data = json.load(f)
    f.close()

    for key in json_data:
        item = json_data[key]
        question = item['question']
        text = item['text']
        for qa in question:
            q,a=qa.split("     ")
            if a in text:
                data={}
                data['qa']=qa
                data['text'] = text
                datas.append(data)
    num_data = len(datas)
    num_test = num_data//5
    random.shuffle(datas)

    test = datas[fold * num_test:(fold + 1) * num_test]
    if fold == 0:
        train = datas[num_test:]
    else:
        train = datas[:num_test * fold]
        train.extend(datas[num_test * (fold + 1):])
    return train, test


class myDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, data,tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item = self.data[index]
        qa,text = item['qa'],item['text']
        q, a = qa.split("     ")
        encode= tokenizer(q, text, add_special_tokens=True, return_tensors="pt")
        input_ids, token_type_ids, token_type_ids = encode['input_ids'],encode['token_type_ids'],encode['token_type_ids']
        return  input_ids.squeeze(), token_type_ids.squeeze(), token_type_ids.squeeze()

    def __len__(self):
        len(self.data)

def train(model, train_dataloader, testdataloader,device,fold):
    pass

for fold in range(5):
    model,tokenizer = get_premodel()

    train_data, test_data = split_data(r"E:\RecipeQA\data\code\text_data.json",fold)
    train_Dataset = myDataset(train_data,tokenizer)
    test_Dataset = myDataset(test_data,tokenizer)

    train_Dataloader = DataLoader(train_Dataset, batch_size=8)
    test_Dataloader = DataLoader(test_Dataset, batch_size=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(model,train_Dataloader,test_Dataloader,device,fold)



