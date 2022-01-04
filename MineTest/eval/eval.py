#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/4 19:00
# @Author  : hit-itnlp-fengmq
# @FileName: eval.py
# @Software: PyCharm
import json
import random
import os
import numpy as np
import torch
# from sentence_transformers import util, SentenceTransformer
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


setup_seed(21)


# 加载模型
def get_premodel(path=r"D:\Anaconda\learn\_Bert\pre_train_model\t5-small"):
    '''
    :param path: 预训练模型在本机的路径
    :return:
    '''
    tokenizer = T5Tokenizer.from_pretrained(path)
    model = T5ForConditionalGeneration.from_pretrained(path)
    print("model load end!")
    return model, tokenizer


class myDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, data_text, data_qa, tokenizer):
        self.data_text = data_text
        self.data_qa = data_qa
        self.tokenizer = tokenizer

        self.max_source_length = 1024

    def __getitem__(self, index):
        item = self.data_qa[index]
        question = item['question']
        doc_id = item['doc_id'].strip()
        q_id = item['q_id']

        texts = self.data_text[doc_id]
        text = "".join(texts)

        q_text = question + " : " + text
        source_encoding = self.tokenizer.encode_plus(q_text,
                                                     add_special_tokens=True,
                                                     max_length=self.max_source_length,
                                                     padding='max_length',
                                                     return_attention_mask=True,
                                                     return_tensors='pt',
                                                     truncation=True)
        input_ids, attention_mask = source_encoding['input_ids'], source_encoding['attention_mask']
        return input_ids.squeeze(), doc_id, q_id

    def __len__(self):
        return len(self.data_qa)


def get_dataloader(tokenizer, qa_path="test_qa.json", text_path=r"test3.json", batchsize=4):
    data_text = json.load(open(text_path, 'r', encoding='utf-8'))
    data_qa = json.load(open(qa_path, 'r', encoding='utf-8'))
    dataset = myDataset(data_text, data_qa, tokenizer)

    batch_size = batchsize
    dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=SequentialSampler(dataset),  # Select batches randomly
        batch_size=batch_size
    )
    print("dataloader load end!")
    return dataloader


def eval(model, tokenizer, test_dataloader):
    result = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            input_ids, doc_id, q_id = data
            doc_id, q_id = doc_id[0], q_id[0]
            input_ids = input_ids.to(device)
            outputs = model.generate(input_ids)
            decode_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if doc_id not in result.keys():
                result[doc_id] = {}
            result[doc_id][q_id] = decode_output

    w = open("r2vq_pred.json", "w", encoding="utf-8")
    json.dump(result, w, ensure_ascii=False, indent=4)
    w.close()

# eval的时候只能设置为1
batchsize = 1
model, tokenizer = get_premodel(r"/home/mqfeng/code/mytest/T5/save/model2")
test_dataloader = get_dataloader(tokenizer, qa_path="test_qa.json", text_path=r"test3.json", batchsize=batchsize)
eval(model, tokenizer, test_dataloader)