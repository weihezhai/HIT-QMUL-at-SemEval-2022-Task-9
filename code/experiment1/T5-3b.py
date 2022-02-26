#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/5 10:34
# @Author  : hit-itnlp-fengmq
# @FileName: T5.py
# @Software: PyCharm

import json
import os
import random
from argparse import ArgumentParser
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from sentence_transformers import util, SentenceTransformer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor, get_linear_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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


class R2QADataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, data_text, data_text_entity, data_qa, tokenizer):
        self.data_text = data_text
        self.data_text_entity = data_text_entity
        self.data_qa = data_qa
        self.tokenizer = tokenizer

        self.max_source_length = 1024
        self.max_target_length = 45

    def __getitem__(self, index):
        item = self.data_qa[index]
        question = item['question']
        answer = item['answer'].strip()
        doc_id = item['doc_id'].strip()

        if 'how many' in question:
            texts = self.data_text_entity[doc_id]
        else:
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
        # 答案
        target_encoding = self.tokenizer.encode_plus(answer,
                                                     add_special_tokens=True,
                                                     max_length=self.max_target_length,
                                                     padding='max_length',
                                                     return_attention_mask=True,
                                                     truncation=True)
        labels = target_encoding.input_ids
        labels = [(label if label != self.tokenizer.pad_token_id else -100) for label in labels]
        labels = torch.tensor(labels)

        return input_ids.squeeze(), attention_mask.squeeze(), labels

    def __len__(self):
        return len(self.data_qa)


def get_dataloader(tokenizer, train_qa_path="train_qa2.json", train_text_path=r"train3.json", train_text_path_entity="",
                   val_qa_path="train_qa2.json", val_text_path=r"train3.json", val_text_path_entity="", batchsize=4):
    train_data_text = json.load(open(train_text_path, 'r', encoding='utf-8'))
    train_data_text_entity = json.load(open(train_text_path_entity, 'r', encoding='utf-8'))
    train_data_qa = json.load(open(train_qa_path, 'r', encoding='utf-8'))
    train_dataset = R2QADataset(train_data_text,train_data_text_entity, train_data_qa, tokenizer)

    val_data_text = json.load(open(val_text_path, 'r', encoding='utf-8'))
    val_data_text_entity = json.load(open(val_text_path_entity, 'r', encoding='utf-8'))
    val_data_qa = json.load(open(val_qa_path, 'r', encoding='utf-8'))
    val_dataset = R2QADataset(val_data_text, val_data_text_entity,val_data_qa, tokenizer)

    batch_size = batchsize
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size
    )
    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=1
    )
    print("dataloader load end!")
    return train_dataloader, validation_dataloader

def eval_val(model, tokenizer,validation_dataloader, device):
    model.eval()
    test_loss = []
    test_acc = []
    with torch.no_grad():
        for data in tqdm(validation_dataloader):
            data = tuple(t.to(device) for t in data)
            input_ids, attention_mask, labels = data
            if torch.cuda.device_count() > 1:
                outputs = model.generate(input_ids, max_length=45)
            else:
                outputs = model.generate(input_ids, max_length=45)
            output_all = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            test_loss.append(output_all.loss.mean().cpu().item())

            labels = [[i.item() for i in label if i != -100] for label in labels]
            decode_labels = tokenizer.decode(labels[0], skip_special_tokens=True)
            decode_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if decode_labels == decode_output:
                test_acc.append(1)
            else:
                test_acc.append(0)
    return np.mean(test_loss),np.mean(test_acc)
def train(model, tokenizer, train_dataloader, validation_dataloader, device_map,epochs=10,
          save_path=r"/home/mqfeng/code/mytest/T5_only_train_dataset/save"):
    # training_logs = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
    # model.parallelize(device_map)
    model.to(device)
   # model.parallelize(device_map)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = nn.DataParallel(model)

    optimizer = Adafactor(model.parameters(),
                          lr=2e-4,
                          eps=(1e-30, 1e-3),
                          clip_threshold=1.0,
                          decay_rate=-0.8,
                          beta1=None,
                          weight_decay=0.0,
                          relative_step=False,
                          scale_parameter=False,
                          warmup_init=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=400, num_training_steps=len(train_dataloader) * epochs)

    logs_dict = {
        "val_acc": [],
        "val_loss": []}
    step = 0
    best_val_acc = 0
    for epoch in range(epochs):
        model.parallelize(device_map)

        for data in tqdm(train_dataloader):
            model.train()
            optimizer.zero_grad()
            data = tuple(t.to(device) for t in data)
            input_ids, attention_mask, labels = data
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = output.loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            step += 1
            if step % 1000 == 0:
                val_loss, val_acc = eval_val(model, tokenizer, validation_dataloader, device)

                logs_dict['val_loss'].append(val_loss)
                logs_dict['val_acc'].append(val_acc)
                # 保存acc和loss的json文件
                w = open('loss.json', "w", encoding="utf-8")
                json.dump(logs_dict, w, ensure_ascii=False, indent=4)
                w.close()

                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                    sub_path = save_path
                    if not os.path.exists(sub_path):
                        os.mkdir(sub_path)

                    if torch.cuda.device_count() > 1:
                        model.save_pretrained(sub_path)
                        tokenizer.save_pretrained(sub_path)
                    else:
                        model.save_pretrained(sub_path)
                        tokenizer.save_pretrained(sub_path)


# device_map = {
#     0: [0, 1, 2,3, 4, 5,6, 7, 8, 9,10,11],
#     1: [12,13,14,15,16,17,18,19,20,21,22,23]
#
# }

device_map = {
    0: [0, 1, 2],
    1: [3, 4, 5, 6, 7, 8, 9],
    2: [10, 11, 12, 13, 14, 15, 16],
    3: [17, 18, 19, 20, 21, 22, 23],
}
batchsize = 4
# model 路径
model, tokenizer = get_premodel(r'/data/scratch/acw664/T5-3b')
#model.parallelize(device_map)
# 训练集和验证集路径
train_dataloader, validation_dataloader = get_dataloader(tokenizer,
                                                         train_qa_path="/data/scratch/acw664/T5-3b/MineTest/utils/data/train_qa.json",
                                                         train_text_path_entity=r"/data/scratch/acw664/T5-3b/MineTest/utils/entity/train3.json",
                                                         train_text_path=r"/data/scratch/acw664/T5-3b/MineTest/utils/data01/train3.json",
                                                         val_qa_path="/data/scratch/acw664/T5-3b/MineTest/utils/data/val_qa.json",
                                                         val_text_path=r"/data/scratch/acw664/T5-3b/MineTest/utils/data01/val3.json",
                                                         val_text_path_entity=r"/data/scratch/acw664/T5-3b/MineTest/utils/entity/val3.json",
                                                         batchsize=batchsize
                                                         )
# 模型保存路径                                                         
train(model, tokenizer, train_dataloader, validation_dataloader,device_map,
                      save_path=r'save',epochs=10)

