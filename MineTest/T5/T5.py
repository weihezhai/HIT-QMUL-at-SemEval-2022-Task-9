#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/1/5 10:34
# @Author  : hit-itnlp-fengmq
# @FileName: T5.py
# @Software: PyCharm

import json
import os
import random

import numpy as np
import pandas as  pd
import torch
import torch.nn as nn
# from sentence_transformers import util, SentenceTransformer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor, get_linear_schedule_with_warmup


# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


setup_seed(2022)


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
    def __init__(self, data_text, data_qa, tokenizer):
        self.data_text = data_text
        self.data_qa = data_qa
        self.tokenizer = tokenizer

        self.max_source_length = 1024
        self.max_target_length = 32

    def __getitem__(self, index):
        item = self.data_qa[index]
        question = item['question']
        answer = item['answer'].strip()
        doc_id = item['doc_id'].strip()

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


def get_dataloader(tokenizer, train_qa_path="train_qa2.json", train_text_path=r"train3.json",
                   val_qa_path="train_qa2.json", val_text_path=r"train3.json", batchsize=4):
    train_data_text = json.load(open(train_text_path, 'r', encoding='utf-8'))
    train_data_qa = json.load(open(train_qa_path, 'r', encoding='utf-8'))
    train_dataset = R2QADataset(train_data_text, train_data_qa, tokenizer)

    val_data_text = json.load(open(val_text_path, 'r', encoding='utf-8'))
    val_data_qa = json.load(open(val_qa_path, 'r', encoding='utf-8'))
    val_dataset = R2QADataset(val_data_text, val_data_qa, tokenizer)

    batch_size = batchsize
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size
    )
    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )
    print("dataloader load end!")
    return train_dataloader, validation_dataloader


def train(model, tokenizer, train_dataloader, validation_dataloader, epochs=5,
          save_path=r"/home/mqfeng/code/mytest/T5_only_train_dataset/save"):
    training_logs = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

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

    for epoch in range(epochs):
        model.train()
        train_loss = []
        for data in tqdm(train_dataloader):
            optimizer.zero_grad()
            data = tuple(t.to(device) for t in data)
            input_ids, attention_mask, labels = data
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = output.loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss.append(loss.item())

        # if epoch >= 2:
        #     s_path = save_path
        #     sub_path = os.path.join(s_path, "model" + str(epoch))
        #     os.mkdir(sub_path)
        #     if torch.cuda.device_count() > 1:
        #         model.module.save_pretrained(sub_path)
        #     else:
        #         model.save_pretrained(sub_path)

        model.eval()
        test_loss = []
        test_acc = []
        with torch.no_grad():
            for data in tqdm(validation_dataloader):
                data = tuple(t.to(device) for t in data)
                input_ids, attention_mask, labels = data
                if torch.cuda.device_count() > 1:
                    outputs = model.module.generate(input_ids)
                else:
                    outputs = model.generate(input_ids)
                output_all = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                test_loss.append(output_all.loss.mean().cpu())

                labels = [[i.item() for i in label if i != -100] for label in labels]
                decode_labels = tokenizer.decode(labels[0], skip_special_tokens=True)
                decode_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if decode_labels == decode_output:
                    test_acc.append(1)
                else:
                    test_acc.append(0)
        print("{}  Train_loss: {}----Test_loss:{} Test_acc:{}".format(epoch, np.mean(train_loss), np.mean(test_loss),
                                                                      np.mean(test_acc)))
        training_logs.append(
            {
                'epoch': epoch + 1,
                'Training Loss': np.mean(train_loss),

                'Valid. Loss': np.mean(test_loss),
                'Valid. Acc': np.mean(test_acc),
            }
        )
    return training_logs


def train_summary(data):
    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data)
    df_stats = df_stats.set_index('epoch')
    df_stats.to_csv('train_summary.csv', sep='\t')


batchsize = 4
# model 路径
model, tokenizer = get_premodel(r"/home/mqfeng/pretrainModel/t5-base")
# 训练集和验证集路径
train_dataloader, validation_dataloader = get_dataloader(tokenizer,
                                                         train_qa_path="train_qa.json",
                                                         train_text_path=r"train3.json",
                                                         val_qa_path="val_qa.json",
                                                         val_text_path=r"val3.json",
                                                         batchsize=batchsize)
# 模型保存路径                                                         
training_logs = train(model, tokenizer, train_dataloader, validation_dataloader,
                      save_path=r"/home/mqfeng/code/mytest/T5/save")
train_summary(training_logs)
