#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/10/29 10:53
# @Author  : hit-itnlp-fengmq
# @FileName: pretrain.py
# @Software: PyCharm
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, WEIGHTS_NAME, CONFIG_NAME
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# 加载模型
def get_premodel(path=r"/home/mqfeng/preModels/albert-base-v2"):
    '''
    :param path: 预训练模型在本机的路径
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path)#Embedding(30000, 128) [MASK]:4
    print("model load end!")
    return model, tokenizer



class myDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, data_path, tokenizer):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        data = self.data[index].strip('\n')
        encode = self.tokenizer.encode_plus(data, add_special_tokens=True,
                                            max_length=512, padding='max_length',
                                            return_attention_mask=True, return_tensors='pt',
                                            truncation=True)
        label_input_ids, token_type_ids, attention_mask = encode['input_ids'], encode['token_type_ids'], encode[
            'attention_mask']
        input_ids ,masked_mlm= self.Masked(label_input_ids,attention_mask)
        return input_ids, token_type_ids[0], attention_mask[0], label_input_ids[0],masked_mlm

    def __len__(self):
        return len(self.data)

    def Masked(self, input_ids, attention_mask):
        masked_ids = input_ids.clone().detach()[0]
        masked_mlm = torch.zeros_like(masked_ids)
        for i in range((attention_mask.sum())):
            if i == 0 or i == len(masked_ids) - 1:  # 确保CLS和Seq不会被MASK
                continue
            if random.random() <= 0.15:
                masked_mlm[i] = 1
                prop = random.random()
                if prop <= 0.8:
                    masked_ids[i] = 4
                elif prop <= 0.9:
                    masked_ids[i] = random.randint(5, 29999)
        return masked_ids, masked_mlm


def train(model, tokenizer, train_dataloader, device, epoch=3):
    model.to(device)
    optim = AdamW(model.parameters(), lr=1e-5, weight_decay=0.2)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=400, num_training_steps=len(train_dataloader) * epoch)
    criterion2 = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epoch):
        train_acc = []
        train_loss = []
        loop = tqdm(train_dataloader, leave=True)
        for data in tqdm(loop):
            optim.zero_grad()
            data = tuple(t.to(device) for t in data)
            input_ids, token_type_ids, attention_mask, y,masked_mlm = data

            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=y)
            logits = outputs.logits
            active_loss = masked_mlm == 1
            active_labels = torch.where(
                active_loss.view((-1)), y.view(-1), torch.tensor(criterion2.ignore_index).type_as(y))
            loss = criterion2(logits.view(-1, 30000), active_labels)
            loss.backward()
            optim.step()
            scheduler.step()

            train_loss.append(loss.item())
            temp = (torch.argmax(logits[masked_mlm.bool()],1)==y[masked_mlm.bool()]).int().tolist()
            train_acc.extend(temp)

            loop.set_description(f'Epoch:{epoch}')
            loop.set_postfix(loss=loss.item(),acc=np.mean(temp))
        print("loss:{} acc:{}".format(np.mean(train_loss),np.mean(train_acc)))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_path = r"epoch" + str(epoch)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        output_model_file = os.path.join(model_path, WEIGHTS_NAME)
        output_config_file = os.path.join(model_path, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        print("epoch {} saved!".format(epoch))
model, tokenizer = get_premodel(r'/home/hwzhai/code/albert-large-v2')
dataset = myDataset(r"recipe_corpora.txt", tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
# device = torch.device( 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE:{}".format(device))
train(model,tokenizer, dataloader, device, epoch=3)