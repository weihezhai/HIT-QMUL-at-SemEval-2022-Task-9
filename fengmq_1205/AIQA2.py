#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/12/11 19:11
# @Author  : hit-itnlp-fengmq
# @FileName: AIQA2.py
# @Software: PyCharm

import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import get_linear_schedule_with_warmup
from collections import namedtuple
from sentence_transformers import SentenceTransformer, util
from transformers import AlbertTokenizer, AlbertForQuestionAnswering



# base
# 1e-5
# 随机数种子
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(2021)
# 加载模型
def get_premodel(path=r"D:\Anaconda\learn\_Bert\pre_train_model\albert-base-v2"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AlbertForQuestionAnswering.from_pretrained(path)
    print("model load end!")
    return model, tokenizer


def split_data(path=r'EM_all_data.json', fold=0):
    f = open(path, 'r', encoding='utf-8')
    json_data = json.load(f)
    f.close()

    datas = []
    for item in json_data:
        qa, texts = item["qa"], item["text"]
        qa['quest'] = qa['quest'].split("=")[1].strip().lower()
        qa['answer'] = qa['answer'].strip().lower()
        text = ""
        for tt in texts:
            text += tt["text"]
        text = text.strip().lower()
        temp_dict = {}
        temp_dict['qa'] = qa
        temp_dict['text'] = text
        datas.append(temp_dict)

    random.shuffle(datas)
    num_data = len(datas)
    num_test = num_data // 5
    test = datas[fold * num_test:(fold + 1) * num_test]
    if fold == 0:
        train = datas[num_test:]
    else:
        train = datas[:num_test * fold]
        train.extend(datas[num_test * (fold + 1):])
    print("split data end !")

    print("nums of train data:{}".format(len(train)))
    print("nums of val data:{}".format(len(test)))
    return train, test


class myDataset(Dataset):  # 需要继承data.Dataset
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item = self.data[index]
        qa = item["qa"]
        text = item["text"]
        q, a = qa["quest"], qa["answer"]
        encode = self.tokenizer.encode_plus(q, text, add_special_tokens=True,
                                            max_length=512,
                                            padding='max_length',
                                            return_attention_mask=True,
                                            return_tensors='pt',
                                            truncation=True)
        input_ids, token_type_ids, attention_mask = encode['input_ids'], encode['token_type_ids'], encode[
            'attention_mask']
        # 获取起始位置
        start, end = self.start_end(a, q, text)
        return input_ids.squeeze(), token_type_ids.squeeze(), attention_mask.squeeze(), torch.tensor(
            start), torch.tensor(end), a

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


def train(model, tokenizer, train_dataloader, testdataloader, device, fold=0, epoch=5):
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    optim = AdamW(model.parameters(), lr=1e-5, weight_decay=0.2)
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=300, num_training_steps=len(train_dataloader) * epoch)

    for epoch in range(epoch):
        model.train()
        train_acc = []
        train_loss = []
        loop = tqdm(train_dataloader, leave=True)
        for data in tqdm(loop):
            optim.zero_grad()
            # a = data[-1][0]
            data = tuple(t.to(device) for t in data[:-1])
            input_ids, token_type_ids, attention_mask, start, end = data

            outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            start_positions=start, end_positions=end)
            loss, start_logits, end_logits = outputs.loss, outputs.start_logits, outputs.end_logits
            loss = loss.mean()
            loss.backward()
            optim.step()
            scheduler.step()

            start_pred = torch.argmax(start_logits, dim=1)
            end_pred = torch.argmax(end_logits, dim=1)
            starts = (np.array((start_pred == start).cpu()))
            ends = (np.array((end_pred == end).cpu()))
            acc = np.all(np.array([starts, ends]).T, axis=-1).astype(int)
            train_acc.extend(acc)
            train_loss.append(loss.item())

            loop.set_description(f'fold:{fold}  Epoch:{epoch}')
            loop.set_postfix(loss=loss.item(), acc=acc)
        # if epoch >=3:
        #     model_to_save = model.module if hasattr(model, 'module') else model
        #     model_path = r"fold" + str(fold) + "_epoch" + str(epoch)
        #     if not os.path.exists(model_path):
        #         os.makedirs(model_path)
        #     torch.save(model_to_save, os.path.join(model_path,'model.pt'))

        model.eval()
        test_loss = []
        test_acc = []
        for data in tqdm(testdataloader):
            with torch.no_grad():

                a = data[-1][0]

                data = tuple(t.to(device) for t in data[:-1])
                input_ids, token_type_ids, attention_mask, start, end = data
                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                start_positions=start, end_positions=end)

                loss = outputs.loss.mean()
                start_pred = torch.argmax(outputs.start_logits, dim=1)
                end_pred = torch.argmax(outputs.end_logits, dim=1)
                pred_ans = (tokenizer.decode(input_ids[0][start_pred:end_pred + 1]))
                if a == pred_ans:
                    test_acc.append(1)
                else:
                    test_acc.append(0)
                test_loss.append(loss.item())
        print("{}, Train_acc:{} Train_loss:{}-----Val_acc:{}  Val_loss:{}".format(
            epoch, np.mean(train_acc), np.mean(train_loss), np.mean(test_acc),
            np.mean(test_loss)))


for fold in range(1):
    model, tokenizer = get_premodel()
    with open(r'QAhead_base.pickle', 'rb') as file:
        QAhead = pickle.load(file)
    myModel = MyModel(model.albert, QAhead)

    train_data, test_data = split_data("EM_all_data.json")

    # 构造DataSet和DataLoader
    train_Dataset = myDataset(train_data, tokenizer)
    test_Dataset = myDataset(test_data, tokenizer)
    # 修改batchsize
    train_Dataloader = DataLoader(train_Dataset, batch_size=2, shuffle=True)
    test_Dataloader = DataLoader(test_Dataset, batch_size=1)
    # 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    train(myModel, tokenizer, train_Dataloader, test_Dataloader, device, fold, epoch=5)

    model.save_pretrained()